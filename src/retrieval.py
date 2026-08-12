"""
retrieval.py
Handles chunking, embedding, indexing, and hybrid (vector + BM25) querying
of the knowledge base.

Embeddings: BAAI/bge-base-en-v1.5, called through the Hugging Face Inference
API (no local torch/sentence-transformers needed). BGE is asymmetric: queries
get a instruction prefix, documents don't -- see BGE_QUERY_PREFIX below.

Vector store: Chroma (persistent, local), cosine distance, embeddings computed
by us and passed in explicitly (bypassing Chroma's auto-embedding) so the
query/document prefix asymmetry is respected.

Hybrid search: vector similarity is fused with BM25 lexical scores via
Reciprocal Rank Fusion. This is specifically to catch "near-miss" chunks --
text that shares keywords with the query but answers a different question
(e.g. wfh_policy.md mentioning "remote" for a question about remote-hiring
zones, which is actually answered by remote_work_zones.md). Pure embedding
similarity can be fooled by this; BM25 alone can too; requiring rank
agreement between both is more robust than either one.
"""

import os
import re
import time
from collections import defaultdict

import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from rank_bm25 import BM25Okapi

from src.state import Evidence

load_dotenv()

# ---- Config ----
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "research_desk_kb_bge_base_en_v15_headerchunked"  # chunking scheme changed -- new collection so stale fixed-size chunks in the old one are never mixed in
CHUNK_SIZE = 150        # fallback word-count cap, only used if a single section is unexpectedly large
CHUNK_OVERLAP = 30      # words overlapping between fallback chunks

# Meta-documentation about the knowledge base itself, not knowledge-base
# content -- indexing these would let retrieval directly surface the planted
# flaws answer key, which defeats the point of the eval.
EXCLUDED_FILES = {"README.md", "planted_flaws.md"}

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_PROVIDER = "hf-inference"
EMBED_BATCH_SIZE = 16
# BGE is asymmetric: this instruction prefix is required on queries (not documents)
# for the model's contrastive training objective to produce good similarity scores.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MAX_EMBED_RETRIES = 3
EMBED_RETRY_BASE_DELAY = 2.0  # seconds; doubles each retry

RRF_K = 60                    # standard Reciprocal Rank Fusion smoothing constant
CANDIDATE_POOL_MULTIPLIER = 4  # each method fetches k * this many candidates before fusion


class RetrievalError(Exception):
    """Raised when the embedding API fails after all retries. Callers (the
    Researcher node) should catch this and route to an honest refusal rather
    than crash or silently return no evidence."""


# ---- BGE embedding client (HF Inference API) ----

class BGEEmbedder:
    def __init__(self):
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RetrievalError(
                "HF_TOKEN is not set. Add it to a local .env file "
                "(see .env.example) -- get a free token at https://huggingface.co/settings/tokens"
            )
        self._client = InferenceClient(model=EMBED_MODEL, provider=EMBED_PROVIDER, token=token)

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        last_err = None
        for attempt in range(1, MAX_EMBED_RETRIES + 1):
            try:
                result = self._client.feature_extraction(texts, normalize=True)
                return result.tolist()
            except (HfHubHTTPError, Exception) as err:  # transient: rate limit, cold-start, network
                last_err = err
                if attempt < MAX_EMBED_RETRIES:
                    delay = EMBED_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    time.sleep(delay)
        raise RetrievalError(
            f"BGE embedding call failed after {MAX_EMBED_RETRIES} attempts: {last_err}"
        ) from last_err

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunk text for indexing. No query prefix."""
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start:start + EMBED_BATCH_SIZE]
            embeddings.extend(self._embed_with_retry(batch))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query. Requires the BGE instruction prefix."""
        return self._embed_with_retry([BGE_QUERY_PREFIX + text])[0]


class _BGEChromaEmbeddingFunction(EmbeddingFunction):
    """Adapter so the Chroma collection has a well-formed embedding_function
    (used by Chroma for consistency checks). Indexing and querying in this
    module always pass precomputed embeddings explicitly and bypass this --
    it exists so the collection is never left without one."""

    def __init__(self, embedder: BGEEmbedder):
        self._embedder = embedder

    def __call__(self, input: Documents) -> Embeddings:
        return self._embedder.embed_documents(list(input))


_embedder = BGEEmbedder()

# ---- Chroma setup (persistent client, cosine distance, explicit embeddings) ----
client = chromadb.PersistentClient(path=CHROMA_DIR)
embedding_fn = _BGEChromaEmbeddingFunction(_embedder)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ---- BM25 side-index (rebuilt from whatever's currently in Chroma) ----
_bm25_index = None
_bm25_chunk_ids: list[str] = []


# Hybrid search — BM25 + Reciprocal Rank Fusion
def _rebuild_bm25_index():
    global _bm25_index, _bm25_chunk_ids
    stored = collection.get(include=["documents"])
    _bm25_chunk_ids = stored["ids"]
    if not _bm25_chunk_ids:
        _bm25_index = None
        return
    tokenized_corpus = [_tokenize(doc) for doc in stored["documents"]]
    _bm25_index = BM25Okapi(tokenized_corpus)


def _ensure_bm25_index():
    if _bm25_index is None and collection.count() > 0:
        _rebuild_bm25_index()


_HEADER_RE = re.compile(r"^#{2,3}\s+.*$", re.MULTILINE)


def _fixed_size_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Word-window fallback, only reached when a single header section is
    still too large to act as one chunk on its own."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap  # move forward with overlap
    return chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits on markdown ## / ### headers so each chunk is one self-contained
    policy sub-topic (e.g. "### 2.1 Sick Leave") instead of a fixed word
    window that has no idea where one topic ends and the next begins. The
    docs in docs/ are written specifically with clean, short sections, so
    this produces focused, on-topic chunks instead of dumping an entire
    document as a single "chunk" -- which is what a 400-word fixed window
    did on every doc here, since none of them exceeded ~400 words.
    Falls back to fixed-size word chunking only for the rare section that's
    still too large on its own (or a doc with no ##/### headers at all).
    """
    header_starts = [m.start() for m in _HEADER_RE.finditer(text)]
    if not header_starts:
        return _fixed_size_chunks(text, chunk_size, overlap)

    sections = []
    if header_starts[0] > 0:
        preamble = text[: header_starts[0]].strip()
        if preamble:
            sections.append(preamble)
    for start, end in zip(header_starts, header_starts[1:] + [len(text)]):
        section = text[start:end].strip()
        if section:
            sections.append(section)

    chunks = []
    for section in sections:
        if len(section.split()) <= chunk_size:
            chunks.append(section)
        else:
            chunks.extend(_fixed_size_chunks(section, chunk_size, overlap))
    return chunks


def build_index(force_rebuild: bool = False):
    """
    Reads all .md files from docs/, chunks them, embeds them with BGE, and
    indexes into Chroma. Idempotent: skips rebuilding if collection already
    has data, unless forced.
    """
    global collection

    existing_count = collection.count()
    if existing_count > 0 and not force_rebuild:
        print(f"Index already exists with {existing_count} chunks. Skipping rebuild.")
        _ensure_bm25_index()
        return

    if force_rebuild and existing_count > 0:
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    ids, documents, metadatas = [], [], []

    for filename in os.listdir(DOCS_DIR):
        if not filename.endswith(".md") or filename in EXCLUDED_FILES:
            continue

        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{idx}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "doc_id": filename,
                "chunk_id": chunk_id,
                "chunk_index": idx
            })

    if not ids:
        raise ValueError(f"No .md files found in {DOCS_DIR}")

    embeddings = _embedder.embed_documents(documents)
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    print(f"Indexed {len(ids)} chunks from {len(set(m['doc_id'] for m in metadatas))} documents.")

    _rebuild_bm25_index()


def _vector_search(query: str, pool_size: int) -> list[tuple[str, str, dict, float]]:
    """Returns (chunk_id, text, metadata, similarity) ranked best-first."""
    query_embedding = _embedder.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=pool_size)

    if not results["ids"][0]:
        return []

    out = []
    for chunk_id, doc_text, meta, dist in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        similarity = 1 - dist  # cosine space: distance = 1 - cosine_similarity
        out.append((chunk_id, doc_text, meta, similarity))
    return out


def _bm25_search(query: str, pool_size: int) -> list[tuple[str, str, dict, float]]:
    """Returns (chunk_id, text, metadata, bm25_score) ranked best-first."""
    _ensure_bm25_index()
    if _bm25_index is None:
        return []

    scores = _bm25_index.get_scores(_tokenize(query))
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:pool_size]

    stored = collection.get(ids=[_bm25_chunk_ids[i] for i in ranked_idx], include=["documents", "metadatas"])
    by_id = {cid: (doc, meta) for cid, doc, meta in zip(stored["ids"], stored["documents"], stored["metadatas"])}

    out = []
    for i in ranked_idx:
        chunk_id = _bm25_chunk_ids[i]
        if scores[i] <= 0 or chunk_id not in by_id:
            continue
        doc_text, meta = by_id[chunk_id]
        out.append((chunk_id, doc_text, meta, float(scores[i])))
    return out


def _reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, str, dict, float]]], k: int
) -> list[Evidence]:
    """Fuses multiple ranked result lists by rank (not raw score, since vector
    similarity and BM25 scores live on different scales). A chunk ranked
    highly by both methods outranks one only one method liked -- this is what
    filters out keyword-only near-misses and embedding-only false positives."""
    fused_scores: dict[str, float] = defaultdict(float)
    chunk_data: dict[str, tuple[str, dict]] = {}

    for ranked in ranked_lists:
        for rank, (chunk_id, doc_text, meta, _score) in enumerate(ranked, start=1):
            fused_scores[chunk_id] += 1.0 / (RRF_K + rank)
            chunk_data.setdefault(chunk_id, (doc_text, meta))

    ranked_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[:k]

    evidence_list = []
    for chunk_id in ranked_ids:
        doc_text, meta = chunk_data[chunk_id]
        evidence_list.append(Evidence(
            doc_id=meta["doc_id"],
            chunk_id=meta["chunk_id"],
            text=doc_text,
            relevance_score=round(fused_scores[chunk_id], 4)
        ))
    return evidence_list


def retrieve(query: str, k: int = 5) -> list[Evidence]:
    """
    Hybrid retrieval: fuses BGE vector similarity with BM25 lexical scores
    via Reciprocal Rank Fusion, returning the top-k matching chunks as
    Evidence objects.
    """
    pool_size = max(k * CANDIDATE_POOL_MULTIPLIER, k)
    vector_results = _vector_search(query, pool_size)
    bm25_results = _bm25_search(query, pool_size)
    return _reciprocal_rank_fusion([vector_results, bm25_results], k)


if __name__ == "__main__":
    # Run this file directly once to build the index:  python -m src.retrieval
    build_index()

    # Quick manual sanity test
    test_query = "sick leave allowance"
    print(f"\nTest query: '{test_query}'")
    results = retrieve(test_query, k=3)
    for ev in results:
        print(f"\n[{ev.doc_id} | {ev.chunk_id} | score={ev.relevance_score}]")
        print(ev.text[:200], "...")
