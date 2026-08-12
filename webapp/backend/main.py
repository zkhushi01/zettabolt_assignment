"""
FastAPI backend for the retrieval test UI.
Thin wrapper around src.retrieval -- no LLM involved, this only exercises
the embedding + BM25 hybrid search pipeline built in src/retrieval.py.
"""

import sys
from pathlib import Path

# Make `src` importable regardless of which directory uvicorn is started from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.retrieval import (
    EMBED_MODEL,
    RetrievalError,
    build_index,
    collection,
    retrieve,
)

app = FastAPI(title="Research Desk Retrieval Test UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    relevance_score: float


class StatusResponse(BaseModel):
    indexed_chunks: int
    embedding_model: str


@app.get("/api/status", response_model=StatusResponse)
def get_status():
    return StatusResponse(indexed_chunks=collection.count(), embedding_model=EMBED_MODEL)


@app.post("/api/search", response_model=list[SearchResult])
def search(req: SearchRequest):
    try:
        evidence = retrieve(req.query, k=req.k)
    except RetrievalError as err:
        raise HTTPException(status_code=502, detail=str(err))
    return [
        SearchResult(
            doc_id=ev.doc_id,
            chunk_id=ev.chunk_id,
            text=ev.text,
            relevance_score=ev.relevance_score,
        )
        for ev in evidence
    ]


@app.post("/api/reindex", response_model=StatusResponse)
def reindex():
    try:
        build_index(force_rebuild=True)
    except RetrievalError as err:
        raise HTTPException(status_code=502, detail=str(err))
    return StatusResponse(indexed_chunks=collection.count(), embedding_model=EMBED_MODEL)
