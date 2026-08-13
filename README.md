
# Research Desk Agent

A multi-agent research assistant built with LangGraph that answers questions
from a small, deliberately imperfect knowledge base (see `docs/README.md`
for the planted contradictions, gaps, and stale-data flaws it has to handle).

Instead of blindly trusting an LLM to answer correctly every time, this
system is built around the real failure modes agents run into in
production -- hallucination, inconsistency, bad state management, and
runaway loops. It uses a pipeline of specialized agents (Clarifier,
Planner, Researcher, Synthesiser, Verifier, Router, Finaliser) that
retrieve evidence, draft citation-backed claims, verify each claim against
its source, and either finish, retry, or honestly say "I don't know" --
instead of guessing. See `graph.png` for the rendered graph and
`REPORT.md` for the full architecture rationale, metrics, and known
limitations.

Key things this project focuses on:
- Every factual claim in the final answer is backed by a citation
- Contradictions between sources are surfaced to the user, not silently resolved
- The system can refuse to answer when the knowledge base has no answer
- A retry loop (with a hard cap) lets the agent fix its own unsupported claims
- Full run traces and an evaluation harness measure accuracy, groundedness,
  hallucination rate, and consistency across runs

Built with LangGraph, LangChain, and Pydantic.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in both keys:
```
HF_TOKEN=          # Hugging Face token, for BGE embeddings (free: https://huggingface.co/settings/tokens)
GOOGLE_API_KEY=    # Gemini key, for the agent graph's LLM calls (free: https://aistudio.google.com/apikey)
```

The Gemini free tier caps out at 500 requests/day -- fine for interactive use and small eval runs,
but the full 45-run eval harness (below) can burn through a large fraction of that in one run (see
`REPORT.md`, Limitations).

## Run it — CLI

```bash
python main.py "How many days of casual leave am I entitled to?"                  # interactive (default)
python main.py "How many days of casual leave am I entitled to?" --non-interactive # never blocks
```

Interactive mode can pause mid-run with a clarifying question (answer it at the prompt); it
resumes the same graph run via LangGraph's `interrupt()`/`Command(resume=...)`. Non-interactive
mode (used by the eval harness) never blocks -- it auto-assumes an interpretation instead and
records that assumption in the output.

Every run prints the full pipeline's intermediate state (clarified question, sub-questions,
retrieved evidence, claims with verification labels, surfaced conflicts, final answer and
confidence) and writes a full JSON trace to `traces/<run_id>.json`.

## Run it — visual graph UI

A separate small web UI for stepping through each node's actual output (Clarifier, Planner,
Researcher, Synthesiser, Verifier, Router, Finaliser) instead of reading CLI text or raw trace
JSON. See `graph_ui/README.md` for setup; in short:

```bash
uvicorn graph_ui.backend.main:app --reload --port 8003   # backend, from repo root
cd graph_ui/frontend && npm install && npm run dev        # frontend, separate terminal
```

Then open the printed Vite URL (default `http://localhost:5174`).

`webapp/` is a separate, older UI that only exercises raw retrieval (no LLM/graph involved) --
kept for reference, not part of the main flow.

## Run the eval harness

```bash
python -m eval.run_eval
```

Runs all 15 questions in `eval/questions.py` three times each (45 runs total, non-interactive),
judges each with an LLM (`eval/judge.py`), and writes `eval/results/raw_results.json` (per-run
detail, including every claim and its verification label) and `eval/report.json` (aggregated
accuracy, groundedness, hallucination rate, correct-abstention rate, and consistency -- see the
docstring at the top of `eval/run_eval.py` for exact metric definitions, and `REPORT.md` for the
latest results and analysis).

## Project structure

```
src/            graph nodes (clarifier, planner, researcher, synthesiser, verifier, router,
                finaliser), state schema, LLM client, retrieval, tracing
main.py         CLI entry point
eval/           eval harness: questions, LLM judge, metric aggregation
graph_ui/       visual step-through UI (FastAPI backend + React frontend)
webapp/         older retrieval-only UI (no LLM), kept for reference
docs/           the knowledge base itself, plus docs/README.md documenting its planted flaws
traces/         full JSON trace per run (gitignored)
REPORT.md       architecture rationale, before/after metrics, root-caused failures, limitations
graph.png       rendered graph topology
```
