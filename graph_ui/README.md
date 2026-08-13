# Agent Graph UI

A small UI for driving `src/graph.py` (Clarifier -> Planner -> Researcher)
node by node, so each node's actual output can be checked visually instead
of only reading CLI text or a raw trace JSON. Separate from `webapp/`,
which only exercises raw retrieval with no LLM involved.

Supports both graph modes from one flow:
- **Interactive** -- Clarifier can pause with a follow-up question; reply in
  the browser and the run resumes from the same point (same `interrupt()` /
  `Command(resume=...)` mechanism `main.py`'s CLI uses).
- **Non-interactive** -- Clarifier auto-assumes instead of asking, same as
  the eval harness.

## Run it

Backend (from repo root, same env as `main.py`):
```
uvicorn graph_ui.backend.main:app --reload --port 8002
```

Frontend:
```
cd graph_ui/frontend
npm install
npm run dev
```

Open the Vite URL (default `http://localhost:5174`).
