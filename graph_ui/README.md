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
uvicorn graph_ui.backend.main:app --reload --port 8003
```
(Port 8003, not 8002 or 8001 -- both got stuck with an orphaned listening socket on Windows that
no tool could kill. If you change `.env` while the backend is already running, restart it --
`--reload` only watches `.py` files, not `.env`, so it won't pick up a new API key on its own.)

Frontend:
```
cd graph_ui/frontend
npm install
npm run dev
```

Open the Vite URL (default `http://localhost:5174`).
