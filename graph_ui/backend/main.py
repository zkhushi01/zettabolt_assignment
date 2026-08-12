"""
graph_ui/backend/main.py
FastAPI backend for the Agent Graph UI -- runs the actual graph from
src/graph.py (Clarifier -> Planner -> Researcher, so far) node by node so
each node's output can be inspected individually, and supports interactive
mode (pause on a clarifying question, resume with the user's reply) via
LangGraph's interrupt()/Command(resume=...) mechanism -- the same mechanism
main.py's CLI uses for interactive mode, just driven over HTTP instead of
stdin/stdout.

Kept separate from webapp/backend, which only exercises raw retrieval (no
LLM, single request/response). This app runs the whole graph, which is
multi-step, resumable across requests, and can hit real API failures
(Gemini/HF) -- a different enough request lifecycle to not squeeze into the
same FastAPI app.
"""

import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.graph import build_graph
from src.llm import LLMConfigError
from src.retrieval import RetrievalError
from src.state import AgentState

app = FastAPI(title="Research Desk -- Agent Graph UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],  # this frontend's Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

# One compiled graph, reused across requests. LangGraph's InMemorySaver
# checkpointer (see src/graph.py) keys all state by thread_id, so concurrent
# runs/threads don't collide on a shared graph object.
_graph = build_graph()


class RunRequest(BaseModel):
    question: str = Field(min_length=1)
    interactive: bool = True


class ResumeRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    reply: str = Field(min_length=1)


class StepOut(BaseModel):
    node: str
    update: dict


class RunResponse(BaseModel):
    thread_id: str
    status: str  # "interrupted" | "done"
    steps: list[StepOut]
    clarifying_question: str | None = None
    state: dict | None = None


def _drain(stream) -> tuple[list[StepOut], str | None]:
    """
    Consumes a graph.stream(..., stream_mode="updates") iterator into a
    frontend-friendly shape: one StepOut per node that actually ran, in
    order, plus the interrupt payload if the run paused instead of
    finishing. This -- not a separate trace system -- is what lets the UI
    show one card per node: state.trace exists in the schema but no node
    writes to it yet, while stream_mode="updates" already gives the real
    per-node output for free.
    """
    steps: list[StepOut] = []
    interrupt_question: str | None = None
    for chunk in stream:
        if "__interrupt__" in chunk:
            interrupt_question = chunk["__interrupt__"][0].value["clarifying_question"]
            continue
        for node_name, update in chunk.items():
            steps.append(StepOut(node=node_name, update=jsonable_encoder(update)))
    return steps, interrupt_question


def _build_response(thread_id: str, steps: list[StepOut], interrupt_question: str | None) -> RunResponse:
    if interrupt_question is not None:
        return RunResponse(thread_id=thread_id, status="interrupted", steps=steps, clarifying_question=interrupt_question)
    snapshot = _graph.get_state({"configurable": {"thread_id": thread_id}})
    final_state = jsonable_encoder(AgentState.model_validate(snapshot.values).model_dump())
    return RunResponse(thread_id=thread_id, status="done", steps=steps, state=final_state)


@app.post("/api/run", response_model=RunResponse)
def run_graph(req: RunRequest):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    state_in = AgentState(raw_question=req.question, interactive=req.interactive)

    try:
        steps, interrupt_question = _drain(_graph.stream(state_in, config=config, stream_mode="updates"))
    except (LLMConfigError, RetrievalError) as err:
        raise HTTPException(status_code=502, detail=str(err))

    return _build_response(thread_id, steps, interrupt_question)


@app.post("/api/resume", response_model=RunResponse)
def resume_graph(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    snapshot = _graph.get_state(config)
    if not snapshot.next:
        raise HTTPException(status_code=400, detail="No pending clarification for this thread_id.")

    try:
        steps, interrupt_question = _drain(
            _graph.stream(Command(resume=req.reply), config=config, stream_mode="updates")
        )
    except (LLMConfigError, RetrievalError) as err:
        raise HTTPException(status_code=502, detail=str(err))

    return _build_response(req.thread_id, steps, interrupt_question)


@app.get("/api/topology")
def topology():
    """Mermaid source for the current graph shape -- lets the UI render
    exactly what's wired in src/graph.py instead of a hand-maintained
    picture that can drift out of sync with the real edges."""
    return {"mermaid": _graph.get_graph().draw_mermaid()}
