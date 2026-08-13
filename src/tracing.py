"""
tracing.py
Wraps every graph node so each execution is recorded -- node name, order,
duration, and exactly what it returned -- without any node needing to know
tracing exists. This is the implementation behind state.trace (declared in
state.py since the very first commit, unused until now): observability is a
cross-cutting concern, so it belongs in one wrapper applied in src/graph.py,
not six lines of bookkeeping copied into all seven node files with seven
chances to drift.

save_run_trace() is called by the two entry points (main.py's CLI, and
graph_ui/backend's /api/run + /api/resume) once a run reaches a terminal
state, writing the accumulated trace to traces/ as one JSON file per run.
"""

import json
import os
import time
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel

from src.state import AgentState

TRACES_DIR = os.path.join(os.path.dirname(__file__), "..", "traces")


def _json_default(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def traced(name: str, node_fn):
    """
    Wraps a node function so its execution is appended to state.trace.
    Transparent to interrupt()-based pausing (src/clarifier.py): interrupt()
    raises an exception that unwinds straight through this wrapper, so
    nothing below the `node_fn(state)` call runs and no trace entry is
    recorded for the paused attempt -- correct, since the node hasn't
    actually returned an output yet. On resume, the node reruns from
    scratch and, if it completes this time, gets its trace entry then.
    """
    def wrapped(state: AgentState) -> dict:
        started = time.time()
        update = node_fn(state)
        entry = {
            "node": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.time() - started, 3),
            # Round-tripped through json so the trace only ever holds plain
            # JSON types, never a live pydantic instance -- it's written to
            # disk verbatim by save_run_trace below.
            "output": json.loads(json.dumps(update, default=_json_default)),
        }
        return {**update, "trace": state.trace + [entry]}
    return wrapped


def save_run_trace(state: AgentState, run_id: str) -> str:
    """
    Writes the full accumulated trace for one run to traces/<run_id>.json.
    Call once a run reaches a terminal state -- Finaliser, or one of the
    early-exit ENDs (refusal, no-retrieval-needed, unresolved clarification)
    -- not on an interrupted/paused state, which isn't finished yet.
    """
    os.makedirs(TRACES_DIR, exist_ok=True)
    path = os.path.join(TRACES_DIR, f"{run_id}.json")
    payload = {
        "run_id": run_id,
        "raw_question": state.raw_question,
        "clarified_question": state.clarified_question,
        "assumptions": state.assumptions,
        "retrieval_needed": state.retrieval_needed,
        "retry_count": state.retry_count,
        "final_answer": state.final_answer,
        "confidence": state.confidence,
        "refused": state.refused,
        "conflicts": state.conflicts,
        "trace": state.trace,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return path
