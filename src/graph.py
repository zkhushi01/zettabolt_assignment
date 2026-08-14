"""
graph.py
Wires the nodes into a LangGraph StateGraph:

  Clarifier -> Planner -> Researcher -> Synthesiser -> Verifier -> Router
                  |  ^ (still_ambiguous,        ^                    |
                  |     interactive)            '-- Router: rewrite_answer --'
                  |                                                  |
                  |                     Router: retry_research -> Researcher
                  |                                                  |
                  |                           Router: finish -> Finaliser -> END
                  |                                                    ^
                  '-- retrieval_needed=False (no evidence pipeline) ---'

Router is the graph's one required real cycle: Verifier -> Router can send
control back to Researcher (new evidence needed) or Synthesiser (same
evidence, re-synthesise), capped by state.max_retries so it provably
terminates (see src/router.py).

Planner's retrieval_needed=False path goes to Finaliser too, not straight to
END -- Finaliser has a separate branch for this (src/finaliser.py's
_answer_without_retrieval) that either answers a purely definitional
question directly or refuses. This used to dead-end at END with no output at
all; the eval harness caught it doing exactly that on a real question
("how many employees does the company have" -- correctly judged as not
needing retrieval, but then nothing existed downstream to answer or refuse
it), so every path now reaches an actual final_answer or an honest refusal.

Checkpointed with InMemorySaver so Clarifier's `interrupt()` calls can pause
mid-run and resume later with the same thread_id -- required for interactive
(CLI) mode. Non-interactive callers (the eval harness) never trigger an
interrupt in the first place (see AgentState.interactive), so the
checkpointer is inert for them, not just unused.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.clarifier import clarifier_node
from src.finaliser import finaliser_node
from src.planner import planner_node
from src.researcher import researcher_node
from src.router import router_node
from src.state import AgentState, RouterDecision
from src.synthesiser import synthesiser_node
from src.tracing import traced
from src.verifier import verifier_node


def _route_after_clarifier(state: AgentState) -> str:
    # clarifier_node always sets clarified_question before returning
    # normally (the only way it *doesn't* return is via interrupt(), which
    # pauses the graph before this router ever runs) -- so this is really
    # just "proceed to planning."
    return "planner"


def _route_after_planner(state: AgentState) -> str:
    # Bounce back to Clarifier only if there's an actual chance of it
    # helping: interactive mode (so the user can genuinely be asked) and
    # still under the round cap. In non-interactive mode, re-entering
    # Clarifier would just re-auto-assume the *same* clarified_question
    # unchanged (nothing new to add) and Planner would very likely flag
    # still_ambiguous again -- bouncing anyway would still terminate (this
    # same cap applies inside Clarifier too, see src/clarifier.py), but only
    # after wasting rounds on calls guaranteed not to change anything, so
    # it's skipped outright rather than relying on the cap to bail out late.
    if state.still_ambiguous and state.interactive and state.clarification_rounds < state.max_clarification_rounds:
        return "clarifier"
    if state.retrieval_needed:
        return "researcher"
    # retrieval_needed == False: Finaliser's _answer_without_retrieval branch
    # handles this -- answer directly if genuinely definitional, else refuse.
    return "finaliser"


def _route_after_researcher(state: AgentState) -> str:
    # refused=True means Researcher hit a hard retrieval-infra failure (see
    # src/researcher.py) and already set final_answer itself -- nothing for
    # Synthesiser to synthesise from, so stop rather than call it on an
    # empty/garbage evidence pool.
    if state.refused:
        return END
    return "synthesiser"


def _route_after_router(state: AgentState) -> str:
    if state.router_decision == RouterDecision.RETRY_RESEARCH:
        return "researcher"
    if state.router_decision == RouterDecision.REWRITE_ANSWER:
        return "synthesiser"
    return "finaliser"


def build_graph():
    graph = StateGraph(AgentState)
    # traced() wraps each node so state.trace records every execution (node
    # name, order, duration, output) without any node file needing to know
    # tracing exists -- see src/tracing.py.
    graph.add_node("clarifier", traced("clarifier", clarifier_node))
    graph.add_node("planner", traced("planner", planner_node))
    graph.add_node("researcher", traced("researcher", researcher_node))
    graph.add_node("synthesiser", traced("synthesiser", synthesiser_node))
    graph.add_node("verifier", traced("verifier", verifier_node))
    graph.add_node("router", traced("router", router_node))
    graph.add_node("finaliser", traced("finaliser", finaliser_node))

    graph.add_edge(START, "clarifier")
    graph.add_conditional_edges("clarifier", _route_after_clarifier, {"planner": "planner"})
    graph.add_conditional_edges(
        "planner", _route_after_planner, {"clarifier": "clarifier", "researcher": "researcher", "finaliser": "finaliser"}
    )
    graph.add_conditional_edges("researcher", _route_after_researcher, {"synthesiser": "synthesiser", END: END})
    graph.add_edge("synthesiser", "verifier")
    graph.add_edge("verifier", "router")
    graph.add_conditional_edges(
        "router", _route_after_router, {"researcher": "researcher", "synthesiser": "synthesiser", "finaliser": "finaliser"}
    )
    graph.add_edge("finaliser", END)

    return graph.compile(checkpointer=InMemorySaver())
