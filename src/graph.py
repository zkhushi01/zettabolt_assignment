"""
graph.py
Wires the nodes into a LangGraph StateGraph. Currently: Clarifier -> Planner,
with Planner able to bounce back to Clarifier if it finds the question still
too ambiguous to plan against (capped by state.max_clarification_rounds, see
src/clarifier.py), or forward to Researcher -> Synthesiser when retrieval is
needed. Verifier/Router/Finaliser are not built yet, so the graph currently
ends once Synthesiser returns claims (or once Planner decides no retrieval
is needed, or Researcher hits a refusal-worthy infra failure -- there's no
downstream node yet to answer from the question alone, so those paths just
stop, same as before).

Checkpointed with InMemorySaver so Clarifier's `interrupt()` calls can pause
mid-run and resume later with the same thread_id -- required for interactive
(CLI) mode. Non-interactive callers (the eval harness) never trigger an
interrupt in the first place (see AgentState.interactive), so the
checkpointer is inert for them, not just unused.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.clarifier import clarifier_node
from src.planner import planner_node
from src.researcher import researcher_node
from src.state import AgentState
from src.synthesiser import synthesiser_node


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
    # retrieval_needed == False: Synthesiser/Finaliser (answer from the
    # question alone, or refuse) aren't built yet, so this path just stops,
    # same as before Researcher existed.
    return END


def _route_after_researcher(state: AgentState) -> str:
    # refused=True means Researcher hit a hard retrieval-infra failure (see
    # src/researcher.py) and already set final_answer itself -- nothing for
    # Synthesiser to synthesise from, so stop rather than call it on an
    # empty/garbage evidence pool.
    if state.refused:
        return END
    return "synthesiser"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("clarifier", clarifier_node)
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("synthesiser", synthesiser_node)

    graph.add_edge(START, "clarifier")
    graph.add_conditional_edges("clarifier", _route_after_clarifier, {"planner": "planner"})
    graph.add_conditional_edges(
        "planner", _route_after_planner, {"clarifier": "clarifier", "researcher": "researcher", END: END}
    )
    graph.add_conditional_edges("researcher", _route_after_researcher, {"synthesiser": "synthesiser", END: END})
    # Verifier isn't built yet -- Synthesiser is the end of the graph for now.
    graph.add_edge("synthesiser", END)

    return graph.compile(checkpointer=InMemorySaver())
