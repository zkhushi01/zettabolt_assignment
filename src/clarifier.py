"""
clarifier.py
Entry node of the graph -- always runs first. Decides whether the user's
question is answerable as written, or whether something essential is
missing (e.g. "what is it?" -- which policy/product?).

Also re-entered when the Planner decides the clarified question is still
too ambiguous to plan against (see `state.still_ambiguous`); in that case
this node skips its own LLM judgment and goes straight to asking (or
recording an assumption for) the Planner's specific ambiguity note, instead
of generating a second, possibly different clarifying question.
"""

from langgraph.types import interrupt

from src.llm import get_llm
from src.state import AgentState, ClarifierOutput

_SYSTEM_PROMPT = """You are the Clarifier for an internal HR-policy Q&A agent.

Decide whether the user's question can be answered as written, or whether
something essential is missing (which policy, which employee group, which
time period, etc.) such that answering now would mean guessing.

Ask for clarification only when genuinely necessary -- most concrete
questions about a named policy topic (leave, WFH, expenses, probation,
notice period, performance reviews, remote hiring zones) do NOT need
clarification even if you don't yet know the answer; "the knowledge base
might not cover this" is not a reason to ask the user, it's a reason to let
retrieval run and possibly refuse later. Only ask when the question itself
is genuinely ambiguous (vague pronouns, no named topic, multiple plausible
readings that would lead to different answers).

If clarification is needed, ask exactly ONE specific, concrete question.
"""


def _ask_or_record_assumption(state: AgentState, question: str, reason: str, fallback_question: str) -> dict:
    """
    Shared by both entry paths (Clarifier's own ambiguity finding, and the
    Planner's still_ambiguous bounce-back) so there is exactly one place
    that implements "ask the user vs. proceed on an assumption" -- one code
    path with a mode flag, not divergent logic per caller.
    """
    at_cap = state.clarification_rounds >= state.max_clarification_rounds
    if not state.interactive or at_cap:
        why_stuck = "clarification round cap reached" if at_cap else "non-interactive mode"
        assumption = (
            f"Auto-assumed interpretation ({why_stuck}): '{fallback_question}'. "
            f"Reason clarification would otherwise have been requested: {reason}"
        )
        return {
            "clarified_question": fallback_question,
            "assumptions": state.assumptions + [assumption],
            "clarifying_question_pending": None,
            "still_ambiguous": False,
            # Counts as a used round even though nobody was actually asked --
            # keeps this field meaning "rounds of ambiguity handling spent"
            # consistently, rather than only counting the interactive branch.
            "clarification_rounds": state.clarification_rounds + 1,
        }

    # Interactive and under the cap: actually pause the graph and ask.
    user_reply = interrupt({"clarifying_question": question})
    combined = f"{state.clarified_question or state.raw_question}\n(clarification: {user_reply})"
    return {
        "clarified_question": combined,
        "clarification_rounds": state.clarification_rounds + 1,
        "clarifying_question_pending": None,
        "still_ambiguous": False,
    }


def clarifier_node(state: AgentState) -> dict:
    if state.still_ambiguous:
        # Re-entered from the Planner -- it already decided clarification is
        # needed and supplied a specific note. Don't re-judge, just act on it.
        question = state.clarifying_question_pending or "Could you clarify what you're asking about?"
        return _ask_or_record_assumption(
            state,
            question=question,
            reason=f"Planner flagged the plan as still too ambiguous: {question}",
            fallback_question=state.clarified_question or state.raw_question,
        )

    llm = get_llm()
    result: ClarifierOutput = llm.with_structured_output(ClarifierOutput).invoke([
        ("system", _SYSTEM_PROMPT),
        ("human", state.raw_question),
    ])

    if not result.needs_clarification:
        return {
            "clarified_question": result.clarified_question or state.raw_question,
            "assumptions": state.assumptions + result.assumptions,
            "clarifying_question_pending": None,
        }

    return _ask_or_record_assumption(
        state,
        question=result.clarifying_question or "Could you clarify your question?",
        reason=result.reason,
        fallback_question=result.clarified_question or state.raw_question,
    )
