"""
planner.py
Decomposes the clarified question into 1-4 sub-questions and decides
whether retrieval is even needed. This is the gate in front of the
Researcher: nothing gets retrieved except through a Planner decision, so a
purely definitional/meta question ("what kind of assistant are you?") never
triggers a KB lookup, and a genuinely unresolvable one can bounce back to
the Clarifier instead of being planned against garbage.
"""

from src.llm import get_llm
from src.state import AgentState, PlannerOutput, SubQuestion

# Gemini's temperature=0 is not bit-exact deterministic -- observed, across
# identical calls, retrieval_needed=True paired with an empty sub_questions
# list (a self-contradictory plan: retrieval is needed but there is nothing
# to search for). Retried rather than trusted on the first response, since
# .with_structured_output() only guarantees the JSON matches the schema's
# *types*, not that the fields are mutually consistent.
MAX_PLAN_RETRIES = 2

_SYSTEM_PROMPT = """You are the Planner for an internal HR-policy Q&A agent
backed by a small knowledge base of policy documents (leave, WFH, expenses,
probation, exit/offboarding, code of conduct, performance reviews, remote
work zones, onboarding, benefits).

Given a clarified question, decide:
1. Whether it needs retrieval from the knowledge base at all. Set
   retrieval_needed=false only for questions answerable purely from the
   question itself (definitional, about the assistant, or basic arithmetic
   with no domain facts needed) -- NOT for "I don't know if the KB covers
   this", since that's Researcher's/Verifier's job to discover.
2. If retrieval is needed, break it into 1-4 concrete, independently
   searchable sub-questions. A single-fact question gets exactly one
   sub-question; a comparison or multi-hop question (e.g. "how did X change
   between the old and new policy") gets one sub-question per fact needed.
3. Whether the clarified question is STILL too ambiguous to plan against at
   all (this should be rare -- the Clarifier already screened for this;
   only flag it if something the Clarifier couldn't have caught makes even
   sub-question decomposition impossible). If so, set still_ambiguous=true
   and ambiguity_note to a specific, concrete "please specify X" message --
   but ALWAYS ALSO fill in sub_questions/retrieval_needed with your
   best-effort plan regardless, in case there's no chance to ask further.
"""


def planner_node(state: AgentState) -> dict:
    llm = get_llm()
    question = state.clarified_question or state.raw_question
    structured_llm = llm.with_structured_output(PlannerOutput)

    result = None
    for attempt in range(1, MAX_PLAN_RETRIES + 1):
        candidate: PlannerOutput = structured_llm.invoke([
            ("system", _SYSTEM_PROMPT),
            ("human", question),
        ])
        if not candidate.retrieval_needed or candidate.sub_questions:
            result = candidate
            break
        # retrieval_needed=True with zero sub_questions -- broken plan, retry.

    if result is None:
        # Every retry came back broken. Don't propagate a plan Researcher
        # can't act on -- fall back to a single sub-question that is just
        # the question itself, and say so explicitly rather than silently
        # dropping retrieval.
        result = PlannerOutput(
            sub_questions=[SubQuestion(id="sq1", text=question)],
            retrieval_needed=True,
            still_ambiguous=False,
            rationale=f"Fallback plan after {MAX_PLAN_RETRIES} attempts returned an empty plan.",
        )

    update = {
        "sub_questions": result.sub_questions,
        "retrieval_needed": result.retrieval_needed,
        "still_ambiguous": result.still_ambiguous,
    }
    if result.still_ambiguous:
        # The router (src/graph.py) decides whether there's still a
        # clarification round budget left to actually act on this note --
        # the best-effort plan above is kept either way as a fallback.
        update["clarifying_question_pending"] = result.ambiguity_note or "Could you specify what you mean?"
    return update
