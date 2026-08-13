"""
finaliser.py
Last node. Turns the verified claim set into the actual answer shown to the
user: SUPPORTED and CONFLICTING_SOURCES claims survive, UNSUPPORTED/
CONTRADICTED ones are dropped (Router already gave them their retry/rewrite
chances -- if they're still bad here, the retry cap was hit and they didn't
recover), conflicts are surfaced explicitly rather than picking a side, and
sub-questions with nothing usable are named rather than silently omitted.

confidence is computed in code from data already in state (surviving claims'
own confidence, penalized per unresolved conflict/gap) rather than asked of
the LLM -- that's arithmetic over known quantities, not a judgment call, and
keeping it deterministic means it's actually auditable (same claim set
always gives the same score) instead of one more thing that can drift
between runs.

The one LLM call here only rephrases already-verified claims into prose --
it is explicitly told not to add any fact that isn't already in the claims
list, since this is the last step before the user sees the answer and the
one place a new hallucination would be hardest to catch.
"""

from collections import defaultdict

from src.llm import get_llm
from src.state import AgentState, ConflictRecord, FinaliserOutput

_SYSTEM_PROMPT = """You are the Finaliser for an internal HR-policy Q&A agent.

You are given a list of already-verified claims (each already carries its citation), a list of
conflicts between sources that must be mentioned explicitly, and a list of sub-questions that
could not be answered.

Write the final answer as fluent prose using ONLY the claims given -- do not add, infer, or guess
any fact not present in them. Keep every citation attached to its fact (they are already correct,
just phrase them naturally, e.g. "(leave_policy_v1.md)"). For conflicts, state both values and
both sources explicitly -- never silently pick one. For anything in the "could not be answered"
list, say so plainly rather than omitting it. If there are no usable claims at all, just say you
don't know.
"""

_CONFLICT_PENALTY = 0.15   # per unresolved cross-source conflict
_GAP_PENALTY = 0.10        # per sub-question with no usable claim at all


def _build_conflicts(conflicting_claims: list, sub_question_text: dict) -> list[ConflictRecord]:
    by_sub_question = defaultdict(list)
    for c in conflicting_claims:
        by_sub_question[c.sub_question_id].append(c)

    conflicts = []
    for sq_id, claims in by_sub_question.items():
        # Synthesiser emits one claim per disagreeing version (see
        # synthesiser.py), so 2+ is the normal case; only the first two are
        # recorded as the representative pair if a sub-question somehow
        # produced more than two disagreeing versions -- the full set is
        # still in the answer prose via claims_block below, just not
        # duplicated across multiple ConflictRecord rows.
        if len(claims) < 2:
            continue
        a, b = claims[0], claims[1]
        conflicts.append(ConflictRecord(
            topic=sub_question_text.get(sq_id, sq_id),
            doc_a=a.citations[0],
            doc_b=b.citations[0],
            description=f'"{a.text}" (citing {a.citations[0]}) vs. "{b.text}" (citing {b.citations[0]})',
        ))
    return conflicts


def finaliser_node(state: AgentState) -> dict:
    supported = [c for c in state.claims if c.verification_status == "SUPPORTED"]
    conflicting = [c for c in state.claims if c.verification_status == "CONFLICTING_SOURCES"]
    usable_claims = supported + conflicting

    sub_question_text = {sq.id: sq.text for sq in state.sub_questions}
    conflicts = _build_conflicts(conflicting, sub_question_text)

    answered_sub_questions = {c.sub_question_id for c in usable_claims}
    all_sub_questions = {sq.id for sq in state.sub_questions}
    gaps = all_sub_questions - answered_sub_questions

    if not usable_claims:
        return {
            "refused": True,
            "final_answer": "I don't know. No verified evidence was found to answer this question.",
            "confidence": 0.0,
            "conflicts": conflicts,
        }

    avg_confidence = sum(c.confidence for c in usable_claims) / len(usable_claims)
    penalty = _CONFLICT_PENALTY * len(conflicts) + _GAP_PENALTY * len(gaps)
    confidence = round(max(0.0, min(1.0, avg_confidence - penalty)), 2)

    claims_block = "\n".join(f"- {c.text} [{', '.join(c.citations)}]" for c in usable_claims)
    conflicts_block = "\n".join(f"- {c.description}" for c in conflicts) or "(none)"
    gaps_block = "\n".join(f"- {sub_question_text.get(sq_id, sq_id)}" for sq_id in gaps) or "(none)"

    llm = get_llm()
    result: FinaliserOutput = llm.with_structured_output(FinaliserOutput).invoke([
        ("system", _SYSTEM_PROMPT),
        ("human", (
            f"Verified claims:\n{claims_block}\n\n"
            f"Conflicts to mention explicitly:\n{conflicts_block}\n\n"
            f"Could not be answered (say so honestly):\n{gaps_block}"
        )),
    ])

    return {
        "final_answer": result.answer,
        "confidence": confidence,
        "conflicts": conflicts,
        "refused": False,
    }
