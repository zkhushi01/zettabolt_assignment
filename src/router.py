"""
router.py
Reads the Verifier's per-claim labels and decides: retry_research,
rewrite_answer, or finish. Deliberately plain code, not an LLM node -- by
this point the Verifier has already done the judgment work (is this claim
actually supported?), so Router only needs to aggregate already-structured
labels and check a counter. That's arithmetic, not reasoning, and putting an
LLM call on this node would mean re-spending an API call every single pass
through the graph's one required retry cycle for a decision that's fully
determined by data already in state -- see the max_retries cap below, which
needs to be trivially provably-terminating, not dependent on a model
reliably returning consistent structured output every time.

Label -> action mapping:
- UNSUPPORTED: the citation didn't actually address the claim -- the
  evidence itself is probably insufficient, so retry_research (new search).
- CONTRADICTED: the claim misrepresents evidence that's otherwise fine --
  a Synthesiser mistake, not an evidence gap, so rewrite_answer (same
  evidence, re-synthesise) rather than searching again for no reason.
- CONFLICTING_SOURCES: not a failure. Two sources genuinely disagree; the
  spec's answer is to surface that, not resolve it -- so this never
  triggers a retry, it flows straight to Finaliser.
"""

from collections import defaultdict

from src.state import AgentState, RouterDecision


def router_node(state: AgentState) -> dict:
    labels_by_sub_question = defaultdict(set)
    for c in state.claims:
        if c.verification_status:
            labels_by_sub_question[c.sub_question_id].add(c.verification_status)

    needs_retry_research = {sq for sq, labels in labels_by_sub_question.items() if "UNSUPPORTED" in labels}
    needs_rewrite = {
        sq for sq, labels in labels_by_sub_question.items() if "CONTRADICTED" in labels
    } - needs_retry_research  # if a sub-question needs both, retry_research subsumes it -- fresh evidence gets re-synthesised anyway

    if not needs_retry_research and not needs_rewrite:
        return {"router_decision": RouterDecision.FINISH, "retry_sub_question_ids": []}

    if state.retry_count >= state.max_retries:
        # Cap reached -- stop looping regardless of what's still unresolved.
        # Finaliser is responsible for dropping/flagging whatever never got
        # a SUPPORTED verdict; this is what makes the cycle provably finite.
        return {"router_decision": RouterDecision.FINISH, "retry_sub_question_ids": []}

    affected = needs_retry_research | needs_rewrite
    # Strip claims for every sub-question being redone so Synthesiser sees
    # them as "not yet covered" and regenerates them (see synthesiser.py) --
    # Router owns this, not Researcher/Synthesiser, since it's the one node
    # that knows which sub-questions are being retried for which reason.
    kept_claims = [c for c in state.claims if c.sub_question_id not in affected]

    if needs_retry_research:
        return {
            "router_decision": RouterDecision.RETRY_RESEARCH,
            "retry_count": state.retry_count + 1,
            "retry_sub_question_ids": list(needs_retry_research),
            "claims": kept_claims,
        }

    # Only rewrite-worthy claims remain -- same evidence, Synthesiser just
    # gets another pass at it, so Researcher is skipped entirely.
    return {
        "router_decision": RouterDecision.REWRITE_ANSWER,
        "retry_count": state.retry_count + 1,
        "retry_sub_question_ids": [],
        "claims": kept_claims,
    }
