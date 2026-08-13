"""
verifier.py
Checks every claim against its own cited evidence and labels it SUPPORTED,
CONTRADICTED, UNSUPPORTED, or CONFLICTING_SOURCES. This is the only node
that gets to see the *whole* claim set at once, on purpose: detecting
CONFLICTING_SOURCES requires comparing sibling claims for the same
sub-question against each other (e.g. "10 days" citing doc A vs. "12 days"
citing doc B -- each is individually well-supported by its own citation, so
that judgment is impossible from any single claim in isolation).

One LLM call for the whole set rather than one call per claim: cheaper, and
the cross-claim check above would need every claim visible anyway. Each
claim is still judged only against the evidence text it actually cites
(the model isn't shown the whole evidence pool), which is what keeps
"independent per claim" true even inside one batched call.

Only unverified claims (verification_status is None) are sent -- a claim
that already survived a prior pass isn't re-spent on again after a retry,
matching how Researcher/Synthesiser only redo the sub-questions that failed.
"""

from src.llm import get_llm
from src.state import AgentState, VerifierOutput

_SYSTEM_PROMPT = """You are the Verifier for an internal HR-policy Q&A agent.

You are given a numbered list of claims, each with the exact evidence snippet(s) it cites.
Decide a label for EACH claim:

- SUPPORTED: the cited evidence clearly states this fact, and no sibling claim for the same
  sub-question disagrees with it.
- CONTRADICTED: the cited evidence itself does not say what the claim says -- the claim
  misrepresents its own citation.
- UNSUPPORTED: the cited evidence does not actually address this fact at all (wrong or
  irrelevant citation, or too vague to support the specific thing claimed).
- CONFLICTING_SOURCES: this claim IS accurately supported by its own citation, but another claim
  in this same list, for the SAME sub-question, states a different value for the same fact citing
  a DIFFERENT source. Label BOTH of the disagreeing claims CONFLICTING_SOURCES, not SUPPORTED --
  do not silently prefer one.

For every claim give a one-sentence `reason`. Be concrete (name what's missing or which sibling
claim it conflicts with) -- this is shown to the user when a claim is dropped or flagged, and
used to retry the search if the claim needs it.
"""


def verifier_node(state: AgentState) -> dict:
    to_verify = [c for c in state.claims if c.verification_status is None]
    if not to_verify:
        return {}

    evidence_text_by_chunk_id = {ev.chunk_id: ev.text for ev in state.evidence}

    claims_block = "\n\n".join(
        f"[{c.id}] Claim: {c.text}\nCited evidence:\n" + "\n".join(
            f"  - [{cid}] {evidence_text_by_chunk_id.get(cid, '(citation not found in evidence pool)')}"
            for cid in c.citations
        )
        for c in to_verify
    )

    llm = get_llm()
    result: VerifierOutput = llm.with_structured_output(VerifierOutput).invoke([
        ("system", _SYSTEM_PROMPT),
        ("human", claims_block),
    ])
    verdict_by_claim_id = {v.claim_id: v for v in result.verdicts}

    # Build updates only for the claims actually sent (to_verify), then apply
    # them over the full list -- iterating state.claims directly here would
    # look up every already-verified claim's id in verdict_by_claim_id too,
    # find nothing (it was never asked about), and wrongly overwrite it via
    # the "model omitted this claim" fallback below.
    updates_by_id = {}
    for c in to_verify:
        verdict = verdict_by_claim_id.get(c.id)
        if verdict is None:
            # Model omitted this claim id from its response. Leaving
            # verification_status at None would let it slip through Router
            # unnoticed (Router only reacts to a set status) and never reach
            # Finaliser as SUPPORTED either -- an invalid-output case treated
            # as UNSUPPORTED rather than silently vanishing.
            updates_by_id[c.id] = c.model_copy(update={
                "verification_status": "UNSUPPORTED",
                "verification_reason": "Verifier did not return a verdict for this claim.",
            })
        else:
            updates_by_id[c.id] = c.model_copy(update={
                "verification_status": verdict.label,
                "verification_reason": verdict.reason,
            })

    updated_claims = [updates_by_id.get(c.id, c) for c in state.claims]
    return {"claims": updated_claims}
