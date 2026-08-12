"""
synthesiser.py
Turns the Researcher's raw evidence pool into atomic, cited claims -- this is
the node that actually judges "does this snippet support this sub-question",
which Researcher deliberately does not do (it just dumps top-k hits).

One LLM call per sub-question, scoped to only that sub-question's evidence,
so citations can be checked against a small known set: any citation the
model returns that isn't one of the chunk_ids it was actually given is
dropped rather than trusted, since a hallucinated citation looks grounded
but isn't -- worse than no citation at all.

A sub-question with zero evidence (see state.unanswered_sub_questions) is
never sent to the LLM at all -- no evidence in, no claim out, so the system
can't fabricate an answer just because it "sounds right" from parametric
knowledge.
"""

from collections import defaultdict

from src.llm import get_llm
from src.state import AgentState, Claim, SynthesiserOutput

_SYSTEM_PROMPT = """You are the Synthesiser for an internal HR-policy Q&A agent.

You are given one sub-question and a numbered list of evidence snippets,
each tagged with its chunk_id. Extract atomic factual claims that directly
answer the sub-question, using ONLY information present in the evidence --
never your own outside knowledge.

Rules:
- One claim = one atomic fact. Never combine multiple facts into one claim
  (e.g. "sick leave is 10 days and casual leave is 8 days" must be two
  claims, not one).
- Every claim must list at least one citation, copied EXACTLY as the
  chunk_id shown next to the evidence it came from.
- If two snippets disagree (e.g. different numbers for the same fact),
  output a separate claim for each version with its own citation -- do not
  merge them or silently pick one, a downstream step resolves conflicts.
- If the evidence doesn't actually answer the sub-question, return no claims
  at all. Do not guess or fill gaps from general knowledge.
- confidence is your own 0-1 estimate of how clearly the cited evidence
  supports this specific claim.
"""


def synthesiser_node(state: AgentState) -> dict:
    evidence_by_sub_question = defaultdict(list)
    for ev in state.evidence:
        evidence_by_sub_question[ev.sub_question_id].append(ev)

    sub_question_text = {sq.id: sq.text for sq in state.sub_questions}

    llm = get_llm()
    structured_llm = llm.with_structured_output(SynthesiserOutput)

    new_claims = []
    claim_counter = len(state.claims)
    for sub_question_id, evidence_list in evidence_by_sub_question.items():
        valid_chunk_ids = {ev.chunk_id for ev in evidence_list}
        evidence_block = "\n\n".join(f"[{ev.chunk_id}] {ev.text}" for ev in evidence_list)
        question_text = sub_question_text.get(sub_question_id, sub_question_id)

        result: SynthesiserOutput = structured_llm.invoke([
            ("system", _SYSTEM_PROMPT),
            ("human", f"Sub-question: {question_text}\n\nEvidence:\n{evidence_block}"),
        ])

        for candidate in result.claims:
            valid_citations = [c for c in candidate.citations if c in valid_chunk_ids]
            if not valid_citations:
                continue
            claim_counter += 1
            new_claims.append(Claim(
                id=f"claim{claim_counter}",
                text=candidate.text,
                sub_question_id=sub_question_id,
                citations=valid_citations,
                confidence=candidate.confidence,
            ))

    return {"claims": state.claims + new_claims}
