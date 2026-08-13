"""
researcher.py
The RAG step -- only reachable through a Planner decision (retrieval_needed
== True), or a Router decision to retry_research. Runs hybrid retrieval
(src/retrieval.py) and returns evidence snippets tagged with doc/chunk/
sub-question IDs, never prose: turning evidence into claims is the
Synthesiser's job, not this one's.

A sub-question that comes back with zero hits is recorded in
unanswered_sub_questions rather than dropped silently -- that's a real
signal ("the KB doesn't cover this") the Synthesiser/Finaliser need to
produce an honest "I don't know" instead of inventing an answer.

Two modes, one function (same "one code path" reasoning as Clarifier's
interactive/non-interactive split):
- First pass (state.retry_sub_question_ids empty): run every sub-question
  with its own text, verbatim.
- Retry pass (Router set state.retry_sub_question_ids): only re-run the
  sub-questions that actually failed verification, and reformulate each of
  their queries first -- re-issuing the identical query would return the
  identical, already-known-bad top-k, which is exactly what the spec says a
  retry must not do.
"""

from src.llm import safe_structured_invoke
from src.retrieval import RetrievalError, retrieve
from src.state import AgentState, Evidence, ReformulatedQuery

# Per sub-question, not a global total -- each sub-question is an
# independent search and deserves its own top-k, matching how
# src/retrieval.py's retrieve() is scoped (one query in, k results out).
# Kept small (2) rather than padding out to a larger k: retrieve() always
# returns exactly k results with no relevance-score cutoff, so a bigger k
# against this small a knowledge base just drags in weakly-related chunks
# from other docs to fill the quota, rather than genuinely more evidence.
# This stays fixed even on retry -- see _reformulate_query below for how a
# retry is made to actually change something instead.
EVIDENCE_PER_SUB_QUESTION = 2

_REFORMULATE_SYSTEM_PROMPT = """You are the query-reformulation step of a Researcher retry loop
for an internal HR-policy Q&A agent.

The search below already ran once for this sub-question, but the Verifier judged the resulting
claim UNSUPPORTED: the retrieved chunk(s) didn't actually contain the answer. You are given the
sub-question and exactly what was retrieved last time (which didn't help).

Rewrite the search query so it is more likely to surface DIFFERENT, more relevant chunks --
change the wording, use more literal keywords from the sub-question's actual topic, drop words
that may have pulled in unrelated chunks. Do not just repeat the sub-question verbatim. Output
only the new query text, not commentary about it.
"""


def _reformulate_query(sub_question_text: str, previous_evidence: list[Evidence]) -> str:
    previous_block = "\n".join(f"- {ev.text}" for ev in previous_evidence) or "(nothing was retrieved)"
    # Fallback on unparseable output is the original sub-question text
    # verbatim -- equivalent to skipping reformulation for this attempt
    # rather than crashing the retry.
    result: ReformulatedQuery = safe_structured_invoke(
        ReformulatedQuery,
        [
            ("system", _REFORMULATE_SYSTEM_PROMPT),
            ("human", f"Sub-question: {sub_question_text}\n\nPreviously retrieved (unhelpful):\n{previous_block}"),
        ],
        fallback=lambda err: ReformulatedQuery(query=sub_question_text),
    )
    return result.query


def researcher_node(state: AgentState) -> dict:
    retrying = bool(state.retry_sub_question_ids)
    target_ids = set(state.retry_sub_question_ids) if retrying else {sq.id for sq in state.sub_questions}
    targets = [sq for sq in state.sub_questions if sq.id in target_ids]

    new_evidence = []
    newly_unanswered = []

    for sub_question in targets:
        query_text = sub_question.text
        if retrying:
            previous_evidence = [ev for ev in state.evidence if ev.sub_question_id == sub_question.id]
            query_text = _reformulate_query(sub_question.text, previous_evidence)

        try:
            results = retrieve(query_text, k=EVIDENCE_PER_SUB_QUESTION)
        except RetrievalError as err:
            # Per retrieval.py's own contract: an embedding-API failure after
            # retries is an infrastructure failure, not "no evidence found",
            # and conflating the two would make a broken retriever look like
            # a genuinely uncovered topic. There's no evidence for Synthesiser
            # to work from either way, so refuse honestly here rather than
            # crash the run or silently return an empty evidence pool.
            return {
                "refused": True,
                "final_answer": f"I don't know. Retrieval failed and no evidence could be gathered: {err}",
                "confidence": 0.0,
                "unanswered_sub_questions": state.unanswered_sub_questions + [sq.id for sq in state.sub_questions],
            }

        if not results:
            newly_unanswered.append(sub_question.id)
            continue

        new_evidence.extend(
            ev.model_copy(update={"sub_question_id": sub_question.id}) for ev in results
        )

    # Drop the stale evidence/unanswered-marker only for the sub-questions
    # actually being redone -- everything else (already-good sub-questions on
    # a retry pass) is carried over untouched.
    kept_evidence = [ev for ev in state.evidence if ev.sub_question_id not in target_ids]
    kept_unanswered = [sq_id for sq_id in state.unanswered_sub_questions if sq_id not in target_ids]

    return {
        "evidence": kept_evidence + new_evidence,
        "unanswered_sub_questions": kept_unanswered + newly_unanswered,
    }
