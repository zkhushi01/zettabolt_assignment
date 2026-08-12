"""
researcher.py
The RAG step -- only reachable through a Planner decision (retrieval_needed
== True). Runs hybrid retrieval (src/retrieval.py) once per sub-question and
returns evidence snippets tagged with doc/chunk/sub-question IDs, never
prose: turning evidence into claims is the Synthesiser's job, not this one's.

A sub-question that comes back with zero hits is recorded in
unanswered_sub_questions rather than dropped silently -- that's a real
signal ("the KB doesn't cover this") the Synthesiser/Finaliser need to
produce an honest "I don't know" instead of inventing an answer.
"""

from src.retrieval import RetrievalError, retrieve
from src.state import AgentState

# Per sub-question, not a global total -- each sub-question is an
# independent search and deserves its own top-k, matching how
# src/retrieval.py's retrieve() is scoped (one query in, k results out).
# Kept small (2) rather than padding out to a larger k: retrieve() always
# returns exactly k results with no relevance-score cutoff, so a bigger k
# against this small a knowledge base just drags in weakly-related chunks
# from other docs to fill the quota, rather than genuinely more evidence.
EVIDENCE_PER_SUB_QUESTION = 2


def researcher_node(state: AgentState) -> dict:
    new_evidence = []
    newly_unanswered = []

    for sub_question in state.sub_questions:
        try:
            results = retrieve(sub_question.text, k=EVIDENCE_PER_SUB_QUESTION)
        except RetrievalError as err:
            # Per retrieval.py's own contract: an embedding-API failure after
            # retries is an infrastructure failure, not "no evidence found",
            # and conflating the two would make a broken retriever look like
            # a genuinely uncovered topic. There's no Finaliser yet to hand
            # this to, so refuse honestly here rather than crash the run or
            # silently return an empty evidence pool.
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

    return {
        "evidence": state.evidence + new_evidence,
        "unanswered_sub_questions": state.unanswered_sub_questions + newly_unanswered,
    }
