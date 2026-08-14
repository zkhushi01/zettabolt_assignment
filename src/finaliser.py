"""
finaliser.py
Last node. Two distinct jobs depending on how it was reached:

1. retrieval_needed=False (Planner decided this question needs no KB
   lookup): there's no evidence/claims pool to work from at all here, so
   this is a narrow, separate judgment call -- can this genuinely be
   answered with no domain fact (definitional/meta), or does it actually
   need a specific fact this system has no business guessing? See
   _answer_without_retrieval. This used to be a dead end (the graph simply
   stopped after Planner, producing no output at all) until the eval
   harness caught it doing exactly that on a real question ("how many
   total employees does the company have" -- Planner correctly judged this
   isn't a retrieval question, but nothing existed to then answer or refuse
   it, so the run just silently produced nothing).

2. retrieval_needed=True (the normal path): turns the verified claim set
   into the actual answer. SUPPORTED and CONFLICTING_SOURCES claims
   survive, UNSUPPORTED/CONTRADICTED ones are dropped (Router already gave
   them their retry/rewrite chances -- if they're still bad here, the retry
   cap was hit and they didn't recover), conflicts are surfaced explicitly
   rather than picking a side, and sub-questions with nothing usable are
   named rather than silently omitted.

confidence (path 2) is computed in code from data already in state
(surviving claims' own confidence, penalized per unresolved conflict/gap)
rather than asked of the LLM -- that's arithmetic over known quantities, not
a judgment call, and keeping it deterministic means it's auditable (same
claim set always gives the same score) instead of one more thing that can
drift between runs.

Both LLM calls here are told never to add a fact beyond what they're given
-- this is the last step before the user sees the answer, and the hardest
place to catch a new hallucination introduced at the last mile.
"""

from collections import defaultdict

from src.llm import safe_structured_invoke
from src.state import AgentState, ConflictRecord, DirectAnswerOutput, FinaliserOutput

_SYSTEM_PROMPT = """You are the Finaliser for an internal HR-policy Q&A agent.

You are given a list of already-verified claims (each already carries its citation), a list of
conflicts between sources that must be mentioned explicitly, and a list of sub-questions that
could not be answered.

Write the final answer as fluent prose using ONLY the claims given -- do not add, infer, or guess
any fact not present in them. Do NOT attach an inline citation to individual facts (no
"(leave_policy_v1.md)" after a sentence) -- the source file names are appended separately after
your answer under a "Sources" heading, so keep the prose itself free of them. For conflicts, state
both values and both sources explicitly -- naming the sources is what makes the conflict itself
intelligible, so this is the one place a file name belongs in the prose; never silently pick one.
For anything in the "could not be answered" list, say so plainly rather than omitting it. If there
are no usable claims at all, just say you don't know.
"""

_DIRECT_ANSWER_SYSTEM_PROMPT = """You are the Finaliser for an internal HR-policy Q&A agent,
handling a question the Planner decided does NOT require knowledge-base retrieval.

Set can_answer_without_retrieval=True and provide an answer ONLY if the question is purely
definitional/meta (e.g. "what kind of assistant are you", "what can you help with") or basic
arithmetic/reasoning with no company-specific fact needed.

Set can_answer_without_retrieval=False for ANY question that would require a specific
company/policy/domain fact to answer correctly -- a number, a name, a date, a rule, a headcount
-- even if you think you might know or could plausibly guess it. Guessing a domain fact here
would bypass the entire evidence/citation pipeline this system is built around. If in doubt,
refuse.
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


def _answer_without_retrieval(state: AgentState) -> dict:
    question = state.clarified_question or state.raw_question
    result: DirectAnswerOutput = safe_structured_invoke(
        DirectAnswerOutput,
        [("system", _DIRECT_ANSWER_SYSTEM_PROMPT), ("human", question)],
        fallback=lambda err: DirectAnswerOutput(can_answer_without_retrieval=False),
    )

    if result.can_answer_without_retrieval and result.answer:
        return {
            "final_answer": result.answer,
            # Not evidence-grounded (no citations exist here by construction --
            # this path only fires for questions with nothing to cite), so
            # this reflects certainty about the assistant's own nature, not a
            # claim-confidence average like the retrieval path below.
            "confidence": 1.0,
            "refused": False,
            "conflicts": [],
        }
    return {
        "final_answer": "I don't know. This question requires information that isn't retrievable from the knowledge base, and I won't guess.",
        "confidence": 0.0,
        "refused": True,
        "conflicts": [],
    }


def finaliser_node(state: AgentState) -> dict:
    if not state.retrieval_needed:
        return _answer_without_retrieval(state)

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

    # Fallback on unparseable output is an honest "internal error" refusal --
    # never guess prose that might drift from the verified claims if the
    # schema didn't parse.
    result: FinaliserOutput = safe_structured_invoke(
        FinaliserOutput,
        [
            ("system", _SYSTEM_PROMPT),
            ("human", (
                f"Verified claims:\n{claims_block}\n\n"
                f"Conflicts to mention explicitly:\n{conflicts_block}\n\n"
                f"Could not be answered (say so honestly):\n{gaps_block}"
            )),
        ],
        fallback=lambda err: FinaliserOutput(answer="I don't know. An internal error occurred while composing the final answer."),
    )

    # Built in code, not asked of the LLM -- the citations already live on
    # usable_claims, so this is a dedup-and-format pass over known data
    # rather than one more thing the model could drop or rephrase wrong.
    sources = []
    for c in usable_claims:
        for cite in c.citations:
            if cite not in sources:
                sources.append(cite)
    sources_block = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources) if sources else ""

    return {
        "final_answer": result.answer.strip() + sources_block,
        "confidence": confidence,
        "conflicts": conflicts,
        "refused": False,
    }
