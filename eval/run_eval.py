"""
run_eval.py
The eval harness: runs all 15 questions (eval/questions.py) 3 times each
(45 total runs, non-interactive mode -- Clarifier auto-assumes instead of
blocking, see AgentState.interactive), judges each with an LLM (eval/judge.py),
and reports, per the task brief's exact metric definitions:

- Accuracy: does the final answer match the reference answer? (LLM-judge,
  per question/run -- this one genuinely needs semantic comparison)
- Groundedness: % of CLAIMS actually supported by their cited text -- a
  claim-level stat computed directly from the Verifier's own labels
  (SUPPORTED/CONFLICTING_SOURCES count as grounded -- both mean the claim's
  OWN citation genuinely backs it; CONFLICTING_SOURCES is a disagreement
  with a *sibling* claim, not with its own evidence). Deliberately NOT
  judge-assessed: the Verifier already produced this exact label per claim,
  so asking a second LLM to re-guess it from the final answer text would be
  strictly less precise, not more.
- Hallucination rate: % of claims that are confident but unsupported --
  the complement of groundedness at the same claim level (UNSUPPORTED /
  CONTRADICTED claims).
- Correct-abstention: % of the 3 unanswerable questions' runs where the
  system actually refused (state.refused) instead of inventing something.
- Consistency: run all 15 questions 3x, check how often the answer changes
  -- reported two ways: exact-text-match rate (the literal reading) and a
  looser same-verdict rate (same judge accuracy label + same refused flag
  across all 3 runs -- catches the CONFLICTING_SOURCES-vs-SUPPORTED
  instability observed manually earlier in this project on the sick-leave
  question, which exact-text-match would flag as "changed" even though the
  underlying facts stated were the same).
- Cost: LLM calls + tokens, average per question (src/llm.py's usage tracker).

One run failing (a real API error, not a graph-level refusal) does not stop
the harness -- it's recorded as an error and the harness moves on, so a
single rate-limit blip doesn't lose 44 other runs' worth of results.

Run: python -m eval.run_eval
"""

import json
import os
import time
from collections import defaultdict

from eval.judge import judge_answer
from eval.questions import QUESTIONS
from main import run
from src.llm import get_usage_snapshot, reset_usage_tracker

RUNS_PER_QUESTION = 3
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "report.json")

# Gentle pacing between runs -- not a documented rate limit workaround for
# any specific tier, just a small buffer since 45 runs back-to-back is a lot
# of rapid-fire API calls for a free-tier key.
PAUSE_BETWEEN_RUNS_SECONDS = 1.0


def _run_one(question: dict, run_index: int) -> dict:
    run_id = f"{question['id']}_run{run_index}"
    reset_usage_tracker()

    try:
        final_state, trace_path = run(question["question"], interactive=False, run_id=run_id)
    except Exception as err:  # noqa: BLE001 -- a real API/infra failure, not a graph-level refusal
        return {
            "question_id": question["id"],
            "category": question["category"],
            "run_index": run_index,
            "run_id": run_id,
            "error": f"{type(err).__name__}: {err}",
        }

    usage = get_usage_snapshot()
    claims_used = [
        c.text for c in final_state.claims
        if c.verification_status in ("SUPPORTED", "CONFLICTING_SOURCES")
    ]

    try:
        verdict = judge_answer(
            question=question["question"],
            category=question["category"],
            reference_answer=question["reference_answer"],
            final_answer=final_state.final_answer,
            refused=final_state.refused,
            claims_used=claims_used,
        )
        judge_result = verdict.model_dump()
    except Exception as err:  # noqa: BLE001 -- judge itself failed; still keep the run's own result
        judge_result = {"accuracy": None, "grounded": None, "reasoning": f"Judge failed: {err}"}

    return {
        "question_id": question["id"],
        "category": question["category"],
        "run_index": run_index,
        "run_id": run_id,
        "final_answer": final_state.final_answer,
        "refused": final_state.refused,
        "confidence": final_state.confidence,
        "retry_count": final_state.retry_count,
        "claims": [c.model_dump() for c in final_state.claims],
        "conflicts": [c.model_dump() for c in final_state.conflicts],
        "usage": usage,
        "judge": judge_result,
        "trace_path": trace_path,
    }


_GROUNDED_LABELS = ("SUPPORTED", "CONFLICTING_SOURCES")
_UNGROUNDED_LABELS = ("UNSUPPORTED", "CONTRADICTED")


def _aggregate(results: list[dict]) -> dict:
    valid = [r for r in results if "error" not in r]
    errored = [r for r in results if "error" in r]

    def category_results(category: str) -> list[dict]:
        return [r for r in valid if r["category"] == category]

    def accuracy_rate(rows: list[dict]) -> float:
        judged = [r for r in rows if r["judge"].get("accuracy") is not None]
        if not judged:
            return 0.0
        score = sum(1.0 if r["judge"]["accuracy"] == "correct" else 0.5 if r["judge"]["accuracy"] == "partial" else 0.0 for r in judged)
        return round(score / len(judged), 3)

    def claims_in(rows: list[dict]) -> list[dict]:
        return [c for r in rows for c in r["claims"]]

    def groundedness_rate(rows: list[dict]) -> float:
        # Claim-level, per the task brief: "% of claims actually supported by
        # their cited text" -- computed directly from the Verifier's own
        # per-claim label, not re-assessed by the judge from the final prose.
        claims = claims_in(rows)
        if not claims:
            return 1.0  # no claims produced (e.g. all refusals) -- nothing ungrounded was stated
        return round(sum(1 for c in claims if c["verification_status"] in _GROUNDED_LABELS) / len(claims), 3)

    categories = sorted({q["category"] for q in QUESTIONS})
    per_category = {
        cat: {
            "n_runs": len(category_results(cat)),
            "accuracy": accuracy_rate(category_results(cat)),
            "groundedness": groundedness_rate(category_results(cat)),
        }
        for cat in categories
    }

    unanswerable_runs = category_results("unanswerable")
    correct_abstention = (
        round(sum(1 for r in unanswerable_runs if r["refused"]) / len(unanswerable_runs), 3)
        if unanswerable_runs else 0.0
    )

    overall_groundedness = groundedness_rate(valid)
    hallucination_rate = round(1 - overall_groundedness, 3)

    # Consistency, reported two ways -- see module docstring for why both:
    # exact_answer_match_rate is the literal "does the answer text change"
    # reading; verdict_match_rate additionally tolerates the same underlying
    # facts being phrased differently but flags a genuine verdict flip
    # (e.g. a contradiction detected in one run and missed in another).
    by_question = defaultdict(list)
    for r in valid:
        by_question[r["question_id"]].append(r)

    exact_match_questions = 0
    verdict_match_questions = 0
    inconsistent_question_ids = []
    for qid, rows in by_question.items():
        if len({r["final_answer"] for r in rows}) == 1:
            exact_match_questions += 1
        verdict_signatures = {(r["judge"].get("accuracy"), r["refused"]) for r in rows}
        if len(verdict_signatures) == 1:
            verdict_match_questions += 1
        else:
            inconsistent_question_ids.append(qid)

    n_questions = len(by_question) or 1
    exact_answer_match_rate = round(exact_match_questions / n_questions, 3)
    verdict_match_rate = round(verdict_match_questions / n_questions, 3)

    total_calls = sum(r["usage"]["llm_calls"] for r in valid)
    total_tokens = sum(r["usage"]["total_tokens"] for r in valid)

    return {
        "n_total_runs": len(results),
        "n_errored_runs": len(errored),
        "overall_accuracy": accuracy_rate(valid),
        "overall_groundedness": overall_groundedness,
        "hallucination_rate": hallucination_rate,
        "correct_abstention_rate": correct_abstention,
        "consistency": {
            "exact_answer_match_rate": exact_answer_match_rate,
            "verdict_match_rate": verdict_match_rate,
            "inconsistent_question_ids": inconsistent_question_ids,
        },
        "per_category": per_category,
        "cost": {
            "total_llm_calls": total_calls,
            "total_tokens": total_tokens,
            "avg_llm_calls_per_question_run": round(total_calls / len(valid), 2) if valid else 0,
            "avg_tokens_per_question_run": round(total_tokens / len(valid), 1) if valid else 0,
        },
        "errored_run_ids": [r["run_id"] for r in errored],
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []

    total = len(QUESTIONS) * RUNS_PER_QUESTION
    done = 0
    for question in QUESTIONS:
        for run_index in range(1, RUNS_PER_QUESTION + 1):
            result = _run_one(question, run_index)
            results.append(result)
            done += 1
            status = "ERROR" if "error" in result else result["judge"].get("accuracy", "?")
            print(f"[{done}/{total}] {result['run_id']} ({question['category']}) -> {status}")
            time.sleep(PAUSE_BETWEEN_RUNS_SECONDS)

    with open(os.path.join(RESULTS_DIR, "raw_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary = _aggregate(results)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    print(json.dumps(summary, indent=2))
    print(f"\nRaw results: {os.path.join(RESULTS_DIR, 'raw_results.json')}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
