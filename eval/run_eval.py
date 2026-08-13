"""
run_eval.py
The eval harness: runs all 15 questions (eval/questions.py) 3 times each
(45 total runs, non-interactive mode -- Clarifier auto-assumes instead of
blocking, see AgentState.interactive), judges each with an LLM (eval/judge.py),
and reports:

- accuracy (overall and per category)
- groundedness / hallucination rate (every fact traces to a real citation?)
- correct-abstention (the 3 unanswerable questions specifically: did the
  system say "I don't know" instead of guessing?)
- consistency (does the same question get the same verdict across all 3 of
  its runs? -- see the CONFLICTING_SOURCES-vs-SUPPORTED instability observed
  manually earlier in this project on the exact same sick-leave question)
- cost (LLM calls + tokens, via src/llm.py's usage tracker)

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
        "conflicts": [c.model_dump() for c in final_state.conflicts],
        "usage": usage,
        "judge": judge_result,
        "trace_path": trace_path,
    }


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

    def grounded_rate(rows: list[dict]) -> float:
        judged = [r for r in rows if r["judge"].get("grounded") is not None]
        if not judged:
            return 0.0
        return round(sum(1 for r in judged if r["judge"]["grounded"]) / len(judged), 3)

    categories = sorted({q["category"] for q in QUESTIONS})
    per_category = {
        cat: {
            "n_runs": len(category_results(cat)),
            "accuracy": accuracy_rate(category_results(cat)),
            "groundedness": grounded_rate(category_results(cat)),
        }
        for cat in categories
    }

    unanswerable_runs = category_results("unanswerable")
    correct_abstention = (
        round(sum(1 for r in unanswerable_runs if r["refused"]) / len(unanswerable_runs), 3)
        if unanswerable_runs else 0.0
    )

    hallucination_rate = round(1 - grounded_rate(valid), 3) if valid else 0.0

    # Consistency: for each question, do all its runs agree on both the
    # judge's accuracy verdict AND the refused flag? A question that flips
    # between CONFLICTING_SOURCES-labeled-as-SUPPORTED-vs-not across runs
    # (observed manually earlier on Q14, the sick-leave question) shows up
    # here as inconsistent.
    by_question = defaultdict(list)
    for r in valid:
        by_question[r["question_id"]].append(r)
    consistent_questions = 0
    inconsistent_question_ids = []
    for qid, rows in by_question.items():
        signatures = {(r["judge"].get("accuracy"), r["refused"]) for r in rows}
        if len(signatures) == 1:
            consistent_questions += 1
        else:
            inconsistent_question_ids.append(qid)
    consistency_rate = round(consistent_questions / len(by_question), 3) if by_question else 0.0

    total_calls = sum(r["usage"]["llm_calls"] for r in valid)
    total_tokens = sum(r["usage"]["total_tokens"] for r in valid)
    n_questions = len(by_question) or 1

    return {
        "n_total_runs": len(results),
        "n_errored_runs": len(errored),
        "overall_accuracy": accuracy_rate(valid),
        "overall_groundedness": grounded_rate(valid),
        "hallucination_rate": hallucination_rate,
        "correct_abstention_rate": correct_abstention,
        "consistency_rate": consistency_rate,
        "inconsistent_question_ids": inconsistent_question_ids,
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
