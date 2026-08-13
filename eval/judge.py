"""
judge.py
LLM-as-judge for the eval harness. Compares the system's actual output
against a hand-written reference_answer (see questions.py, grounded in
docs/planted_flaws.md) and returns a structured verdict -- not a second
opinion on domain facts, just "does this match what the reference says
should have happened for this category of question."

Judging logic is category-aware because "correct" means something different
per category:
- single_doc / multi_hop: does the answer state the same facts as the
  reference?
- unanswerable: did the system refuse / say it doesn't know, WITHOUT
  stating a specific invented fact? (a confident wrong answer is worse than
  a vague one here, and both are "incorrect")
- contradictory: does the answer surface BOTH conflicting values and
  sources, rather than confidently stating just one as if uncontested?

One judge call per (question, run) -- not part of the system's own cost
metric (see eval/run_eval.py, which resets/reads src/llm.py's usage tracker
around the *system* run only, before the judge call happens).
"""

from typing import Literal

from pydantic import BaseModel

from src.llm import get_llm

_JUDGE_SYSTEM_PROMPT = """You are grading the output of an HR-policy Q&A agent against a
reference answer written by a human who knows the ground truth. You are NOT grading writing
style -- only whether the agent's answer is factually correct and appropriately confident (or
appropriately uncertain) for the question's category.

Category-specific grading rules:

- single_doc / multi_hop: "correct" if the answer states the same key facts as the reference
  (numbers, durations, eligibility criteria). "partial" if it gets some facts right but misses or
  garbles others. "incorrect" if it states a wrong fact, or a near-miss fact from the wrong
  document (e.g. answering a remote-hiring-zone question with WFH-eligibility content).

- unanswerable: "correct" ONLY if the agent says it doesn't know / cannot find this information,
  and states no specific invented fact. "incorrect" if it confidently states any specific
  fact/number for something the reference says isn't in the knowledge base at all -- this is the
  single worst failure mode (hallucination on an unanswerable question), grade it strictly.

- contradictory: "correct" ONLY if the answer explicitly states BOTH conflicting values and both
  sources (per the reference). "partial" if it mentions the conflict exists but only gives one
  value, or gives both values but drops one source. "incorrect" if it confidently states only one
  value as if there were no disagreement.

Also judge `grounded`: true only if EVERY factual claim in the agent's answer traces back to one
of the "Claims available to the agent" listed below (each already has real citations) -- false if
the answer states anything, however small, that isn't among those claims. If the agent correctly
refused (no claims used), grounded is true by default (nothing was stated that could be
ungrounded).
"""


class JudgeVerdict(BaseModel):
    accuracy: Literal["correct", "partial", "incorrect"]
    grounded: bool
    reasoning: str


def judge_answer(
    question: str,
    category: str,
    reference_answer: str,
    final_answer: str | None,
    refused: bool,
    claims_used: list[str],
) -> JudgeVerdict:
    llm = get_llm()
    claims_block = "\n".join(f"- {c}" for c in claims_used) or "(none -- agent refused/found nothing usable)"

    human_prompt = (
        f"Category: {category}\n\n"
        f"Question: {question}\n\n"
        f"Reference answer (ground truth): {reference_answer}\n\n"
        f"Agent's final answer: {final_answer or '(no answer -- agent refused)'}\n"
        f"Agent refused: {refused}\n\n"
        f"Claims available to the agent when it wrote this answer:\n{claims_block}"
    )

    return llm.with_structured_output(JudgeVerdict).invoke([
        ("system", _JUDGE_SYSTEM_PROMPT),
        ("human", human_prompt),
    ])
