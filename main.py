"""
main.py
CLI entry point for the Research Desk agent graph. Currently drives it only
as far as Clarifier -> Planner -> Researcher -> Synthesiser (Verifier onward
isn't built yet), so a run prints the resulting plan, evidence, and claims
rather than a final, verified answer.

Two modes:
  python main.py "question"                    interactive (default) --
                                                 can pause and ask a follow-up
  python main.py "question" --non-interactive   never blocks; auto-assumes
                                                 instead of asking
"""

import argparse
import uuid

from langgraph.types import Command

from src.graph import build_graph
from src.state import AgentState


def run(question: str, interactive: bool = True) -> AgentState:
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state_in = AgentState(raw_question=question, interactive=interactive)

    result = graph.invoke(state_in, config=config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"\nClarifying question: {payload['clarifying_question']}")
        reply = input("Your answer: ")
        result = graph.invoke(Command(resume=reply), config=config)

    return AgentState.model_validate(result)


def main():
    parser = argparse.ArgumentParser(description="Research Desk agent (Clarifier -> Planner -> Researcher -> Synthesiser, so far)")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="Never ask follow-up questions; record an assumption and proceed instead",
    )
    args = parser.parse_args()

    final_state = run(args.question, interactive=not args.non_interactive)

    print("\n--- Result ---")
    print(f"Clarified question: {final_state.clarified_question}")
    if final_state.assumptions:
        print("Assumptions made:")
        for assumption in final_state.assumptions:
            print(f"  - {assumption}")
    print(f"Retrieval needed: {final_state.retrieval_needed}")
    if final_state.sub_questions:
        print("Sub-questions:")
        for sub_q in final_state.sub_questions:
            print(f"  [{sub_q.id}] {sub_q.text}")
    if final_state.refused:
        print(f"\nRefused: {final_state.final_answer}")
    if final_state.evidence:
        print(f"\nEvidence ({len(final_state.evidence)} chunks):")
        for ev in final_state.evidence:
            print(f"  [{ev.sub_question_id}] {ev.doc_id} ({ev.chunk_id}, score={ev.relevance_score})")
            print(f"    {ev.text[:150]}...")
    if final_state.unanswered_sub_questions:
        print("\nUnanswered sub-questions (no evidence found):")
        for sq_id in final_state.unanswered_sub_questions:
            print(f"  - {sq_id}")
    if final_state.claims:
        print(f"\nClaims ({len(final_state.claims)}):")
        for claim in final_state.claims:
            print(f"  [{claim.id} | {claim.sub_question_id} | confidence={claim.confidence}] {claim.text}")
            print(f"    citations: {', '.join(claim.citations)}")


if __name__ == "__main__":
    main()
