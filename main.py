"""
main.py
CLI entry point for the Research Desk agent graph: Clarifier -> Planner ->
Researcher -> Synthesiser -> Verifier -> Router -> Finaliser, with Router
able to loop back to Researcher/Synthesiser (capped retries) before a run
finishes.

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
from src.tracing import save_run_trace


def run(question: str, interactive: bool = True, run_id: str = None) -> tuple[AgentState, str]:
    """Returns (final_state, trace_file_path). run_id defaults to a fresh
    uuid but can be passed explicitly -- the eval harness will want to name
    runs after their question index/repeat number rather than a random id."""
    run_id = run_id or str(uuid.uuid4())
    graph = build_graph()
    config = {"configurable": {"thread_id": run_id}}
    state_in = AgentState(raw_question=question, interactive=interactive)

    result = graph.invoke(state_in, config=config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"\nClarifying question: {payload['clarifying_question']}")
        reply = input("Your answer: ")
        result = graph.invoke(Command(resume=reply), config=config)

    final_state = AgentState.model_validate(result)
    trace_path = save_run_trace(final_state, run_id)
    return final_state, trace_path


def main():
    parser = argparse.ArgumentParser(
        description="Research Desk agent (Clarifier -> Planner -> Researcher -> Synthesiser -> Verifier -> Router -> Finaliser)"
    )
    parser.add_argument("question", help="Question to ask")
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="Never ask follow-up questions; record an assumption and proceed instead",
    )
    args = parser.parse_args()

    final_state, trace_path = run(args.question, interactive=not args.non_interactive)

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
        print(f"\nClaims ({len(final_state.claims)}, {final_state.retry_count} retries used):")
        for claim in final_state.claims:
            print(
                f"  [{claim.id} | {claim.sub_question_id} | {claim.verification_status} | "
                f"confidence={claim.confidence}] {claim.text}"
            )
            print(f"    citations: {', '.join(claim.citations)}")
            if claim.verification_reason:
                print(f"    reason: {claim.verification_reason}")
    if final_state.conflicts:
        print(f"\nConflicts surfaced ({len(final_state.conflicts)}):")
        for conflict in final_state.conflicts:
            print(f"  [{conflict.topic}] {conflict.description}")
    if final_state.final_answer:
        print(f"\n--- Final answer (confidence={final_state.confidence}) ---")
        print(final_state.final_answer)
    print(f"\nFull trace saved to: {trace_path}")


if __name__ == "__main__":
    main()
