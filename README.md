
# Research Desk Agent

A multi-agent research assistant built with LangGraph that answers questions
from a small, deliberately imperfect knowledge base.

Instead of blindly trusting an LLM to answer correctly every time, this
system is built around the real failure modes agents run into in
production -- hallucination, inconsistency, bad state management, and
runaway loops. It uses a pipeline of specialized agents (Clarifier,
Planner, Researcher, Synthesiser, Verifier, Router, Finaliser) that
retrieve evidence, draft citation-backed claims, verify each claim against
its source, and either finish, retry, or honestly say "I don't know" --
instead of guessing.

Key things this project focuses on:
- Every factual claim in the final answer is backed by a citation
- Contradictions between sources are surfaced to the user, not silently resolved
- The system can refuse to answer when the knowledge base has no answer
- A retry loop (with a hard cap) lets the agent fix its own unsupported claims
- Full run traces and an evaluation harness measure accuracy, groundedness,
  hallucination rate, and consistency across runs

Built with LangGraph, LangChain, and Pydantic.
