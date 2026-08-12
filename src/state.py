from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---- Sub-units ----

class SubQuestion(BaseModel):
    id: str
    text: str


# Kept in the shape src/retrieval.py already produces and webapp/backend/main.py
# already consumes (doc_id, chunk_id, text, relevance_score) -- renaming
# `text` to the spec's `snippet` would only be a label change and touches two
# working call sites for no behavioral gain, so it stays. `sub_question_id`
# is the one field retrieval.py can't fill in (it doesn't know which
# sub-question it was called for) -- the Researcher node stamps it on after
# calling retrieve(), which is why it's optional here and required in
# practice everywhere the Researcher populates state.evidence.
class Evidence(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    relevance_score: float
    sub_question_id: Optional[str] = None


VerificationLabel = Literal["SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "CONFLICTING_SOURCES"]


class Claim(BaseModel):
    id: str
    text: str
    sub_question_id: str
    citations: list[str]  # doc_id/chunk_id references
    confidence: float
    # Set by the Verifier node (not built yet) -- None until then.
    verification_status: Optional[VerificationLabel] = None


class ConflictRecord(BaseModel):
    topic: str
    doc_a: str
    doc_b: str
    description: str


# Router's three options per spec (retry research / rewrite the answer /
# finish) -- refusal is not a router branch, it's Finaliser setting
# `refused=True` when nothing survives verification.
class RouterDecision(str, Enum):
    FINISH = "finish"
    RETRY_RESEARCH = "retry_research"
    REWRITE_ANSWER = "rewrite_answer"


# ---- Structured LLM outputs ----
# Every LLM-backed node returns one of these, validated by .with_structured_output()
# rather than hand-parsed JSON -- so a malformed response is a raised exception
# the caller can catch, not a silent bad state update.

class ClarifierOutput(BaseModel):
    needs_clarification: bool
    clarifying_question: Optional[str] = None
    clarified_question: Optional[str] = None
    assumptions: list[str] = Field(default_factory=list)
    reason: str


class PlannerOutput(BaseModel):
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    retrieval_needed: bool
    still_ambiguous: bool
    ambiguity_note: Optional[str] = None
    rationale: str


# One LLM call per sub-question's evidence group (see src/synthesiser.py), so
# citations don't carry an id yet -- the node stamps id/sub_question_id on
# after validating each citation actually names a chunk_id it was given.
class SynthesisedClaim(BaseModel):
    text: str
    citations: list[str]
    confidence: float


class SynthesiserOutput(BaseModel):
    claims: list[SynthesisedClaim] = Field(default_factory=list)


# ---- Top-level state ----
# Pydantic model (not a bare dict / TypedDict) so every node's input and
# output is validated at the boundary -- a node returning a malformed update
# fails fast instead of corrupting state silently three nodes later.

class AgentState(BaseModel):
    # mode: interactive (CLI, can pause and ask the user) vs non-interactive
    # (eval harness -- must never block, records an assumption instead).
    # One state field driving one graph, per the "single code path" rule --
    # not two divergent graphs for the two modes.
    interactive: bool = True

    # question / clarification
    raw_question: str
    clarified_question: Optional[str] = None
    clarifying_question_pending: Optional[str] = None
    # Single counter shared by BOTH the Clarifier's own re-ask loop and the
    # Planner's "still too vague, ask the user" bounce-back. The spec listed
    # these as two counters (clarification_turns, clarification_rounds) but
    # both represent the same thing -- "how many times we've interrupted the
    # user this run" -- and the spec itself says the Planner path reuses "the
    # same round cap" as the Clarifier, so a second counter would just be
    # dead weight tracking the same number.
    clarification_rounds: int = 0
    max_clarification_rounds: int = 2
    assumptions: list[str] = Field(default_factory=list)

    # planning
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    retrieval_needed: Optional[bool] = None
    still_ambiguous: bool = False

    # retrieval -- populated by the Researcher node (src/researcher.py).
    evidence: list[Evidence] = Field(default_factory=list)
    unanswered_sub_questions: list[str] = Field(default_factory=list)

    # synthesis / verification -- same as above, not wired up yet.
    claims: list[Claim] = Field(default_factory=list)
    conflicts: list[ConflictRecord] = Field(default_factory=list)

    # control flow
    retry_count: int = 0
    max_retries: int = 2
    router_decision: Optional[RouterDecision] = None

    # output
    final_answer: Optional[str] = None
    confidence: float = 0.0
    refused: bool = False

    # observability: one entry per node call, every field explicit rather
    # than a loose dict grab-bag, so the eval harness can read this without
    # guessing what keys might be present.
    trace: list[dict] = Field(default_factory=list)
