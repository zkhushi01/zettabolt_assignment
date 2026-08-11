from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


class Evidence(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    relevance_score: float

class Claim(BaseModel):
    text: str
    supporting_evidence: list[str]  # doc_ids/chunk_ids
    verification_status: Literal["grounded", "unsupported", "contradicted"]
    confidence: float


class ConflictRecord(BaseModel):
    topic: str
    doc_a: str
    doc_b: str
    description: str


class RouterDecision(str, Enum):
    PASS = "pass"
    RETRY_RESEARCH = "retry_research"
    RETRY_PLAN = "retry_plan"
    REFUSE = "refuse"


class AgentState(BaseModel):
    # input side
    original_question: str
    clarified_question: Optional[str] = None
    assumptions: list[str] = Field(default_factory=list)

    # planning
    sub_questions: list[str] = Field(default_factory=list)

    # retrieval
    evidence: list[Evidence] = Field(default_factory=list)

    # synthesis
    draft_answer: Optional[str] = None
    claims: list[Claim] = Field(default_factory=list)

    # verification
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    hallucination_flags: list[str] = Field(default_factory=list)
    overall_confidence: float = 0.0

    # control flow
    retry_count: int = 0
    max_retries: int = 2
    router_decision: Optional[RouterDecision] = None
    is_answerable: Optional[bool] = None

    # output
    final_answer: Optional[str] = None
    citations: list[str] = Field(default_factory=list)
    trace_log: list[dict] = Field(default_factory=list)