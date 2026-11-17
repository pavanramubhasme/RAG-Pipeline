"""API request/response models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatTurn(BaseModel):
    """One earlier message of the conversation, sent by the client (the API stays stateless)."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)
    history: list[ChatTurn] = Field(
        default_factory=list,
        max_length=20,
        description="previous turns, oldest first; lets follow-ups like 'what are their addresses?' work",
    )


class SearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=20)


class Passage(BaseModel):
    source: str
    section: str | None = None
    page: int | None = None
    classification: str | None = Field(default=None, description='e.g. "INTERNAL", "CONFIDENTIAL:HR"')
    score: float | None = None
    text: str


class Citation(Passage):
    ref: int


class AccessInfo(BaseModel):
    """The authorization decision for this request: who asked, and which labels were searched."""

    user: str
    role: str
    scopes: list[str] = Field(description='classification labels this caller may see; ["*"] = all')


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool
    access: AccessInfo | None = Field(default=None, description="None when access control is off")
    cached: bool = Field(
        default=False, description="answered from the answer cache: no retrieval, no LLM call"
    )
    blocked: bool = Field(default=False, description="stopped by the input guardrails")
    guardrails: list[str] = Field(
        default_factory=list,
        description='guardrail actions, e.g. "redacted:email", "blocked:prompt_injection"',
    )
    grounded: bool | None = Field(description="LLM check that the answer is supported; None if skipped")
    standalone_question: str | None = Field(
        default=None, description="the follow-up rewritten with conversation context, if history was used"
    )
    search_query: str | None = Field(description="rewritten query, if the agent had to retry")
    steps: list[str] = Field(description="path taken through the LangGraph workflow")
    timings_ms: dict[str, float]


class SearchResponse(BaseModel):
    results: list[Passage]


class IngestResponse(BaseModel):
    files: int
    chunks_added: int
    skipped_unchanged: int
    quarantined_chunks: int = Field(
        default=0, description="chunks dropped because they contain injection text"
    )
    masked: dict[str, int] = Field(
        default_factory=dict, description='chunks with masked values per kind, e.g. {"secret": 2}'
    )
    classifications: dict[str, int] = Field(
        default_factory=dict, description='new chunks per label, e.g. {"CONFIDENTIAL:HR": 11}'
    )
    total_chunks: int
