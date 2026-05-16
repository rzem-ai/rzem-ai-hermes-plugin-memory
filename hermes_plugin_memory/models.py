"""Pydantic models for memory provider results."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ObservationType = Literal["temporal", "preference", "sequence", "contextual"]
RelevanceMode = Literal["similarity", "recency_weighted", "recent", "since"]
TaskStatus = Literal["pending", "running", "done", "failed"]


class _Permissive(BaseModel):
    """Base model that tolerates extra fields from the server."""

    model_config = ConfigDict(extra="ignore")


class Thought(_Permissive):
    """A single thought returned from `search_memory`."""

    id: str | None = None
    agent_id: str | None = None
    content: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    similarity: float | None = None
    score: float | None = None


class SearchResult(_Permissive):
    """Result of a `search_memory` call."""

    mode: str
    thoughts: list[Thought]
    raw: str


class CaptureResult(_Permissive):
    """Result of a `capture_memory` call (MCP tool or REST `/capture`)."""

    id: str | None = None
    skipped: bool = False
    superseded: int = 0
    message: str


class IngestResult(_Permissive):
    """Result of an `ingest_article` call."""

    doc_id: str
    chunks: int
    replaced: int
    agent_id: str
    message: str


class KVEntry(_Permissive):
    """A KV store entry."""

    key: str
    value: Any = None
    version: int | None = None
    updated_at: datetime | None = None


class Observation(_Permissive):
    """A behavioural observation row."""

    id: int | None = None
    agent_id: str | None = None
    observation_type: ObservationType
    context: dict[str, Any] = Field(default_factory=dict)
    action: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] | None = None
    observed_at: datetime | None = None


class Pattern(_Permissive):
    """A mined behavioural pattern row."""

    id: int
    agent_id: str | None = None
    pattern_type: str | None = None
    description: str | None = None
    confidence: float | None = None
    observation_count: int | None = None
    context_tags: list[str] = Field(default_factory=list)
    similarity: float | None = None
    last_observed_at: datetime | None = None
    is_active: bool | None = None


class MineResult(_Permissive):
    """Result of a `mine_patterns` call."""

    agent_id: str
    status: Literal["completed", "skipped", "error"]
    observations_analysed: int = 0
    patterns_new: int = 0
    patterns_reinforced: int = 0
    patterns_decayed: int = 0
    message: str | None = None


class Task(_Permissive):
    """A task queue entry."""

    id: str | None = None
    agent_id: str
    title: str
    description: str
    task_type: str = "general"
    priority: int = 0
    status: TaskStatus | None = None
    created_at: datetime | None = None


class UsageSummary(_Permissive):
    """Aggregated usage / metering result."""

    agent_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    calls: int | None = None
    tool_calls: int | None = None
    raw: str | None = None
