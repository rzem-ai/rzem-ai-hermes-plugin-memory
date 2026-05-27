"""Hermes MemoryProvider adapter for ``rzem-ai-hermes-plugin-memory``.

Bridges the async :class:`AgentMemoryProvider` (an MCP client for the
``agent-memory-mcp`` server) into the synchronous
:class:`agent.memory_provider.MemoryProvider` ABC that Hermes's conversation
loop calls each turn.

Discovery contract — Hermes scans ``$HERMES_HOME/plugins/<name>/__init__.py``
for the string ``register_memory_provider`` or ``MemoryProvider`` and then
invokes :func:`register` to harvest the provider instance. The directory
name (``rzem-ai-hermes-plugin-memory``) becomes the key written to
``memory.provider`` in ``config.yaml``.

A daemon-thread event loop hosts the long-lived ``AgentMemoryProvider``
async context; each sync call bridges across via
``asyncio.run_coroutine_threadsafe``. That avoids re-establishing the MCP
session per turn while keeping the public surface synchronous.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Optional

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ConfigDict, Field

try:
    from agent.memory_provider import MemoryProvider
    from tools.registry import tool_error
except ImportError:
    MemoryProvider = object  # type: ignore[misc,assignment]

    def tool_error(message: str) -> str:  # type: ignore[misc]
        return json.dumps({"error": message})


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

Transport = Literal["http", "stdio", "sse"]


@dataclass
class MemoryProviderConfig:
    """Connection settings for the agent-memory MCP server."""

    transport: Transport = "http"
    agent_id: str = "default"

    # HTTP / SSE transport
    base_url: str = "http://127.0.0.1:3002"
    bearer_token: str | None = None
    request_timeout_seconds: float = 30.0

    # stdio transport (subprocess)
    stdio_command: str = "node"
    stdio_args: list[str] = field(default_factory=lambda: ["dist/stdio.js"])
    stdio_cwd: str | None = None
    stdio_env: dict[str, str] | None = None

    @classmethod
    def from_env(cls, prefix: str = "HERMES_MEMORY_") -> MemoryProviderConfig:
        """Build a config from `HERMES_MEMORY_*` environment variables."""

        def env(key: str, default: str | None = None) -> str | None:
            return os.environ.get(f"{prefix}{key}", default)

        transport = (env("TRANSPORT", "http") or "http").lower()
        if transport not in ("http", "stdio", "sse"):
            raise ValueError(
                f"Invalid {prefix}TRANSPORT={transport!r}; must be 'http', 'sse', or 'stdio'"
            )

        cfg = cls(
            transport=transport,  # type: ignore[arg-type]
            agent_id=env("AGENT_ID", "default") or "default",
            base_url=env("BASE_URL", "http://127.0.0.1:3002") or "http://127.0.0.1:3002",
            bearer_token=env("BEARER_TOKEN"),
            request_timeout_seconds=float(env("TIMEOUT_SECONDS", "30") or "30"),
            stdio_command=env("STDIO_COMMAND", "node") or "node",
            stdio_cwd=env("STDIO_CWD"),
        )
        stdio_args = env("STDIO_ARGS")
        if stdio_args:
            cfg.stdio_args = shlex.split(stdio_args)
        return cfg


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MemoryProviderError(Exception):
    """Base class for memory provider errors."""


class MemoryConnectionError(MemoryProviderError):
    """Raised when the connection to the MCP server cannot be established or used."""


class MemoryToolError(MemoryProviderError):
    """Raised when an MCP tool invocation reports an error."""

    def __init__(self, tool: str, message: str) -> None:
        super().__init__(f"{tool}: {message}")
        self.tool = tool
        self.message = message


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# MCP client
# ---------------------------------------------------------------------------


class MCPMemoryClient:
    """Thin async wrapper around the `mcp` Python SDK.

    Use as an async context manager:

        async with MCPMemoryClient(config) as client:
            text = await client.call_tool("kv_get", {"agent_id": "a", "key": "k"})
    """

    def __init__(self, config: MemoryProviderConfig) -> None:
        self.config = config
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> MCPMemoryClient:
        stack = AsyncExitStack()
        try:
            if self.config.transport == "stdio":
                params = StdioServerParameters(
                    command=self.config.stdio_command,
                    args=list(self.config.stdio_args),
                    env=self.config.stdio_env,
                    cwd=self.config.stdio_cwd,
                )
                streams = await stack.enter_async_context(stdio_client(params))
            elif self.config.transport == "http":
                from mcp.client.streamable_http import streamablehttp_client

                headers = self._auth_headers()
                ctx = streamablehttp_client(
                    f"{self.config.base_url.rstrip('/')}/mcp",
                    headers=headers or None,
                    timeout=self.config.request_timeout_seconds,
                )
                streams = await stack.enter_async_context(ctx)
            elif self.config.transport == "sse":
                from mcp.client.sse import sse_client

                headers = self._auth_headers()
                ctx = sse_client(
                    f"{self.config.base_url.rstrip('/')}/sse",
                    headers=headers or None,
                )
                streams = await stack.enter_async_context(ctx)
            else:
                raise MemoryConnectionError(f"Unknown transport: {self.config.transport}")

            # Streamable HTTP yields (read, write, get_session_id); the others yield (read, write).
            read, write = streams[0], streams[1]
            session = ClientSession(read, write)
            self._session = await stack.enter_async_context(session)
            await self._session.initialize()
        except Exception as exc:
            await stack.aclose()
            self._session = None
            raise MemoryConnectionError(
                f"Failed to connect to agent-memory MCP server: {exc}"
            ) from exc
        self._stack = stack
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        stack = self._stack
        self._stack = None
        self._session = None
        if stack is not None:
            await stack.aclose()

    def _auth_headers(self) -> dict[str, str]:
        if self.config.bearer_token:
            return {"Authorization": f"Bearer {self.config.bearer_token}"}
        return {}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke an MCP tool and return the concatenated text content.

        Raises :class:`MemoryToolError` when the server marks the response as
        an error (``isError = true``).
        """
        if self._session is None:
            raise MemoryConnectionError(
                "client is not connected; use as `async with MCPMemoryClient(...) as client:`"
            )
        result = await self._session.call_tool(name, arguments)
        text_parts: list[str] = []
        for item in result.content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
        body = "\n".join(text_parts)
        if getattr(result, "isError", False):
            raise MemoryToolError(name, body or "tool returned an error")
        return body

    async def call_tool_json(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke an MCP tool and parse the response body as JSON."""
        body = await self.call_tool(name, arguments)
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise MemoryToolError(name, f"expected JSON response, got: {body!r}") from exc


# ---------------------------------------------------------------------------
# Provider (high-level async surface)
# ---------------------------------------------------------------------------

# Matches a single line of `search_memory` output:
#   [N] (score: X.XXX, sim: Y.YYY, YYYY-MM-DD) [agent_id] content | tags: tag1, tag2
# `score:` is only present in modes that compute a composite score; tags are optional.
_SEARCH_LINE_RE = re.compile(
    r"^\[\d+\]\s+\("
    r"(?:score:\s*(?P<score>[0-9.]+),\s*)?"
    r"sim:\s*(?P<sim>[0-9.]+|N/A),\s*"
    r"(?P<date>\d{4}-\d{2}-\d{2})"
    r"\)\s*"
    r"\[(?P<agent>[^\]]+)\]\s*"
    r"(?P<content>.*?)"
    r"(?:\s*\|\s*tags:\s*(?P<tags>.*))?$"
)

# Matches a task header line from `task_list`:
#   [STATUS] title → agent (id: <uuid>, priority: N)
_TASK_HEADER_RE = re.compile(
    r"^\[(?P<status>\w+)\]\s+(?P<title>.+?)\s+→\s+(?P<agent>\S+)\s+"
    r"\(id:\s+(?P<id>[^,]+),\s+priority:\s+(?P<priority>-?\d+)\)$"
)

_CAPTURE_ID_RE = re.compile(r"id:\s*(?P<id>[0-9a-f-]{36})", re.IGNORECASE)
_SUPERSEDED_RE = re.compile(r"superseded\s+(?P<n>\d+)", re.IGNORECASE)
_INGEST_DOC_RE = re.compile(r"doc_id:\s*(?P<doc_id>[0-9a-f]+)")
_INGEST_CHUNKS_RE = re.compile(r"chunks:\s*(?P<n>\d+)")
_INGEST_REPLACED_RE = re.compile(r"replaced:\s*(?P<n>\d+)")
_INGEST_AGENT_RE = re.compile(r"agent:\s*(?P<agent>[^)\s]+)")


class AgentMemoryProvider:
    """High-level memory provider for Hermes agents.

    Wraps every tool exposed by the agent-memory MCP server with a typed,
    pythonic interface. Each instance is bound to a default ``agent_id``;
    individual calls accept an ``agent_id`` override to address other agents'
    namespaces. Pass ``agent_id='*'`` to :meth:`search` to run an unscoped
    search across all agents.

    Usage::

        config = MemoryProviderConfig.from_env()
        async with AgentMemoryProvider(config) as memory:
            await memory.capture("user prefers dark mode", tags=["ui", "pref"])
            results = await memory.search("ui preferences", limit=3)
    """

    def __init__(self, config: MemoryProviderConfig) -> None:
        self.config = config
        self._client: MCPMemoryClient | None = None

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> AgentMemoryProvider:
        self._client = MCPMemoryClient(self.config)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.__aexit__(exc_type, exc, tb)

    def _require_client(self) -> MCPMemoryClient:
        if self._client is None:
            raise MemoryConnectionError(
                "provider is not connected; use `async with AgentMemoryProvider(...) as memory:`"
            )
        return self._client

    def _agent(self, agent_id: str | None) -> str:
        return agent_id if agent_id is not None else self.config.agent_id

    # -- thoughts / semantic memory ----------------------------------------

    async def search(
        self,
        query: str,
        *,
        agent_id: str | None = None,
        limit: int = 5,
        relevance_mode: RelevanceMode | None = None,
        relevance_value: float | None = None,
    ) -> SearchResult:
        """Semantic vector search over stored thoughts.

        Pass ``agent_id='*'`` to run an unscoped search across all agents.
        """
        args: dict[str, Any] = {"query": query, "limit": limit}
        if agent_id != "*":
            args["agent_id"] = self._agent(agent_id)
        if relevance_mode is not None:
            args["relevance_mode"] = relevance_mode
        if relevance_value is not None:
            args["relevance_value"] = relevance_value
        body = await self._require_client().call_tool("search_memory", args)
        return _parse_search_result(body)

    async def capture(
        self,
        content: str,
        *,
        tags: Iterable[str] = (),
        agent_id: str | None = None,
    ) -> CaptureResult:
        """Store a thought with an embedding for later retrieval.

        Uses the server's REST ``/capture`` fast path when the transport is
        HTTP or SSE (to skip the MCP round-trip), falling back to the
        ``capture_memory`` MCP tool if the REST call fails.
        """
        agent = self._agent(agent_id)
        tag_list = list(tags)
        if self.config.transport in ("http", "sse"):
            try:
                return await self._capture_via_rest(content, agent, tag_list)
            except MemoryConnectionError:
                # Fall through to the MCP tool path.
                pass
        body = await self._require_client().call_tool(
            "capture_memory",
            {"content": content, "agent_id": agent, "tags": tag_list},
        )
        return _parse_capture_result(body)

    async def _capture_via_rest(
        self,
        content: str,
        agent_id: str,
        tags: list[str],
    ) -> CaptureResult:
        url = f"{self.config.base_url.rstrip('/')}/capture"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.bearer_token:
            headers["Authorization"] = f"Bearer {self.config.bearer_token}"
        try:
            async with httpx.AsyncClient(timeout=self.config.request_timeout_seconds) as client:
                resp = await client.post(
                    url,
                    headers=headers,
                    json={"content": content, "agent_id": agent_id, "tags": tags},
                )
        except httpx.HTTPError as exc:
            raise MemoryConnectionError(f"REST /capture failed: {exc}") from exc
        if resp.status_code >= 400:
            raise MemoryToolError("capture_memory", f"HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        skipped = bool(data.get("skipped", False))
        captured_id = data.get("id")
        superseded = int(data.get("superseded") or 0)
        if skipped:
            message = "Memory skipped — near-duplicate (cosine ≥ 0.85 within 48h)."
        elif superseded:
            message = f"Memory captured (id: {captured_id}, superseded {superseded} stale thoughts)"
        else:
            message = f"Memory captured (id: {captured_id})"
        return CaptureResult(
            id=captured_id, skipped=skipped, superseded=superseded, message=message
        )

    async def ingest_article(
        self,
        *,
        title: str,
        content: str,
        url: str | None = None,
        author: str | None = None,
        published_at: datetime | str | None = None,
        topics: Iterable[str] = (),
        agent_id: str | None = None,
    ) -> IngestResult:
        """Chunk and embed a long-form article into the library namespace.

        ``agent_id`` defaults to ``library`` on the server side; pass an
        explicit value to override.
        """
        args: dict[str, Any] = {"title": title, "content": content}
        if url is not None:
            args["url"] = url
        if author is not None:
            args["author"] = author
        if published_at is not None:
            args["published_at"] = (
                published_at.isoformat() if isinstance(published_at, datetime) else published_at
            )
        topic_list = list(topics)
        if topic_list:
            args["topics"] = topic_list
        if agent_id is not None:
            args["agent_id"] = agent_id
        body = await self._require_client().call_tool("ingest_article", args)
        return _parse_ingest_result(body)

    async def forget(self, thought_id: str) -> str:
        """Soft-delete a thought by UUID."""
        return await self._require_client().call_tool(
            "forget_thought", {"thought_id": thought_id}
        )

    # -- agent state -------------------------------------------------------

    async def get_agent_state(self, agent_id: str | None = None) -> Any:
        """Retrieve the persistent state blob for an agent.

        Returns ``None`` when the agent has no state row.
        """
        body = await self._require_client().call_tool(
            "get_agent_state", {"agent_id": self._agent(agent_id)}
        )
        if body.startswith("No state found"):
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body

    # -- KV store ----------------------------------------------------------

    async def kv_get(self, key: str, *, agent_id: str | None = None) -> Any:
        """Read a key from the per-agent KV store. Returns ``None`` if missing."""
        body = await self._require_client().call_tool(
            "kv_get", {"agent_id": self._agent(agent_id), "key": key}
        )
        if body.startswith("No value found"):
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body

    async def kv_set(self, key: str, value: Any, *, agent_id: str | None = None) -> str:
        """Upsert a key. The server auto-increments the version on update."""
        return await self._require_client().call_tool(
            "kv_set",
            {"agent_id": self._agent(agent_id), "key": key, "value": value},
        )

    async def kv_delete(self, key: str, *, agent_id: str | None = None) -> str:
        return await self._require_client().call_tool(
            "kv_delete", {"agent_id": self._agent(agent_id), "key": key}
        )

    async def kv_list(self, *, agent_id: str | None = None) -> list[KVEntry]:
        body = await self._require_client().call_tool(
            "kv_list", {"agent_id": self._agent(agent_id)}
        )
        return _parse_kv_list(body)

    # -- usage / metering --------------------------------------------------

    async def usage_today(self, *, agent_id: str | None = None) -> UsageSummary:
        """Return today's LLM cost across all agents, or a specific one."""
        args: dict[str, Any] = {}
        if agent_id is not None:
            args["agent_id"] = agent_id
        body = await self._require_client().call_tool("usage_today", args)
        cost = _extract_first_float(body)
        return UsageSummary(agent_id=agent_id, cost_usd=cost, raw=body)

    async def usage_summary(self, *, agent_id: str | None = None) -> UsageSummary:
        """Return the all-time LLM usage summary."""
        args: dict[str, Any] = {}
        if agent_id is not None:
            args["agent_id"] = agent_id
        body = await self._require_client().call_tool("usage_summary", args)
        return _parse_usage_summary(body, agent_id)

    # -- task queue --------------------------------------------------------

    async def task_post(
        self,
        *,
        title: str,
        description: str,
        agent_id: str | None = None,
        task_type: str = "general",
        priority: int = 0,
    ) -> str:
        return await self._require_client().call_tool(
            "task_post",
            {
                "agent_id": self._agent(agent_id),
                "title": title,
                "description": description,
                "task_type": task_type,
                "priority": priority,
            },
        )

    async def task_list(
        self,
        *,
        agent_id: str | None = None,
        status: TaskStatus | None = None,
    ) -> list[Task]:
        args: dict[str, Any] = {}
        if agent_id is not None:
            args["agent_id"] = agent_id
        if status is not None:
            args["status"] = status
        body = await self._require_client().call_tool("task_list", args)
        return _parse_task_list(body)

    # -- pattern learning --------------------------------------------------

    async def capture_observation(
        self,
        *,
        observation_type: ObservationType,
        context: dict[str, Any],
        action: dict[str, Any],
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        args: dict[str, Any] = {
            "agent_id": self._agent(agent_id),
            "observation_type": observation_type,
            "context": context,
            "action": action,
        }
        if metadata is not None:
            args["metadata"] = metadata
        return await self._require_client().call_tool("capture_observation", args)

    async def get_recent_observations(
        self,
        *,
        agent_id: str | None = None,
        days: int = 30,
        observation_type: ObservationType | None = None,
        limit: int = 100,
    ) -> list[Observation]:
        args: dict[str, Any] = {
            "agent_id": self._agent(agent_id),
            "days": days,
            "limit": limit,
        }
        if observation_type is not None:
            args["observation_type"] = observation_type
        body = await self._require_client().call_tool("get_recent_observations", args)
        if body.startswith("No observations"):
            return []
        return [Observation(**row) for row in _parse_json_list(body)]

    async def get_active_patterns(
        self,
        *,
        agent_id: str | None = None,
        min_confidence: float = 0.0,
        pattern_type: ObservationType | None = None,
        limit: int = 50,
    ) -> list[Pattern]:
        args: dict[str, Any] = {
            "agent_id": self._agent(agent_id),
            "min_confidence": min_confidence,
            "limit": limit,
        }
        if pattern_type is not None:
            args["pattern_type"] = pattern_type
        body = await self._require_client().call_tool("get_active_patterns", args)
        if body.startswith("No active patterns"):
            return []
        return [Pattern(**row) for row in _parse_json_list(body)]

    async def search_patterns(
        self,
        query: str,
        *,
        agent_id: str | None = None,
        min_confidence: float = 0.5,
        limit: int = 7,
    ) -> list[Pattern]:
        args: dict[str, Any] = {
            "query": query,
            "agent_id": self._agent(agent_id),
            "min_confidence": min_confidence,
            "limit": limit,
        }
        body = await self._require_client().call_tool("search_patterns", args)
        if body.startswith("No matching patterns"):
            return []
        return [Pattern(**row) for row in _parse_json_list(body)]

    async def update_pattern_confidence(
        self,
        pattern_id: int,
        confidence: float,
        *,
        increment_count: bool = False,
    ) -> str:
        return await self._require_client().call_tool(
            "update_pattern_confidence",
            {
                "pattern_id": pattern_id,
                "confidence": confidence,
                "increment_count": increment_count,
            },
        )

    async def forget_pattern(self, pattern_id: int) -> str:
        return await self._require_client().call_tool(
            "forget_pattern", {"pattern_id": pattern_id}
        )

    async def mine_patterns(self, *, agent_id: str | None = None) -> MineResult:
        """Run one mining round on demand for the given (or default) agent."""
        body = await self._require_client().call_tool(
            "mine_patterns", {"agent_id": self._agent(agent_id)}
        )
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise MemoryProviderError(f"unexpected mine_patterns response: {body!r}") from exc
        return MineResult(**payload)


# ---- parsers ---------------------------------------------------------------


def _parse_search_result(body: str) -> SearchResult:
    mode = "similarity"
    thoughts: list[Thought] = []
    if body.strip() == "No matching memories found.":
        return SearchResult(mode=mode, thoughts=[], raw=body)
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("mode:"):
            mode = line.split(":", 1)[1].strip()
            continue
        m = _SEARCH_LINE_RE.match(line)
        if not m:
            continue
        tags_str = m.group("tags") or ""
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        score = m.group("score")
        sim_raw = m.group("sim")
        sim = float(sim_raw) if sim_raw and sim_raw != "N/A" else None
        try:
            created_at: datetime | None = (
                datetime.fromisoformat(m.group("date")) if m.group("date") else None
            )
        except ValueError:
            created_at = None
        thoughts.append(
            Thought(
                agent_id=m.group("agent"),
                content=(m.group("content") or "").strip(),
                tags=tags,
                created_at=created_at,
                similarity=sim,
                score=float(score) if score else None,
            )
        )
    return SearchResult(mode=mode, thoughts=thoughts, raw=body)


def _parse_capture_result(body: str) -> CaptureResult:
    text = body.strip()
    if "skipped" in text.lower():
        return CaptureResult(skipped=True, message=text)
    id_match = _CAPTURE_ID_RE.search(text)
    sup_match = _SUPERSEDED_RE.search(text)
    return CaptureResult(
        id=id_match.group("id") if id_match else None,
        superseded=int(sup_match.group("n")) if sup_match else 0,
        message=text,
    )


def _parse_ingest_result(body: str) -> IngestResult:
    text = body.strip()
    doc = _INGEST_DOC_RE.search(text)
    chunks = _INGEST_CHUNKS_RE.search(text)
    replaced = _INGEST_REPLACED_RE.search(text)
    agent = _INGEST_AGENT_RE.search(text)
    return IngestResult(
        doc_id=doc.group("doc_id") if doc else "",
        chunks=int(chunks.group("n")) if chunks else 0,
        replaced=int(replaced.group("n")) if replaced else 0,
        agent_id=agent.group("agent") if agent else "",
        message=text,
    )


def _parse_kv_list(body: str) -> list[KVEntry]:
    text = body.strip()
    if not text or text.startswith("No KV entries"):
        return []
    out: list[KVEntry] = []
    for line in text.splitlines():
        if ": " not in line:
            continue
        key, _, raw_value = line.partition(": ")
        try:
            value: Any = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        out.append(KVEntry(key=key.strip(), value=value))
    return out


def _parse_json_list(body: str) -> list[dict[str, Any]]:
    text = body.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return []


def _parse_task_list(body: str) -> list[Task]:
    text = body.strip()
    if not text or text.startswith("No tasks"):
        return []
    out: list[Task] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _TASK_HEADER_RE.match(line)
        if not m:
            i += 1
            continue
        description = ""
        if i + 1 < len(lines) and not _TASK_HEADER_RE.match(lines[i + 1]):
            description = lines[i + 1].strip().rstrip("…")
            i += 2
        else:
            i += 1
        status = m.group("status").lower()
        out.append(
            Task(
                id=m.group("id").strip(),
                agent_id=m.group("agent"),
                title=m.group("title").strip(),
                description=description,
                priority=int(m.group("priority")),
                status=status if status in ("pending", "running", "done", "failed") else None,
            )
        )
    return out


def _parse_usage_summary(body: str, agent_id: str | None) -> UsageSummary:
    def num(label: str) -> int | None:
        m = re.search(rf"{label}\s+([\d,]+)", body)
        if not m:
            return None
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            return None

    cost_match = re.search(r"\$([\d.]+)", body)
    return UsageSummary(
        agent_id=agent_id,
        calls=num("Calls:"),
        tool_calls=num("Tool calls:"),
        input_tokens=num("Input tokens:"),
        output_tokens=num("Output tokens:"),
        cost_usd=float(cost_match.group(1)) if cost_match else None,
        raw=body,
    )


def _extract_first_float(body: str) -> float | None:
    m = re.search(r"\$?(\d+(?:\.\d+)?)", body)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Hermes plugin descriptor
# ---------------------------------------------------------------------------

PLUGIN_NAME = "memory"
PLUGIN_VERSION = "0.1.0"
__version__ = PLUGIN_VERSION


@dataclass
class HermesMemoryPlugin:
    """Plugin descriptor returned to the Hermes host."""

    name: str = PLUGIN_NAME
    version: str = PLUGIN_VERSION

    def register(self, host: Any) -> AgentMemoryProvider:
        """Attach an :class:`AgentMemoryProvider` to a Hermes agent host.

        ``host`` is expected to optionally expose:

        * ``host.config.memory`` (or ``host.config['memory']``) — a sub-block
          whose attributes / keys mirror :class:`MemoryProviderConfig`.
        * ``host.agent_id`` — the agent's identifier; overrides whatever
          ``MemoryProviderConfig.agent_id`` resolved to.
        * ``host.register_service(name, instance)`` — optional hook the
          provider is registered against as ``memory``.

        Falls back to ``HERMES_MEMORY_*`` environment variables for anything
        the host doesn't supply.
        """
        config = _resolve_config(host)
        provider = AgentMemoryProvider(config)
        register = getattr(host, "register_service", None)
        if callable(register):
            register(self.name, provider)
        return provider


def _resolve_config(host: Any) -> MemoryProviderConfig:
    base = MemoryProviderConfig.from_env()

    agent_id = getattr(host, "agent_id", None)
    if agent_id:
        base.agent_id = agent_id

    host_config = getattr(host, "config", None)
    memory_cfg: Any = None
    if host_config is not None:
        if isinstance(host_config, dict):
            memory_cfg = host_config.get("memory")
        else:
            memory_cfg = getattr(host_config, "memory", None)
    if memory_cfg is None:
        return base

    def _get(key: str, default: Any = None) -> Any:
        if isinstance(memory_cfg, dict):
            return memory_cfg.get(key, default)
        return getattr(memory_cfg, key, default)

    for key in (
        "transport",
        "agent_id",
        "base_url",
        "bearer_token",
        "stdio_command",
        "stdio_cwd",
    ):
        value = _get(key)
        if value is not None:
            setattr(base, key, value)

    args = _get("stdio_args")
    if args is not None:
        base.stdio_args = list(args)
    timeout = _get("request_timeout_seconds")
    if timeout is not None:
        base.request_timeout_seconds = float(timeout)
    env = _get("stdio_env")
    if env is not None:
        base.stdio_env = dict(env)

    return base


plugin = HermesMemoryPlugin()


# ---------------------------------------------------------------------------
# Tool schemas & synchronous Hermes adapter
# ---------------------------------------------------------------------------

PROVIDER_NAME = "rzem-ai-hermes-plugin-memory"

_SEARCH_SCHEMA = {
    "name": "memory_search",
    "description": (
        "Semantic vector search over thoughts stored in the agent-memory backend. "
        "Returns excerpts ranked by similarity to `query`. Use this to recall facts "
        "about the user, past decisions, or prior context across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {
                "type": "integer",
                "description": "Maximum thoughts to return (default 5, max 20).",
            },
            "agent_id": {
                "type": "string",
                "description": "Override the agent namespace. Pass '*' to search across all agents.",
            },
        },
        "required": ["query"],
    },
}

_CAPTURE_SCHEMA = {
    "name": "memory_capture",
    "description": (
        "Persist a thought into the agent-memory store with an embedding so it can "
        "be recalled later via memory_search. The server dedupes near-duplicates "
        "(cosine >= 0.85 within 48h)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact or note to remember."},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags used for later filtering.",
            },
        },
        "required": ["content"],
    },
}

_FORGET_SCHEMA = {
    "name": "memory_forget",
    "description": "Soft-delete a thought by its UUID. Use the id returned by memory_capture.",
    "parameters": {
        "type": "object",
        "properties": {
            "thought_id": {"type": "string", "description": "UUID of the thought."},
        },
        "required": ["thought_id"],
    },
}

_KV_GET_SCHEMA = {
    "name": "memory_kv_get",
    "description": "Read a structured value from the per-agent KV store. Returns null if absent.",
    "parameters": {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
}

_KV_SET_SCHEMA = {
    "name": "memory_kv_set",
    "description": "Upsert a JSON-serializable value into the per-agent KV store.",
    "parameters": {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "value": {
                "description": "Any JSON-serializable value (string, number, object, array, bool, null).",
            },
        },
        "required": ["key", "value"],
    },
}

_KV_DELETE_SCHEMA = {
    "name": "memory_kv_delete",
    "description": "Delete a key from the per-agent KV store.",
    "parameters": {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
}

_KV_LIST_SCHEMA = {
    "name": "memory_kv_list",
    "description": "List every key/value pair in the per-agent KV store.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

_USAGE_SCHEMA = {
    "name": "memory_usage_today",
    "description": "Return today's LLM cost (USD) for the active agent namespace.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

_ALL_SCHEMAS = [
    _SEARCH_SCHEMA,
    _CAPTURE_SCHEMA,
    _FORGET_SCHEMA,
    _KV_GET_SCHEMA,
    _KV_SET_SCHEMA,
    _KV_DELETE_SCHEMA,
    _KV_LIST_SCHEMA,
    _USAGE_SCHEMA,
]


class RzemMemoryProvider(MemoryProvider):
    """Synchronous Hermes adapter around the async AgentMemoryProvider."""

    def __init__(self) -> None:
        self._config: Optional[MemoryProviderConfig] = None
        self._provider: Optional[AgentMemoryProvider] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_ready = threading.Event()
        self._ctx_entered = False
        self._initialized = False

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        """Config-only readiness probe (no network)."""
        try:
            cfg = MemoryProviderConfig.from_env()
        except Exception:
            return False
        if cfg.transport == "stdio":
            return bool(cfg.stdio_command)
        return bool(cfg.base_url)

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "base_url",
                "description": "agent-memory MCP server URL (HTTP/SSE transport)",
                "default": "http://127.0.0.1:3000",
                "env_var": "HERMES_MEMORY_BASE_URL",
            },
            {
                "key": "agent_id",
                "description": "Default agent namespace",
                "default": "default",
                "env_var": "HERMES_MEMORY_AGENT_ID",
            },
            {
                "key": "bearer_token",
                "description": "Bearer token if the server is behind auth",
                "secret": True,
                "env_var": "HERMES_MEMORY_BEARER_TOKEN",
            },
        ]

    def _start_loop(self) -> None:
        if self._loop_thread is not None:
            return

        def _runner() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop_ready.set()
            try:
                self._loop.run_forever()
            finally:
                self._loop.close()

        self._loop_thread = threading.Thread(
            target=_runner, daemon=True, name="rzem-memory-loop"
        )
        self._loop_thread.start()
        self._loop_ready.wait(timeout=5)

    def _run(self, coro: Any, *, timeout: Optional[float] = None) -> Any:
        if self._loop is None:
            raise RuntimeError("rzem-memory event loop not started")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        if timeout is None and self._config is not None:
            timeout = self._config.request_timeout_seconds
        return fut.result(timeout=timeout or 30.0)

    def initialize(self, session_id: str, **kwargs) -> None:
        try:
            cfg = MemoryProviderConfig.from_env()
        except Exception as exc:
            logger.warning("rzem-memory: invalid HERMES_MEMORY_* config: %s", exc)
            return

        identity = (
            kwargs.get("agent_identity")
            or kwargs.get("user_id")
            or cfg.agent_id
        )
        if identity:
            cfg.agent_id = identity

        self._config = cfg
        self._provider = AgentMemoryProvider(cfg)
        self._start_loop()

        try:
            self._run(self._provider.__aenter__(), timeout=10.0)
        except Exception as exc:
            logger.warning("rzem-memory: connect to %s failed: %s", cfg.base_url, exc)
            self._provider = None
            return

        self._ctx_entered = True
        self._initialized = True
        logger.info(
            "rzem-memory: connected to %s as agent_id=%s",
            cfg.base_url if cfg.transport != "stdio" else cfg.stdio_command,
            cfg.agent_id,
        )

    def system_prompt_block(self) -> str:
        if not self._initialized:
            return ""
        return (
            "# Persistent memory (rzem-ai/agent-memory-mcp)\n"
            "Cross-session memory backed by PostgreSQL + pgvector. Tools:\n"
            "- `memory_search(query, limit?)` — recall past thoughts by similarity\n"
            "- `memory_capture(content, tags?)` — persist a new fact\n"
            "- `memory_forget(thought_id)` — soft-delete a thought\n"
            "- `memory_kv_get/set/delete/list` — structured per-agent state\n"
            "- `memory_usage_today` — today's LLM cost for this agent_id"
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return list(_ALL_SCHEMAS) if self._initialized else []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._initialized or self._provider is None:
            return tool_error("rzem-memory is not initialized.")
        try:
            return self._dispatch(tool_name, args)
        except MemoryConnectionError as exc:
            return tool_error(f"memory backend unreachable: {exc}")
        except MemoryToolError as exc:
            return tool_error(f"memory tool failed: {exc}")
        except asyncio.TimeoutError:
            return tool_error(f"{tool_name} timed out")
        except Exception as exc:
            logger.exception("rzem-memory tool %s failed", tool_name)
            return tool_error(f"{tool_name} failed: {exc}")

    def _dispatch(self, tool_name: str, args: Dict[str, Any]) -> str:
        provider = self._provider
        assert provider is not None

        if tool_name == "memory_search":
            query = (args.get("query") or "").strip()
            if not query:
                return tool_error("Missing required parameter: query")
            limit = max(1, min(int(args.get("limit", 5)), 20))
            agent_id = args.get("agent_id")
            result = self._run(provider.search(query, limit=limit, agent_id=agent_id))
            return json.dumps(
                {
                    "mode": result.mode,
                    "thoughts": [
                        {
                            "content": t.content,
                            "tags": t.tags,
                            "agent_id": t.agent_id,
                            "created_at": t.created_at.isoformat() if t.created_at else None,
                            "similarity": t.similarity,
                        }
                        for t in result.thoughts
                    ],
                }
            )

        if tool_name == "memory_capture":
            content = (args.get("content") or "").strip()
            if not content:
                return tool_error("Missing required parameter: content")
            tags = list(args.get("tags") or [])
            result = self._run(provider.capture(content, tags=tags))
            return json.dumps(
                {
                    "id": result.id,
                    "skipped": result.skipped,
                    "superseded": result.superseded,
                    "message": result.message,
                }
            )

        if tool_name == "memory_forget":
            tid = (args.get("thought_id") or "").strip()
            if not tid:
                return tool_error("Missing required parameter: thought_id")
            body = self._run(provider.forget(tid))
            return json.dumps({"result": body})

        if tool_name == "memory_kv_get":
            key = (args.get("key") or "").strip()
            if not key:
                return tool_error("Missing required parameter: key")
            value = self._run(provider.kv_get(key))
            return json.dumps({"key": key, "value": value})

        if tool_name == "memory_kv_set":
            key = (args.get("key") or "").strip()
            if not key:
                return tool_error("Missing required parameter: key")
            if "value" not in args:
                return tool_error("Missing required parameter: value")
            body = self._run(provider.kv_set(key, args["value"]))
            return json.dumps({"result": body})

        if tool_name == "memory_kv_delete":
            key = (args.get("key") or "").strip()
            if not key:
                return tool_error("Missing required parameter: key")
            body = self._run(provider.kv_delete(key))
            return json.dumps({"result": body})

        if tool_name == "memory_kv_list":
            entries = self._run(provider.kv_list())
            return json.dumps(
                {"entries": [{"key": e.key, "value": e.value} for e in entries]}
            )

        if tool_name == "memory_usage_today":
            summary = self._run(provider.usage_today())
            return json.dumps(
                {
                    "agent_id": summary.agent_id,
                    "cost_usd": summary.cost_usd,
                    "raw": summary.raw,
                }
            )

        return tool_error(f"Unknown tool: {tool_name}")

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._initialized or self._provider is None:
            return ""
        text = (query or "").strip()
        if not text:
            return ""
        try:
            result = self._run(self._provider.search(text, limit=3), timeout=5.0)
        except Exception as exc:
            logger.debug("rzem-memory prefetch failed: %s", exc)
            return ""
        if not result.thoughts:
            return ""
        lines = ["## Recalled thoughts (rzem-memory)"]
        for t in result.thoughts:
            tags = f" [{', '.join(t.tags)}]" if t.tags else ""
            lines.append(f"- {t.content}{tags}")
        return "\n".join(lines)

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if action != "add" or target != "user" or not content:
            return
        if not self._initialized or self._provider is None or self._loop is None:
            return
        tags = ["user-profile"]
        try:
            asyncio.run_coroutine_threadsafe(
                self._provider.capture(content, tags=tags), self._loop
            )
        except Exception as exc:
            logger.debug("rzem-memory mirror failed: %s", exc)

    def shutdown(self) -> None:
        if self._provider is not None and self._ctx_entered and self._loop is not None:
            try:
                self._run(self._provider.__aexit__(None, None, None), timeout=5.0)
            except Exception:
                pass
            self._ctx_entered = False
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=3.0)


def register(ctx) -> None:
    """Hermes plugin entry point — invoked by ``plugins.memory._ProviderCollector``."""
    ctx.register_memory_provider(RzemMemoryProvider())


__all__ = [
    "AgentMemoryProvider",
    "CaptureResult",
    "HermesMemoryPlugin",
    "IngestResult",
    "KVEntry",
    "MCPMemoryClient",
    "MemoryConnectionError",
    "MemoryProviderConfig",
    "MemoryProviderError",
    "MemoryToolError",
    "MineResult",
    "Observation",
    "ObservationType",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "PROVIDER_NAME",
    "Pattern",
    "RelevanceMode",
    "RzemMemoryProvider",
    "SearchResult",
    "Task",
    "TaskStatus",
    "Thought",
    "UsageSummary",
    "plugin",
    "register",
]
