"""Memory provider — the high-level surface a Hermes agent talks to."""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Iterable

import httpx

from .client import MCPMemoryClient
from .config import MemoryProviderConfig
from .errors import MemoryConnectionError, MemoryProviderError, MemoryToolError
from .models import (
    CaptureResult,
    IngestResult,
    KVEntry,
    MineResult,
    Observation,
    ObservationType,
    Pattern,
    RelevanceMode,
    SearchResult,
    Task,
    TaskStatus,
    Thought,
    UsageSummary,
)

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
