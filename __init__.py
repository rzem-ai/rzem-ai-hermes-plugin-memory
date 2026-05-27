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
import threading
from typing import Any, Dict, List, Optional

from hermes_plugin_memory import (
    AgentMemoryProvider,
    MemoryConnectionError,
    MemoryProviderConfig,
    MemoryToolError,
)

try:
    from agent.memory_provider import MemoryProvider
    from tools.registry import tool_error
except ImportError:
    MemoryProvider = object  # type: ignore[misc,assignment]

    def tool_error(message: str) -> str:  # type: ignore[misc]
        return json.dumps({"error": message})

logger = logging.getLogger(__name__)

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


__all__ = ["PROVIDER_NAME", "RzemMemoryProvider", "register"]
