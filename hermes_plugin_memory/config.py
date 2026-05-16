"""Configuration loader for the memory provider plugin."""
from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Literal

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
