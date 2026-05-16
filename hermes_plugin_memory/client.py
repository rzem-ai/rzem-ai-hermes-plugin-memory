"""MCP client wrapper supporting stdio, Streamable HTTP, and legacy SSE transports."""
from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import MemoryProviderConfig
from .errors import MemoryConnectionError, MemoryToolError


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
