# rzem-ai-hermes-plugin-memory

Memory provider plugin for the [Hermes](https://github.com/rzem-ai) AI agent
platform. Wraps every tool offered by the
[`agent-memory-mcp`](https://github.com/rzem-ai/rzem-ai-agent-memory-mcp) server
behind a typed async Python interface so a Hermes agent can read and write its
shared memory store (PostgreSQL + pgvector) without speaking MCP directly.

## What you get

`AgentMemoryProvider` exposes the full surface of the agent-memory MCP server:

| Area | Methods |
|---|---|
| Semantic memory | `search`, `capture`, `ingest_article`, `forget` |
| Agent state | `get_agent_state` |
| KV store | `kv_get`, `kv_set`, `kv_delete`, `kv_list` |
| Usage metering | `usage_today`, `usage_summary` |
| Task queue | `task_post`, `task_list` |
| Pattern learning | `capture_observation`, `get_recent_observations`, `get_active_patterns`, `search_patterns`, `update_pattern_confidence`, `forget_pattern`, `mine_patterns` |

Results are parsed into Pydantic models (`Thought`, `SearchResult`,
`CaptureResult`, `Pattern`, `Observation`, `Task`, `UsageSummary`,
`MineResult`, `KVEntry`, `IngestResult`) — no scraping the raw MCP text in
caller code.

## Install

```bash
pip install -e .
```

Runtime dependencies: `mcp>=1.27.0`, `pydantic>=2.7.0`, `httpx>=0.27.0`.
Requires Python 3.11+.

## Quick start

```python
import asyncio

from hermes_plugin_memory import AgentMemoryProvider, MemoryProviderConfig


async def main() -> None:
    config = MemoryProviderConfig(
        transport="http",
        base_url="http://memory.local:3002",
        agent_id="angus",
    )
    async with AgentMemoryProvider(config) as memory:
        await memory.capture("user prefers dark mode", tags=["ui", "preference"])
        results = await memory.search("ui preferences", limit=3)
        for thought in results.thoughts:
            print(f"{thought.created_at}: {thought.content}")

        await memory.kv_set("last_seen", "2026-05-16T12:00:00Z")
        seen = await memory.kv_get("last_seen")

        await memory.capture_observation(
            observation_type="preference",
            context={"topic": "theme"},
            action={"choice": "dark"},
        )
        mined = await memory.mine_patterns()
        print(mined)


asyncio.run(main())
```

## Configuration

The provider is configured via `MemoryProviderConfig`. When loaded via
`MemoryProviderConfig.from_env()` the following environment variables are
read (prefix `HERMES_MEMORY_`):

| Variable | Default | Notes |
|---|---|---|
| `HERMES_MEMORY_TRANSPORT` | `http` | `http` (Streamable HTTP) / `sse` (legacy) / `stdio` |
| `HERMES_MEMORY_AGENT_ID` | `default` | Default agent namespace |
| `HERMES_MEMORY_BASE_URL` | `http://127.0.0.1:3002` | MCP server base URL (HTTP / SSE transports) |
| `HERMES_MEMORY_BEARER_TOKEN` | unset | Optional Bearer token for the `Authorization` header |
| `HERMES_MEMORY_TIMEOUT_SECONDS` | `30` | Per-request timeout |
| `HERMES_MEMORY_STDIO_COMMAND` | `node` | Subprocess command (stdio transport) |
| `HERMES_MEMORY_STDIO_ARGS` | `dist/stdio.js` | Space-separated subprocess args |
| `HERMES_MEMORY_STDIO_CWD` | unset | Working directory for the subprocess |

When `transport` is `http` or `sse`, `capture()` posts to the server's
REST `/capture` endpoint to skip the MCP round-trip on the hot path, falling
back to the MCP `capture_memory` tool if the REST call fails.

## Hermes plugin discovery

The package registers under the `hermes.plugins` entry-point group:

```toml
[project.entry-points."hermes.plugins"]
memory = "hermes_plugin_memory.plugin:plugin"
```

The Hermes host calls `plugin.register(host)`. The plugin resolves its
config from `host.config.memory` (object or dict), then falls back to
`HERMES_MEMORY_*` env vars. If the host exposes a `register_service(name,
instance)` hook the provider is registered there as `memory`; the
provider instance is also returned.

## Transports

- **`http`** — Streamable HTTP at `<base_url>/mcp` (default; current MCP
  protocol version).
- **`sse`** — legacy SSE at `<base_url>/sse` for older MCP clients.
- **`stdio`** — spawns the MCP server as a subprocess and talks over its
  stdio pipes.

All three are session-managed by the `mcp` Python SDK; the provider holds a
`ClientSession` open for its lifetime as an async context manager.

## Tests

```bash
pip install -e .[dev]
pytest
```

The test suite exercises the response parsers and config loader against
fixture text taken from the agent-memory MCP server's documented output
formats.
