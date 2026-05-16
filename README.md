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

## Installation

The plugin is a client. It needs a running `agent-memory-mcp` server to talk
to. The full path from a clean machine to "operational" is below.

### 1. Prerequisites

| Component | Version | Notes |
|---|---|---|
| Python | 3.11+ | The plugin and its async runtime |
| `pip` / `venv` | bundled with Python | Or any equivalent (`uv`, `pipx`, Poetry) |
| Git | any recent | For cloning |
| `agent-memory-mcp` server | latest `main` | Reachable on the network or as a subprocess |
| PostgreSQL | 16+ | Required by the MCP server, not by this plugin |
| pgvector | 0.8+ | Postgres extension; required by the MCP server |
| Embedding gateway | LiteLLM or Ollama | Required by the MCP server (768-dim) |

The plugin itself only needs Python; the rest are listed because nothing
useful happens end-to-end without them. If you already have the MCP server
running and reachable, you can skip the server bring-up step.

### 2. Clone the plugin

```bash
git clone git@github.com:rzem-ai/rzem-ai-hermes-plugin-memory.git
cd rzem-ai-hermes-plugin-memory
```

### 3. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
python -m pip install --upgrade pip
```

### 4. Install the plugin

For use as a library:

```bash
pip install -e .
```

For development (adds `pytest`, `pytest-asyncio`, `ruff`):

```bash
pip install -e ".[dev]"
```

Runtime dependencies are pinned through `pyproject.toml`: `mcp>=1.27.0`,
`pydantic>=2.7.0`, `httpx>=0.27.0`.

### 5. Bring up (or point to) the agent-memory MCP server

The plugin is a thin client over MCP. Skip this step if a server is already
deployed at a URL you can reach.

```bash
git clone git@github.com:rzem-ai/rzem-ai-agent-memory-mcp.git
cd rzem-ai-agent-memory-mcp
npm install
cp mcp.example.toml mcp.toml
# edit mcp.toml — database, embeddings provider, LLM gateway
npm run build
npm run start:http        # or start:stdio for subprocess mode
```

Defaults: HTTP on `:3002`, exposing `/mcp`, `/sse`, `/messages`, `/health`,
and `/capture`. Full server setup (Postgres schema, systemd unit,
embedding/LLM gateways) is documented in that repo's `docs/deployment.md`.

Confirm the server is alive:

```bash
curl http://127.0.0.1:3002/health
# {"status":"ok"}
```

### 6. Configure the plugin

The provider reads `HERMES_MEMORY_*` environment variables via
`MemoryProviderConfig.from_env()`. The minimal HTTP setup:

```bash
export HERMES_MEMORY_TRANSPORT=http
export HERMES_MEMORY_BASE_URL=http://127.0.0.1:3002
export HERMES_MEMORY_AGENT_ID=angus            # your agent's id
# export HERMES_MEMORY_BEARER_TOKEN=secret     # if the server is behind auth
```

For the stdio transport (server spawned as a subprocess):

```bash
export HERMES_MEMORY_TRANSPORT=stdio
export HERMES_MEMORY_STDIO_COMMAND=node
export HERMES_MEMORY_STDIO_ARGS="dist/stdio.js --config /etc/agent-memory/config.toml"
export HERMES_MEMORY_STDIO_CWD=/path/to/rzem-ai-agent-memory-mcp
```

Full variable list is in [Configuration](#configuration).

### 7. Verify the install

Run the test suite — exercises the parsers and config loader, no network
required:

```bash
pytest
```

Then run an end-to-end smoke against the live MCP server:

```bash
python -c "
import asyncio
from hermes_plugin_memory import AgentMemoryProvider, MemoryProviderConfig

async def main():
    async with AgentMemoryProvider(MemoryProviderConfig.from_env()) as memory:
        result = await memory.capture('plugin install smoke test', tags=['smoke'])
        print('capture:', result.message)
        hits = await memory.search('smoke test', limit=1)
        print('search:', hits.thoughts)

asyncio.run(main())
"
```

A successful run prints a `Memory captured` line (or a `Memory skipped`
line on the second run, which is also correct — the server's 48 h dedup
kicked in) and a single search hit. At that point the plugin is
operational.

### 8. Wire into Hermes

Once installed in the same Python environment Hermes is running in, the
host will discover the plugin automatically via the `hermes.plugins`
entry point (see [Hermes plugin discovery](#hermes-plugin-discovery)) and
call `plugin.register(host)`. No extra registration code is required on
the Hermes side.

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

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `MemoryConnectionError: Failed to connect …` on HTTP | Server not running, wrong `BASE_URL`, or firewall | `curl <base_url>/health`; check the server's logs |
| Same error on stdio | Bad `STDIO_COMMAND` / `STDIO_ARGS` / `STDIO_CWD`, or the server crashed on boot | Run the configured command by hand and read stderr |
| `MemoryToolError: capture_memory: HTTP 401` | Server is behind auth, no bearer token configured | Set `HERMES_MEMORY_BEARER_TOKEN` |
| `Memory skipped — near-duplicate …` | Dedup fired (cosine ≥ 0.85 within 48 h) — not an error | Working as designed; vary the input or wait out the window |
| Embeddings / mining errors in the server log | LiteLLM or Ollama unreachable | Check `[embeddings]` / `[llm]` sections of `mcp.toml` |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The test suite exercises the response parsers and config loader against
fixture text taken from the agent-memory MCP server's documented output
formats.
