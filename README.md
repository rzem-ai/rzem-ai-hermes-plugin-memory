# rzem-ai-hermes-plugin-memory

Memory provider plugin for the [Hermes](https://github.com/rzem-ai) AI agent
platform. Two complementary surfaces in one package:

1. **A Hermes-native `MemoryProvider`** that the conversation loop drives
   each turn (recall on `prefetch`, persistence on `sync_turn`, the
   `memory_*` tools exposed to the model).
2. **A typed async Python library** (`AgentMemoryProvider`) wrapping every
   tool of the [`agent-memory-mcp`](https://github.com/rzem-ai/rzem-ai-agent-memory-mcp)
   server so other plugins, skills, and scripts can talk to the shared
   PostgreSQL + pgvector memory store without speaking MCP directly.

## What you get

### Tools exposed to the agent

When activated as the memory provider, the plugin registers these tools
on the agent's model:

| Tool | Purpose |
|---|---|
| `memory_search(query, limit?, agent_id?)` | Semantic vector recall over stored thoughts |
| `memory_capture(content, tags?)` | Persist a fact with embedding + dedup |
| `memory_forget(thought_id)` | Soft-delete a thought by UUID |
| `memory_kv_get(key)` / `memory_kv_set(key, value)` | Read/write structured per-agent KV state |
| `memory_kv_delete(key)` / `memory_kv_list()` | Manage the KV store |
| `memory_usage_today()` | Today's LLM cost for this `agent_id` |

A `prefetch()` hook also runs each turn — when the user's message returns
relevant thoughts, they are injected into the system prompt for that turn
under a `## Recalled thoughts (rzem-memory)` heading. Built-in user-profile
writes are mirrored as captures via `on_memory_write`.

### Library surface

`AgentMemoryProvider` exposes the full surface of the agent-memory MCP
server for direct use:

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

The plugin is a client. It needs a running `agent-memory-mcp` server to
talk to. The full path from a clean machine to "operational" is below.

### 1. Prerequisites

| Component | Version | Notes |
|---|---|---|
| Hermes | latest `main` | Host agent; ships the `MemoryProvider` ABC this plugin implements |
| Python | 3.11+ | The plugin and its async runtime |
| `uv` | any recent | Used to install into Hermes's venv |
| Git | any recent | For cloning |
| `agent-memory-mcp` server | latest `main` | Reachable on the network or as a subprocess |
| PostgreSQL | 16+ | Required by the MCP server, not by this plugin |
| pgvector | 0.8+ | Postgres extension; required by the MCP server |
| Embedding gateway | LiteLLM or Ollama | Required by the MCP server (768-dim) |

The plugin itself only needs Python; the rest are listed because nothing
useful happens end-to-end without them. If you already have the MCP server
running and reachable, you can skip the server bring-up step.

### 2. Clone into the Hermes plugin directory

```bash
hermes plugins install https://github.com/rzem-ai/rzem-ai-hermes-plugin-memory.git --no-enable
```

That clones into `$HERMES_HOME/plugins/rzem-ai-hermes-plugin-memory/`.
Equivalent manual install:

```bash
git clone https://github.com/rzem-ai/rzem-ai-hermes-plugin-memory.git \
  "$HERMES_HOME/plugins/rzem-ai-hermes-plugin-memory"
```

> `hermes plugins install` also leaves the plugin marked under
> `plugins.enabled:` in `config.yaml`. Memory providers are gated by a
> different key (`memory.provider`, see step 5), so remove it from
> `plugins.enabled:` — leaving it there is harmless but misleading.

### 3. Install the package into Hermes's venv

The plugin code (under `hermes_plugin_memory/`) ships pinned dependencies
(`mcp`, `pydantic`, `httpx`) via `pyproject.toml`. Install it editable into
the Hermes venv so the imports inside the top-level `__init__.py` resolve:

```bash
uv pip install --python <hermes-venv>/bin/python -e \
  "$HERMES_HOME/plugins/rzem-ai-hermes-plugin-memory"
```

For a default `scripts/install.sh` install the venv is
`~/.hermes/bin/venv`. On a split install
(`--dir /srv/sprites/hermes/bin --hermes-home /srv/sprites/hermes/agents/angus`)
it lives at `<install-dir>/venv`.

For development (adds `pytest`, `pytest-asyncio`, `ruff`):

```bash
uv pip install --python <hermes-venv>/bin/python -e \
  "$HERMES_HOME/plugins/rzem-ai-hermes-plugin-memory[dev]"
```

### 4. Bring up (or point to) the agent-memory MCP server

The plugin is a thin client over MCP. Skip this step if a server is
already deployed at a URL you can reach.

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

### 5. Activate it as the memory provider

Edit `$HERMES_HOME/config.yaml`:

```yaml
memory:
  provider: rzem-ai-hermes-plugin-memory
plugins:
  enabled: []      # leave empty for this plugin — it's selected via memory.provider
  disabled: []
```

Or use the wizard:

```bash
hermes memory setup
```

It picks up the schema from `RzemMemoryProvider.get_config_schema()` and
prompts for `HERMES_MEMORY_BASE_URL`, `HERMES_MEMORY_AGENT_ID`, and an
optional bearer token.

Only one external memory provider can be active at a time — that's the
`kind: exclusive` contract declared in `plugin.yaml`. Switching providers
is a single edit to `memory.provider`.

### 6. Configure connection settings

Append to `$HERMES_HOME/.env`:

```bash
# rzem-ai memory provider → local agent-memory-mcp on :3002
HERMES_MEMORY_AGENT_ID=<your-agent-id>      # e.g. 'angus'
# HERMES_MEMORY_BASE_URL=http://127.0.0.1:3002  # default — only set to override
# HERMES_MEMORY_BEARER_TOKEN=secret             # only if the server is behind auth
```

The defaults handle a local server on the standard port, so `AGENT_ID` is
typically the only variable you need to set. The full variable list is in
[Configuration](#configuration).

### 7. Verify the install

Run the test suite — exercises the parsers and config loader, no network
required:

```bash
pytest "$HERMES_HOME/plugins/rzem-ai-hermes-plugin-memory"
```

Then run an end-to-end smoke against the live MCP server through the
Hermes discovery path:

```bash
HERMES_HOME=$HERMES_HOME HERMES_MEMORY_AGENT_ID=<your-agent-id> \
  <hermes-venv>/bin/python -c "
import sys
sys.path.insert(0, '<hermes-install-dir>/bin')
from plugins.memory import load_memory_provider
p = load_memory_provider('rzem-ai-hermes-plugin-memory')
p.initialize('smoke-test', platform='cli', agent_context='primary')
print(p.handle_tool_call('memory_capture', {'content': 'plugin smoke test', 'tags': ['smoke']}))
print(p.handle_tool_call('memory_search', {'query': 'smoke test', 'limit': 1}))
p.shutdown()
"
```

A successful run prints a `Memory captured` line (or a `Memory skipped`
line on the second run, which is also correct — the server's 48h dedup
kicked in) and a single search hit. At that point the plugin is
operational and Hermes will pick it up on the next session.

## Discovery & lifecycle

The plugin is discovered by Hermes's memory provider scanner at
`plugins/memory/__init__.py`: it walks `$HERMES_HOME/plugins/<name>/`,
detects directories whose `__init__.py` mentions the string
`register_memory_provider` (cheap text scan, no import), and calls the
package's `register(ctx)` to harvest a provider instance.

The directory name becomes the registry key written to
`memory.provider` in `config.yaml`. Provider discovery does NOT use
Python entry points — the `[project.entry-points."hermes.plugins"]`
metadata in `pyproject.toml` is preserved for forwards-compatibility but
is not what activates the plugin inside a running Hermes.

Lifecycle methods invoked by `MemoryManager` (see
`agent/memory_provider.py`):

| Method | When | Implementation notes |
|---|---|---|
| `is_available()` | Discovery (no network) | Returns True when `HERMES_MEMORY_BASE_URL` (or `_STDIO_COMMAND`) resolves |
| `initialize(session_id, **kwargs)` | Agent startup | Opens an async context on a daemon-thread event loop |
| `system_prompt_block()` | System prompt assembly | Static tool inventory block |
| `prefetch(query)` | Before each turn | Synchronous `search(query, limit=3)` with a 5s ceiling |
| `get_tool_schemas()` | Once after init | The eight `memory_*` schemas |
| `handle_tool_call(name, args)` | Per tool call | Bridged sync via `asyncio.run_coroutine_threadsafe` |
| `on_memory_write(action, target, content)` | Built-in memory writes | Mirrors user-profile adds as captures with `tags=['user-profile']` |
| `shutdown()` | Agent exit | `__aexit__` the provider, stop the loop, join the thread |

Async↔sync bridge: all calls into the `mcp` SDK happen on a single daemon
event loop spawned in `initialize()`. The `AgentMemoryProvider`'s
`__aenter__/__aexit__` runs on that loop; sync callers block via
`fut.result(timeout=...)`.

## Library usage (outside the Hermes loop)

If you only want the typed MCP client (e.g. from a skill, a test, or
another plugin), import the inner package directly:

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

The Hermes adapter does NOT expose the pattern learning or task queue
tools to the model — they are still accessible from Python via this
library surface.

## Configuration

`RzemMemoryProvider.get_config_schema()` declares three fields walked by
`hermes memory setup`. Internally the provider builds a
`MemoryProviderConfig` via `MemoryProviderConfig.from_env()`, reading
`HERMES_MEMORY_*` variables:

| Variable | Default | Notes |
|---|---|---|
| `HERMES_MEMORY_TRANSPORT` | `http` | `http` (Streamable HTTP) / `sse` (legacy) / `stdio` |
| `HERMES_MEMORY_AGENT_ID` | `default` | Default agent namespace — also auto-overridden by Hermes's `agent_identity` / `user_id` kwargs |
| `HERMES_MEMORY_BASE_URL` | `http://127.0.0.1:3002` | MCP server base URL (HTTP / SSE transports) |
| `HERMES_MEMORY_BEARER_TOKEN` | unset | Optional Bearer token for the `Authorization` header |
| `HERMES_MEMORY_TIMEOUT_SECONDS` | `30` | Per-request timeout |
| `HERMES_MEMORY_STDIO_COMMAND` | `node` | Subprocess command (stdio transport) |
| `HERMES_MEMORY_STDIO_ARGS` | `dist/stdio.js` | Space-separated subprocess args |
| `HERMES_MEMORY_STDIO_CWD` | unset | Working directory for the subprocess |

When `transport` is `http` or `sse`, `capture()` posts to the server's
REST `/capture` endpoint to skip the MCP round-trip on the hot path,
falling back to the MCP `capture_memory` tool if the REST call fails.

## Transports

- **`http`** — Streamable HTTP at `<base_url>/mcp` (default; current MCP
  protocol version).
- **`sse`** — legacy SSE at `<base_url>/sse` for older MCP clients.
- **`stdio`** — spawns the MCP server as a subprocess and talks over its
  stdio pipes.

All three are session-managed by the `mcp` Python SDK; the provider holds
a `ClientSession` open for its lifetime as an async context manager.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `hermes plugins list` doesn't show the plugin | `plugin.yaml` missing from the top of the cloned dir | Re-run `hermes plugins install …` or restore the file |
| Loaded by discovery but `is_available()` returns False | `HERMES_MEMORY_BASE_URL` unset and stdio not configured | Set the env var in `$HERMES_HOME/.env` |
| `rzem-memory: connect to http://… failed` in logs | Server not running, wrong URL, or firewall | `curl <base_url>/health`; check the server's logs |
| `rzem-memory is not initialized.` from a tool call | `initialize()` failed earlier (see warning above) | Fix the underlying connection error then restart the session |
| `MemoryToolError: capture_memory: HTTP 401` | Server is behind auth, no bearer token configured | Set `HERMES_MEMORY_BEARER_TOKEN` |
| `Memory skipped — near-duplicate …` | Dedup fired (cosine ≥ 0.85 within 48h) — not an error | Working as designed; vary the input or wait out the window |
| Embeddings / mining errors in the server log | LiteLLM or Ollama unreachable | Check `[embeddings]` / `[llm]` sections of `mcp.toml` |

## Tests

```bash
cd "$HERMES_HOME/plugins/rzem-ai-hermes-plugin-memory"
uv pip install --python <hermes-venv>/bin/python -e ".[dev]"
pytest
```

The test suite exercises the response parsers and config loader against
fixture text taken from the agent-memory MCP server's documented output
formats. The Hermes adapter in the top-level `__init__.py` is exercised
by the smoke test in [step 7 above](#7-verify-the-install).
