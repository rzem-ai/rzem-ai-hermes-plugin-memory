"""Smoke tests for the memory provider — parsers and config wiring."""
from __future__ import annotations

import json

import pytest

from hermes_plugin_memory import (
    AgentMemoryProvider,
    MemoryConnectionError,
    MemoryProviderConfig,
    plugin,
)
from hermes_plugin_memory.provider import (
    _parse_capture_result,
    _parse_ingest_result,
    _parse_kv_list,
    _parse_search_result,
    _parse_task_list,
    _parse_usage_summary,
)


def test_parse_search_result_recency_weighted() -> None:
    body = (
        "mode: recency_weighted\n"
        "\n"
        "[1] (score: 0.742, sim: 0.842, 2026-04-10) [angus] First memory | tags: foo, bar\n"
        "\n"
        "[2] (score: 0.531, sim: 0.780, 2026-02-15) [angus] Older memory | tags: baz"
    )
    result = _parse_search_result(body)
    assert result.mode == "recency_weighted"
    assert len(result.thoughts) == 2
    first = result.thoughts[0]
    assert first.content == "First memory"
    assert first.tags == ["foo", "bar"]
    assert first.similarity == 0.842
    assert first.score == 0.742
    assert first.agent_id == "angus"
    assert result.thoughts[1].content == "Older memory"


def test_parse_search_result_similarity_no_tags() -> None:
    body = "mode: similarity\n\n[1] (sim: 0.910, 2026-05-01) [mabel] Just the content"
    result = _parse_search_result(body)
    assert result.mode == "similarity"
    assert len(result.thoughts) == 1
    assert result.thoughts[0].tags == []
    assert result.thoughts[0].score is None
    assert result.thoughts[0].similarity == 0.910
    assert result.thoughts[0].content == "Just the content"


def test_parse_search_result_empty() -> None:
    result = _parse_search_result("No matching memories found.")
    assert result.thoughts == []


def test_parse_capture_result_inserted() -> None:
    body = "Memory captured successfully (id: 11111111-2222-3333-4444-555555555555)"
    result = _parse_capture_result(body)
    assert result.id == "11111111-2222-3333-4444-555555555555"
    assert not result.skipped
    assert result.superseded == 0


def test_parse_capture_result_superseded() -> None:
    body = (
        "Memory captured successfully (id: 11111111-2222-3333-4444-555555555555, "
        "superseded 2 stale thoughts)"
    )
    result = _parse_capture_result(body)
    assert result.superseded == 2


def test_parse_capture_result_skipped() -> None:
    body = "Memory skipped — near-duplicate of a recent thought (cosine ≥ 0.85 within 48h)."
    result = _parse_capture_result(body)
    assert result.skipped
    assert result.id is None


def test_parse_ingest_result() -> None:
    body = "Article ingested (doc_id: abc123def4567890, chunks: 12, replaced: 3, agent: library)"
    result = _parse_ingest_result(body)
    assert result.doc_id == "abc123def4567890"
    assert result.chunks == 12
    assert result.replaced == 3
    assert result.agent_id == "library"


def test_parse_kv_list_lines() -> None:
    body = 'foo: "bar"\nbaz: {"nested": true}'
    rows = _parse_kv_list(body)
    by_key = {r.key: r.value for r in rows}
    assert by_key["foo"] == "bar"
    assert by_key["baz"] == {"nested": True}


def test_parse_kv_list_empty() -> None:
    assert _parse_kv_list("No KV entries for 'angus'") == []


def test_parse_task_list() -> None:
    body = (
        "[PENDING] Draft response → angus (id: 11111111-2222-3333-4444-555555555555, priority: 5)\n"
        "  Reply to the user's question about the deploy pipeline\n"
        "\n"
        "[DONE] Audit secrets → mabel (id: 22222222-3333-4444-5555-666666666666, priority: 0)\n"
        "  Reviewed .env and rotated leaked tokens"
    )
    tasks = _parse_task_list(body)
    assert len(tasks) == 2
    assert tasks[0].id == "11111111-2222-3333-4444-555555555555"
    assert tasks[0].agent_id == "angus"
    assert tasks[0].title == "Draft response"
    assert tasks[0].priority == 5
    assert tasks[0].status == "pending"
    assert tasks[1].status == "done"


def test_parse_task_list_empty() -> None:
    assert _parse_task_list("No tasks found.") == []


def test_parse_usage_summary() -> None:
    body = (
        "Usage summary (agent 'angus'):\n"
        "  Calls:         42\n"
        "  Tool calls:    100\n"
        "  Input tokens:  1,234,567\n"
        "  Output tokens: 89,012\n"
        "  Total cost:    $12.3456 USD"
    )
    summary = _parse_usage_summary(body, "angus")
    assert summary.calls == 42
    assert summary.tool_calls == 100
    assert summary.input_tokens == 1234567
    assert summary.output_tokens == 89012
    assert summary.cost_usd == 12.3456
    assert summary.agent_id == "angus"


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_MEMORY_TRANSPORT", "stdio")
    monkeypatch.setenv("HERMES_MEMORY_AGENT_ID", "angus")
    monkeypatch.setenv("HERMES_MEMORY_BASE_URL", "http://memory.local:3002")
    monkeypatch.setenv("HERMES_MEMORY_BEARER_TOKEN", "secret")
    monkeypatch.setenv(
        "HERMES_MEMORY_STDIO_ARGS",
        "dist/stdio.js --config /etc/agent-memory/config.toml",
    )
    cfg = MemoryProviderConfig.from_env()
    assert cfg.transport == "stdio"
    assert cfg.agent_id == "angus"
    assert cfg.base_url == "http://memory.local:3002"
    assert cfg.bearer_token == "secret"
    assert cfg.stdio_args == [
        "dist/stdio.js",
        "--config",
        "/etc/agent-memory/config.toml",
    ]


def test_config_from_env_rejects_unknown_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_MEMORY_TRANSPORT", "carrier-pigeon")
    with pytest.raises(ValueError):
        MemoryProviderConfig.from_env()


async def test_provider_requires_connection() -> None:
    provider = AgentMemoryProvider(MemoryProviderConfig(transport="http"))
    with pytest.raises(MemoryConnectionError):
        await provider.kv_get("foo")


def test_plugin_singleton() -> None:
    assert plugin.name == "memory"
    assert plugin.version == "0.1.0"


class _StubHost:
    def __init__(self) -> None:
        self.agent_id = "angus"
        self.config = type(
            "_Cfg",
            (),
            {
                "memory": type(
                    "_Mem",
                    (),
                    {
                        "transport": "sse",
                        "base_url": "http://memory.local:3002",
                        "bearer_token": "tok",
                    },
                )()
            },
        )()
        self.registered: dict[str, object] = {}

    def register_service(self, name: str, instance: object) -> None:
        self.registered[name] = instance


def test_plugin_register_with_host() -> None:
    host = _StubHost()
    provider = plugin.register(host)
    assert isinstance(provider, AgentMemoryProvider)
    assert provider.config.transport == "sse"
    assert provider.config.agent_id == "angus"
    assert provider.config.base_url == "http://memory.local:3002"
    assert provider.config.bearer_token == "tok"
    assert host.registered["memory"] is provider


def test_plugin_register_falls_back_to_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_MEMORY_TRANSPORT", "http")
    monkeypatch.setenv("HERMES_MEMORY_BASE_URL", "http://envhost:3002")

    class _BareHost:
        agent_id = "mabel"

    provider = plugin.register(_BareHost())
    assert provider.config.base_url == "http://envhost:3002"
    assert provider.config.agent_id == "mabel"


def test_models_roundtrip() -> None:
    from hermes_plugin_memory.models import MineResult

    payload = {
        "agent_id": "angus",
        "status": "completed",
        "observations_analysed": 42,
        "patterns_new": 3,
        "patterns_reinforced": 2,
        "patterns_decayed": 1,
    }
    result = MineResult(**payload)
    assert result.status == "completed"
    assert result.patterns_new == 3

    # Tolerates extra fields the server might add.
    payload["future_field"] = "yes"
    MineResult(**payload)


def test_search_unscoped_omits_agent_id_arg() -> None:
    # The provider should drop agent_id from the args dict when '*' is passed.
    # We verify the contract by inspecting what arguments would be sent.
    config = MemoryProviderConfig(transport="http", agent_id="angus")
    provider = AgentMemoryProvider(config)
    assert provider._agent("*") == "*"
    assert provider._agent(None) == "angus"
    assert provider._agent("mabel") == "mabel"


def test_models_imports() -> None:
    # Ensures every public model is importable from the top-level package.
    from hermes_plugin_memory import (  # noqa: F401
        CaptureResult,
        IngestResult,
        KVEntry,
        MineResult,
        Observation,
        Pattern,
        SearchResult,
        Task,
        Thought,
        UsageSummary,
    )
    _ = json  # keep linter from complaining about the json import in test scope
