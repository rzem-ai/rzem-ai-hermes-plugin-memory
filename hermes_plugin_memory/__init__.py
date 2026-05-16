"""Memory provider plugin for Hermes, backed by the agent-memory MCP server."""
from .config import MemoryProviderConfig
from .errors import (
    MemoryConnectionError,
    MemoryProviderError,
    MemoryToolError,
)
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
from .plugin import PLUGIN_NAME, PLUGIN_VERSION, HermesMemoryPlugin, plugin
from .provider import AgentMemoryProvider

__version__ = PLUGIN_VERSION

__all__ = [
    "AgentMemoryProvider",
    "CaptureResult",
    "HermesMemoryPlugin",
    "IngestResult",
    "KVEntry",
    "MemoryConnectionError",
    "MemoryProviderConfig",
    "MemoryProviderError",
    "MemoryToolError",
    "MineResult",
    "Observation",
    "ObservationType",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "Pattern",
    "RelevanceMode",
    "SearchResult",
    "Task",
    "TaskStatus",
    "Thought",
    "UsageSummary",
    "plugin",
]
