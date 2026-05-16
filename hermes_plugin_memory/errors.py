"""Exception types for the memory provider plugin."""
from __future__ import annotations


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
