"""Hermes plugin entry point.

Hermes discovers plugins via the ``hermes.plugins`` entry-point group. The
module-level :data:`plugin` singleton is exported there. The platform calls
:meth:`HermesMemoryPlugin.register` once with the agent host; the plugin
binds an :class:`AgentMemoryProvider` to the agent and (if the host exposes
a ``register_service`` hook) registers it as the ``memory`` service.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import MemoryProviderConfig
from .provider import AgentMemoryProvider

PLUGIN_NAME = "memory"
PLUGIN_VERSION = "0.1.0"


@dataclass
class HermesMemoryPlugin:
    """Plugin descriptor returned to the Hermes host."""

    name: str = PLUGIN_NAME
    version: str = PLUGIN_VERSION

    def register(self, host: Any) -> AgentMemoryProvider:
        """Attach an :class:`AgentMemoryProvider` to a Hermes agent host.

        ``host`` is expected to optionally expose:

        * ``host.config.memory`` (or ``host.config['memory']``) — a sub-block
          whose attributes / keys mirror :class:`MemoryProviderConfig`.
        * ``host.agent_id`` — the agent's identifier; overrides whatever
          ``MemoryProviderConfig.agent_id`` resolved to.
        * ``host.register_service(name, instance)`` — optional hook the
          provider is registered against as ``memory``.

        Falls back to ``HERMES_MEMORY_*`` environment variables for anything
        the host doesn't supply.
        """
        config = _resolve_config(host)
        provider = AgentMemoryProvider(config)
        register = getattr(host, "register_service", None)
        if callable(register):
            register(self.name, provider)
        return provider


def _resolve_config(host: Any) -> MemoryProviderConfig:
    base = MemoryProviderConfig.from_env()

    agent_id = getattr(host, "agent_id", None)
    if agent_id:
        base.agent_id = agent_id

    host_config = getattr(host, "config", None)
    memory_cfg: Any = None
    if host_config is not None:
        if isinstance(host_config, dict):
            memory_cfg = host_config.get("memory")
        else:
            memory_cfg = getattr(host_config, "memory", None)
    if memory_cfg is None:
        return base

    def _get(key: str, default: Any = None) -> Any:
        if isinstance(memory_cfg, dict):
            return memory_cfg.get(key, default)
        return getattr(memory_cfg, key, default)

    for key in (
        "transport",
        "agent_id",
        "base_url",
        "bearer_token",
        "stdio_command",
        "stdio_cwd",
    ):
        value = _get(key)
        if value is not None:
            setattr(base, key, value)

    args = _get("stdio_args")
    if args is not None:
        base.stdio_args = list(args)
    timeout = _get("request_timeout_seconds")
    if timeout is not None:
        base.request_timeout_seconds = float(timeout)
    env = _get("stdio_env")
    if env is not None:
        base.stdio_env = dict(env)

    return base


plugin = HermesMemoryPlugin()
