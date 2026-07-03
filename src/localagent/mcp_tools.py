"""Optional Model Context Protocol (MCP) tool loading.

If an ``mcp.json`` file is present (project root, or the path in ``MCP_CONFIG``),
its servers are connected and their tools are exposed as LangChain tools so the
tool-calling agents can use them. Missing config or errors degrade gracefully to
an empty list.

Config format (mirrors the common ``mcpServers`` convention)::

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:/data"]
        },
        "my-http": {
          "url": "http://localhost:8000/mcp",
          "transport": "streamable_http"
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from localagent.config import PROJECT_ROOT

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from localagent.logging_setup import AgentLogger


def _config_path() -> Path:
    return Path(os.environ.get("MCP_CONFIG", PROJECT_ROOT / "mcp.json"))


def _normalize_servers(raw: dict) -> dict:
    """Accept either ``{"mcpServers": {...}}``, ``{"servers": {...}}`` or a flat map."""
    servers = raw.get("mcpServers") or raw.get("servers") or raw
    normalized: dict[str, dict] = {}
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            continue
        entry = dict(cfg)
        if "transport" not in entry:
            entry["transport"] = "stdio" if "command" in entry else "streamable_http"
        normalized[name] = entry
    return normalized


def load_mcp_tools(logger: AgentLogger | None = None) -> list[BaseTool]:
    """Load tools from configured MCP servers, or return an empty list."""
    config = _config_path()
    if not config.exists():
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        if logger:
            logger.error("langchain-mcp-adapters not installed; skipping MCP tools")
        return []

    try:
        servers = _normalize_servers(json.loads(config.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        if logger:
            logger.error(f"failed to read {config.name}: {exc}")
        return []

    tools: list[BaseTool] = []
    for name, cfg in servers.items():
        try:
            server_tools = asyncio.run(MultiServerMCPClient({name: cfg}).get_tools())
        except Exception as exc:  # noqa: BLE001 - one bad server must not break the others
            if logger:
                logger.error(f"MCP server '{name}' unavailable: {exc}")
            continue
        tools.extend(server_tools)
        if logger:
            logger.info("mcp server loaded", server=name, count=len(server_tools))
    return tools
