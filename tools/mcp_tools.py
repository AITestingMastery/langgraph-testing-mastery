"""tools/mcp_tools.py — load live Jira + Gmail MCP servers as LangChain tools.

PERSISTENT SESSIONS (the speed fix): `MultiServerMCPClient.get_tools()` creates a
NEW session for EVERY tool call — for stdio servers that means re-launching
`uvx mcp-atlassian` and redoing the MCP handshake each time (seconds per call).
Here each server gets ONE session, opened once and held open on the shared
background loop (async_bridge.py), so tool calls reuse it.

Set MCP_PERSISTENT_SESSIONS=false to fall back to per-call sessions.
Each server loads independently — one failing server doesn't disable the other.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from async_bridge import get_loop, run_async

log = logging.getLogger(__name__)

CONFIG = Path(__file__).parent.parent / "config" / "mcp_servers.json"
_holders: list[concurrent.futures.Future] = []   # keeps persistent sessions alive


def _expand_env(connections: dict[str, Any]) -> dict[str, Any]:
    for _name, cfg in connections.items():
        if isinstance(cfg, dict) and cfg.get("env"):
            resolved = dict(os.environ)
            for k, v in cfg["env"].items():
                if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                    resolved[k] = os.environ.get(v[2:-1], "")
                else:
                    resolved[k] = v
            cfg["env"] = resolved
    return connections


def read_config(path: Path = CONFIG) -> dict[str, Any]:
    if not path.exists():
        return {}
    conns = {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}
    return _expand_env(conns)


async def _hold_sessions(client, names: list[str], ready: concurrent.futures.Future) -> None:
    """Open one session per server, report the tools, then keep them open forever."""
    from langchain_mcp_adapters.tools import load_mcp_tools as load_from_session
    tools: list[Any] = []
    errors: dict[str, str] = {}
    try:
        async with AsyncExitStack() as stack:
            for name in names:
                try:
                    session = await stack.enter_async_context(client.session(name))
                    tools += await load_from_session(session, server_name=name)
                except Exception as exc:  # noqa: BLE001 — one bad server shouldn't kill the rest
                    log.exception("MCP server %s failed to start", name)
                    errors[name] = f"{type(exc).__name__}: {exc}"
            ready.set_result((tools, errors))
            await asyncio.Event().wait()          # hold the sessions open
    except Exception as exc:  # noqa: BLE001
        if not ready.done():
            ready.set_exception(exc)


def load_mcp_tools(config_path: Path = CONFIG, persistent: bool | None = None,
                   timeout: float = 120) -> tuple[list[Any], dict[str, str]]:
    """Connect to the configured MCP servers. Returns (tools, {server: error})."""
    conns = read_config(config_path)
    if not conns:
        return [], {}
    from langchain_mcp_adapters.client import MultiServerMCPClient
    client = MultiServerMCPClient(conns)
    if persistent is None:
        persistent = os.getenv("MCP_PERSISTENT_SESSIONS", "true").lower() != "false"

    if not persistent:
        tools, errors = [], {}
        for name in conns:
            try:
                tools += run_async(client.get_tools(server_name=name), timeout)
            except Exception as exc:  # noqa: BLE001
                errors[name] = f"{type(exc).__name__}: {exc}"
        return tools, errors

    ready: concurrent.futures.Future = concurrent.futures.Future()
    holder = asyncio.run_coroutine_threadsafe(_hold_sessions(client, list(conns), ready), get_loop())
    _holders.append(holder)
    return ready.result(timeout)


def tools_by_prefix(tools: list, prefix: str) -> list:
    """Filter loaded MCP tools by name prefix, e.g. 'jira' or 'gmail'."""
    return [t for t in tools if t.name.startswith(prefix)]