"""agents/_helpers.py — a tiny tool-calling loop shared by the specialist agents.

Each specialist gets a model + its own subset of tools and a task. This runs the
model, executes any tool calls, and returns the text.

  * guard    — tool-call-level guardrail: sees the REAL arguments and can refuse
               the call before the tool runs.
  * blocked  — collects guardrail refusals (shown in the trail).
  * tool_log — collects one entry per tool call: agent, tool, args, status, time,
               sources. The UI's "tools used / sources" panel is built from this.

Async (MCP) tools run on the shared background loop (async_bridge.py), which is
what lets their sessions stay open between calls.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from async_bridge import run_async
from observability import extract_sources

log = logging.getLogger(__name__)

Guard = Callable[[str, dict], "tuple[bool, str]"]


def _invoke_tool(tool, args: dict) -> str:
    """Call a tool that may be sync or async (MCP tools are async-only)."""
    try:
        return str(tool.invoke(args))
    except NotImplementedError:
        return str(run_async(tool.ainvoke(args)))


def _text(msg) -> str:
    return msg.content if isinstance(msg.content, str) else str(msg.content)


def _short(args: dict, limit: int = 300) -> str:
    s = json.dumps(args, default=str, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "…"


def run_mini_agent(model, tools: list, system: str, task: str, max_steps: int = 4,
                   guard: Guard | None = None, blocked: list | None = None,
                   tool_log: list | None = None, agent: str = "") -> str:
    """A minimal reason -> call-tools -> answer loop for one specialist."""
    by_name = {t.name: t for t in tools}
    bound = model.bind_tools(tools) if tools else model
    messages: list[Any] = [("system", system), ("human", task)]

    for _ in range(max_steps):
        ai = bound.invoke(messages)
        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            return _text(ai)
        messages.append(ai)
        for c in calls:
            name, args = c["name"], c.get("args", {}) or {}
            tool = by_name.get(name)
            status, t0 = "ok", time.perf_counter()
            if tool is None:
                status, result = "error", f"(no tool {name})"
            else:
                ok, msg = guard(name, args) if guard else (True, "")
                if not ok:
                    log.warning("tool call blocked: %s %s", name, msg)
                    if blocked is not None:
                        blocked.append(msg)
                    status, result = "blocked", f"BLOCKED BY GUARDRAIL — this call was NOT executed: {msg}"
                else:
                    try:
                        result = _invoke_tool(tool, args)
                    except Exception as exc:  # noqa: BLE001 — report tool errors to the model
                        log.exception("tool %s failed", name)
                        status, result = "error", f"TOOL ERROR ({name}): {exc}"
            if tool_log is not None:
                tool_log.append({
                    "agent": agent, "tool": name, "args": _short(args), "status": status,
                    "ms": int((time.perf_counter() - t0) * 1000),
                    "sources": extract_sources(name, args, result) if status == "ok" else [],
                    "preview": result[:300],
                })
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result[:4000]})
    final = bound.invoke(messages + [("human", "Give your final answer now.")])
    return _text(final)