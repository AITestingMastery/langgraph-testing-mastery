"""async_bridge.py — one long-lived asyncio loop on a background thread.

Streamlit and our graph nodes are synchronous, but MCP tools are async and their
sessions are bound to the event loop that opened them. Running every coroutine on
ONE persistent loop lets MCP sessions stay open across tool calls (fast), instead
of spinning up a new loop + new server process per call (slow).
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine

_loop: asyncio.AbstractEventLoop | None = None
_lock = threading.Lock()


def get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, name="async-bridge", daemon=True).start()
        return _loop


def run_async(coro: Coroutine[Any, Any, Any], timeout: float | None = 180) -> Any:
    """Run a coroutine on the shared loop from sync code and wait for the result."""
    return asyncio.run_coroutine_threadsafe(coro, get_loop()).result(timeout)
