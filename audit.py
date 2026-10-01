"""audit.py — (6) the audit log: a permanent, append-only record of what happened.

One JSON object per line in logs/audit.jsonl (git-ignored):
  request · guard_block (layer 1/2/3) · budget_block · tool_result_cleaned (layer 4)
  · outbound_redacted · action (real Jira/Gmail calls) · approval · output_guard (layer 5)

Answers the questions a QA lead or auditor asks: *who approved what, what did the
agents actually do, and what did each guard block?* Logging never breaks the app —
any file error is swallowed.

Config: AUDIT_LOG_ENABLED (default true), AUDIT_LOG_PATH (default logs/audit.jsonl).
"""
from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from pathlib import Path

_thread: contextvars.ContextVar = contextvars.ContextVar("audit_thread", default=None)
_lock = threading.Lock()


def set_thread(thread_id: str | None) -> None:
    """Called by the graph before each node so every event carries its request's thread id."""
    _thread.set(thread_id)


def enabled() -> bool:
    return os.getenv("AUDIT_LOG_ENABLED", "true").lower() != "false"


def log_path() -> Path:
    return Path(os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl"))


def audit(event: str, **fields) -> None:
    if not enabled():
        return
    record = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event,
              "thread": fields.pop("thread", None) or _thread.get(), **fields}
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _lock, path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def read_recent(n: int = 50) -> list[dict]:
    """Most recent n events, newest first."""
    try:
        lines = log_path().read_text(encoding="utf-8").splitlines()[-n:]
    except OSError:
        return []
    out = []
    for line in reversed(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
