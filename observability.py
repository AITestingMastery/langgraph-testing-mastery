"""observability.py — what the UI shows under each answer.

  * extract_sources()  — turn a tool result into citable sources
                         (doc files, Jira keys with links, email recipients)
  * summarize_turn()   — tools used, sources, per-node timing for one request
  * LangSmith helpers  — is tracing on, and a direct link to a run's trace

Everything here is derived from graph STATE (tool_log, timings), never from the
LLM's own claims — so "sources" means what tools actually returned.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache

DOC_SOURCE_RE = re.compile(r"\[source:\s*([^\]]+)\]")
ISSUE_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")
EMAIL_RE = re.compile(r"[\w.\-+]+@[\w.\-]+\.\w+")
BUILTIN_TOOLS = {"format_bug_report", "check_duplicate_bug", "generate_test_cases"}


# ---------------------------------------------------------------- sources
def extract_sources(tool: str, args: dict, result: str) -> list[dict]:
    """Sources a single tool call contributed: [{'kind','label','url'}]."""
    out: list[dict] = []
    if tool == "search_docs":
        for name in dict.fromkeys(DOC_SOURCE_RE.findall(result)):
            out.append({"kind": "doc", "label": name.strip(), "url": None})
    elif tool.startswith("jira"):
        # only real keys from the RESULT, in our Jira project — not doc IDs like
        # BUG-101 that the agent happened to put in its search query
        base = os.getenv("JIRA_URL", "").rstrip("/")
        project = os.getenv("JIRA_PROJECT_KEY", "TEST").strip().upper()
        keys = [k for k in dict.fromkeys(ISSUE_KEY_RE.findall(result))
                if k.startswith(f"{project}-")]
        for key in keys:
            out.append({"kind": "jira", "label": key, "url": f"{base}/browse/{key}" if base else None})
        if not keys:
            out.append({"kind": "jira", "label": f"{tool} (no matching issues)", "url": None,
                        "empty": True})
    elif tool.startswith("gmail") and ("send" in tool or "draft" in tool):
        for addr in dict.fromkeys(EMAIL_RE.findall(json.dumps(args, default=str))):
            out.append({"kind": "email", "label": addr, "url": None})
    return out


def _dedupe(sources: list[dict]) -> list[dict]:
    seen, out = set(), []
    for s in sources:
        if (s["kind"], s["label"]) not in seen:
            seen.add((s["kind"], s["label"]))
            out.append(s)
    return out


def _cited_sources(state: dict, log: list) -> list[dict]:
    """Docs that were retrieved + Jira issues the answer actually USES.

    A Jira search can return 10 tickets; listing all of them as 'sources' overstates
    what the answer relied on. Issues that appear in the answer/research/bug/jira text
    are listed individually; the rest collapse into one '+N more retrieved' line."""
    all_src = _dedupe([s for e in log if e.get("status") == "ok" for s in e.get("sources", [])])
    text = " ".join(state.get(k, "") or "" for k in
                    ("final", "research", "bug_report", "jira_result", "email_result"))
    out, uncited = [], 0
    for s in all_src:
        if s["kind"] == "jira" and not s.get("empty") and s["label"] not in text:
            uncited += 1
        elif not s.get("empty"):
            out.append(s)
    if uncited:
        out.append({"kind": "jira", "label": f"+{uncited} more Jira issue(s) retrieved but not cited",
                    "url": None, "uncited": True})
    if not out and any(s.get("empty") for s in all_src):
        out.append({"kind": "jira", "label": "Jira searched — no matching issues", "url": None,
                    "uncited": True})
    return out


def summarize_turn(state: dict) -> dict:
    """Everything the details panel needs, computed from state."""
    log = state.get("tool_log", []) or []
    timings = state.get("timings", []) or []
    per_node: dict[str, dict] = defaultdict(lambda: {"calls": 0, "ms": 0})
    for t in timings:
        per_node[t["node"]]["calls"] += 1
        per_node[t["node"]]["ms"] += t["ms"]
    total_ms = sum(t["ms"] for t in timings)
    slowest = max(per_node.items(), key=lambda kv: kv[1]["ms"])[0] if per_node else None
    return {
        "tools": log,
        "tool_counts": Counter(e["tool"] for e in log),
        "sources": _cited_sources(state, log),
        "timing": dict(per_node),
        "total_ms": total_ms,
        "slowest": slowest,
        "llm_calls": sum(e.get("llm_calls", 0) for e in timings) or None,
        "loops": sum("loop back" in t for t in state.get("trail", [])),
    }


# ---------------------------------------------------------------- LangSmith
def langsmith_enabled() -> bool:
    tracing = (os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or "").lower()
    key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    return tracing == "true" and bool(key) and "your-key" not in key


def langsmith_project() -> str:
    return os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "default"


def ui_host() -> str:
    """API endpoint → web UI host (api.smith… → smith…, eu.api.smith… → eu.smith…)."""
    endpoint = (os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com").rstrip("/")
    return re.sub(r"^(https?://)((?:[\w-]+\.)*?)api\.", r"\1\2", endpoint, count=1)


@lru_cache(maxsize=4)
def _project_ids(project: str) -> tuple[str, str]:
    """(tenant_id, project_id). Only successful lookups are cached — if the project
    doesn't exist yet (first trace still uploading), we retry next render."""
    from langsmith import Client
    p = Client().read_project(project_name=project)
    return str(p.tenant_id), str(p.id)


def project_url() -> str:
    try:
        tenant, pid = _project_ids(langsmith_project())
        return f"{ui_host()}/o/{tenant}/projects/p/{pid}"
    except Exception:  # noqa: BLE001
        return ui_host()


def trace_url(run_id: str) -> str | None:
    """Direct link to one run's trace. ?poll=true makes the page wait if the trace
    is still uploading."""
    try:
        tenant, pid = _project_ids(langsmith_project())
        return f"{ui_host()}/o/{tenant}/projects/p/{pid}/r/{run_id}?poll=true"
    except Exception:  # noqa: BLE001
        return None