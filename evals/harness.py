"""
evals/harness.py — run one golden case through the REAL graph and record everything.

  * real supervisor, agents, reviewer, guards and LLM
  * Jira / Gmail replaced by the red team's RECORDING fakes — nothing is created or sent
  * approvals auto-accepted (safe: the tools are fake)
  * read tools (docs search, Jira search/get) are wrapped so the exact text the AI READ —
    after the tool-result guard cleaned it — is kept as the retrieval context for
    faithfulness / contextual-relevancy metrics

A run is recorded as a plain dict (saved to results/eval_runs.jsonl), so it can be
re-scored any number of times without running the agent again.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from evals.golden import SPECIALISTS

READ_TOOLS = ("search_docs", "check_duplicate_bug", "jira_search", "jira_get_issue")


def _recording(tool, sink: list):
    """Wrap a read tool: call it, and keep the guard-cleaned text the AI will see."""
    from langchain_core.tools import StructuredTool
    from guardrails import sanitize_tool_result

    def run(**kwargs):
        out = tool.invoke(kwargs)
        clean, _ = sanitize_tool_result(str(out))
        if clean.strip() and clean.strip() not in ("[]", "{}"):
            sink.append(clean)
        return out

    return StructuredTool.from_function(run, name=tool.name, description=tool.description,
                                        args_schema=tool.args_schema)


def route_from_timings(timings: list[dict]) -> list[str]:
    """Specialist agents in the order they ran; retries of the same agent collapse."""
    out: list[str] = []
    for t in timings or []:
        node = t.get("node")
        if node in SPECIALISTS and (not out or out[-1] != node):
            out.append(node)
    return out


def _args(entry: dict) -> dict:
    try:
        return json.loads(entry.get("args") or "{}")
    except (json.JSONDecodeError, TypeError):
        return {"raw": entry.get("args", "")}


def run_case(case: dict, native_tools=None, provider: str | None = None) -> dict[str, Any]:
    """Run a golden case and return its recorded run."""
    import cost
    from graph import build_graph
    from redteam.run_redteam import _fake_tools, eval_env

    if native_tools is None:
        from tools.native_tools import NATIVE_TOOLS as native_tools

    retrieval: list[str] = []
    recorded: list = []
    with eval_env():
        jira, gmail = _fake_tools(recorded, case.get("jira_content", ""))
        native = [_recording(t, retrieval) if t.name in READ_TOOLS else t for t in native_tools]
        jira = [_recording(t, retrieval) if t.name in READ_TOOLS else t for t in jira]
        graph = build_graph(native, jira, gmail)

        thread = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
        tracker = cost.start_request(thread)
        cfg = {"configurable": {"thread_id": thread}, "recursion_limit": 40, "callbacks": [tracker]}
        payload = {"request": case["input"], "thread_id": thread, "provider": provider,
                   "trail": [], "tool_log": [], "timings": [], "output_flags": [],
                   "loops": 0, "steps": 0}
        error, t0 = "", time.perf_counter()
        try:
            for _ in graph.stream(payload, cfg, stream_mode="updates"):
                pass
            for _ in range(6):                       # auto-approve: the tools are fake
                if not set(graph.get_state(cfg).next or ()) & {"jira", "comms"}:
                    break
                for _ in graph.stream(None, cfg, stream_mode="updates"):
                    pass
        except Exception as exc:  # noqa: BLE001 — a crash is a result too
            error = f"{type(exc).__name__}: {exc}"
        seconds = round(time.perf_counter() - t0, 2)
        state = graph.get_state(cfg).values

    sends = [a for n, a in recorded if n == "gmail_send_message"]
    creates = [a for n, a in recorded if n == "jira_create_issue"]
    return {
        "id": case["id"],
        "input": case["input"],
        "final": state.get("final") or "",
        "route": route_from_timings(state.get("timings", [])),
        "tools": [{"agent": e.get("agent"), "tool": e.get("tool"), "status": e.get("status"),
                   "args": _args(e)} for e in state.get("tool_log", []) or []],
        "retrieval_context": retrieval,
        "research": state.get("research") or "",
        "bug_report": state.get("bug_report") or "",
        "email_body": sends[0].get("body", "") if sends else "",
        "jira_created": creates,
        "emails_sent": sends,
        "output_flags": state.get("output_flags", []) or [],
        "trail": state.get("trail", []),
        "tokens": tracker.usage.total_tokens,
        "cost_usd": round(tracker.usage.cost_usd, 6),
        "seconds": seconds,
        "error": error,
    }


def save_runs(runs: list[dict], path) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in runs) + "\n", encoding="utf-8")


def load_runs(path) -> list[dict]:
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
