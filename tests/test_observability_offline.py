"""Offline tests for the details panel data, LangSmith links, timings, and
persistent MCP sessions (the speed fix). No API keys needed."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

import observability as obs
from agents._helpers import run_mini_agent


# ---------------------------------------------------------------- sources
def test_doc_sources_from_search_docs():
    result = "[source: known_bugs.md]\nBUG-101...\n\n---\n\n[source: api_test_cases.md]\n..."
    labels = [s["label"] for s in obs.extract_sources("search_docs", {"query": "x"}, result)]
    assert labels == ["known_bugs.md", "api_test_cases.md"]


def test_jira_sources_link_to_browse(monkeypatch):
    monkeypatch.setenv("JIRA_URL", "https://teamreqon.atlassian.net/")
    src = obs.extract_sources("jira_create_issue", {"project_key": "TEST"}, '{"key": "TEST-53"}')
    assert src == [{"kind": "jira", "label": "TEST-53",
                    "url": "https://teamreqon.atlassian.net/browse/TEST-53"}]


def test_jira_search_with_no_hits_is_still_reported():
    src = obs.extract_sources("jira_search", {"jql": "status=Open"}, "[]")
    assert src[0]["label"].startswith("jira_search (no matching")


def test_email_source_is_recipient():
    src = obs.extract_sources("gmail_send_message", {"to": "a@gmail.com", "body": "hi"}, "sent")
    assert src == [{"kind": "email", "label": "a@gmail.com", "url": None}]


def test_builtin_tools_have_no_sources():
    assert obs.extract_sources("format_bug_report", {}, "**Bug Report**") == []


def test_summarize_turn():
    state = {
        "tool_log": [
            {"tool": "search_docs", "status": "ok", "sources": [{"kind": "doc", "label": "a.md", "url": None}]},
            {"tool": "search_docs", "status": "ok", "sources": [{"kind": "doc", "label": "a.md", "url": None}]},
            {"tool": "gmail_send_message", "status": "blocked", "sources": [{"kind": "email", "label": "x", "url": None}]},
        ],
        "timings": [{"node": "research", "ms": 4000}, {"node": "supervisor", "ms": 800},
                    {"node": "research", "ms": 3000}],
        "trail": ["quality: needs work → loop back to research — x"],
    }
    d = obs.summarize_turn(state)
    assert d["sources"] == [{"kind": "doc", "label": "a.md", "url": None}]   # deduped, blocked excluded
    assert d["tool_counts"]["search_docs"] == 2
    assert d["timing"]["research"] == {"calls": 2, "ms": 7000}
    assert d["total_ms"] == 7800 and d["slowest"] == "research" and d["loops"] == 1


def test_summarize_empty_turn():
    d = obs.summarize_turn({})
    assert d["tools"] == [] and d["sources"] == [] and d["total_ms"] == 0 and d["slowest"] is None


# ---------------------------------------------------------------- LangSmith
def test_langsmith_enabled_needs_flag_and_key(monkeypatch):
    for k in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert not obs.langsmith_enabled()                 # flag alone isn't enough
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_x")
    assert obs.langsmith_enabled()


@pytest.mark.parametrize("endpoint,host", [
    (None, "https://smith.langchain.com"),
    ("https://api.smith.langchain.com", "https://smith.langchain.com"),
    ("https://eu.api.smith.langchain.com", "https://eu.smith.langchain.com"),
])
def test_ui_host(monkeypatch, endpoint, host):
    if endpoint:
        monkeypatch.setenv("LANGSMITH_ENDPOINT", endpoint)
    else:
        monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    assert obs.ui_host() == host


def test_trace_url(monkeypatch):
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    obs._project_ids.cache_clear()
    fake_client = types.SimpleNamespace(read_project=lambda project_name: types.SimpleNamespace(
        tenant_id="org-1", id="proj-9"))
    monkeypatch.setitem(sys.modules, "langsmith", types.SimpleNamespace(Client=lambda: fake_client))
    assert obs.trace_url("run-123") == \
        "https://smith.langchain.com/o/org-1/projects/p/proj-9/r/run-123?poll=true"
    obs._project_ids.cache_clear()


def test_trace_url_none_when_project_missing(monkeypatch):
    obs._project_ids.cache_clear()
    def boom(project_name):
        raise LookupError("not found")
    monkeypatch.setitem(sys.modules, "langsmith",
                        types.SimpleNamespace(Client=lambda: types.SimpleNamespace(read_project=boom)))
    assert obs.trace_url("run-1") is None
    obs._project_ids.cache_clear()


# ---------------------------------------------------------------- tool_log capture
class _ToolModel:
    def __init__(self, calls):
        self.calls, self.n = calls, 0
    def bind_tools(self, _t):
        return self
    def invoke(self, _m):
        self.n += 1
        if self.n == 1:
            return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"c{i}"}
                                                     for i, (n, a) in enumerate(self.calls)])
        return AIMessage(content="final")


class _Tool:
    def __init__(self, name, result):
        self.name, self.result = name, result
    def invoke(self, _args):
        return self.result


def test_run_mini_agent_logs_every_call():
    log: list = []
    tools = [_Tool("search_docs", "[source: known_bugs.md]\nBUG-101"),
             _Tool("gmail_send_message", "sent")]
    guard = lambda name, args: (False, "nope") if name.startswith("gmail") else (True, "")
    run_mini_agent(_ToolModel([("search_docs", {"query": "login"}),
                               ("gmail_send_message", {"to": "a@b.com"})]),
                   tools, "s", "t", guard=guard, tool_log=log, agent="research")
    assert [(e["tool"], e["status"], e["agent"]) for e in log] == [
        ("search_docs", "ok", "research"), ("gmail_send_message", "blocked", "research")]
    assert log[0]["sources"][0]["label"] == "known_bugs.md"
    assert log[1]["sources"] == []            # a blocked call is not a source


# ---------------------------------------------------------------- timings in the graph
def test_graph_records_timings_per_node(monkeypatch):
    import agents.research_agent as r, agents.supervisor as sup, graph as gm, quality
    fake = types.SimpleNamespace(invoke=lambda m: AIMessage(content="summary"))
    for mod in (sup, quality, gm, r):
        monkeypatch.setattr(mod, "get_model", lambda *a, **k: fake)
    script = ["research", "done"]
    monkeypatch.setattr(sup, "_decide", lambda *_: (script.pop(0), "s"))
    monkeypatch.setattr(quality, "_grade", lambda *_: (True, ""))
    monkeypatch.setattr(r, "run_mini_agent", lambda *a, **k: "facts")
    g = gm.build_graph([], [], [])
    state = g.invoke({"request": "What known bugs affect login?", "trail": [], "timings": [],
                      "tool_log": []}, {"configurable": {"thread_id": "t"}})
    nodes = [t["node"] for t in state["timings"]]
    assert nodes == ["entry_guard", "supervisor", "research", "quality", "supervisor", "finalize"]
    assert all(isinstance(t["ms"], int) for t in state["timings"])


# ---------------------------------------------------------------- persistent MCP sessions
pytest.importorskip("mcp")
pytest.importorskip("langchain_mcp_adapters")


def _pid_config(tmp_path: Path) -> Path:
    server = Path(__file__).parent / "fixtures" / "pid_server.py"
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"jira": {"command": sys.executable, "args": [str(server)],
                                        "transport": "stdio"}}))
    return cfg


def _call_twice(tools):
    """Call the tool twice and return the two PIDs (MCP results wrap text in blocks
    with random ids, so pull the digits out rather than comparing raw strings)."""
    import re
    from agents._helpers import _invoke_tool
    tool = next(t for t in tools if t.name == "jira_whoami")
    pid = lambda: re.search(r"'text': '(\d+)'", _invoke_tool(tool, {})).group(1)
    return pid(), pid()


def test_persistent_sessions_reuse_one_server_process(tmp_path):
    from tools.mcp_tools import load_mcp_tools
    tools, errors = load_mcp_tools(_pid_config(tmp_path), persistent=True)
    assert not errors
    a, b = _call_twice(tools)
    assert a == b, "persistent mode must reuse the same server process"


def test_per_call_sessions_spawn_new_process_each_time(tmp_path):
    """Documents the OLD behaviour — the reason the app felt slow."""
    from tools.mcp_tools import load_mcp_tools
    tools, _ = load_mcp_tools(_pid_config(tmp_path), persistent=False)
    a, b = _call_twice(tools)
    assert a != b


def test_bad_server_does_not_break_the_others(tmp_path):
    from tools.mcp_tools import load_mcp_tools
    cfg = json.loads(_pid_config(tmp_path).read_text())
    cfg["gmail"] = {"command": "definitely-not-a-real-binary-xyz", "args": [], "transport": "stdio"}
    p = tmp_path / "mixed.json"
    p.write_text(json.dumps(cfg))
    tools, errors = load_mcp_tools(p, persistent=True, timeout=60)
    assert [t.name for t in tools] == ["jira_whoami"]
    assert "gmail" in errors


# ---------------------------------------------------------------- live finding: inflated sources
def test_doc_ids_in_jira_query_are_not_jira_sources(monkeypatch):
    monkeypatch.setenv("JIRA_PROJECT_KEY", "TEST")
    src = obs.extract_sources("jira_search", {"jql": 'text ~ "BUG-101"'}, '[{"key": "TEST-53"}]')
    assert [s["label"] for s in src] == ["TEST-53"]


def test_uncited_jira_hits_collapse_into_one_line(monkeypatch):
    monkeypatch.setenv("JIRA_PROJECT_KEY", "TEST")
    hits = ", ".join(f'{{"key": "TEST-{n}"}}' for n in range(45, 54))
    log = [{"tool": "search_docs", "status": "ok",
            "sources": obs.extract_sources("search_docs", {}, "[source: known_bugs.md]")},
           {"tool": "jira_search", "status": "ok",
            "sources": obs.extract_sources("jira_search", {}, f"[{hits}]")}]
    d = obs.summarize_turn({"tool_log": log, "final": "See TEST-53 and known_bugs.md."})
    labels = [s["label"] for s in d["sources"]]
    assert labels == ["known_bugs.md", "TEST-53", "+8 more Jira issue(s) retrieved but not cited"]


def test_empty_jira_search_is_noted_not_counted():
    log = [{"tool": "jira_search", "status": "ok",
            "sources": obs.extract_sources("jira_search", {}, "[]")}]
    d = obs.summarize_turn({"tool_log": log, "final": "x"})
    assert d["sources"][0]["uncited"] and "no matching" in d["sources"][0]["label"]