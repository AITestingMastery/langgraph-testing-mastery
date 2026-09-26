"""
graph.py — assembles the multi-agent graph. THE CENTERPIECE.

This is where LangGraph earns its place. We wire specialist agents as nodes,
route between them with the supervisor (conditional edges), loop back on a failed
quality check (a cycle), and pause before real actions (human-in-the-loop). A
single create_agent can't express this; a graph can.

Flow:
  START → entry_guard → supervisor ──▶ research / bug        ─┐
                            ▲    └──▶ guardrail → jira/comms ─┤ (jira/comms pause for approval)
                            │                                  ▼
                            └───────────── quality ◀───────────┘
                                   (grades the LAST step; weak → supervisor retries it)
  supervisor → done → finalize → output_guard → END   (only the supervisor can end the run)

Real actions (jira, comms) are gated: the graph interrupts BEFORE those nodes so
the app can ask the user to approve.
"""

from __future__ import annotations

import functools
import time

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from state import QAState
from llm import get_model
from agents.supervisor import supervisor_node
from agents.research_agent import make_research_node
from agents.bug_agent import make_bug_node
from agents.jira_agent import make_jira_node
from agents.comms_agent import make_comms_node
from quality import quality_node
from guardrails import entry_guard_node, guardrail_node, output_guard_node


def _finalize_node(state: dict) -> dict:
    """Write the final answer from ONLY what actually happened.

    If a guardrail blocked the request, say so and stop. Otherwise summarize
    strictly from the results present in state — never claim a ticket or email
    unless jira_result / email_result actually exist (prevents hallucination).
    """
    # blocked at the entry guardrail — final was already set there
    if state.get("guardrail_block") and state.get("final"):
        return {"trail": ["🏁 finalized (blocked)"]}

    # build a FACTUAL list of what really ran
    facts = []
    if state.get("research"): facts.append("Research findings:\n" + state["research"])
    if state.get("bug_report"): facts.append("Bug report:\n" + state["bug_report"])
    if state.get("jira_result"): facts.append("Jira result:\n" + state["jira_result"])
    if state.get("email_result"): facts.append("Email result:\n" + state["email_result"])

    if not facts:
        return {"final": "No action was taken for this request.",
                "trail": ["🏁 finalized (nothing to report)"]}

    model = get_model(state.get("provider"))
    msg = model.invoke([
        ("system", "Summarize the outcome for the user using ONLY the facts given. "
                   "Do NOT claim any Jira ticket or email unless it appears in the facts. "
                   "If an action was declined by the user or blocked by a guardrail, say "
                   "so plainly — never describe it as unnecessary or as your own choice."),
        ("human", f"Request: {state['request']}\n\nFacts (only these happened):\n"
                  + "\n\n".join(facts)),
    ])
    final = msg.content if isinstance(msg.content, str) else str(msg.content)
    status = _action_status(state)
    if status:
        final += "\n\n---\n**Action status**\n" + "\n".join(f"- {s}" for s in status)
    return {"final": final, "trail": ["🏁 finalized"]}


def _action_status(state: dict) -> list[str]:
    """Deterministic status lines for real actions that did NOT happen — built from
    state, never from the LLM, so a decline is never reworded as 'unnecessary'."""
    out = []
    for label, key in (("Jira", "jira_result"), ("Email", "email_result")):
        first = (state.get(key) or "").splitlines()[0] if state.get(key) else ""
        if first.startswith("(declined"):
            out.append(f"⛔ {label}: declined by you — not performed")
        elif first.startswith("(blocked"):
            out.append(f"🛡️ {label}: {first.strip('()')}")
        elif first.startswith("(not performed"):
            out.append(f"🔴 {label}: {first.strip('()')}")
    return out


def _timed(name: str, fn):
    """Wrap a node so every run records how long it took (state['timings']).
    This is what the UI's timing panel shows — it makes 'why is it slow?' answerable."""
    @functools.wraps(fn)
    def wrapper(state):
        t0 = time.perf_counter()
        out = fn(state) or {}
        return {**out, "timings": [{"node": name, "ms": int((time.perf_counter() - t0) * 1000)}]}
    return wrapper


def route_from_supervisor(state: dict) -> str:
    """Conditional edge: send control to whichever agent the supervisor picked."""
    return state.get("next_agent", "done")


def route_from_entry_guard(state: dict) -> str:
    """Entry guardrail: block unsafe requests at the door, else proceed."""
    return "blocked" if state.get("guardrail_block") else "ok"


def route_from_guardrail(state: dict) -> str:
    """After the guardrail: if it blocked the action, return to the supervisor (so
    other requested work can continue); otherwise run the chosen gated action."""
    if state.get("guardrail_block"):
        return "blocked"
    return state.get("next_agent", "blocked")  # 'jira' or 'comms'


def build_graph(native_tools, jira_tools, gmail_tools, checkpointer=None):
    """Construct and compile the multi-agent graph.

    Gated nodes (jira, comms) get an interrupt_before so the app can pause for
    approval. Returns the compiled graph (with an in-memory checkpointer).
    """
    # Research may only READ Jira (search/get) — never create/update. Real actions
    # belong ONLY to the gated jira/comms nodes, so they can't bypass approval.
    jira_read = [t for t in jira_tools if any(k in t.name for k in ("search", "get"))]
    jira_write = [t for t in jira_tools if any(k in t.name for k in ("create", "update", "add", "transition"))]
    research_tools = list(native_tools) + jira_read
    bug_tools = [t for t in native_tools if t.name == "format_bug_report"]

    g = StateGraph(QAState)

    # nodes
    g.add_node("entry_guard", _timed("entry_guard", entry_guard_node))   # request-level safety, runs FIRST
    g.add_node("supervisor", _timed("supervisor", supervisor_node))
    g.add_node("research", _timed("research", make_research_node(research_tools)))
    g.add_node("bug", _timed("bug", make_bug_node(bug_tools)))
    g.add_node("guardrail", _timed("guardrail", guardrail_node))   # action-level safety, before jira/comms
    g.add_node("jira", _timed("jira", make_jira_node(jira_write)))
    # comms may only SEND — never read, trash, or delete mail (least privilege)
    gmail_send = [t for t in gmail_tools
                  if t.name in ("gmail_send_message", "gmail_create_draft", "gmail_send_draft")]
    g.add_node("comms", _timed("comms", make_comms_node(gmail_send)))
    g.add_node("quality", _timed("quality", quality_node))
    g.add_node("finalize", _timed("finalize", _finalize_node))
    g.add_node("output_guard", _timed("output_guard", output_guard_node))

    # edges — every request passes the entry guardrail BEFORE the supervisor
    g.add_edge(START, "entry_guard")
    g.add_conditional_edges("entry_guard", route_from_entry_guard, {
        "ok": "supervisor", "blocked": "finalize",
    })
    # supervisor routes; jira/comms go via the guardrail first
    g.add_conditional_edges("supervisor", route_from_supervisor, {
        "research": "research", "bug": "bug",
        "jira": "guardrail", "comms": "guardrail",
        "done": "finalize",
    })
    # guardrail decides: proceed to the chosen action, or hand back to the supervisor if blocked
    g.add_conditional_edges("guardrail", route_from_guardrail, {
        "jira": "jira", "comms": "comms", "blocked": "supervisor",
    })
    # read/format specialists report straight to quality
    for specialist in ["research", "bug"]:
        g.add_edge(specialist, "quality")
    # the gated action nodes also report to quality
    g.add_edge("jira", "quality")
    g.add_edge("comms", "quality")
    # quality ALWAYS hands back to the supervisor (the CYCLE). The supervisor then
    # retries a weak step, moves on to the next one, or ends with 'done'.
    g.add_edge("quality", "supervisor")
    # output guard: last check on the answer (redaction + claim verification)
    g.add_edge("finalize", "output_guard")
    g.add_edge("output_guard", END)

    # gate the real-action nodes: pause before they run so the app can approve
    return g.compile(
        checkpointer=checkpointer or InMemorySaver(),
        interrupt_before=["jira", "comms"],
    )