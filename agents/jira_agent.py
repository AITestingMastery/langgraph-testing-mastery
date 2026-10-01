"""agents/jira_agent.py — creates/updates Jira tickets (real action, gated)."""
from __future__ import annotations
from llm import get_model
from agents._helpers import guard_trail, run_mini_agent
from guardrails import action_kind, allowed_jira_project, check_jira_args, scrub_outbound_args


def _system() -> str:
    return ("You are a Jira agent. Use the Jira tools to create or update the ticket "
            "the request asks for, using the research/bug report as content. "
            f"Always use project key {allowed_jira_project()}. "
            "Report the ticket key and URL you created. If a tool call is BLOCKED or "
            "fails, say so plainly — never claim a ticket exists unless a tool returned it.")


def make_jira_node(tools):
    def node(state: dict) -> dict:
        # no tools = no ticket. Never let the LLM "pretend" it filed one.
        if not tools:
            return {"jira_result": "(not performed — Jira MCP tools are not connected)",
                    "trail": ["🗂️ jira: no Jira tools connected — skipped"]}
        model = get_model(state.get("provider"))
        content = state.get("bug_report") or state.get("research") or ""
        task = (f"Request: {state['request']}\n\nContent to use:\n{content}\n\n"
                "Create or update the Jira ticket accordingly.")
        blocked: list[str] = []
        tool_log: list = []
        out = run_mini_agent(model, tools, _system(), task,
                             guard=check_jira_args, blocked=blocked,
                             tool_log=tool_log, agent="jira",
                             scrub=lambda n, a: scrub_outbound_args(n, a, state.get("request", "")))
        trail = guard_trail(tool_log) + [f"🛡️ tool-level guardrail BLOCKED jira call → {b}" for b in blocked
                                         if not b.startswith("Action budget")]
        done = sum(1 for e in tool_log if e.get("status") == "ok" and action_kind(e.get("tool", "")))
        if blocked and done:        # some actions happened, then a guard stopped the rest
            out = f"(partial: {done} done, then blocked — {blocked[0]})\n{out}"
        elif blocked:
            out = f"(blocked: {blocked[0]})\n{out}"
        trail.append("🗂️ jira agent acted on a ticket")
        return {"jira_result": out, "tool_log": tool_log, "trail": trail}
    return node