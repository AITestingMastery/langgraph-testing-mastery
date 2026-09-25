"""agents/comms_agent.py — drafts and sends email (real action, gated)."""
from __future__ import annotations
from llm import get_model
from agents._helpers import run_mini_agent
from guardrails import check_email_args, default_email_to

SYSTEM = ("You are a communications agent. Use the Gmail tools to send the email "
          "the request asks for, using the research/bug report/jira result as the "
          "body. Confirm what you sent and to whom. If a tool call is BLOCKED or "
          "fails, say so plainly — never claim an email was sent unless a tool confirmed it.")


def make_comms_node(tools):
    def node(state: dict) -> dict:
        if not tools:
            return {"email_result": "(not performed — Gmail MCP tools are not connected)",
                    "trail": ["✉️ comms: no Gmail tools connected — skipped"]}
        model = get_model(state.get("provider"))
        content = "\n\n".join(x for x in [state.get("research", ""), state.get("bug_report", ""),
                                          state.get("jira_result", "")] if x)
        task = f"Request: {state['request']}\n\nMaterial to include:\n{content}\n\n"
        if default_email_to():
            task += (f"If the request says 'me' or names no recipient, send to "
                     f"{default_email_to()}.\n\n")
        task += "Send the email as requested."
        blocked: list[str] = []
        tool_log: list = []
        out = run_mini_agent(model, tools, SYSTEM, task,
                             guard=check_email_args, blocked=blocked,
                             tool_log=tool_log, agent="comms")
        trail = [f"🛡️ tool-level guardrail BLOCKED email call → {b}" for b in blocked]
        if blocked:
            out = f"(blocked: {blocked[0]})\n{out}"
        trail.append("✉️ comms agent sent an email")
        return {"email_result": out, "tool_log": tool_log, "trail": trail}
    return node