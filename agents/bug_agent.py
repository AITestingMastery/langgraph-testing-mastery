"""agents/bug_agent.py — turns research into a clean bug report."""
from __future__ import annotations
from llm import get_model
from agents._helpers import run_mini_agent

SYSTEM = ("You are a QA bug-writing agent. Using the research provided, call "
          "format_bug_report to produce a clean, standardized bug report. "
          "Return the formatted report.")


def make_bug_node(tools):
    def node(state: dict) -> dict:
        model = get_model(state.get("provider"))
        task = (f"Request: {state['request']}\n\nResearch:\n{state.get('research','(none)')}\n\n"
                "Produce a formatted bug report.")
        if state.get("quality_notes"):
            task += f"\n\nThe reviewer asked for improvements: {state['quality_notes']}"
        tool_log: list = []
        out = run_mini_agent(model, tools, SYSTEM, task, tool_log=tool_log, agent="bug")
        return {"bug_report": out, "tool_log": tool_log, "trail": ["🐞 bug agent wrote a report"]}
    return node