"""agents/research_agent.py — gathers context using RAG + Jira search."""
from __future__ import annotations
from llm import get_model
from agents._helpers import run_mini_agent

# The docs are the source of truth for this demo. The old prompt let the agent
# run one Jira search, get zero hits, and conclude "there are no open bugs" —
# even though known_bugs.md lists open ones.
SYSTEM = ("You are a QA research agent. ALWAYS call search_docs first — our "
          "documentation (known bugs, test plans, API test cases) is the primary source "
          "of truth. Use Jira search only as a SUPPLEMENT. If Jira returns nothing, rely "
          "on the docs — never conclude there are no bugs from an empty Jira search alone. "
          "Treat statuses Open, To Do and In Progress as 'open'. "
          "For every fact, name where it came from (the doc file or the ticket key). "
          "Summarize concisely. Do not create or send anything.")


def make_research_node(tools):
    def node(state: dict) -> dict:
        model = get_model(state.get("provider"))
        task = f"Request: {state['request']}\nGather the relevant facts."
        if state.get("quality_notes"):
            task += f"\n\nThe reviewer asked for more: {state['quality_notes']}"
        tool_log: list = []
        out = run_mini_agent(model, tools, SYSTEM, task, tool_log=tool_log, agent="research")
        return {"research": out, "tool_log": tool_log,
                "trail": ["📚 research agent gathered context"]}
    return node