"""
state.py — the shared State that flows through the whole graph.

In LangGraph, every node reads this object and returns updates to it. This is the
heart of the graph: the request, what each agent found, the drafts, the routing
decision, and the quality verdict all live here and travel between nodes.

Contrast with the LangChain agent: there, state was hidden inside create_agent.
Here we define it ourselves — that's the "under the hood" control LangGraph gives.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class QAState(TypedDict, total=False):
    # the user's request
    request: str

    # which model provider to use (set by the app)
    provider: str

    # conversation history (messages accumulate — Annotated + operator.add appends)
    messages: Annotated[list, operator.add]

    # a running log of which node ran, for the live graph view (also appends)
    trail: Annotated[list, operator.add]

    # what the research agent gathered (RAG + Jira context)
    research: str

    # the formatted bug report, if the bug agent ran
    bug_report: str

    # results of real actions
    jira_result: str
    email_result: str

    # supervisor's routing decision: which agent to run next
    next_agent: str

    # quality check
    quality_ok: bool
    quality_notes: str
    loops: int  # consecutive retries of the CURRENT step (resets when a step passes)
    steps: int  # total supervisor decisions this request (hard cap: MAX_STEPS)

    # guardrails
    guardrail_block: bool
    guardrail_note: str

    # observability (both append across nodes)
    tool_log: Annotated[list, operator.add]  # one entry per tool call (tool, args, sources, ms)
    timings: Annotated[list, operator.add]   # one entry per node run ({"node", "ms"})

    # the final answer shown to the user
    final: str