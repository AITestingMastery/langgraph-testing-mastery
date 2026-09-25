"""
agents/supervisor.py — the router.

The supervisor reads the request and the work done so far, and decides which
specialist runs next — or that the work is complete. It is the ONLY node that can
end the run (quality always hands control back here).

The LLM makes the decision, but six deterministic rules sit on top of it:
  1. step cap          — never more than MAX_STEPS routing decisions per request
  2. no repeat actions — a jira/comms step that already ran, was declined, or was
                         blocked is never run again (no duplicate tickets/emails)
  3. no early 'done'   — if the request clearly asks for a ticket/email that hasn't
                         been handled yet, 'done' is overridden
  4. no idle repeats   — a research/bug step that JUST passed quality isn't re-run
                         back-to-back (only a failed quality check justifies a retry)
  5. no unrequested   — jira/comms only run when the request explicitly asks for a
     actions             ticket/email (the LLM can't decide on its own to file one)
  6. no unneeded bug   — the bug agent only runs if a bug report was asked for, or a
     report               ticket is about to be filed (saves ~5s of LLM calls otherwise)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Literal

from pydantic import BaseModel, Field

from guardrails import EMAIL_RE
from llm import get_model

log = logging.getLogger(__name__)

ROUTES = ["research", "bug", "jira", "comms", "done"]
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))

SUPERVISOR_PROMPT = """You are the supervisor of a QA assistant team. Decide the \
SINGLE next step for this request, based on what's already been done.

Team members you can route to:
- research : gather context from the docs (RAG) and Jira SEARCH (use FIRST for most tasks)
- bug      : format findings into a clean bug report
- jira     : create or update a Jira ticket (a real action — the ONLY way to create tickets)
- comms    : draft and send an email (a real action — the ONLY way to send email)
- done     : everything the request asked for is complete; write the final answer

Routing rules:
- Identify EVERY action the request asks for, and do them in a sensible order
  (usually research → bug → jira → comms).
- If quality feedback says the last step was weak, route to that step again.
- A jira/comms step that shows as done, DECLINED by the user, or BLOCKED by a
  guardrail counts as handled — never route to it again.
- Choose 'done' only when every requested item is handled."""


class Decision(BaseModel):
    next: Literal["research", "bug", "jira", "comms", "done"]
    why: str = Field(description="one short reason for this routing decision")


# heuristics for "did the user explicitly ask for a real action?"
_JIRA_ASK = re.compile(
    r"\b(create|file|raise|open|log|update)\b[^.]*\b(jira|ticket)\b"
    r"|\blog\b[^.]*\bas an? (bug|issue)\b"
    r"|\b(update|comment on|add a comment to|close)\b[^.]*\b[A-Z][A-Z0-9]+-\d+\b",
    re.I)
_BUG_ASK = re.compile(r"\b(bug report|format\w*|write[- ]?up|write (it )?up|report it)\b", re.I)
_EMAIL_ASK = re.compile(r"\b(email|e-mail|mail|send)\b", re.I)


def pending_actions(state: dict) -> list[str]:
    """Real actions the request explicitly asks for that haven't been handled yet."""
    req = state.get("request", "")
    out = []
    if _JIRA_ASK.search(req) and not state.get("jira_result"):
        out.append("jira")
    wants_email = _EMAIL_ASK.search(req) and (EMAIL_RE.search(req) or re.search(r"\bme\b", req, re.I))
    if wants_email and not state.get("email_result"):
        out.append("comms")
    return out


def _already_handled(state: dict, action: str) -> bool:
    return bool(state.get({"jira": "jira_result", "comms": "email_result"}[action]))


def _decide(model, messages) -> tuple[str, str]:
    """Ask the LLM for a routing decision. Structured output first, JSON fallback."""
    try:
        d = model.with_structured_output(Decision).invoke(messages)
        return d.next, d.why
    except Exception as exc:  # noqa: BLE001
        log.warning("structured routing failed (%s), falling back to JSON", exc)
    try:
        msg = model.invoke(messages + [("human", 'Reply ONLY with JSON: {"next": "...", "why": "..."}')])
        raw = msg.content if isinstance(msg.content, str) else str(msg.content)
        d = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return str(d.get("next", "done")), str(d.get("why", ""))
    except Exception:  # noqa: BLE001
        return "done", "could not parse decision"


def supervisor_node(state: dict) -> dict:
    steps = state.get("steps", 0) + 1
    if steps > MAX_STEPS:
        return {"next_agent": "done", "steps": steps,
                "trail": [f"🧭 supervisor → done (step limit {MAX_STEPS} reached)"]}

    model = get_model(state.get("provider"))
    nxt, why = _decide(model, [
        ("system", SUPERVISOR_PROMPT),
        ("human", f"Request: {state['request']}\n\nWork so far:\n{_work_so_far(state)}"),
    ])
    if nxt not in ROUTES:
        nxt, why = "done", f"invalid route '{nxt}'"

    pending = pending_actions(state)
    gathered = bool(state.get("research") or state.get("bug_report"))

    # rule 2: never repeat a real action
    if nxt in ("jira", "comms") and _already_handled(state, nxt):
        why = f"{nxt} already handled — not repeating"
        nxt = pending[0] if pending else "done"
    # rule 5: real actions only when the request explicitly asked for them
    elif nxt in ("jira", "comms") and nxt not in pending:
        why = f"{nxt} was not requested — skipping"
        nxt = pending[0] if (pending and gathered) else "done"
    # rule 6: the bug agent only when a report was asked for or will feed a ticket
    elif (nxt == "bug" and not _BUG_ASK.search(state.get("request", ""))
          and "jira" not in pending):
        why = "no bug report was requested — skipping"
        nxt = pending[0] if (pending and gathered) else "done"
    # rule 4: a step that just passed isn't re-run back-to-back without a reason
    elif (nxt in ("research", "bug") and nxt == state.get("next_agent")
          and state.get("quality_ok")):
        why = f"{nxt} just passed quality — not re-running"
        nxt = pending[0] if pending else "done"
    # rule 3: don't finish while a requested action is still pending
    elif nxt == "done" and pending:
        target = pending[0] if gathered else "research"
        why = f"request still needs {pending[0]} — overriding 'done'"
        nxt = target

    return {"next_agent": nxt, "steps": steps,
            "trail": [f"🧭 supervisor → {nxt} ({why})"]}


def _work_so_far(state: dict) -> str:
    bits = []
    if state.get("research"): bits.append("- research: done")
    if state.get("bug_report"): bits.append("- bug report: written")
    if state.get("jira_result"): bits.append("- jira: " + state["jira_result"][:120])
    if state.get("email_result"): bits.append("- email: " + state["email_result"][:120])
    if state.get("quality_notes"):
        bits.append(f"- quality feedback on last step ({state.get('next_agent','?')}): "
                    + state["quality_notes"][:160])
    return "\n".join(bits) or "(nothing yet)"