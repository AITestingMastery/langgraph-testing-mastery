"""
quality.py — the grader node that drives the self-correction CYCLE.

It grades ONLY the step that just ran — not whether the whole request is finished
(that's the supervisor's job). Control always returns to the supervisor:
  - pass  → supervisor picks the next step (or 'done')
  - fail  → supervisor re-routes to the same step with the reviewer's notes

`loops` counts consecutive retries of the CURRENT step and resets on a pass, so a
multi-step request (research → bug → jira → comms) never runs out of loop budget.

Real actions (jira, comms) are never auto-retried: re-running them would file a
duplicate ticket or send a second email.
"""
from __future__ import annotations

import json
import logging
import os

from pydantic import BaseModel, Field

from llm import get_model

log = logging.getLogger(__name__)

MAX_LOOPS = int(os.getenv("MAX_LOOPS", "2"))  # max retries per step
MAX_REVIEW_CHARS = 12000  # enough for a full research answer
ACTION_STEPS = {"jira", "comms"}

SYSTEM = """You are a QA reviewer. Judge ONLY the output of the step that just ran —
not whether the whole request is finished (other steps may still follow).
Approve if this step's output is accurate and useful for its part of the request.
If it's weak, say specifically what's missing."""


class Verdict(BaseModel):
    ok: bool
    notes: str = Field(default="", description="what's missing, if anything")


def _step_output(state: dict, step: str) -> str:
    key = {"research": "research", "bug": "bug_report",
           "jira": "jira_result", "comms": "email_result"}.get(step, "")
    return state.get(key, "") if key else ""


def _for_review(output: str) -> str:
    """Give the reviewer the whole output; if we must cut, say so explicitly
    so the cut itself is never graded as a defect (this caused false loop-backs)."""
    if len(output) <= MAX_REVIEW_CHARS:
        return output
    return (output[:MAX_REVIEW_CHARS]
            + "\n\n[... truncated for review length — do NOT treat this cut-off as a defect]")


def _grade(model, request: str, step: str, output: str) -> tuple[bool, str]:
    msgs = [("system", SYSTEM),
            ("human", f"Request: {request}\n\nStep that just ran: {step}\n\nIts output:\n{_for_review(output)}")]
    try:
        v = model.with_structured_output(Verdict).invoke(msgs)
        return bool(v.ok), v.notes or ""
    except Exception as exc:  # noqa: BLE001
        log.warning("structured grading failed (%s), falling back to JSON", exc)
    try:
        raw = model.invoke(msgs + [("human", 'Reply ONLY with JSON: {"ok": true|false, "notes": "..."}')]).content
        raw = raw if isinstance(raw, str) else str(raw)
        v = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return bool(v.get("ok", True)), str(v.get("notes", ""))
    except Exception:  # noqa: BLE001
        return True, ""


def quality_node(state: dict) -> dict:
    step = state.get("next_agent", "")

    if step in ACTION_STEPS:
        return {"quality_ok": True, "quality_notes": "", "loops": 0,
                "trail": [f"✅ quality: {step} result recorded (real actions are never auto-retried)"]}

    output = _step_output(state, step)
    ok, notes = _grade(get_model(state.get("provider")), state["request"], step, output)
    loops = state.get("loops", 0)

    if not ok and loops >= MAX_LOOPS:
        return {"quality_ok": True, "quality_notes": "", "loops": 0,
                "trail": [f"✅ quality: accepting {step} after {loops} retries (loop limit)"]}
    if ok:
        return {"quality_ok": True, "quality_notes": "", "loops": 0,
                "trail": [f"✅ quality: {step} passed"]}

    reason = (notes[:70] + "…") if len(notes) > 70 else (notes or "needs more detail")
    return {"quality_ok": False, "quality_notes": notes or "needs more detail",
            "loops": loops + 1,
            "trail": [f"quality: needs work → loop back to {step} — \"{reason}\""]}