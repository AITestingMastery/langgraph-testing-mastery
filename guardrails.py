"""
guardrails.py — safety checks that run BEFORE real actions.

Two layers (defense in depth):
  1. guardrail_node      — a visible graph node before jira/comms. Checks the
                           request itself (early, shows in the trail).
  2. check_*_args()      — enforced on the ACTUAL tool-call arguments inside the
                           tool loop (agents/_helpers.py). Even if the LLM decides
                           to email someone else or file into another project,
                           the call is refused before it reaches the tool.

All config is read at CALL time (not import time) so values in .env always apply,
no matter when load_dotenv() runs — this avoids the import-order bug class.
"""

from __future__ import annotations

import json
import os
import re

EMAIL_RE = re.compile(r"[\w.\-+]+@[\w.\-]+\.\w+")
SECRET_RE = re.compile(r"(sk-[a-z0-9_\-]{10,}|api[_-]?key\s*[:=]|ATATT[a-z0-9_\-]{10,})", re.I)

# crude prompt-injection / unsafe-instruction signals in the user's request
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|the\s+)?(previous\s+|prior\s+)?(instructions|rules|prompts)",
    r"disregard .* (instructions|policy|rules)",
    r"reveal .* (system prompt|api key|token|secret|password)",
    r"delete (all|everything|the (whole )?project|the database)",
    r"forget (all|your|the) (instructions|rules)",
]


# ---------------- config (read at call time) ----------------
def allowed_email_domains() -> list[str]:
    raw = os.getenv("ALLOWED_EMAIL_DOMAINS", "gmail.com,example.com")
    return [d.strip().lower() for d in raw.split(",") if d.strip()]


def allowed_jira_project() -> str:
    return os.getenv("JIRA_PROJECT_KEY", "TEST").strip().upper()


def default_email_to() -> str:
    return os.getenv("DEFAULT_EMAIL_TO", "").strip()


# ---------------- basic checks ----------------
def check_request(request: str) -> tuple[bool, str]:
    """Guardrail on the incoming request: catch obvious injection/unsafe asks."""
    low = request.lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, low):
            return False, ("this request looks like a prompt-injection or an unsafe "
                           "instruction, so it was refused.")
    return True, "request ok"


def check_email(to: str, body: str) -> tuple[bool, str]:
    """Recipient domain must be allow-listed; body must not leak a secret."""
    if not to or "@" not in to:
        return False, "Email blocked: no valid recipient."
    domains = allowed_email_domains()
    domain = to.rsplit("@", 1)[-1].lower()
    if domain not in domains:
        return False, (f"Email blocked by guardrail: '{domain}' is not in the allowed "
                       f"domains ({', '.join(domains)}).")
    if body and SECRET_RE.search(body):
        return False, "Email blocked: body appears to contain a secret/API key."
    return True, "email ok"


def check_jira(summary: str, project_key: str) -> tuple[bool, str]:
    """Require a real summary and the allowed project key."""
    if not summary or len(summary.strip()) < 5:
        return False, "Jira blocked by guardrail: summary is missing or too short."
    allowed = allowed_jira_project()
    if not project_key or project_key.strip().upper() != allowed:
        return False, (f"Jira blocked by guardrail: project '{project_key or '(none)'}' "
                       f"is not allowed (only {allowed}).")
    return True, "jira ok"


# ---------------- tool-call level checks (used in agents/_helpers.py) ----------------
def check_email_args(tool_name: str, args: dict) -> tuple[bool, str]:
    """Validate the REAL arguments of a Gmail tool call: every address must be
    allow-listed, and nothing in the payload may look like a secret."""
    blob = json.dumps(args, default=str)
    addrs = EMAIL_RE.findall(blob)
    for addr in addrs:
        ok, msg = check_email(addr, "")
        if not ok:
            return ok, msg
    if SECRET_RE.search(blob):
        return False, "Email blocked: payload appears to contain a secret/API key."
    return True, "email args ok"


def check_jira_args(tool_name: str, args: dict) -> tuple[bool, str]:
    """Validate the REAL arguments of a Jira write call (mcp-atlassian arg names,
    with fallbacks)."""
    allowed = allowed_jira_project()
    if "create" in tool_name:
        project = str(args.get("project_key") or args.get("project") or "")
        return check_jira(str(args.get("summary", "")), project)
    issue_key = str(args.get("issue_key") or args.get("issueKey") or args.get("key") or "")
    if issue_key and not issue_key.upper().startswith(f"{allowed}-"):
        return False, f"Jira blocked by guardrail: {issue_key} is outside project {allowed}."
    return True, "jira args ok"


# ---------------- graph nodes ----------------
def entry_guard_node(state: dict) -> dict:
    """Runs FIRST, before the supervisor — checks the incoming request itself."""
    ok, msg = check_request(state.get("request", ""))
    if not ok:
        return {"guardrail_block": True, "guardrail_note": msg,
                "final": f"🛡️ Request blocked by guardrail. {msg}",
                "trail": [f"🛡️ entry guardrail BLOCKED: {msg}"]}
    return {"guardrail_block": False,
            "trail": ["🛡️ entry guardrail: request passed"]}


def guardrail_node(state: dict) -> dict:
    """Runs right before the gated action (jira or comms) — the early, visible check.
    If it blocks, the action's result is recorded as blocked and control returns to
    the supervisor (so other requested work can still continue)."""
    action = state.get("next_agent", "")
    req = state.get("request", "")

    if action == "comms":
        m = EMAIL_RE.search(req)
        to = m.group(0) if m else default_email_to()
        ok, msg = check_email(to, "")
        if not ok:
            return {"guardrail_block": True, "guardrail_note": msg,
                    "email_result": f"(blocked: {msg})",
                    "trail": [f"🛡️ guardrail BLOCKED email → {msg}"]}
        return {"guardrail_block": False,
                "trail": [f"🛡️ guardrail: email to {to} allowed"]}

    if action == "jira":
        if not (state.get("bug_report") or state.get("research")):
            return {"guardrail_block": True,
                    "guardrail_note": "nothing gathered to file a ticket from",
                    "jira_result": "(blocked: nothing to file)",
                    "trail": ["🛡️ guardrail BLOCKED jira → no content to file"]}
        return {"guardrail_block": False,
                "trail": [f"🛡️ guardrail: jira action allowed (project {allowed_jira_project()})"]}

    return {"guardrail_block": False, "trail": ["🛡️ guardrail: passed"]}