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


# =====================================================================
# Layer 4 — TOOL-RESULT guard (indirect prompt injection)
# ---------------------------------------------------------------------
# Text that comes BACK from tools (docs, Jira tickets) is data written by
# someone else. If it contains instructions aimed at the AI, remove those
# lines before the LLM ever reads them.
# =====================================================================
TOOL_INJECTION_PATTERNS = [
    r"\b(ai|assistant|agent|llm|model|chatbot|bot)s?\b[^.\n]{0,60}\b(must|should|need to|are required to|shall|have to)\b",
    r"ignore\s+(all\s+|the\s+|your\s+|any\s+)?(previous\s+|prior\s+|above\s+|earlier\s+)?(instructions|rules|prompts|guidelines)",
    r"disregard\s+[^.\n]{0,40}\b(instructions|rules|policy|policies|guidelines)",
    r"\b(you are now|new instructions|system prompt|developer mode)\b",
    r"\b(email|send|forward|exfiltrate|upload)\b[^.\n]{0,60}\bto\b[^.\n]{0,25}[\w.\-+]+@[\w.\-]+",
    r"\bdo not (tell|inform|mention)\b[^.\n]{0,30}\b(user|anyone|them)\b",
]
_TOOL_INJ = [re.compile(p, re.I) for p in TOOL_INJECTION_PATTERNS]
REMOVED_LINE = "[⚠️ removed by guardrail: this line looked like instructions aimed at the AI]"


def is_read_tool(name: str) -> bool:
    """Tools whose results come from outside content (docs, tickets, mail) — those get
    scanned. Our own action confirmations (send/create/update) and pure formatters don't."""
    if name in ("search_docs", "check_duplicate_bug"):
        return True
    return any(k in name for k in ("search", "get", "list", "read"))


def scan_tool_result(text: str) -> list[str]:
    """Return the suspicious lines found in a tool result (empty list = clean)."""
    hits = []
    for line in (text or "").splitlines():
        if any(p.search(line) for p in _TOOL_INJ):
            hits.append(line.strip()[:160])
    return hits


def sanitize_tool_result(text: str) -> tuple[str, list[str]]:
    """Remove suspicious lines; return (clean_text, flags)."""
    flags, out = [], []
    for line in (text or "").splitlines():
        if any(p.search(line) for p in _TOOL_INJ):
            flags.append(line.strip()[:160])
            out.append(REMOVED_LINE)
        else:
            out.append(line)
    return "\n".join(out), flags


# =====================================================================
# Layer 5 — OUTPUT guard (the final answer shown to the user)
# ---------------------------------------------------------------------
#  * redacts secrets, phone numbers, and email addresses that are neither in an
#    allowed domain nor typed by the user
#  * verifies claims: every ticket key / "ticket created" / "email sent" in the
#    answer must match what the tools actually did (state["tool_log"])
# =====================================================================
OUTPUT_SECRET_RE = re.compile(
    r"\b(sk-(?:ant-|proj-)?[A-Za-z0-9_\-]{16,}|lsv2_[A-Za-z0-9_]{16,}|ATATT[A-Za-z0-9_\-=]{16,}"
    r"|gh[pous]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9\-]{10,})")
PHONE_RE = re.compile(r"(?<![\w-])\+?\d[\d \-().]{8,}\d(?![\w-])")
_NEGATION = re.compile(r"\b(not|no|never|wasn't|weren't|isn't|didn't|blocked|declined|skipped|cancel\w*|failed)\b", re.I)
_EMAIL_CLAIM = re.compile(r"\b(e-?mail|message)\b[^.\n]{0,50}\b(sent|delivered)\b|\bsent\b[^.\n]{0,40}\be-?mail\b", re.I)
_TICKET_CLAIM = re.compile(r"\b(created|filed|opened|raised|logged)\b[^.\n]{0,50}"
                           r"(\b(ticket|issue)\b|\b[A-Z][A-Z0-9]+-\d+\b)"
                           r"|\b(ticket|issue)\b[^.\n]{0,50}\b(created|filed|opened|raised)\b", re.I)


def output_guard_enabled() -> bool:
    return os.getenv("OUTPUT_GUARD", "true").lower() != "false"


def redact_output(text: str, request: str = "") -> tuple[str, list[str]]:
    """Redact secrets, phone numbers and unapproved email addresses."""
    flags: list[str] = []

    def _secret(m):
        flags.append("redacted a secret / API key")
        return "[REDACTED secret]"
    text = OUTPUT_SECRET_RE.sub(_secret, text)

    allowed_domains = set(allowed_email_domains())
    typed = {a.lower() for a in EMAIL_RE.findall(request or "")}

    def _email(m):
        addr = m.group(0)
        if addr.lower() in typed or addr.rsplit("@", 1)[-1].lower() in allowed_domains:
            return addr
        flags.append(f"redacted an email address outside the allowed domains")
        return "[REDACTED email]"
    text = EMAIL_RE.sub(_email, text)

    def _phone(m):
        digits = re.sub(r"\D", "", m.group(0))
        if 10 <= len(digits) <= 15:
            flags.append("redacted a phone number")
            return "[REDACTED phone]"
        return m.group(0)
    text = PHONE_RE.sub(_phone, text)
    return text, flags


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def verify_claims(text: str, state: dict) -> tuple[str, list[str]]:
    """Check what the answer CLAIMS against what the tools actually DID."""
    log = state.get("tool_log", []) or []
    ok = [e for e in log if e.get("status") == "ok"]
    verified_keys = {s["label"] for e in ok for s in e.get("sources", []) if s.get("kind") == "jira"}
    sent = any(("send" in e["tool"]) and e["tool"].startswith("gmail") for e in ok)
    created = any(e["tool"] in ("jira_create_issue",) for e in ok)
    flags: list[str] = []

    project = allowed_jira_project()
    for key in sorted(set(re.findall(rf"\b{re.escape(project)}-\d+\b", text))):
        if key not in verified_keys:
            flags.append(f"{key} is mentioned but no tool returned it")
            text = re.sub(rf"\b{re.escape(key)}\b(?! \(⚠️)", f"{key} (⚠️ unverified)", text)

    for sentence in _sentences(text):
        if _NEGATION.search(sentence):
            continue
        if _EMAIL_CLAIM.search(sentence) and not sent:
            flags.append("the answer says an email was sent, but no send was recorded")
            break
    for sentence in _sentences(text):
        if _NEGATION.search(sentence):
            continue
        if _TICKET_CLAIM.search(sentence) and not created and not verified_keys:
            flags.append("the answer says a ticket was created, but no ticket was created")
            break
    return text, flags


def output_guard_node(state: dict) -> dict:
    """Last node before END: clean and fact-check the final answer."""
    if not output_guard_enabled():
        return {"trail": ["🛡️ output guard: off (OUTPUT_GUARD=false)"]}
    final = state.get("final", "") or ""
    # the Action status block is built from state by code — leave it untouched
    body, sep, status = final.partition("\n\n---\n**Action status**")
    body, redactions = redact_output(body, state.get("request", ""))
    body, claims = verify_claims(body, state)
    flags = list(dict.fromkeys(redactions + claims))
    new_final = body + (sep + status if sep else "")
    if not flags:
        return {"final": new_final, "trail": ["🛡️ output guard: answer passed"]}
    new_final += "\n\n---\n**🛡️ Output guard**\n" + "\n".join(f"- {f}" for f in flags)
    return {"final": new_final, "output_flags": flags,
            "trail": [f"🛡️ output guard: {len(flags)} issue(s) fixed or flagged — " + "; ".join(flags)[:140]]}