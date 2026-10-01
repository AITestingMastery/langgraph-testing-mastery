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
_PROJECT_ASK = re.compile(
    r"\bproject\s+([A-Z][A-Z0-9]{1,9})\b"
    r"|\b(?:ticket|issue|bug)\b[^.\n]{0,60}?\b(?:in|into|under)\s+([A-Z][A-Z0-9]{1,9})\b"
    r"(?=\s*(?:[.,;!?]|$|\s+(?:and|for|with|to|please)\b))")


def requested_project(request: str) -> str | None:
    """The Jira project key the user explicitly named, if any ('… in PROD', 'project PROD')."""
    m = _PROJECT_ASK.search(request or "")
    return (m.group(1) or m.group(2)) if m else None


def check_request(request: str) -> tuple[bool, str]:
    """Guardrail on the incoming request: catch obvious injection/unsafe asks."""
    normalized, notes = normalize_text(request)      # see "Hidden-text normalization"
    low = normalized.lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, low):
            return False, ("this request looks like a prompt-injection or an unsafe "
                           f"instruction{_disguise_note(notes)}, so it was refused.")
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
RECIPIENT_KEYS = ("to", "cc", "bcc", "recipient", "recipients", "to_email", "email")


def _recipients(args: dict) -> list[str]:
    """Addresses the email will be SENT to. Addresses inside the body are content —
    the outbound scrub redacts those instead of blocking the whole email."""
    found = []
    for k, v in (args or {}).items():
        if k.lower() in RECIPIENT_KEYS:
            found += EMAIL_RE.findall(json.dumps(v, default=str))
    return found


def check_email_args(tool_name: str, args: dict) -> tuple[bool, str]:
    """Validate the REAL arguments of a Gmail tool call: every RECIPIENT must be
    allow-listed, and nothing in the payload may look like a secret."""
    blob = json.dumps(args, default=str)
    # recipients from the recipient fields; if a tool uses unknown field names,
    # fall back to every address in the payload (safe default)
    addrs = _recipients(args) or EMAIL_RE.findall(blob)
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
        from audit import audit
        audit("guard_block", layer=1, reason=msg, request=state.get("request", "")[:200])
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
            from audit import audit
            audit("guard_block", layer=2, action="email", reason=msg)
            return {"guardrail_block": True, "guardrail_note": msg,
                    "email_result": f"(blocked: {msg})",
                    "trail": [f"🛡️ guardrail BLOCKED email → {msg}"]}
        return {"guardrail_block": False,
                "trail": [f"🛡️ guardrail: email to {to} allowed"]}

    if action == "jira":
        asked = requested_project(req)
        if asked and asked != allowed_jira_project():
            from audit import audit
            msg = (f"you asked for project {asked}, but this assistant may only file tickets "
                   f"in {allowed_jira_project()} (JIRA_PROJECT_KEY)")
            audit("guard_block", layer=2, action="jira", reason=msg)
            return {"guardrail_block": True, "guardrail_note": msg,
                    "jira_result": f"(blocked: {msg})",
                    "trail": [f"🛡️ guardrail BLOCKED jira → {msg}"]}
        if not (state.get("bug_report") or state.get("research")):
            from audit import audit
            audit("guard_block", layer=2, action="jira", reason="nothing to file")
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
    # text ADDRESSED to an AI — not ordinary words like "user agent", "bot detection",
    # "data model" (the red-team set found those as false positives)
    r"\b(ai|llm|chatbot)s?\b[^.\n]{0,60}\b(must|should|need to|are required to|shall|have to)\b",
    r"\b(assistants?|agents?|models?|bots?|llms?)\s+(reading|processing|summari[sz]ing|parsing|handling)\s+this\b",
    r"\bnote\s+(for|to)\s+(the\s+)?(ai|assistants?|agents?|bots?|llms?)\b",
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


def _suspicious(line: str) -> bool:
    normalized, _ = normalize_text(line)
    return any(p.search(normalized) for p in _TOOL_INJ)


def scan_tool_result(text: str) -> list[str]:
    """Return the suspicious lines found in a tool result (empty list = clean)."""
    return [line.strip()[:160] for line in (text or "").splitlines() if _suspicious(line)]


def sanitize_tool_result(text: str) -> tuple[str, list[str]]:
    """Remove suspicious lines; return (clean_text, flags)."""
    flags, out = [], []
    for line in (text or "").splitlines():
        if _suspicious(line):
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
_NEGATION = re.compile(r"\b(not|no|never|wasn't|weren't|isn't|didn't|doesn't|cannot|can't|couldn't|unable|"
                       r"unavailable|unknown|blocked|declined|skipped|cancel\w*|failed)\b", re.I)
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
    created_keys = {s["label"] for e in ok if e["tool"] == "jira_create_issue"
                    for s in e.get("sources", []) if s.get("kind") == "jira"}
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
    # a key that the answer says was CREATED must be one the create tool returned —
    # not just one that appeared in a search (live finding: an old ticket's key was
    # reported as the new ticket)
    for sentence in _sentences(text):
        if _NEGATION.search(sentence) or not _TICKET_CLAIM.search(sentence):
            continue
        for key in dict.fromkeys(re.findall(rf"\b{re.escape(project)}-\d+\b", sentence)):
            if created_keys and key not in created_keys:
                real = ", ".join(sorted(created_keys))
                flags.append(f"{key} is reported as created, but the create tool returned {real}")
                text = text.replace(sentence, sentence.replace(
                    key, f"{key} (⚠️ the ticket actually created is {real})"), 1)

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
    body, ids = verify_doc_ids(body, state)
    flags = list(dict.fromkeys(redactions + claims + ids))
    new_final = body + (sep + status if sep else "")
    if not flags:
        return {"final": new_final, "trail": ["🛡️ output guard: answer passed"]}
    new_final += "\n\n---\n**🛡️ Output guard**\n" + "\n".join(f"- {f}" for f in flags)
    from audit import audit
    audit("output_guard", layer=5, flags=flags)
    return {"final": new_final, "output_flags": flags,
            "trail": [f"🛡️ output guard: {len(flags)} issue(s) fixed or flagged — " + "; ".join(flags)[:140]]}


# =====================================================================
# BATCH 1 HARDENING
# =====================================================================

# ---------------------------------------------------------------------
# (1) Hidden-text normalization — used by layers 1 and 4
# Attackers disguise trigger words so a regex can't see them:
#   "Ig\u200bnore"  (invisible zero-width space)   "Іgnore" (Cyrillic І)
#   "I g n o r e"   (spaced letters)               "Ｉｇｎｏｒｅ" (full-width)
# We scan a NORMALIZED copy. The original text is never changed.
# ---------------------------------------------------------------------
import time as _time
import unicodedata
from collections import deque

_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064\ufeff\u00ad"), None)
_HOMOGLYPHS = str.maketrans({
    # Cyrillic
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "ѕ": "s", "ԁ": "d", "һ": "h", "ӏ": "l", "ԛ": "q", "ԝ": "w",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C",
    "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S",
    # Greek
    "α": "a", "ο": "o", "ρ": "p", "ι": "i", "κ": "k", "ν": "v",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N",
    "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Latin look-alikes
    "ɡ": "g", "ı": "i",
})
_SPACED = re.compile(r"(?<![A-Za-z])(?:[A-Za-z][ .\-_*]){3,}[A-Za-z](?![A-Za-z])")


_LEET_MAP = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"}
_LEET = re.compile(r"(?<=[A-Za-z])[013457](?=[A-Za-z0-9]*[A-Za-z])|(?<![A-Za-z0-9])[013457](?=[A-Za-z]{2,})")


def normalize_text(text: str) -> tuple[str, list[str]]:
    """Return (normalized_copy, disguises_found). Only used for SCANNING."""
    notes: list[str] = []
    t = unicodedata.normalize("NFKC", text or "")
    if any(0xFF01 <= ord(c) <= 0xFF5E for c in (text or "")):
        notes.append("full-width letters")
    t2 = t.translate(_INVISIBLE)
    if t2 != t:
        notes.append("invisible characters")
    t3 = t2.translate(_HOMOGLYPHS)
    if t3 != t2:
        notes.append("look-alike letters")
    t4 = _SPACED.sub(lambda m: re.sub(r"[ .\-_*]", "", m.group(0)), t3)
    if t4 != t3:
        notes.append("spaced-out letters")
    # leetspeak: digits standing in for letters INSIDE words ("1gn0re" → "ignore")
    t5 = t4
    for _ in range(4):                       # repeat: "prev10us" needs two passes
        nxt = _LEET.sub(lambda m: _LEET_MAP[m.group(0)], t5)
        if nxt == t5:
            break
        t5 = nxt
    if t5 != t4:
        notes.append("number-for-letter swaps")
    return t5, notes


def _disguise_note(notes: list[str]) -> str:
    return f" (disguised with {', '.join(notes)})" if notes else ""


# ---------------------------------------------------------------------
# (2) Outbound content scrub — extends layer 3
# Before an email / ticket is SENT, redact phone numbers, outside email addresses
# and secrets from its text fields (recipients and keys are left alone).
# ---------------------------------------------------------------------
_NON_CONTENT_KEYS = {"to", "cc", "bcc", "recipient", "recipients", "from", "reply_to",
                     "project_key", "project", "issue_key", "issuekey", "key", "issue_type",
                     "priority", "labels", "assignee"}


def scrub_outbound_args(tool_name: str, args: dict, request: str = "") -> tuple[dict, list[str]]:
    """Return (scrubbed_args, what_was_redacted) for a real email / ticket call."""
    flags: list[str] = []
    out: dict = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and k.lower() not in _NON_CONTENT_KEYS:
            new, f = redact_output(v, request)
            flags += f
            out[k] = new
        else:
            out[k] = v
    return out, list(dict.fromkeys(flags))


# ---------------------------------------------------------------------
# (3) Grounded document IDs — extends layer 5
# Every BUG-123-style ID in the answer must appear in something a tool returned.
# ---------------------------------------------------------------------
def doc_id_re() -> re.Pattern:
    return re.compile(os.getenv("DOC_ID_PATTERN", r"\bBUG-\d+\b"))


def verify_doc_ids(text: str, state: dict) -> tuple[str, list[str]]:
    known = {i for e in (state.get("tool_log") or []) if e.get("status") == "ok"
             for i in e.get("ids", [])}
    flags: list[str] = []
    for sentence in _sentences(text):
        if _NEGATION.search(sentence):
            continue
        bad = [i for i in dict.fromkeys(doc_id_re().findall(sentence)) if i not in known]
        if bad:
            marked = sentence
            for i in bad:
                marked = re.sub(rf"\b{re.escape(i)}\b(?! \(⚠️)", f"{i} (⚠️ not in any source)", marked)
                flags.append(f"{i} is mentioned but isn't in any document or ticket the tools returned")
            text = text.replace(sentence, marked, 1)
    return text, list(dict.fromkeys(flags))


# ---------------------------------------------------------------------
# (4) Action budget — extends the supervisor rules
# Limits real actions per request (per agent run) and per hour (this app process).
# ---------------------------------------------------------------------
_ACTION_TIMES: deque = deque()


def action_kind(tool_name: str) -> str | None:
    if tool_name == "jira_create_issue":
        return "ticket"
    if tool_name in ("gmail_send_message", "gmail_send_draft"):
        return "email"
    if tool_name in ("jira_update_issue", "jira_add_comment"):
        return "update"
    return None


def _limit(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def check_budget(tool_name: str, run_log: list) -> tuple[bool, str]:
    kind = action_kind(tool_name)
    if kind is None:
        return True, "not an action"
    per_request = {"ticket": _limit("MAX_TICKETS_PER_REQUEST", 1),
                   "email": _limit("MAX_EMAILS_PER_REQUEST", 1),
                   "update": _limit("MAX_UPDATES_PER_REQUEST", 3)}[kind]
    done = sum(1 for e in run_log if e.get("status") == "ok" and action_kind(e.get("tool", "")) == kind)
    if done >= per_request:
        return False, (f"Action budget: at most {per_request} {kind}"
                       f"{'s' if per_request != 1 else ''} per request (MAX_{kind.upper()}S_PER_REQUEST).")
    hourly = _limit("MAX_ACTIONS_PER_HOUR", 10)
    now = _time.time()
    while _ACTION_TIMES and now - _ACTION_TIMES[0] > 3600:
        _ACTION_TIMES.popleft()
    if len(_ACTION_TIMES) >= hourly:
        return False, f"Action budget: {hourly} real actions in the last hour (MAX_ACTIONS_PER_HOUR)."
    return True, "within budget"


def record_action(tool_name: str) -> None:
    if action_kind(tool_name):
        _ACTION_TIMES.append(_time.time())


def reset_action_budget() -> None:
    _ACTION_TIMES.clear()


# ---------------------------------------------------------------------
# (5) Scope guard — what to say when nothing in the request is QA work
# ---------------------------------------------------------------------
def scope_message() -> str:
    return (
        "🧭 That's outside what this QA assistant does, so nothing was run.\n\n"
        "**I can help you:**\n"
        "- research our QA docs and Jira — *\"What known bugs affect the login page?\"*\n"
        "- write a standard bug report — *\"Format the Chrome login bug as a bug report.\"*\n"
        f"- file a Jira ticket in **{allowed_jira_project()}** — after you approve\n"
        "- email a summary — after you approve\n"
        "- answer general QA questions — *\"What's the difference between severity and priority?\"*")