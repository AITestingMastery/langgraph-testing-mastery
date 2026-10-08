"""Make the repo root importable and give every test a clean, key-free env."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OPENAI_API_KEY", "sk-test-offline-not-real")

# Settings from YOUR .env that change guard / budget behaviour. app.py calls
# load_dotenv(), so without this a value like MAX_TOKENS_PER_REQUEST=3000 (left over
# from a manual test) leaks into every test after the first app smoke test.
# Same lesson as the Advance-RAG OPENWEATHER_API_KEY test: tests must not depend on .env.
_GUARD_SETTINGS = (
    "MAX_TOKENS_PER_REQUEST", "MAX_COST_PER_REQUEST_USD", "MAX_COST_PER_DAY_USD",
    "MAX_TICKETS_PER_REQUEST", "MAX_EMAILS_PER_REQUEST", "MAX_UPDATES_PER_REQUEST",
    "MAX_ACTIONS_PER_HOUR", "MAX_LOOPS", "MAX_STEPS", "RAG_TOP_K", "OUTPUT_GUARD",
    "DOC_ID_PATTERN", "ALLOWED_EMAIL_DOMAINS", "JIRA_PROJECT_KEY", "DEFAULT_EMAIL_TO",
    "AUDIT_LOG_ENABLED", "EVAL_JUDGE_MODEL", "EVAL_WORKERS", "EVAL_GATE_JUDGE",
    "LIVE_EVAL", "LIVE_EVAL_JUDGE", "LIVE_EVAL_SAMPLE", "LIVE_EVAL_THRESHOLD",
    "LIVE_EVAL_TOKEN_BUDGET", "LIVE_EVAL_SECONDS_BUDGET",
)


@pytest.fixture(autouse=True)
def _isolate_guard_state(tmp_path, monkeypatch):
    """Each test starts from the documented defaults: a fresh action budget, its own
    audit/usage files, and none of your .env guard or budget overrides."""
    import dotenv
    import guardrails
    guardrails.reset_action_budget()
    # app.py (run by the smoke tests) calls load_dotenv() DURING a test, which would
    # re-read your .env after the cleanup below — so tests never load .env at all.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for name in _GUARD_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    for name in [k for k in os.environ if k.startswith("PRICE_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("USAGE_LOG_PATH", str(tmp_path / "usage.json"))
    # never talk to your real LangSmith account from a test: no traces, no feedback
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    # real-time evaluation: off unless a test turns it on, and never writes into the repo
    monkeypatch.setenv("LIVE_EVAL", "off")
    monkeypatch.setenv("LIVE_EVAL_PATH", str(tmp_path / "live_evals.jsonl"))
    monkeypatch.setenv("LIVE_FEEDBACK_PATH", str(tmp_path / "user_feedback.jsonl"))
    monkeypatch.setenv("GOLDEN_CANDIDATES_PATH", str(tmp_path / "golden_candidates.jsonl"))
    yield
    guardrails.reset_action_budget()