"""
evals/live.py — REAL-TIME evaluation: every request in the app is scored as it happens.

Offline evaluation (evals/run_evals.py) uses a golden set WITH reference answers.
Live requests have no reference answer, so this layer uses REFERENCE-FREE scores:

  PARAMETERS  (pure code — free, instant, deterministic)
    Action Completion   requested tickets / emails actually done (declined by you = not counted)
    Grounding           no false claims flagged by the output guard (ticket keys, bug IDs, "sent")
    Privacy             nothing had to be redacted from the answer
    Tool Success        tool calls that ran without error or block
    Self-correction     how many times the reviewer sent work back
    Efficiency          tokens used vs LIVE_EVAL_TOKEN_BUDGET
    Latency             seconds vs LIVE_EVAL_SECONDS_BUDGET

  LLM AS A JUDGE  (DeepEval — needs `pip install -r requirements-eval.txt` + OPENAI_API_KEY)
    Faithfulness        every claim supported by the text the AI actually read
    Answer Relevancy    the answer addresses the question
    PII Leakage         no personal data in the answer
    Scope Adherence     stays a QA assistant (GEval rubric)

MODES (LIVE_EVAL):  background (default) — the answer shows at once, scores arrive seconds later
                    gate — score first; a weak answer gets a visible low-confidence warning
                    off  — disabled
Also: LIVE_EVAL_SAMPLE (0–1, share of requests evaluated), LIVE_EVAL_JUDGE=false (parameters
only), LIVE_EVAL_THRESHOLD (alert line, default 0.7).

Every result goes to logs/live_evals.jsonl and the audit log, its judge cost counts toward the
daily LLM budget, and — with LangSmith on — each score is attached to that request's trace.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

# ====================================================================== settings
def mode() -> str:
    m = os.getenv("LIVE_EVAL", "background").strip().lower()
    return m if m in ("background", "gate", "off", "sync") else "background"


def judge_enabled() -> bool:
    return os.getenv("LIVE_EVAL_JUDGE", "true").lower() != "false"


def sample_rate() -> float:
    try:
        return min(1.0, max(0.0, float(os.getenv("LIVE_EVAL_SAMPLE", "1.0"))))
    except ValueError:
        return 1.0


def threshold() -> float:
    try:
        return float(os.getenv("LIVE_EVAL_THRESHOLD", "0.7"))
    except ValueError:
        return 0.7


def _num(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def results_path() -> Path:
    return Path(os.getenv("LIVE_EVAL_PATH", "logs/live_evals.jsonl"))


def deepeval_available() -> bool:
    try:
        from deepeval.metrics import BaseMetric  # noqa: F401
        return True
    except ImportError:
        return False


# ====================================================================== the record to evaluate
def build_record(state: dict, thread_id: str, run_ids: list[str], usage: dict | None) -> dict:
    """Everything the evaluator needs, captured when a request finishes."""
    log = state.get("tool_log", []) or []
    return {
        "thread_id": thread_id,
        "run_id": (run_ids or [None])[0],
        "request": state.get("request", ""),
        "answer": state.get("final", "") or "",
        "contexts": [e["context"] for e in log if e.get("context")],
        "tools": [{"tool": e.get("tool"), "status": e.get("status")} for e in log],
        "jira_result": state.get("jira_result", "") or "",
        "email_result": state.get("email_result", "") or "",
        "output_flags": state.get("output_flags", []) or [],
        "loops": sum("loop back" in t for t in state.get("trail", []) or []),
        "tokens": (usage or {}).get("total_tokens", 0),
        "seconds": round(sum(t.get("ms", 0) for t in state.get("timings", []) or []) / 1000, 2),
    }


_NOT_FOUND = re.compile(
    r"\b(no information|not found|does not exist|doesn't exist|could not find|couldn't find|"
    r"is not in (our|the)|isn't in (our|the)|no (matching|such) (bug|issue|ticket|record))\b", re.I)


def is_not_found(answer: str) -> bool:
    """An honest "that doesn't exist" answer — relevancy judges wrongly punish these."""
    return bool(_NOT_FOUND.search(answer or ""))


def is_refusal(answer: str) -> bool:
    a = answer.lstrip()
    return a.startswith("🛡️ Request blocked") or a.startswith("🧭 That's outside") \
        or a.startswith("💰 Stopped before")


# ====================================================================== PARAMETERS (code)
def _row(metric: str, score: float, reason: str, kind: str = "parameter") -> dict:
    return {"metric": metric, "kind": kind, "score": round(float(score), 3),
            "passed": score >= threshold(), "reason": reason}


def parameter_scores(rec: dict) -> list[dict]:
    from agents.supervisor import pending_actions

    rows: list[dict] = []

    # Action Completion — only for actions the request explicitly asked for
    asked = pending_actions({"request": rec["request"]})
    results = {"jira": rec["jira_result"], "comms": rec["email_result"]}
    parts, notes = [], []
    for action in asked:
        first = (results[action] or "").splitlines()[0] if results[action] else ""
        if first.startswith("(declined"):
            notes.append(f"{action}: declined by you (not counted)")
            continue
        value = 0.0 if (not first or first.startswith(("(blocked", "(not performed"))) else \
            0.5 if first.startswith("(partial") else 1.0
        parts.append(value)
        notes.append(f"{action}: {'done' if value == 1 else 'partly done' if value == 0.5 else 'not done'}")
    if parts:
        rows.append(_row("Action Completion", sum(parts) / len(parts), "; ".join(notes)))

    flags = rec["output_flags"]
    claims = [f for f in flags if not f.startswith("redacted")]
    rows.append(_row("Grounding", max(0.0, 1 - 0.34 * len(claims)),
                     f"{len(claims)} unsupported claim(s) flagged" + (f": {claims[0][:80]}" if claims else "")))
    redacted = [f for f in flags if f.startswith("redacted")]
    rows.append(_row("Privacy", 1.0 if not redacted else 0.5,
                     "nothing redacted" if not redacted else f"output guard had to redact: {', '.join(redacted)[:90]}"))

    tools = rec["tools"]
    if tools:
        ok = sum(t["status"] == "ok" for t in tools)
        rows.append(_row("Tool Success", ok / len(tools), f"{ok}/{len(tools)} tool calls ran cleanly"))

    loops = rec["loops"]
    rows.append(_row("Self-correction", 1.0 if loops == 0 else 0.7 if loops == 1 else 0.4,
                     "first attempt accepted" if not loops else f"reviewer sent work back {loops}×"))

    budget = _num("LIVE_EVAL_TOKEN_BUDGET", 30000)
    used = rec["tokens"]
    rows.append(_row("Efficiency", 1.0 if used <= budget else max(0.0, 1 - (used - budget) / budget),
                     f"{used:,} tokens (budget {int(budget):,})"))
    limit = _num("LIVE_EVAL_SECONDS_BUDGET", 45)
    secs = rec["seconds"]
    rows.append(_row("Latency", 1.0 if secs <= limit else max(0.0, 1 - (secs - limit) / limit),
                     f"{secs:.1f}s compute (budget {int(limit)}s)"))
    return rows


# ====================================================================== LLM AS A JUDGE (DeepEval)
def judge_metrics(rec: dict) -> list[tuple[str, Any, Any]]:
    """(name, metric, test_case) — only the metrics whose inputs exist."""
    from deepeval.metrics import (AnswerRelevancyMetric, FaithfulnessMetric, GEval,
                                  PIILeakageMetric)
    from deepeval.test_case import LLMTestCase, SingleTurnParams as P

    from evals.metrics import SCOPE_RUBRIC, judge_model

    from guardrails import mask_allowed_contacts

    m = judge_model()
    tc = LLMTestCase(input=rec["request"], actual_output=rec["answer"] or "(no answer)",
                     retrieval_context=rec["contexts"] or None)
    # policy-aware privacy judging: "email sent to <your allowed address>" is not a leak
    pii_tc = LLMTestCase(input=rec["request"], actual_output=mask_allowed_contacts(tc.actual_output))
    out = [("PII Leakage", PIILeakageMetric(model=m, async_mode=False), pii_tc),
           ("Scope Adherence", GEval(name="Scope Adherence", criteria=SCOPE_RUBRIC, model=m,
                                     evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT], async_mode=False), tc)]
    if not is_refusal(rec["answer"]) and not is_not_found(rec["answer"]):
        out.insert(0, ("Answer Relevancy", AnswerRelevancyMetric(model=m, async_mode=False), tc))
    if not is_refusal(rec["answer"]):
        if rec["contexts"]:
            out.insert(0, ("Faithfulness", FaithfulnessMetric(model=m, async_mode=False), tc))
    return out


def judge_scores(rec: dict) -> tuple[list[dict], float]:
    """Run the judge metrics IN PARALLEL — each is a few network calls, so the total time is
    about the slowest metric, not the sum (live finding: 4 in a row took ~28 s)."""
    def one(item):
        name, metric, tc = item
        try:
            metric.measure(tc)
            score = float(metric.score or 0.0)
            return _row(name, score, (metric.reason or "")[:240], kind="judge"), \
                float(getattr(metric, "evaluation_cost", 0) or 0)
        except Exception as exc:  # noqa: BLE001 — a judge failure must never break the app
            return {"metric": name, "kind": "judge", "score": None, "passed": False,
                    "reason": f"judge error: {type(exc).__name__}: {str(exc)[:160]}"}, 0.0

    items = judge_metrics(rec)
    if not items:
        return [], 0.0
    with ThreadPoolExecutor(max_workers=len(items), thread_name_prefix="live-judge") as pool:
        results = list(pool.map(one, items))                     # map keeps the order
    return [r for r, _ in results], sum(c for _, c in results)


# ====================================================================== evaluate one request
def evaluate(rec: dict) -> dict:
    t0 = time.perf_counter()
    rows = parameter_scores(rec)
    judge_cost, judge_note = 0.0, ""
    if not judge_enabled():
        judge_note = "LLM judge off (LIVE_EVAL_JUDGE=false)"
    elif not deepeval_available():
        judge_note = "LLM judge needs DeepEval — pip install -r requirements-eval.txt"
    elif not os.getenv("OPENAI_API_KEY"):
        judge_note = "LLM judge needs OPENAI_API_KEY"
    else:
        judged, judge_cost = judge_scores(rec)
        rows += judged
    scored = [r["score"] for r in rows if r["score"] is not None]
    overall = round(sum(scored) / len(scored), 3) if scored else None
    verdict = ("good" if overall is not None and overall >= 0.8 else
               "review" if overall is not None and overall >= 0.6 else "poor")
    return {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "thread_id": rec["thread_id"],
            "run_id": rec["run_id"], "request": rec["request"][:300], "overall": overall,
            "verdict": verdict, "scores": rows, "judge_cost_usd": round(judge_cost, 6),
            "judge_note": judge_note, "eval_seconds": round(time.perf_counter() - t0, 2)}


def _persist(result: dict) -> None:
    """Write the result everywhere it belongs. Never raises."""
    try:
        p = results_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    except OSError:
        pass
    try:
        from audit import audit
        audit("live_eval", thread=result["thread_id"], overall=result["overall"], verdict=result["verdict"],
              low=[r["metric"] for r in result["scores"] if r["score"] is not None and not r["passed"]])
    except Exception:  # noqa: BLE001
        pass
    if result["judge_cost_usd"]:
        try:
            import cost
            cost._DAILY.add(result["judge_cost_usd"])           # judge spend counts toward the daily budget
        except Exception:  # noqa: BLE001
            pass
    send_to_langsmith(result)


def _feedback_target(run_id: str) -> dict:
    """Which trace + project the feedback belongs to. Passing the project (session_id) and
    the trace (trace_id) is the current LangSmith API — and trace_id lets the client upload
    feedback in the background, so the app never waits on it."""
    kw = {"run_id": run_id, "trace_id": run_id}            # our run_id is the root run = the trace
    try:
        from observability import _project_ids, langsmith_project
        kw["session_id"] = _project_ids(langsmith_project())[1]
    except Exception:  # noqa: BLE001 — project not found yet: send without it
        pass
    return kw


def send_to_langsmith(result: dict) -> int:
    """Attach every score to the request's LangSmith trace as feedback. Returns how many."""
    try:
        from observability import langsmith_enabled
        if not (langsmith_enabled() and result.get("run_id")):
            return 0
        from langsmith import Client
        client, sent, target = Client(), 0, _feedback_target(result["run_id"])
        for r in result["scores"]:
            if r["score"] is None:
                continue
            client.create_feedback(**target, key=f"live_eval.{r['metric']}",
                                   score=r["score"], comment=r["reason"][:500])
            sent += 1
        client.create_feedback(**target, key="live_eval.overall", score=result["overall"])
        return sent + 1
    except Exception:  # noqa: BLE001 — tracing is optional; never break the app
        return 0


# ====================================================================== scheduling
_LOCK = threading.Lock()
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="live-eval")
_JOBS: dict[str, Future] = {}


def should_evaluate() -> bool:
    return mode() != "off" and random.random() < sample_rate()


def submit(rec: dict) -> Future:
    """Background mode: evaluate on a worker thread; the UI polls with status()."""
    def job():
        res = evaluate(rec)
        _persist(res)
        return res
    fut = _POOL.submit(job)
    _JOBS[rec["thread_id"]] = fut
    return fut


def run_now(rec: dict) -> dict:
    """Gate / sync mode: evaluate before the answer is shown."""
    res = evaluate(rec)
    _persist(res)
    return res


def status(thread_id: str) -> tuple[str, dict | None]:
    """('pending'|'done'|'none'|'error', result)."""
    fut = _JOBS.get(thread_id)
    if fut is None:
        return "none", None
    if not fut.done():
        return "pending", None
    try:
        return "done", fut.result()
    except Exception as exc:  # noqa: BLE001
        return "error", {"error": f"{type(exc).__name__}: {exc}"}


def gate_warning(result: dict) -> str | None:
    """Gate mode: a visible warning when the answer shouldn't be trusted as-is."""
    weak = [r for r in result["scores"]
            if r["metric"] in ("Faithfulness", "Grounding", "Answer Relevancy")
            and r["score"] is not None and r["score"] < threshold()]
    if not weak:
        return None
    names = ", ".join(f"{r['metric']} {r['score']:.2f}" for r in weak)
    return (f"⚠️ **Low confidence** — the live evaluation scored this answer below "
            f"{threshold():.2f} on {names}. Check it against the sources before relying on it.")


# ====================================================================== the dashboard + the feedback loop
def recent(n: int = 20) -> list[dict]:
    try:
        lines = results_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def rolling(n: int = 20) -> dict:
    """Average per metric over the last n evaluations, plus which are below the threshold."""
    rows = [r for res in recent(n) for r in res.get("scores", []) if r.get("score") is not None]
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r["metric"], []).append(r["score"])
    avgs = {k: round(sum(v) / len(v), 3) for k, v in by.items()}
    overall = [res["overall"] for res in recent(n) if res.get("overall") is not None]
    return {"count": len(recent(n)), "overall": round(sum(overall) / len(overall), 3) if overall else None,
            "metrics": avgs, "alerts": sorted(k for k, v in avgs.items() if v < threshold())}


def record_feedback(thread_id: str, run_id: str | None, rating: int, request: str, answer: str) -> None:
    """👍 (1) / 👎 (0) from the user — saved, audited, and sent to LangSmith."""
    entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "thread_id": thread_id, "rating": rating,
             "request": request[:300], "answer": answer[:1000]}
    try:
        p = Path(os.getenv("LIVE_FEEDBACK_PATH", "logs/user_feedback.jsonl"))
        p.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        from audit import audit
        audit("user_feedback", thread=thread_id, rating=rating)
    except Exception:  # noqa: BLE001
        pass
    try:
        from observability import langsmith_enabled
        if langsmith_enabled() and run_id:
            from langsmith import Client
            Client().create_feedback(**_feedback_target(run_id), key="user_rating", score=rating)
    except Exception:  # noqa: BLE001
        pass


def add_golden_candidate(request: str, answer: str, route: list[str], result: dict | None) -> Path:
    """Turn a live request into an offline test-case candidate (results/golden_candidates.jsonl).
    Review it, then copy it into evals/golden.py with the expected route, tools and facts."""
    p = Path(os.getenv("GOLDEN_CANDIDATES_PATH", "results/golden_candidates.jsonl"))
    p.parent.mkdir(parents=True, exist_ok=True)
    cand = {"id": f"LIVE-{time.strftime('%m%d%H%M%S')}", "category": "from live", "input": request,
            "expected_route": route, "expected_tools": [], "must_include": [],
            "expected_output": "TODO: write the correct answer",
            "observed_answer": answer[:1500],
            "live_overall": (result or {}).get("overall"),
            "low_scores": [r["metric"] for r in (result or {}).get("scores", [])
                           if r.get("score") is not None and not r.get("passed")]}
    with _LOCK, p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(cand, ensure_ascii=False) + "\n")
    return p