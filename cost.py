"""cost.py — INFRASTRUCTURE guardrail: an LLM token & cost budget.

Every LLM call inside the graph (supervisor, agents, reviewer, finalize — including
structured-output calls) is counted by a LangChain callback attached to the run
config. No agent code needs to know about it.

Limits (all optional, read at call time from .env):
  MAX_TOKENS_PER_REQUEST    default 150000   tokens one request may use
  MAX_COST_PER_REQUEST_USD  default 0.25     dollars one request may spend
  MAX_COST_PER_DAY_USD      default 5.00     dollars this app may spend per day
                                             (persisted in logs/usage.json, so a
                                              restart doesn't reset it)

When a limit is hit, the supervisor stops routing new work and the answer says why.
The call already in flight finishes — limits are checked between steps, which keeps
the graph in a clean, explainable state.

Prices: per 1M tokens, (input, output) in USD. Defaults are list prices at the time
of writing — CHECK your provider's pricing page and override with
PRICE_<MODEL>=input,output, e.g.  PRICE_GPT_4O_MINI=0.15,0.60
A model with no known price is counted in tokens and shown as "price unknown".
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

DEFAULT_PRICES = {            # USD per 1M tokens: (input, output) — verify before relying on it
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def max_tokens_per_request() -> int:
    return int(_env_float("MAX_TOKENS_PER_REQUEST", 150_000))


def max_cost_per_request() -> float:
    return _env_float("MAX_COST_PER_REQUEST_USD", 0.25)


def max_cost_per_day() -> float:
    return _env_float("MAX_COST_PER_DAY_USD", 5.00)


def _env_prices() -> dict[str, tuple[float, float]]:
    """PRICE_GPT_4O_MINI=0.15,0.60 → {'gpt-4o-mini': (0.15, 0.60)}"""
    out = {}
    for k, v in os.environ.items():
        if k.startswith("PRICE_") and "," in v:
            try:
                i, o = (float(x) for x in v.split(","))
            except ValueError:
                continue
            out[k[6:].lower().replace("_", "-")] = (i, o)
    return out


def price_for(model_name: str | None) -> tuple[float, float] | None:
    """(input, output) USD per 1M tokens, or None if unknown. .env overrides the
    defaults; the LONGEST matching prefix wins, so 'gpt-4o-mini-2024-07-18'
    matches 'gpt-4o-mini', not 'gpt-4o'."""
    if not model_name:
        return None
    prices = {**DEFAULT_PRICES, **_env_prices()}
    name = model_name.lower()
    for key in sorted(prices, key=len, reverse=True):
        if name.startswith(key):
            return prices[key]
    return None


# ---------------------------------------------------------------- per-request tracking
@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    unpriced_tokens: int = 0
    by_model: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "total_tokens": self.total_tokens,
                "cost_usd": round(self.cost_usd, 6), "unpriced_tokens": self.unpriced_tokens,
                "by_model": self.by_model}


class UsageTracker(BaseCallbackHandler):
    """Counts tokens and cost for ONE request (all its run segments, incl. resumes)."""

    raise_error = False

    def __init__(self) -> None:
        self.usage = Usage()
        self._lock = threading.Lock()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for gens in getattr(response, "generations", []) or []:
            for g in gens:
                msg = getattr(g, "message", None)
                um = getattr(msg, "usage_metadata", None) or {}
                if not um:
                    continue
                meta = getattr(msg, "response_metadata", None) or {}
                model = meta.get("model_name") or meta.get("model") or \
                    ((getattr(response, "llm_output", None) or {}).get("model_name")) or "unknown"
                self.add(model, int(um.get("input_tokens", 0)), int(um.get("output_tokens", 0)))

    def add(self, model: str, inp: int, out: int) -> None:
        price = price_for(model)
        cost = (inp * price[0] + out * price[1]) / 1_000_000 if price else 0.0
        with self._lock:
            u = self.usage
            u.calls += 1
            u.input_tokens += inp
            u.output_tokens += out
            u.cost_usd += cost
            if not price:
                u.unpriced_tokens += inp + out
            m = u.by_model.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                             "cost_usd": 0.0, "priced": bool(price)})
            m["calls"] += 1
            m["input_tokens"] += inp
            m["output_tokens"] += out
            m["cost_usd"] = round(m["cost_usd"] + cost, 6)
        _DAILY.add(cost)


_TRACKERS: dict[str, UsageTracker] = {}
_TRACKERS_LOCK = threading.Lock()


def start_request(thread_id: str) -> UsageTracker:
    """Create (or reset) the tracker for a request. The app passes it as a callback."""
    with _TRACKERS_LOCK:
        t = _TRACKERS[thread_id] = UsageTracker()
        if len(_TRACKERS) > 200:                       # don't grow forever
            for k in list(_TRACKERS)[:-100]:
                _TRACKERS.pop(k, None)
        return t


def tracker_for(thread_id: str | None) -> UsageTracker | None:
    return _TRACKERS.get(thread_id or "")


# ---------------------------------------------------------------- daily spend (persisted)
class _DailySpend:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def _path(self) -> Path:
        return Path(os.getenv("USAGE_LOG_PATH", "logs/usage.json"))

    def _read(self) -> dict:
        try:
            return json.loads(self._path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def today(self) -> float:
        return float(self._read().get(time.strftime("%Y-%m-%d"), 0.0))

    def add(self, cost: float) -> None:
        if cost <= 0:
            return
        with self._lock:
            data = self._read()
            day = time.strftime("%Y-%m-%d")
            data[day] = round(float(data.get(day, 0.0)) + cost, 6)
            data = dict(sorted(data.items())[-31:])     # keep a month
            try:
                p = self._path()
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(data, indent=1), encoding="utf-8")
            except OSError:
                pass                                     # never break the app over this


_DAILY = _DailySpend()


def spent_today() -> float:
    return _DAILY.today()


# ---------------------------------------------------------------- the guard
def check_cost_budget(thread_id: str | None) -> tuple[bool, str]:
    """Called by the supervisor before routing the next step."""
    daily_limit = max_cost_per_day()
    if spent_today() >= daily_limit:
        return False, (f"daily LLM budget reached: ${spent_today():.4f} of ${daily_limit:.2f} "
                       "(MAX_COST_PER_DAY_USD)")
    t = tracker_for(thread_id)
    if t is None:
        return True, "not tracked"
    u = t.usage
    if u.total_tokens >= max_tokens_per_request():
        return False, (f"this request used {u.total_tokens:,} tokens "
                       f"(limit {max_tokens_per_request():,}, MAX_TOKENS_PER_REQUEST)")
    if u.cost_usd >= max_cost_per_request():
        return False, (f"this request cost ${u.cost_usd:.4f} "
                       f"(limit ${max_cost_per_request():.2f}, MAX_COST_PER_REQUEST_USD)")
    return True, "within budget"
