"""Offline tests for the INFRASTRUCTURE guardrail: LLM token & cost budget (cost.py)."""
from __future__ import annotations

import types

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult

import cost


class UsageFake(BaseChatModel):
    """A chat model that reports token usage like OpenAI does."""
    content: str = '{"next": "done", "why": "x"}'

    @property
    def _llm_type(self):
        return "usage-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        msg = AIMessage(content=self.content,
                        usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200},
                        response_metadata={"model_name": "gpt-4o-mini-2024-07-18"})
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kw):
        return self


# ---------------------------------------------------------------- prices
def test_longest_prefix_wins():
    assert cost.price_for("gpt-4o-mini-2024-07-18") == (0.15, 0.60)
    assert cost.price_for("gpt-4o-2024-08-06") == (2.50, 10.00)


def test_unknown_model_has_no_price():
    assert cost.price_for("claude-sonnet-5") is None and cost.price_for(None) is None


def test_env_overrides_and_adds_prices(monkeypatch):
    monkeypatch.setenv("PRICE_GPT_4O_MINI", "0.2,0.8")
    monkeypatch.setenv("PRICE_CLAUDE_SONNET_5", "3,15")
    assert cost.price_for("gpt-4o-mini") == (0.2, 0.8)
    assert cost.price_for("claude-sonnet-5-20260101") == (3.0, 15.0)


# ---------------------------------------------------------------- counting
def test_cost_is_computed_from_tokens():
    t = cost.UsageTracker()
    t.add("gpt-4o-mini", 1_000_000, 1_000_000)
    assert t.usage.cost_usd == pytest.approx(0.75)        # 0.15 + 0.60
    assert t.usage.total_tokens == 2_000_000 and t.usage.calls == 1


def test_unpriced_model_counts_tokens_not_dollars():
    t = cost.UsageTracker()
    t.add("some-new-model", 500, 100)
    assert t.usage.cost_usd == 0 and t.usage.unpriced_tokens == 600
    assert t.usage.by_model["some-new-model"]["priced"] is False


def test_callback_reads_usage_from_llm_result():
    t = cost.UsageTracker()
    msg = AIMessage(content="x", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    response_metadata={"model_name": "gpt-4o-mini"})
    t.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))
    assert t.usage.input_tokens == 10 and t.usage.output_tokens == 5


def test_callback_counts_every_call_inside_the_graph(monkeypatch):
    """The tracker is attached to the RUN config only — it still sees the supervisor's
    structured-output call and finalize, with no agent code changes."""
    import agents.supervisor as sup, graph as gm, quality
    fake = UsageFake()
    for mod in (sup, quality, gm):
        monkeypatch.setattr(mod, "get_model", lambda *a, **k: fake)
    t = cost.start_request("cost-thread")
    g = gm.build_graph([], [], [])
    # an off-topic request: the supervisor decides "done" → only real LLM calls are counted
    for _ in g.stream({"request": "Book me a flight to Goa next Friday.", "thread_id": "cost-thread",
                       "trail": [], "timings": [], "tool_log": [], "output_flags": []},
                      {"configurable": {"thread_id": "c1"}, "callbacks": [t]}, stream_mode="updates"):
        pass
    assert t.usage.calls >= 1 and t.usage.total_tokens >= 1200
    assert t.usage.cost_usd > 0


# ---------------------------------------------------------------- the guard
def test_untracked_request_passes():
    assert cost.check_cost_budget("nobody")[0]


def test_token_limit(monkeypatch):
    monkeypatch.setenv("MAX_TOKENS_PER_REQUEST", "1000")
    t = cost.start_request("tok")
    t.add("gpt-4o-mini", 900, 200)
    ok, msg = cost.check_cost_budget("tok")
    assert not ok and "1,100 tokens" in msg and "MAX_TOKENS_PER_REQUEST" in msg


def test_cost_limit(monkeypatch):
    monkeypatch.setenv("MAX_COST_PER_REQUEST_USD", "0.01")
    t = cost.start_request("usd")
    t.add("gpt-4o", 10_000, 0)                            # 10k × $2.50/1M = $0.025
    ok, msg = cost.check_cost_budget("usd")
    assert not ok and "MAX_COST_PER_REQUEST_USD" in msg


def test_daily_limit_survives_a_restart(monkeypatch):
    monkeypatch.setenv("MAX_COST_PER_DAY_USD", "0.05")
    cost.UsageTracker().add("gpt-4o", 30_000, 0)          # $0.075, written to usage.json
    fresh = cost._DailySpend()                            # like a new app process
    assert fresh.today() == pytest.approx(0.075)
    ok, msg = cost.check_cost_budget("anything")
    assert not ok and "daily LLM budget" in msg


def test_supervisor_stops_and_the_answer_says_why(monkeypatch):
    import agents.supervisor as sup, graph as gm, quality
    fake = types.SimpleNamespace(invoke=lambda m: AIMessage(content="x"))
    for mod in (sup, quality, gm):
        monkeypatch.setattr(mod, "get_model", lambda *a, **k: fake)
    monkeypatch.setattr(sup, "_decide", lambda *_: ("research", "s"))
    monkeypatch.setenv("MAX_TOKENS_PER_REQUEST", "100")
    cost.start_request("over").add("gpt-4o-mini", 500, 0)
    g = gm.build_graph([], [], [])
    state = g.invoke({"request": "What known bugs affect login?", "thread_id": "over", "trail": [],
                      "timings": [], "tool_log": [], "output_flags": []},
                     {"configurable": {"thread_id": "o2"}})
    assert "Stopped before any work was done" in state["final"]
    assert any(t.startswith("💰 cost budget: stopping") for t in state["trail"])


def test_partial_work_shows_stopped_early(monkeypatch):
    from graph import _action_status
    assert _action_status({"budget_stop": "this request used 160,000 tokens"}) == [
        "💰 Stopped early: this request used 160,000 tokens"]