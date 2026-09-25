"""llm.py — one place to get the chat model (provider-swappable)."""
from __future__ import annotations

import os
from functools import lru_cache

from langchain.chat_models import init_chat_model

PROVIDERS = {
    "OpenAI · gpt-4o-mini": "openai:gpt-4o-mini",
    "OpenAI · gpt-4o": "openai:gpt-4o",
    "Anthropic · Claude Sonnet": "anthropic:claude-sonnet-5",
}
DEFAULT_PROVIDER = "OpenAI · gpt-4o-mini"


@lru_cache(maxsize=8)
def _build(model_id: str, temperature: float):
    # one client per (model, temperature) — not a new client in every node call
    return init_chat_model(model_id, temperature=temperature)


def get_model(provider_label: str | None = None, temperature: float = 0):
    model_id = PROVIDERS.get(provider_label or DEFAULT_PROVIDER, PROVIDERS[DEFAULT_PROVIDER])
    provider = model_id.split(":", 1)[0]
    key_env = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider)
    if key_env and not os.getenv(key_env):
        raise RuntimeError(f"{key_env} is not set — needed for provider '{provider}'.")
    return _build(model_id, temperature)