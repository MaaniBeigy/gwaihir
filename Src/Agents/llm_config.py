"""Shared LLM endpoint resolver for the coach agents.

When `OPENAI_API_KEY` and `OPENAI_MODEL` are both set, all agents route to OpenAI.
Otherwise each agent falls back to OpenRouter with its per-agent `*_MODEL` env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class LlmConfig:
    api_base: str
    api_key: str
    model: str
    provider: str  # "openai" or "openrouter"

    @property
    def chat_completions_url(self) -> str:
        return f"{self.api_base}/chat/completions"

    def headers(self, title: str = "GameBus Coach") -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            h["HTTP-Referer"] = "https://github.com/habit-agent"
            h["X-Title"] = title
        return h


def resolve_llm_config(
    per_agent_model_env: str,
    *,
    fallback_model: str = "openai/gpt-oss-120b:free",
    explicit_api_key: Optional[str] = None,
    explicit_model: Optional[str] = None,
) -> LlmConfig:
    """Pick the LLM endpoint for this agent.

    Args:
        per_agent_model_env: name of the OpenRouter-side env var, e.g. `STANDARDIZER_MODEL`.
        fallback_model: model id when no per-agent or explicit override is set.
        explicit_api_key: caller override for the API key.
        explicit_model: caller override for the model id.
    """
    env_openai_key = os.getenv("OPENAI_API_KEY")
    env_openai_model = os.getenv("OPENAI_MODEL")
    if env_openai_key and env_openai_model:
        # OpenAI ignores OpenRouter-flavoured model ids, so the env override wins here.
        return LlmConfig(
            api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/"),
            api_key=env_openai_key,
            model=env_openai_model,
            provider="openai",
        )

    api_key = explicit_api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Neither OPENAI_API_KEY+OPENAI_MODEL nor OPENROUTER_API_KEY is set")
    return LlmConfig(
        api_base=os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1").rstrip("/"),
        api_key=api_key,
        model=explicit_model or os.getenv(per_agent_model_env) or fallback_model,
        provider="openrouter",
    )
