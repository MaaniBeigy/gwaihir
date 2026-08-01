"""Endpoint resolution and header behavior in `src.agents.llm_config`."""

from __future__ import annotations

import pytest

from src.agents.llm_config import LlmConfig, resolve_llm_config


def test_resolve_llm_config_raises_when_no_keys_set(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Neither OPENAI"):
        resolve_llm_config("STANDARDIZER_MODEL")


def test_resolve_llm_config_uses_openai_when_env_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    cfg = resolve_llm_config("STANDARDIZER_MODEL")
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o-mini"
    assert cfg.api_key == "sk-openai"


def test_openrouter_headers_have_attribution():
    cfg = LlmConfig(api_base="https://openrouter.ai/api/v1", api_key="k", model="m", provider="openrouter")
    headers = cfg.headers(title="Test")
    assert headers["HTTP-Referer"].startswith("https://")
    assert headers["X-Title"] == "Test"


def test_openai_headers_omit_attribution():
    cfg = LlmConfig(api_base="https://api.openai.com/v1", api_key="k", model="m", provider="openai")
    headers = cfg.headers()
    assert "HTTP-Referer" not in headers
    assert "X-Title" not in headers


def test_chat_completions_url_appends_path():
    cfg = LlmConfig(api_base="https://example.test/v1", api_key="k", model="m", provider="openai")
    assert cfg.chat_completions_url == "https://example.test/v1/chat/completions"
