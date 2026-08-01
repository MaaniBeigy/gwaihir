"""Defensive paths in `ControllerAgent`: instructions loading, JSON extraction, LLM retries."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
import requests

from src.agents.controller_core import ControllerAgent, _extract_assignments


@pytest.fixture(autouse=True)
def _stable_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setattr("src.agents.controller_core.time.sleep", lambda _s: None)


def test_extract_assignments_returns_empty_when_nested_assignments_is_not_a_list():
    """Envelope payload with a non-list `schedule_json.assignments` returns `[]`."""
    payload = {"schedule_json": {"assignments": "not a list"}}
    assert _extract_assignments(payload) == []


def test_extract_assignments_returns_empty_when_envelope_lacks_nested_assignments():
    """Envelope payload whose nested object lacks `assignments` returns `[]`."""
    payload = {"schedule_json": {"other": "value"}}
    assert _extract_assignments(payload) == []


def test_extract_assignments_returns_empty_when_top_assignments_is_none():
    """No `assignments` and no `schedule_json` returns `[]`."""
    assert _extract_assignments({"foo": "bar"}) == []


def test_load_instructions_raises_when_template_file_missing(monkeypatch):
    """Missing controller-instructions template raises `RuntimeError`."""
    monkeypatch.setattr("src.agents.controller_core.os.path.exists", lambda path: False)
    with pytest.raises(RuntimeError, match="Controller instructions not found"):
        ControllerAgent(enable_llm=False)


def test_load_instructions_raises_when_template_file_is_empty(monkeypatch):
    """Empty template file raises `RuntimeError`."""
    monkeypatch.setattr("src.agents.controller_core.os.path.exists", lambda path: True)

    real_open = open

    def _fake_open(path, *args, **kwargs):
        if str(path).endswith("Controller_instructions.txt"):
            from io import StringIO

            return StringIO("")
        return real_open(path, *args, **kwargs)

    with patch("builtins.open", _fake_open):
        with pytest.raises(RuntimeError, match="Controller instructions file is empty"):
            ControllerAgent(enable_llm=False)


def test_parse_json_like_returns_none_for_python_only_scalar():
    """`True` parses via `ast.literal_eval` but is not a dict or list, so returns `None`."""
    agent = ControllerAgent(enable_llm=False)
    assert agent._parse_json_like("True") is None


def test_parse_json_like_returns_none_for_python_none_literal():
    """`None` parses via `ast.literal_eval` but is not a container, so returns `None`."""
    agent = ControllerAgent(enable_llm=False)
    assert agent._parse_json_like("None") is None


def test_parse_json_like_returns_none_for_tuple_literal():
    """Tuples are valid Python literals but not JSON-shaped containers; rejected."""
    agent = ControllerAgent(enable_llm=False)
    assert agent._parse_json_like("(1, 2, 3)") is None


def test_extract_json_candidate_skips_fenced_block_that_is_a_list():
    """A fenced list block is skipped; the trailing brace scan returns the dict."""
    agent = ControllerAgent(enable_llm=False)
    text = 'preamble\n``json\n[1, 2, 3]\n``\ntrailing {"is_valid": true}'
    parsed = agent._extract_json_candidate(text)
    assert parsed == {"is_valid": True}


def test_extract_json_candidate_returns_none_when_brace_scan_yields_non_dict():
    """Brace substring parses to a set (not a dict), so extraction returns `None`."""
    agent = ControllerAgent(enable_llm=False)
    assert agent._extract_json_candidate("noise {1, 2, 3} trailing") is None


def test_extract_json_candidate_returns_none_when_text_has_no_braces_or_brackets():
    """No structural characters; every parsing strategy fails."""
    agent = ControllerAgent(enable_llm=False)
    assert agent._extract_json_candidate("nothing here") is None


class _Resp:
    def __init__(self, payload: Any, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)

    def json(self) -> Any:
        return self._payload


def test_call_llm_raises_after_three_consecutive_http_errors(monkeypatch):
    """All three attempts return non-200; the final aggregate `RuntimeError` is raised."""
    calls = []

    def _post(*args, **kwargs):
        calls.append(1)
        return _Resp({"error": "boom"}, status_code=500)

    monkeypatch.setattr("src.agents.controller_core.requests.post", _post)
    agent = ControllerAgent(enable_llm=True)
    with pytest.raises(RuntimeError, match="Controller LLM call failed after retries"):
        agent._call_llm("prompt")
    assert len(calls) == 3


def test_call_llm_raises_after_three_consecutive_request_exceptions(monkeypatch):
    """Network errors surface as the aggregate retry failure after three tries."""
    calls = []

    def _post(*args, **kwargs):
        calls.append(1)
        raise requests.exceptions.ConnectionError("network down")

    monkeypatch.setattr("src.agents.controller_core.requests.post", _post)
    agent = ControllerAgent(enable_llm=True)
    with pytest.raises(RuntimeError, match="Controller LLM call failed after retries"):
        agent._call_llm("prompt")
    assert len(calls) == 3


def test_call_llm_retries_when_response_has_no_choices(monkeypatch):
    """A 200 response without `choices` raises and triggers a retry."""
    calls = []

    def _post(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            return _Resp({"choices": None})
        return _Resp({"choices": [{"message": {"content": '{"is_valid": true}'}}]})

    monkeypatch.setattr("src.agents.controller_core.requests.post", _post)
    agent = ControllerAgent(enable_llm=True)
    parsed = agent._call_llm("prompt")
    assert parsed == {"is_valid": True}
    assert len(calls) == 3


def test_call_llm_retries_when_content_is_empty(monkeypatch):
    """An empty `content` field raises and triggers a retry."""
    calls = []

    def _post(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            return _Resp({"choices": [{"message": {"content": "   "}}]})
        return _Resp({"choices": [{"message": {"content": '{"is_valid": true}'}}]})

    monkeypatch.setattr("src.agents.controller_core.requests.post", _post)
    agent = ControllerAgent(enable_llm=True)
    parsed = agent._call_llm("prompt")
    assert parsed == {"is_valid": True}
    assert len(calls) == 3


def test_call_llm_retries_when_message_is_not_a_dict(monkeypatch):
    """`choices[0].message` being a non-dict triggers the empty-content guard
    and forces a retry."""
    calls = []

    def _post(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            return _Resp({"choices": [{"message": "not-a-dict"}]})
        return _Resp({"choices": [{"message": {"content": '{"is_valid": true}'}}]})

    monkeypatch.setattr("src.agents.controller_core.requests.post", _post)
    agent = ControllerAgent(enable_llm=True)
    parsed = agent._call_llm("prompt")
    assert parsed == {"is_valid": True}


def test_call_llm_retries_when_choices_first_item_is_not_a_dict(monkeypatch):
    """`choices[0]` being a string is also treated as missing content."""
    calls = []

    def _post(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            return _Resp({"choices": ["junk"]})
        return _Resp({"choices": [{"message": {"content": '{"is_valid": true}'}}]})

    monkeypatch.setattr("src.agents.controller_core.requests.post", _post)
    agent = ControllerAgent(enable_llm=True)
    parsed = agent._call_llm("prompt")
    assert parsed == {"is_valid": True}


def test_controller_agent_without_keys_falls_back(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = ControllerAgent()
    assert agent.api_key is None
    assert agent.model
