"""Unit tests for the StandardizerAgent core logic with a mocked LLM."""

from __future__ import annotations

import json
from typing import List

import pytest

from src.agents.standardizer_core import StandardizerAgent


@pytest.fixture(autouse=True)
def _quiet_logger(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _activities_payload(activities):
    return {"choices": [{"message": {"content": json.dumps({"Activities": activities})}}]}


def test_standardize_returns_clean_json_when_response_is_valid(monkeypatch):
    activities = [
        {
            "name": "go to the gym",
            "description": "go to the gym every Monday at 19:00 for 1 hour",
            "IsInstantaneous": False,
            "IsRegular": True,
            "frequency": {"type": "specific_days", "days": ["mon"], "count": 1},
            "IsRepetitive": True,
            "activity_duration": 60,
            "start_date": "2026-04-13",
            "IsArbitrary": False,
            "time": {"preferred": "19:00"},
            "notes": None,
        }
    ]

    captured: List = []

    def _post(url, headers=None, json=None, timeout=None):
        captured.append({"url": url, "json": json})
        return _FakeResponse(_activities_payload(activities))

    monkeypatch.setattr("src.agents.standardizer_core.requests.post", _post)

    agent = StandardizerAgent()
    raw = agent.standardize("go to the gym every monday at 7pm for 1 hour", clear_history=True)
    parsed = json.loads(raw)
    assert parsed["Activities"][0]["name"] == "go to the gym"
    assert captured  # the LLM was called


def test_standardize_normalizes_duration_from_explicit_time_range(monkeypatch):
    activities = [
        {
            "name": "study",
            "description": "study from 10:00 to 12:00",
            "IsInstantaneous": False,
            "IsRegular": True,
            "frequency": {"type": "daily"},
            "IsRepetitive": True,
            "activity_duration": 30,
            "start_date": "2026-04-13",
            "IsArbitrary": False,
            "time": {},
            "notes": None,
        }
    ]
    monkeypatch.setattr(
        "src.agents.standardizer_core.requests.post",
        lambda *args, **kwargs: _FakeResponse(_activities_payload(activities)),
    )

    agent = StandardizerAgent()
    raw = agent.standardize("study 10:00 to 12:00 every day", clear_history=True)
    parsed = json.loads(raw)
    assert parsed["Activities"][0]["activity_duration"] == 120
    assert parsed["Activities"][0]["time"]["preferred"] == "10:00"


def test_standardize_returns_raw_response_when_invalid_json(monkeypatch):
    monkeypatch.setattr(
        "src.agents.standardizer_core.requests.post",
        lambda *args, **kwargs: _FakeResponse({"choices": [{"message": {"content": "this is not json"}}]}),
    )

    agent = StandardizerAgent()
    result = agent.standardize("test", clear_history=True)
    assert result == "this is not json"


def test_standardize_strips_markdown_fence(monkeypatch):
    fenced = "``json\n" + json.dumps({"Activities": []}) + "\n``"
    monkeypatch.setattr(
        "src.agents.standardizer_core.requests.post",
        lambda *args, **kwargs: _FakeResponse({"choices": [{"message": {"content": fenced}}]}),
    )

    agent = StandardizerAgent()
    result = agent.standardize("test", clear_history=True)
    parsed = json.loads(result)
    assert parsed["Activities"] == []


def test_standardize_raises_on_http_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(
        "src.agents.standardizer_core.requests.post",
        lambda *args, **kwargs: _FakeResponse({"error": "nope"}, status_code=500),
    )

    agent = StandardizerAgent()
    with pytest.raises(RuntimeError, match="openrouter API error 500"):
        agent.standardize("test", clear_history=True)


def test_reset_clears_conversation_history(monkeypatch):
    monkeypatch.setattr(
        "src.agents.standardizer_core.requests.post",
        lambda *args, **kwargs: _FakeResponse(_activities_payload([])),
    )
    agent = StandardizerAgent()
    agent.standardize("test", clear_history=True)
    assert agent.get_conversation_history()
    agent.reset()
    assert agent.get_conversation_history() == []


def test_get_conversation_history_returns_copy(monkeypatch):
    monkeypatch.setattr(
        "src.agents.standardizer_core.requests.post",
        lambda *args, **kwargs: _FakeResponse(_activities_payload([])),
    )
    agent = StandardizerAgent()
    agent.standardize("test", clear_history=True)
    history = agent.get_conversation_history()
    history.append({"role": "user", "content": "tampered"})
    assert {"role": "user", "content": "tampered"} not in agent.get_conversation_history()
