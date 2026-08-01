"""Time-range parsing and duration normalisation in `StandardizerAgent`."""

from __future__ import annotations

import json

import pytest

from src.agents.standardizer_core import StandardizerAgent


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")


def test_parse_time_range_extracts_two_explicit_times():
    agent = StandardizerAgent()
    result = agent._parse_time_range("workout from 06:30 to 07:30")
    assert result == (6 * 60 + 30, 7 * 60 + 30)


def test_parse_time_range_supports_dash_separator():
    agent = StandardizerAgent()
    result = agent._parse_time_range("study 14:00 - 16:00")
    assert result == (14 * 60, 16 * 60)


def test_parse_time_range_returns_none_when_no_match():
    agent = StandardizerAgent()
    assert agent._parse_time_range("vague description with no times") is None


def test_parse_time_range_returns_none_for_invalid_clock():
    agent = StandardizerAgent()
    assert agent._parse_time_range("from 25:99 to 30:00") is None


def test_parse_time_range_handles_overnight_times():
    agent = StandardizerAgent()
    # 23:00 to 06:00 wraps past midnight; end is normalised by adding 24h.
    start, end = agent._parse_time_range("sleep from 23:00 to 06:00")
    assert end > start


def test_normalize_durations_keeps_existing_close_duration():
    agent = StandardizerAgent()
    activities = [
        {
            "name": "x",
            "description": "from 09:00 to 09:30",
            "activity_duration": 32,  # within 5 minutes of computed 30, kept
            "time": {"preferred": "09:00"},
            "IsInstantaneous": False,
        }
    ]
    agent._normalize_durations(activities)
    assert activities[0]["activity_duration"] == 32


def test_normalize_durations_skips_instantaneous_activities():
    agent = StandardizerAgent()
    activities = [
        {
            "name": "weigh",
            "description": "weigh from 06:00 to 06:30",
            "IsInstantaneous": True,
            "activity_duration": 0,
            "time": {},
        }
    ]
    agent._normalize_durations(activities)
    assert activities[0]["activity_duration"] == 0


def test_validate_json_response_accepts_double_encoded_json():
    agent = StandardizerAgent()
    # Some models return a JSON string inside a JSON string.
    inner = json.dumps({"Activities": [{"name": "walk"}]})
    outer = json.dumps(inner)
    is_valid, activities = agent._validate_json_response(outer)
    assert is_valid is True
    assert activities == [{"name": "walk"}]


def test_validate_json_response_rejects_non_string_input():
    agent = StandardizerAgent()
    is_valid, activities = agent._validate_json_response(42)
    assert is_valid is False
    assert activities is None


def test_validate_json_response_rejects_non_dict_root():
    agent = StandardizerAgent()
    is_valid, activities = agent._validate_json_response("[1, 2, 3]")
    assert is_valid is False


def test_validate_json_response_rejects_missing_activities_key():
    agent = StandardizerAgent()
    is_valid, activities = agent._validate_json_response('{"Other": []}')
    assert is_valid is False


def test_send_to_openrouter_handles_no_choices(monkeypatch):
    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": []}

    monkeypatch.setattr(
        "src.agents.standardizer_core.requests.post",
        lambda *a, **kw: _Resp(),
    )
    agent = StandardizerAgent()
    with pytest.raises(Exception):
        agent.standardize("test", clear_history=True)
