"""Defensive paths in `StandardizerAgent`: logger init, JSON re-decode failure,
time-range fallback, malformed time parsing, non-positive duration."""

from __future__ import annotations

import json
import logging
import re
from typing import List, Tuple
from unittest.mock import patch

import pytest

from src.agents.standardizer_core import StandardizerAgent


@pytest.fixture(autouse=True)
def _stable_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")


# --------------------------------------------------------------------------------------------
# __init__: `if not self.logger.handlers` False arm
# --------------------------------------------------------------------------------------------


def test_init_skips_basic_config_when_logger_already_has_handlers(monkeypatch):
    """When the `standardizer` logger already has a handler, the `__init__`
    skips the `logging.basicConfig` call."""
    target = logging.getLogger("standardizer")
    sentinel = logging.NullHandler()
    target.addHandler(sentinel)
    try:
        called = {"basic_config": False}

        def _record(**_kwargs):
            called["basic_config"] = True

        monkeypatch.setattr("logging.basicConfig", _record)
        StandardizerAgent()
        assert called["basic_config"] is False
    finally:
        target.removeHandler(sentinel)


# --------------------------------------------------------------------------------------------
# _validate_json_response: inner re-decode of a JSON-encoded string fails
# --------------------------------------------------------------------------------------------


def test_validate_json_response_returns_false_when_inner_string_is_not_json():
    """A response where the outer JSON decodes to a string, but that string
    is itself not valid JSON, returns `(False, None)`."""
    agent = StandardizerAgent()
    # Outer JSON: a quoted string. Inner content is not parseable JSON.
    raw = json.dumps("this is not parseable as inner json")
    is_valid, activities = agent._validate_json_response(raw)
    assert is_valid is False
    assert activities is None


# --------------------------------------------------------------------------------------------
# _parse_time_range: pattern fallback that picks the first two `HH:MM` matches
# --------------------------------------------------------------------------------------------


def test_parse_time_range_falls_back_to_first_two_times_with_keyword():
    """When none of the explicit patterns match but the text has at least two
    `HH:MM` tokens and a `start|begin|finish|end` keyword, the function
    pairs the first two times found."""
    agent = StandardizerAgent()
    # Pattern 1 needs an explicit "to/-/until/till"; pattern 2 needs "from ...";
    # pattern 3 needs literal `start...`. None match, so the fallback fires.
    result = agent._parse_time_range(
        "I begin around 09:00 with a stretch, then 10:00 calls me to do something else."
    )
    assert result == (9 * 60, 10 * 60)


# _parse_time_range with malformed captured groups (int(...) raises) returns None.


def test_parse_time_range_returns_none_when_captured_groups_are_not_integers(monkeypatch):
    """If a regex match returns groups whose digits don't parse as integers,
    the `ValueError` arm returns `None`. The first `re.search` is
    monkeypatched to return such a fake match."""
    agent = StandardizerAgent()

    class _FakeMatch:
        def groups(self) -> Tuple[str, str]:
            return ("ab:cd", "ef:gh")

    real_search = re.search
    calls = {"count": 0}

    def _fake_search(pattern: str, text: str, flags: int = 0):
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeMatch()
        return real_search(pattern, text, flags)

    monkeypatch.setattr("src.agents.standardizer_core.re.search", _fake_search)

    assert agent._parse_time_range("placeholder text 09:00 to 10:00") is None


# --------------------------------------------------------------------------------------------
# _normalize_durations: time range that yields a non-positive duration is skipped
# --------------------------------------------------------------------------------------------


def test_normalize_durations_skips_activity_when_duration_is_non_positive(monkeypatch):
    """When `_parse_time_range` returns a range where end equals start, the
    computed duration is zero and the activity is left unchanged."""
    agent = StandardizerAgent()
    monkeypatch.setattr(agent, "_parse_time_range", lambda text: (600, 600))

    activities: List[dict] = [
        {
            "name": "stretch",
            "description": "some description",
            "activity_duration": 30,
            "time": {"preferred": "10:00"},
            "IsInstantaneous": False,
        }
    ]
    agent._normalize_durations(activities)
    # Original duration preserved; `time.preferred` not overwritten.
    assert activities[0]["activity_duration"] == 30
    assert activities[0]["time"]["preferred"] == "10:00"


def test_normalize_durations_skips_activity_when_duration_is_negative(monkeypatch):
    """Defensive coverage: a negative computed duration also takes the
    `continue` branch."""
    agent = StandardizerAgent()
    monkeypatch.setattr(agent, "_parse_time_range", lambda text: (600, 540))

    activities = [
        {
            "name": "x",
            "description": "anything",
            "activity_duration": 45,
            "time": {},
            "IsInstantaneous": False,
        }
    ]
    agent._normalize_durations(activities)
    assert activities[0]["activity_duration"] == 45


def _resp(payload=None, *, status=200, text=""):
    class _R:
        def __init__(self):
            self.status_code = status
            self.text = text or json.dumps(payload or {})

        def json(self):
            return payload or {}

    return _R()


def test_standardizer_send_retries_on_empty_content_then_raises(monkeypatch):
    responses = [
        _resp({"choices": [{"message": {"content": ""}}]}, text="raw1"),
        _resp({"choices": [{"message": {"content": None}}]}, text="raw2"),
        _resp({"choices": [{"message": {"content": "   "}}]}, text=""),
    ]
    calls = iter(responses)
    monkeypatch.setattr("src.agents.standardizer_core.requests.post", lambda *a, **k: next(calls))
    agent = StandardizerAgent()
    with pytest.raises(RuntimeError, match="returned empty content after 3 attempts"):
        agent._send_to_openrouter([{"role": "user", "content": "x"}])


def test_validate_json_response_handles_string_payload_that_reparses_to_dict():
    agent = StandardizerAgent()
    inner = json.dumps({"Activities": [{"name": "x"}]})
    is_valid, activities = agent._validate_json_response(json.dumps(inner))
    assert is_valid is True
    assert activities == [{"name": "x"}]


def test_validate_json_response_returns_false_when_reparse_fails():
    agent = StandardizerAgent()
    with patch(
        "src.agents.standardizer_core.json_repair.loads",
        side_effect=["nested-string", ValueError("boom")],
    ):
        is_valid, _ = agent._validate_json_response('"nested"')
    assert is_valid is False


def test_validate_json_response_strips_markdown_json_fence():
    agent = StandardizerAgent()
    fenced = '```json\n{"Activities": [{"name": "x"}]}\n```'
    is_valid, activities = agent._validate_json_response(fenced)
    assert is_valid is True
    assert activities == [{"name": "x"}]


def test_validate_json_response_returns_false_when_repair_raises():
    agent = StandardizerAgent()
    with patch("src.agents.standardizer_core.json_repair.loads", side_effect=ValueError("bad")):
        is_valid, activities = agent._validate_json_response('{"Activities": []}')
    assert is_valid is False
    assert activities is None


def test_validate_json_response_strips_plain_fence_without_json_marker():
    agent = StandardizerAgent()
    fenced = '```\n{"Activities": [{"name": "x"}]}\n```'
    is_valid, activities = agent._validate_json_response(fenced)
    assert is_valid is True
    assert activities == [{"name": "x"}]
