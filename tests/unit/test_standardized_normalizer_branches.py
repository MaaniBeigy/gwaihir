"""Branch-coverage gap tests for `src.agents.tools.standardized_normalizer`."""

import json

from src.agents.tools.standardized_normalizer import (
    _explicit_days_in_text,
    _normalize_one,
    _set_frequency,
    normalize_standardized_activities,
)


def test_explicit_days_handles_empty_and_none():
    assert _explicit_days_in_text("") == []
    assert _explicit_days_in_text(None) == []


def test_explicit_days_dedupes_repeated_mentions():
    # The same weekday mentioned twice exercises the dedup branch.
    assert _explicit_days_in_text("Monday and Mon and Tuesday") == ["mon", "tue"]


def test_normalize_one_returns_for_non_dict():
    # Must not raise.
    _normalize_one("not-a-dict")


def test_set_frequency_keeps_existing_nonzero_count():
    activity = {
        "name": "x",
        "frequency": {"type": "daily", "days": None, "count": 5},
    }
    _set_frequency(activity, days=["mon"], reason="test")
    assert activity["frequency"]["count"] == 5


def test_normalize_returns_payload_for_unparseable_string():
    # The string is not JSON, so the function logs a warning and returns input unchanged.
    out = normalize_standardized_activities("this is not json")
    assert out == "this is not json"


def test_normalize_returns_payload_when_shape_unknown():
    # A bare scalar JSON value parses but is not a dict-with-Activities or list of dicts.
    out = normalize_standardized_activities(json.dumps(42))
    assert out == json.dumps(42)


def test_normalize_handles_bare_list_input():
    payload = [
        {
            "name": "work",
            "description": "On workdays I work from 9 to 5.",
            "frequency": {"type": "daily", "days": None, "count": 1},
        }
    ]
    out = normalize_standardized_activities(payload)
    assert out[0]["frequency"]["days"] == ["mon", "tue", "wed", "thu", "fri"]


def test_normalize_drops_non_string_days_before_check():
    payload = {
        "Activities": [
            {
                "name": "n",
                "description": "I work on workdays",
                "frequency": {"type": "daily", "days": [None, 42, "junk"], "count": 1},
            }
        ],
    }
    out = normalize_standardized_activities(payload)
    assert out["Activities"][0]["frequency"]["days"] == [
        "mon",
        "tue",
        "wed",
        "thu",
        "fri",
    ]


def test_normalize_returns_input_unchanged_for_non_collection():
    assert normalize_standardized_activities(42) == 42


def test_normalize_handles_dict_without_activities():
    out = normalize_standardized_activities({"foo": "bar"})
    assert out == {"foo": "bar"}


def test_normalize_skips_activity_when_frequency_not_dict():
    payload = {
        "Activities": [
            {
                "name": "n",
                "description": "I work on workdays",
                "frequency": "string-not-dict",
            }
        ],
    }
    out = normalize_standardized_activities(payload)
    # Has no `days` change because freq is not a dict to begin with;
    # but the helper still proceeds and writes new dict.
    assert out["Activities"][0]["frequency"]["days"] == [
        "mon",
        "tue",
        "wed",
        "thu",
        "fri",
    ]
