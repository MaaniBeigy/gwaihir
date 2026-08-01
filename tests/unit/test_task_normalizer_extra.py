"""Helpers, env parsing, and activity alignment in `task_normalizer`."""

from __future__ import annotations

from datetime import date

from src.agents.tools import task_normalizer


def test_get_scheduling_period_days_clamps_to_valid_range(monkeypatch):
    monkeypatch.setenv("SCHEDULING_PERIOD_WEEKS", "0")
    assert task_normalizer._get_scheduling_period_days() == 7
    monkeypatch.setenv("SCHEDULING_PERIOD_WEEKS", "1000")
    assert task_normalizer._get_scheduling_period_days() == 364


def test_get_scheduling_period_days_falls_back_on_garbage(monkeypatch):
    monkeypatch.setenv("SCHEDULING_PERIOD_WEEKS", "garbage")
    assert task_normalizer._get_scheduling_period_days() == 7


def test_load_json_returns_input_when_dict_or_list():
    assert task_normalizer._load_json({"a": 1}) == {"a": 1}
    assert task_normalizer._load_json(["x"]) == ["x"]
    assert task_normalizer._load_json(None) is None
    assert task_normalizer._load_json('{"x":1}') == {"x": 1}


def test_extract_activities_handles_missing_or_invalid_input():
    assert task_normalizer._extract_activities(None) == []
    assert task_normalizer._extract_activities([]) == []
    assert task_normalizer._extract_activities({"Activities": []}) == []
    assert task_normalizer._extract_activities([{"name": "x"}]) == [{"name": "x"}]


def test_normalize_activity_time_preference_handles_aliases():
    assert task_normalizer._normalize_activity_time_preference("morning") == "morning"
    assert task_normalizer._normalize_activity_time_preference("12:30") == "12:30"
    assert task_normalizer._normalize_activity_time_preference("nonsense") is None
    assert task_normalizer._normalize_activity_time_preference(None) is None


def test_normalize_days_returns_canonical_three_letter_codes():
    assert task_normalizer._normalize_days(["Tuesday", "thurs"]) == ["tue", "thu"]
    assert task_normalizer._normalize_days(["unknown", "MON", "mon"]) == ["mon"]


def test_get_routine_schedule_mode_returns_unspecified_when_no_time():
    assert task_normalizer._get_routine_schedule_mode({}) == "unspecified"
    assert task_normalizer._get_routine_schedule_mode({"time": {"preferred": "morning"}}) == "window"
    assert task_normalizer._get_routine_schedule_mode({"time": {"preferred": "07:00"}}) == "fixed"


def test_safe_duration_minutes_handles_invalid_input():
    assert task_normalizer._safe_duration_minutes({"activity_duration": "not-a-number"}) == 5
    assert task_normalizer._safe_duration_minutes({"activity_duration": -10}) == 5
    assert task_normalizer._safe_duration_minutes({"IsInstantaneous": True, "activity_duration": 999}) == 5
    assert task_normalizer._safe_duration_minutes({"activity_duration": 30}) == 30


def test_compute_preferred_end_time_handles_missing_or_invalid_start():
    assert task_normalizer._compute_preferred_end_time(None, 30) is None
    assert task_normalizer._compute_preferred_end_time("not a time", 30) is None
    assert task_normalizer._compute_preferred_end_time("09:00", 60) == "10:00"


def test_parse_iso_date_strips_time_suffix():
    parsed = task_normalizer._parse_iso_date("2026-04-15T10:00:00Z")
    assert parsed == date(2026, 4, 15)


def test_parse_iso_date_returns_none_for_garbage():
    assert task_normalizer._parse_iso_date(None) is None
    assert task_normalizer._parse_iso_date("") is None
    assert task_normalizer._parse_iso_date("not-a-date") is None


def test_get_next_monday_skips_to_next_week_when_already_monday():
    next_monday = task_normalizer._get_next_monday(date(2026, 4, 13))  # Monday
    assert next_monday == date(2026, 4, 20)


def test_align_activity_clamps_past_start_dates_to_window_start():
    aligned = task_normalizer._align_activity_start_date_to_schedule_window(
        {"start_date": "2025-01-01", "frequency": {"days": ["mon"]}},
        schedule_start=date(2026, 4, 13),
        schedule_end=date(2026, 4, 19),
    )
    # Mon Apr 13 is the window start and a Monday, so it stays.
    assert aligned["start_date"] == "2026-04-13"
