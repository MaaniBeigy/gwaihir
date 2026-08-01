"""Branch-coverage gap tests for `src.agents.tools.routine_expander`."""

from datetime import date
from unittest.mock import patch

from src.agents.tools.routine_expander import (
    _activity_target_dates,
    _add_minutes,
    _dates_in_window,
    _entry_name_key,
    _format_hhmm,
    _load_activities,
    _matching_dates,
    _normalize_days,
    _parse_hhmm,
    _parse_iso_date,
    _resolve_time_preference,
    _safe_duration_minutes,
    expand_routines_to_assignments,
    merge_assignments,
)


def test_parse_hhmm_returns_none_for_non_matching_pattern():
    assert _parse_hhmm("not-a-time") is None
    assert _parse_hhmm(None) is None
    assert _parse_hhmm("25:99") == (25, 99)


def test_parse_hhmm_handles_value_error():
    with patch("src.agents.tools.routine_expander._HHMM_PATTERN") as fake_pattern:
        fake_pattern.match.return_value = True
        assert _parse_hhmm("aa:bb") is None


def test_resolve_time_preference_returns_none_for_unknown_value():
    assert _resolve_time_preference("definitely-unknown") == (None, "none")
    assert _resolve_time_preference(123) == (None, "none")


def test_resolve_time_preference_alias_path():
    hhmm, source = _resolve_time_preference("Morning")
    assert hhmm == (9, 0)
    assert source == "alias:morning"


def test_add_minutes_wraps_past_midnight():
    assert _add_minutes((23, 30), 60) == (0, 30)


def test_safe_duration_minutes_handles_none_and_negative():
    assert _safe_duration_minutes({"activity_duration": None}) == 5
    assert _safe_duration_minutes({"activity_duration": "not-int"}) == 5
    assert _safe_duration_minutes({"activity_duration": -10}) == 5
    assert _safe_duration_minutes({"activity_duration": 30, "IsInstantaneous": True}) == 5


def test_parse_iso_date_strips_time_part():
    assert _parse_iso_date("2026-05-16T10:00:00") == date(2026, 5, 16)
    assert _parse_iso_date("") is None
    assert _parse_iso_date("not-a-date") is None
    assert _parse_iso_date(None) is None


def test_dates_in_window_inclusive_range():
    out = _dates_in_window(date(2026, 5, 18), date(2026, 5, 20))
    assert out == [date(2026, 5, 18), date(2026, 5, 19), date(2026, 5, 20)]


def test_matching_dates_empty_when_end_before_start():
    assert _matching_dates(date(2026, 5, 20), date(2026, 5, 18), ["mon"]) == []
    assert _matching_dates(date(2026, 5, 18), date(2026, 5, 20), []) == []


def test_load_activities_handles_invalid_inputs():
    assert _load_activities(None) == []
    assert _load_activities("not-json") == []
    assert _load_activities(123) == []
    # Dict without Activities key.
    assert _load_activities({"foo": "bar"}) == []
    # Bare list of dicts.
    assert _load_activities([{"a": 1}, "skip", {"b": 2}]) == [{"a": 1}, {"b": 2}]


def test_activity_target_dates_end_before_window_returns_empty():
    activity = {
        "frequency": {"type": "daily"},
        "end_date": "2026-05-10",
    }
    assert _activity_target_dates(activity, date(2026, 5, 18), date(2026, 5, 24)) == []


def test_activity_target_dates_clamps_raw_end_to_window_end():
    activity = {
        "frequency": {"type": "daily"},
        "end_date": "2026-12-31",
        "start_date": "2026-05-18",
    }
    out = _activity_target_dates(activity, date(2026, 5, 18), date(2026, 5, 20))
    assert out == [date(2026, 5, 18), date(2026, 5, 19), date(2026, 5, 20)]


def test_activity_target_dates_weekly_uses_start_weekday():
    activity = {
        "frequency": {"type": "weekly"},
        "start_date": "2026-05-19",
    }
    out = _activity_target_dates(activity, date(2026, 5, 18), date(2026, 5, 31))
    assert all(d.weekday() == 1 for d in out)


def test_activity_target_dates_no_freq_and_no_days_acts_daily():
    activity = {"start_date": "2026-05-18"}
    out = _activity_target_dates(activity, date(2026, 5, 18), date(2026, 5, 20))
    assert out == [date(2026, 5, 18), date(2026, 5, 19), date(2026, 5, 20)]


def test_activity_target_dates_unknown_freq_returns_empty():
    activity = {"frequency": {"type": "fortnightly"}, "start_date": "2026-05-18"}
    out = _activity_target_dates(activity, date(2026, 5, 18), date(2026, 5, 20))
    assert out == []


def test_activity_target_dates_raw_end_before_raw_start_returns_empty():
    activity = {
        "frequency": {"type": "daily"},
        "start_date": "2026-05-20",
        "end_date": "2026-05-19",
    }
    assert _activity_target_dates(activity, date(2026, 5, 18), date(2026, 5, 21)) == []


def test_expand_skips_routine_without_recurrence_pattern():
    payload = {
        "Activities": [
            {
                "name": "ad hoc",
                "frequency": {"type": "weekly"},
                "time": {"preferred": None},
                "activity_duration": 30,
                "start_date": "2026-05-18",
            }
        ],
    }
    out = expand_routines_to_assignments(
        payload,
        schedule_start=date(2026, 5, 18),
        schedule_end=date(2026, 5, 24),
    )
    assert out == []


def test_merge_assignments_skips_non_dict_and_missing_keys():
    routine = [
        "not-a-dict",
        {"date": "", "start": "07:00", "name": "skipped"},
        {"date": "2026-05-18", "start": "", "name": "skipped"},
        {"date": "2026-05-18", "start": "07:00", "name": "kept"},
        # Duplicate of the kept entry, deduped.
        {"date": "2026-05-18", "start": "07:00", "name": "kept"},
    ]
    scheduler = [
        "not-a-dict",
        {"date": "", "start": "10:00"},
        {"date": "2026-05-18", "start": ""},
    ]
    merged = merge_assignments(routine, scheduler)
    assert [m["name"] for m in merged] == ["kept"]


def test_merge_assignments_handles_none_lists():
    assert merge_assignments(None, None) == []


def test_entry_name_key_falls_back_to_activity_type():
    assert _entry_name_key({"activityType": "GYM"}) == "gym"


def test_normalize_days_filters_invalid_tokens():
    assert _normalize_days(["mon", "xxx", "Mon", None]) == ["mon"]
    assert _normalize_days(None) == []


def test_format_hhmm_zero_pads():
    assert _format_hhmm((9, 5)) == "09:05"


def test_expand_routines_skips_when_only_alias_resolves_to_none():
    payload = {
        "Activities": [
            {
                "name": "skip",
                "frequency": {"type": "weekly"},
                "activity_duration": 30,
                "time": {"preferred": "totally-unknown"},
                "start_date": "2026-05-18",
            }
        ],
    }
    out = expand_routines_to_assignments(
        payload, schedule_start=date(2026, 5, 18), schedule_end=date(2026, 5, 24)
    )
    assert out == []


def test_merge_assignments_routine_with_empty_name_still_kept():
    routine = [{"date": "2026-05-18", "start": "07:00"}]
    assert len(merge_assignments(routine, [])) == 1


def test_merge_assignments_scheduler_dup_hits_seen_set_check():
    scheduler = [
        {"date": "2026-05-18", "start": "10:00", "name": "deep work"},
        {"date": "2026-05-18", "start": "10:00", "name": "deep work"},
    ]
    assert len(merge_assignments([], scheduler)) == 1


def test_activity_target_dates_clamps_start_date_to_schedule_start():
    activity = {"frequency": {"type": "daily"}, "start_date": "2026-04-01"}
    out = _activity_target_dates(activity, date(2026, 5, 18), date(2026, 5, 20))
    assert out == [date(2026, 5, 18), date(2026, 5, 19), date(2026, 5, 20)]
