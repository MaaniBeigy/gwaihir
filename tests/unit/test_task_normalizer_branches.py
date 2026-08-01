"""Unit tests for `task_normalizer` helpers."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict

import pytest

from src.agents.tools import task_normalizer

# --------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------


@pytest.fixture
def simplified_challenge() -> Dict[str, Any]:
    """A canonical already-simplified GameBus challenge envelope."""
    return {
        "challenge_id": 99,
        "name": "Walk W1",
        "description": "Walk this week",
        "type": "TASKS_COLLECTION",
        "start_date": "2026-04-13",
        "end_date": "2026-04-19",
        "tasks": [
            {
                "task_id": 1001,
                "name": "Take a 30-minute walk",
                "description": "walk",
                "est_duration_min": 30,
                "max_repeats": 1,
                "cooldown_days": 0,
            }
        ],
    }


@pytest.fixture
def fixed_routine_activity() -> Dict[str, Any]:
    """Fixed-mode weekday routine with a concrete preferred start time."""
    return {
        "name": "morning walk",
        "description": "weekday morning walk",
        "IsInstantaneous": False,
        "frequency": {"type": "specific_days", "days": ["mon", "tue", "wed", "thu", "fri"], "count": 1},
        "activity_duration": 30,
        "start_date": "2026-04-13",
        "time": {"preferred": "07:00"},
    }


@pytest.fixture
def saturday_only_activity() -> Dict[str, Any]:
    """Routine that only fires on Saturdays."""
    return {
        "name": "long run",
        "frequency": {"type": "specific_days", "days": ["sat"], "count": 1},
        "activity_duration": 60,
        "start_date": "2026-04-13",
        "time": {"preferred": "07:00"},
    }


# --------------------------------------------------------------------------------------------
# _load_json
# --------------------------------------------------------------------------------------------


def test_load_json_returns_none_for_int_input():
    assert task_normalizer._load_json(42) is None


def test_load_json_returns_none_for_object_input():
    assert task_normalizer._load_json(object()) is None


def test_load_json_returns_none_for_bool_input():
    assert task_normalizer._load_json(True) is None


# --------------------------------------------------------------------------------------------
# _extract_activities
# --------------------------------------------------------------------------------------------


def test_extract_activities_returns_empty_for_int_input():
    assert task_normalizer._extract_activities(42) == []


def test_extract_activities_returns_empty_for_dict_without_activities_key():
    assert task_normalizer._extract_activities({"foo": "bar"}) == []


# --------------------------------------------------------------------------------------------
# _iter_matching_dates
# --------------------------------------------------------------------------------------------


def test_iter_matching_dates_returns_empty_when_end_before_start():
    assert (
        task_normalizer._iter_matching_dates(
            date(2026, 4, 19),
            date(2026, 4, 13),
            ["mon"],
        )
        == []
    )


def test_iter_matching_dates_returns_empty_when_allowed_days_empty():
    assert (
        task_normalizer._iter_matching_dates(
            date(2026, 4, 13),
            date(2026, 4, 19),
            [],
        )
        == []
    )


def test_iter_matching_dates_handles_single_day_window():
    assert task_normalizer._iter_matching_dates(
        date(2026, 4, 13),
        date(2026, 4, 13),
        ["mon"],
    ) == [date(2026, 4, 13)]


# --------------------------------------------------------------------------------------------
# _align_activity_start_date_to_schedule_window
# --------------------------------------------------------------------------------------------


def test_align_activity_keeps_start_when_no_matching_day_in_window(saturday_only_activity):
    """When the allowed weekdays do not appear in the schedule window the original
    start date is preserved."""
    aligned = task_normalizer._align_activity_start_date_to_schedule_window(
        saturday_only_activity,
        schedule_start=date(2026, 4, 13),
        schedule_end=date(2026, 4, 14),
    )
    assert aligned["start_date"] == "2026-04-13"


def test_align_activity_advances_to_first_matching_day(fixed_routine_activity):
    """Activities that start before the schedule window are advanced to the first
    allowed weekday inside it."""
    aligned = task_normalizer._align_activity_start_date_to_schedule_window(
        {**fixed_routine_activity, "start_date": "2025-01-01"},
        schedule_start=date(2026, 4, 14),
        schedule_end=date(2026, 4, 19),
    )
    assert aligned["start_date"] == "2026-04-14"


def test_align_activity_handles_non_dict_frequency():
    """A non-dict `frequency` value is treated as having no day filter."""
    aligned = task_normalizer._align_activity_start_date_to_schedule_window(
        {"start_date": "2026-04-13", "frequency": "not-a-dict"},
        schedule_start=date(2026, 4, 13),
        schedule_end=date(2026, 4, 19),
    )
    assert aligned["start_date"] == "2026-04-13"


# --------------------------------------------------------------------------------------------
# _expand_activity_to_unified_tasks
# --------------------------------------------------------------------------------------------


def test_expand_skips_clamp_when_schedule_start_is_none(fixed_routine_activity):
    """Without a schedule window the expansion still runs; the base task's start
    date is left untouched."""
    tasks = task_normalizer._expand_activity_to_unified_tasks(
        fixed_routine_activity,
        source_uuid="src-1",
        index=1,
        schedule_start=None,
        schedule_end=None,
    )
    assert tasks
    assert any(t.get("start_date") for t in tasks)


def test_expand_keeps_valid_start_when_within_window(fixed_routine_activity):
    """A valid start date inside the schedule window is preserved."""
    tasks = task_normalizer._expand_activity_to_unified_tasks(
        fixed_routine_activity,
        source_uuid="src-2",
        index=1,
        schedule_start=date(2026, 4, 13),
        schedule_end=date(2026, 4, 17),
    )
    earliest = min(t["start_date"] for t in tasks)
    assert earliest == "2026-04-13"


def test_expand_floors_start_when_before_window(fixed_routine_activity):
    """An activity that starts before the schedule window is clamped to its start;
    the expansion never produces dates earlier than that."""
    activity = {**fixed_routine_activity, "start_date": "2025-01-01"}
    tasks = task_normalizer._expand_activity_to_unified_tasks(
        activity,
        source_uuid="src-3",
        index=1,
        schedule_start=date(2026, 4, 13),
        schedule_end=date(2026, 4, 17),
    )
    earliest = min(t["start_date"] for t in tasks)
    assert earliest >= "2026-04-13"


def test_expand_does_not_cap_horizon_when_schedule_end_is_none(fixed_routine_activity):
    """Without a schedule end the expansion horizon comes from
    `SCHEDULING_PERIOD_WEEKS` (default of one week, i.e. five weekdays)."""
    tasks = task_normalizer._expand_activity_to_unified_tasks(
        fixed_routine_activity,
        source_uuid="src-4",
        index=1,
        schedule_start=date(2026, 4, 13),
        schedule_end=None,
    )
    assert len(tasks) == 5


def test_expand_returns_base_task_when_no_dates_match_window(saturday_only_activity):
    """When no allowed weekday falls inside the schedule window the un-expanded
    base task is returned as a single-element list."""
    tasks = task_normalizer._expand_activity_to_unified_tasks(
        saturday_only_activity,
        source_uuid="src-5",
        index=1,
        schedule_start=date(2026, 4, 13),
        schedule_end=date(2026, 4, 15),
    )
    assert len(tasks) == 1
    assert tasks[0]["scheduling_required"] is True


# --------------------------------------------------------------------------------------------
# _coerce_to_simplified_challenges
# --------------------------------------------------------------------------------------------


def test_coerce_simplified_skips_non_dict_items_in_list():
    """Non-dict entries in the input list are skipped and the remaining valid
    challenges are simplified."""
    result = task_normalizer._coerce_to_simplified_challenges(
        [
            None,
            "junk",
            42,
            {
                "challenge_id": 7,
                "name": "Real challenge",
                "type": "TASKS_COLLECTION",
                "tasks": [{"task_id": 1, "name": "task", "description": "do the thing"}],
            },
        ]
    )
    assert len(result) == 1
    assert result[0]["challenge_id"] == 7


def test_coerce_simplified_returns_empty_for_none():
    assert task_normalizer._coerce_to_simplified_challenges(None) == []


def test_coerce_simplified_skips_dict_without_id_or_challenge_id():
    """Dicts without `id` or `challenge_id` are dropped."""
    result = task_normalizer._coerce_to_simplified_challenges([{"name": "no-id"}])
    assert result == []


# --------------------------------------------------------------------------------------------
# _extract_gamebus_with_control_loop
# --------------------------------------------------------------------------------------------


def test_control_loop_unwraps_simplified_gamebus_envelope(simplified_challenge):
    payload = {"simplified_gamebus": {"challenges": [simplified_challenge]}}
    simplified, result = task_normalizer._extract_gamebus_with_control_loop(payload)
    assert result["control"]["is_available"] is True
    assert result["control"]["source_used"] in {"direct", "simplified_gamebus"}
    assert result["control"]["tasks_found"] == 1


def test_control_loop_unwraps_challenges_key(simplified_challenge):
    payload = {"challenges": [simplified_challenge]}
    simplified, result = task_normalizer._extract_gamebus_with_control_loop(payload)
    assert result["control"]["is_available"] is True
    assert result["control"]["tasks_found"] == 1


def test_control_loop_unwraps_data_key(simplified_challenge):
    payload = {"data": {"challenges": [simplified_challenge]}}
    simplified, result = task_normalizer._extract_gamebus_with_control_loop(payload)
    assert result["control"]["is_available"] is True
    assert result["control"]["tasks_found"] == 1


def test_control_loop_reports_unavailable_for_empty_dict():
    simplified, result = task_normalizer._extract_gamebus_with_control_loop({})
    assert result["control"]["is_available"] is False
    assert result["control"]["source_used"] == "none"
    assert result["control"]["tasks_found"] == 0


def test_control_loop_reports_input_present_false_for_none():
    simplified, result = task_normalizer._extract_gamebus_with_control_loop(None)
    assert result["control"]["input_present"] is False


def test_control_loop_handles_non_dict_payload():
    simplified, result = task_normalizer._extract_gamebus_with_control_loop("not-a-dict")
    assert result["control"]["is_available"] is False


# --------------------------------------------------------------------------------------------
# build_unified_scheduler_tasks orchestration
# --------------------------------------------------------------------------------------------


def test_build_unified_scheduler_tasks_combines_routines_and_gamebus(
    simplified_challenge, fixed_routine_activity
):
    result = task_normalizer.build_unified_scheduler_tasks(
        standardized_activities_payload={"Activities": [fixed_routine_activity]},
        gamebus_payload={"challenges": [simplified_challenge]},
        schedule_start=date(2026, 4, 13),
        schedule_end=date(2026, 4, 17),
    )
    assert result["gamebus_tasks"] == 1
    assert result["routine_tasks"] >= 1
    types = {t.get("source_type") for t in result["tasks"]}
    assert types == {"routine", "gamebus"}
