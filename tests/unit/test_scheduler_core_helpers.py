"""Private helpers of `src.agents.scheduler_core`."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from src.agents import scheduler_core


def test_summarize_assignments_returns_safe_default_for_non_dict():
    assert scheduler_core._summarize_assignments_for_logs({"assignments": "nope"}) == {
        "count": 0,
        "preview": [],
    }


def test_summarize_assignments_truncates_preview():
    schedule = {
        "assignments": [
            {"task_id": f"t{i}", "type": "gamebus", "date": "2026-04-15", "start": "09:00", "end": "10:00"}
            for i in range(20)
        ]
    }
    summary = scheduler_core._summarize_assignments_for_logs(schedule, max_items=3)
    assert summary["count"] == 20
    assert len(summary["preview"]) == 3


def test_compact_controller_feedback_truncates_long_strings_and_caps_count():
    feedback = ["x" * 300, "ok", "", None, "y" * 50, "z" * 50, "a"] + [str(i) for i in range(20)]
    compact = scheduler_core._compact_controller_feedback(feedback, max_items=4)
    assert len(compact) == 4
    assert compact[0].endswith("…") or len(compact[0]) <= 220


def test_compact_controller_feedback_returns_empty_for_non_list():
    assert scheduler_core._compact_controller_feedback("not-a-list") == []


def test_slim_task_sources_replaces_full_gamebus_list():
    payload = {
        "all_tasks": ["A", "B", "C"],
        "gamebus_tasks": ["heavy"],
        "unscheduled_gamebus": ["light"],
        "routine_tasks": ["r1"],
    }
    slim = scheduler_core._slim_task_sources_for_llm(payload)
    assert slim["gamebus_tasks"] == ["light"]
    assert slim["all_tasks"] == ["r1", "light"]
    assert "heavy" not in slim["all_tasks"]


def test_normalize_busy_intervals_filters_invalid_entries():
    busy = scheduler_core._normalize_busy_intervals(
        [
            {"startDate": "2026-04-15T08:00:00Z", "endDate": "2026-04-15T09:00:00Z"},
            {
                "start": "2026-04-15T09:00:00Z",
                "end": "2026-04-15T09:30:00Z",
                "source": "user",
                "activityType": "MEETING",
            },
            {"startDate": "2026-04-15T10:00:00Z"},  # missing end; dropped
            "not-a-dict",
        ]
    )
    assert len(busy) == 2
    assert busy[0]["source"] == "USER"
    assert busy[1]["activityType"] == "MEETING"


def test_normalize_busy_intervals_returns_empty_for_non_list():
    assert scheduler_core._normalize_busy_intervals(None) == []
    assert scheduler_core._normalize_busy_intervals({"not": "a list"}) == []


def test_build_schedule_window_handles_two_weeks():
    window = scheduler_core._build_schedule_window("2026-04-13", 2)
    assert window == {"start_date": "2026-04-13", "end_date": "2026-04-26"}


def test_get_next_monday_returns_following_monday():
    from datetime import date

    assert scheduler_core._get_next_monday(date(2026, 4, 13)).isoformat() == "2026-04-20"
    assert scheduler_core._get_next_monday(date(2026, 4, 17)).isoformat() == "2026-04-20"


def test_weekday_short_returns_three_letter_code():
    assert scheduler_core._weekday_short("2026-04-13") == "mon"
    assert scheduler_core._weekday_short("not-a-date") is None


def test_normalize_days_dedupes_and_maps_aliases():
    assert scheduler_core._normalize_days(["Monday", "mon", "FRIDAY"]) == ["mon", "fri"]
    assert scheduler_core._normalize_days(None) == []


def test_normalize_llm_provider_supports_both_known_providers():
    assert scheduler_core._normalize_llm_provider("openrouter") == "openrouter"
    assert scheduler_core._normalize_llm_provider("openai") == "openai"


def test_normalize_llm_provider_rejects_unknown_provider():
    with pytest.raises(RuntimeError):
        scheduler_core._normalize_llm_provider("totally-not-a-real-provider")


def test_extract_json_candidate_parses_fenced_code():
    candidate = scheduler_core._extract_json_candidate(
        "preamble\n``json\n" + json.dumps({"hello": "world"}) + "\n``\nepilogue"
    )
    assert candidate == {"hello": "world"}


def test_extract_json_candidate_falls_back_to_brace_scan():
    candidate = scheduler_core._extract_json_candidate('noise{"x":1}trailing')
    assert candidate == {"x": 1}


def test_extract_json_candidate_returns_none_for_empty_input():
    assert scheduler_core._extract_json_candidate("   ") is None


def test_normalize_llm_schedule_promotes_assignments_list():
    parsed = scheduler_core._normalize_llm_schedule(
        [{"date": "2026-04-15", "start": "09:00", "task_id": "g1"}]
    )
    assert parsed == {"assignments": [{"date": "2026-04-15", "start": "09:00", "task_id": "g1"}]}


def test_normalize_llm_schedule_unwraps_envelope():
    payload = {"schedule": {"assignments": [{"date": "2026-04-15", "start": "09:00", "task_id": "g1"}]}}
    parsed = scheduler_core._normalize_llm_schedule(payload)
    assert isinstance(parsed, dict)
    assert parsed["assignments"][0]["task_id"] == "g1"


def test_normalize_task_sources_payload_promotes_tasks_alias():
    payload = {"tasks": [{"task_id": "a"}]}
    normalized = scheduler_core._normalize_task_sources_payload(payload)
    assert normalized["all_tasks"][0]["task_id"] == "a"


def test_normalize_task_sources_payload_uses_nested_when_outer_missing():
    payload = {"task_sources_json": {"all_tasks": [{"task_id": "x"}]}}
    normalized = scheduler_core._normalize_task_sources_payload(payload)
    assert normalized["all_tasks"][0]["task_id"] == "x"


def test_normalize_task_sources_payload_returns_input_when_not_dict():
    assert scheduler_core._normalize_task_sources_payload("string") == "string"


def test_enrich_assignments_attaches_routine_metadata_from_sources():
    schedule = {
        "assignments": [
            {"task_id": "r1", "date": "2026-04-13", "start": "07:00", "end": "07:30"},
        ]
    }
    routine_task = {
        "task_id": "r1",
        "source_type": "routine",
        "name": "Morning walk",
        "description": "Walk every morning",
        "frequency": {"type": "specific_days", "days": ["mon"], "count": 1},
        "scheduling_required": True,
    }
    enriched = scheduler_core._enrich_assignments_with_task_metadata(schedule, [routine_task])
    assignment = enriched["assignments"][0]
    assert assignment["type"] == "routine"
    assert assignment["name"] == "Morning walk"
    assert assignment["frequency"]["days"] == ["mon"]


def test_call_llm_retries_then_returns_content(monkeypatch):
    calls: List[Any] = []

    def _post(url, headers=None, json=None, timeout=None):
        calls.append(url)
        if len(calls) < 3:

            class _ErrResponse:
                status_code = 500
                text = "boom"

                def json(self):
                    return {}

            return _ErrResponse()

        class _OkResponse:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "  ok  "}}]}

        return _OkResponse()

    monkeypatch.setattr("src.agents.scheduler_core.requests.post", _post)
    monkeypatch.setattr("src.agents.scheduler_core.time.sleep", lambda _s: None)

    result = scheduler_core._call_llm("prompt", "model", "key", "openrouter")
    assert result == "ok"
    assert len(calls) == 3


def test_call_llm_raises_after_retries_exhaust(monkeypatch):
    class _ErrResponse:
        status_code = 500
        text = "boom"

        def json(self):
            return {}

    monkeypatch.setattr("src.agents.scheduler_core.requests.post", lambda *args, **kwargs: _ErrResponse())
    monkeypatch.setattr("src.agents.scheduler_core.time.sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="LLM call failed"):
        scheduler_core._call_llm("prompt", "model", "key", "openrouter")


def test_call_llm_requires_api_key():
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        scheduler_core._call_llm("prompt", "model", "", "openrouter")


def test_schedule_from_inputs_runs_pipeline(monkeypatch):
    monkeypatch.setattr(
        "src.agents.scheduler_core._call_llm",
        lambda *args, **kwargs: json.dumps(
            {
                "assignments": [
                    {
                        "task_id": "g1",
                        "type": "gamebus",
                        "date": "2026-04-15",
                        "start": "09:00",
                        "end": "10:00",
                    }
                ]
            }
        ),
    )

    task_sources = {
        "all_tasks": [{"task_id": "g1", "source_type": "gamebus"}],
        "gamebus_tasks": [{"task_id": "g1", "source_type": "gamebus"}],
        "routine_tasks": [],
    }

    result = scheduler_core.schedule_from_inputs(
        scheduler_task_sources_input=task_sources,
        controller_feedback=None,
        scheduling_window={
            "scheduleStartDate": "2026-04-13",
            "scheduleEndDate": "2026-04-19",
            "schedulingPeriodWeeks": 1,
            "currentDate": "2026-04-13",
            "currentDayname": "mon",
        },
        busy_intervals=[
            {"startDate": "2026-04-15T08:00:00Z", "endDate": "2026-04-15T09:00:00Z"},
        ],
        llm_api_key="sk-or-v1-test",
    )

    assert result["scheduled_assignments"] == 1
    assert result["assignments"][0]["task_id"] == "g1"
    assert result["schedule_start_date"] == "2026-04-13"
    assert result["schedule_end_date"] == "2026-04-19"


def test_schedule_from_inputs_rejects_window_without_dates():
    with pytest.raises(RuntimeError, match="scheduling_window"):
        scheduler_core.schedule_from_inputs(
            scheduler_task_sources_input={"all_tasks": [{"task_id": "g1"}]},
            scheduling_window={"schedulingPeriodWeeks": 1},
            llm_api_key="sk-or-v1-test",
        )


def test_schedule_from_inputs_rejects_empty_task_sources():
    with pytest.raises(RuntimeError, match="No schedulable tasks"):
        scheduler_core.schedule_from_inputs(
            scheduler_task_sources_input={"all_tasks": []},
            scheduling_window={
                "scheduleStartDate": "2026-04-13",
                "scheduleEndDate": "2026-04-19",
                "schedulingPeriodWeeks": 1,
                "currentDate": "2026-04-13",
                "currentDayname": "mon",
            },
            llm_api_key="sk-or-v1-test",
        )
