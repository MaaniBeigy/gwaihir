"""Edge-case behaviour for `scheduler_core`: enrich, `_load_json`, LLM error paths."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

import pytest

from src.agents import scheduler_core

# -------- _enrich_assignments_with_task_metadata edge cases ---------------------------------------


def test_enrich_assignments_skips_tasks_with_no_id():
    """Tasks without task_id are skipped during the by_id index build."""
    schedule = {"assignments": [{"task_id": "r1", "type": "routine", "date": "2026-04-13"}]}
    routine_tasks = [
        {"task_id": None, "source_type": "routine", "name": "ignored"},
        {"task_id": "r1", "source_type": "routine", "name": "Morning walk"},
    ]
    enriched = scheduler_core._enrich_assignments_with_task_metadata(schedule, routine_tasks)
    assert enriched["assignments"][0]["name"] == "Morning walk"


def test_enrich_assignments_skips_assignments_with_no_id():
    schedule = {
        "assignments": [
            {"task_id": "r1", "date": "2026-04-13"},
            "not-a-dict",
            {"date": "2026-04-13"},  # no task_id
        ]
    }
    routine_tasks = [{"task_id": "r1", "source_type": "routine", "name": "Walk"}]
    enriched = scheduler_core._enrich_assignments_with_task_metadata(schedule, routine_tasks)
    # Only the first item gets enriched; the others remain unchanged.
    assert enriched["assignments"][0].get("name") == "Walk"


def test_enrich_assignments_routine_without_name_or_description_is_safe():
    schedule = {"assignments": [{"task_id": "r1", "date": "2026-04-13"}]}
    routine_tasks = [{"task_id": "r1", "source_type": "routine"}]
    enriched = scheduler_core._enrich_assignments_with_task_metadata(schedule, routine_tasks)
    assert enriched["assignments"][0]["scheduling_required"] is False


# -------- _load_json branches --------------------------------------------------------------------


def test_load_json_returns_none_for_unknown_type():
    assert scheduler_core._load_json(42) is None


def test_load_json_reads_existing_file_path():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        fh.write('{"hello": "world"}')
        path = fh.name
    try:
        result = scheduler_core._load_json(path)
        assert result == {"hello": "world"}
    finally:
        os.unlink(path)


def test_load_json_parses_inline_json_string():
    assert scheduler_core._load_json('{"x": 1}') == {"x": 1}


# -------- _normalize_task_sources_payload edge cases ----------------------------------------------


def test_normalize_task_sources_payload_returns_payload_unchanged_when_no_aliases():
    payload = {"foo": "bar"}
    assert scheduler_core._normalize_task_sources_payload(payload) is payload


# -------- _normalize_llm_schedule deeper branches -------------------------------------------------


def test_normalize_llm_schedule_unwraps_result_envelope():
    raw = {"result": {"assignments": [{"task_id": "g1", "date": "2026-04-15", "start": "09:00"}]}}
    parsed = scheduler_core._normalize_llm_schedule(raw)
    assert parsed is not None
    assert parsed["assignments"][0]["task_id"] == "g1"


def test_normalize_llm_schedule_drops_non_dict_assignment_items():
    raw = {"assignments": ["junk", 42, {"task_id": "g1", "date": "2026-04-15", "start": "09:00"}]}
    parsed = scheduler_core._normalize_llm_schedule(raw)
    assert parsed is not None
    assert len(parsed["assignments"]) == 1


def test_normalize_llm_schedule_returns_none_when_assignments_is_not_list():
    raw = {"assignments": "not a list"}
    assert scheduler_core._normalize_llm_schedule(raw) is None


# -------- _call_llm error paths -------------------------------------------------------------------


class _Resp:
    def __init__(self, payload, status_code=200, raises_on_json=False):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload) if not isinstance(payload, str) else payload
        self._raises_on_json = raises_on_json

    def json(self):
        if self._raises_on_json:
            raise ValueError("not json")
        return self._payload


def test_call_llm_retries_on_no_choices(monkeypatch):
    calls = []

    def _post(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            return _Resp({"choices": []})
        return _Resp({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("src.agents.scheduler_core.requests.post", _post)
    monkeypatch.setattr("src.agents.scheduler_core.time.sleep", lambda _s: None)

    result = scheduler_core._call_llm("p", "m", "k", "openrouter")
    assert result == "ok"
    assert len(calls) == 3


def test_call_llm_retries_on_empty_content(monkeypatch):
    calls = []

    def _post(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            return _Resp({"choices": [{"message": {"content": ""}}]})
        return _Resp({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("src.agents.scheduler_core.requests.post", _post)
    monkeypatch.setattr("src.agents.scheduler_core.time.sleep", lambda _s: None)

    result = scheduler_core._call_llm("p", "m", "k", "openrouter")
    assert result == "ok"


def test_call_llm_uses_openai_provider_naming_in_errors(monkeypatch):
    monkeypatch.setattr(
        "src.agents.scheduler_core.requests.post", lambda *a, **kw: _Resp({}, status_code=500)
    )
    monkeypatch.setattr("src.agents.scheduler_core.time.sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="LLM call failed"):
        scheduler_core._call_llm("p", "m", "k", "openai")


def test_call_llm_handles_request_exception(monkeypatch):
    import requests as _real_requests

    calls = []

    def _post(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise _real_requests.exceptions.ConnectionError("network down")
        return _Resp({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("src.agents.scheduler_core.requests.post", _post)
    monkeypatch.setattr("src.agents.scheduler_core.time.sleep", lambda _s: None)

    result = scheduler_core._call_llm("p", "m", "k", "openrouter")
    assert result == "ok"


# -------- schedule_gamebus default-window branch (no scheduling_window passed) --------------------


def test_schedule_gamebus_uses_defaults_when_no_window_passed(monkeypatch):
    monkeypatch.setattr(
        scheduler_core,
        "_call_llm",
        lambda *a, **kw: json.dumps(
            {"assignments": [{"task_id": "g1", "date": "2026-04-15", "start": "09:00"}]}
        ),
    )

    agent = scheduler_core.SchedulingAgent(llm_api_key="sk-or-v1-test")
    result = agent.schedule_gamebus(
        scheduler_task_sources={
            "all_tasks": [{"task_id": "g1", "source_type": "gamebus"}],
            "gamebus_tasks": [{"task_id": "g1", "source_type": "gamebus"}],
            "routine_tasks": [],
        },
    )
    assert result is not None
    assert result["schedule_start_date"]


def test_schedule_gamebus_derives_routine_lists_from_all_tasks(monkeypatch):
    """When task_sources omits routine_tasks/gamebus_tasks, they're derived from all_tasks."""
    monkeypatch.setattr(
        scheduler_core,
        "_call_llm",
        lambda *a, **kw: json.dumps(
            {"assignments": [{"task_id": "g1", "date": "2026-04-15", "start": "09:00"}]}
        ),
    )

    agent = scheduler_core.SchedulingAgent(llm_api_key="sk-or-v1-test")
    result = agent.schedule_gamebus(
        scheduler_task_sources={
            "all_tasks": [
                {"task_id": "r1", "source_type": "routine", "scheduling_required": True},
                {"task_id": "g1", "source_type": "gamebus"},
            ],
            # routine_tasks / gamebus_tasks / routine_schedulable_tasks omitted; derived from all_tasks.
        },
        scheduling_window={
            "scheduleStartDate": "2026-04-13",
            "scheduleEndDate": "2026-04-19",
            "schedulingPeriodWeeks": 1,
            "currentDate": "2026-04-13",
            "currentDayname": "mon",
        },
    )
    assert result["routine_tasks"] == 1
    assert result["gamebus_tasks"] == 1


# -------- _normalize_busy_intervals branches ------------------------------------------------------


def test_normalize_busy_intervals_skips_dict_without_start_or_end():
    busy = scheduler_core._normalize_busy_intervals(
        [
            {"startDate": "2026-04-15T08:00:00Z"},  # no end
            {"endDate": "2026-04-15T08:00:00Z"},  # no start
            {"start": "2026-04-15T09:00:00Z", "end": "2026-04-15T10:00:00Z"},
        ]
    )
    assert len(busy) == 1


def test_normalize_busy_intervals_omits_alldays_when_none():
    busy = scheduler_core._normalize_busy_intervals(
        [{"start": "2026-04-15", "end": "2026-04-16", "allDay": None}]
    )
    assert "allDay" not in busy[0]
