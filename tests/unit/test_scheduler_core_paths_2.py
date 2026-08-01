"""Remaining edge cases in `scheduler_core`: helpers, parsers, agent fallbacks."""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from src.agents import scheduler_core

# --------------------------------------------------------------------------------------------
# _load_json: None input
# --------------------------------------------------------------------------------------------


def test_load_json_returns_none_for_none_input():
    """A literal `None` input is returned unchanged via the early return."""
    assert scheduler_core._load_json(None) is None


# --------------------------------------------------------------------------------------------
# _normalize_task_sources_payload: nested task_sources_json without `all_tasks`
# --------------------------------------------------------------------------------------------


def test_normalize_task_sources_payload_skips_nested_when_no_all_tasks_list():
    """When `task_sources_json` is a dict but lacks `all_tasks` as a list,
    the helper falls through to the `tasks` alias check or returns the
    original payload."""
    payload = {"task_sources_json": {"unrelated": "value"}}
    result = scheduler_core._normalize_task_sources_payload(payload)
    assert result is payload


def test_normalize_task_sources_payload_uses_tasks_alias_when_nested_lacks_all_tasks():
    """If `task_sources_json` doesn't supply `all_tasks`, but a top-level
    `tasks` list exists, that alias is promoted."""
    payload = {
        "task_sources_json": {"unrelated": "value"},
        "tasks": [{"task_id": "x"}],
    }
    result = scheduler_core._normalize_task_sources_payload(payload)
    assert result["all_tasks"] == [{"task_id": "x"}]


# --------------------------------------------------------------------------------------------
# _load_instructions: missing or empty template file
# --------------------------------------------------------------------------------------------


def test_load_instructions_raises_when_template_file_missing(monkeypatch):
    monkeypatch.setattr("src.agents.scheduler_core.os.path.exists", lambda path: False)
    with pytest.raises(RuntimeError, match="Scheduler instructions not found"):
        scheduler_core._load_instructions()


def test_load_instructions_raises_when_template_file_is_empty(monkeypatch):
    monkeypatch.setattr("src.agents.scheduler_core.os.path.exists", lambda path: True)

    real_open = open
    from io import StringIO

    def _fake_open(path, *args, **kwargs):
        if str(path).endswith("Scheduler_instructions.txt"):
            return StringIO("")
        return real_open(path, *args, **kwargs)

    with patch("builtins.open", _fake_open):
        with pytest.raises(RuntimeError, match="Scheduler instructions file is empty"):
            scheduler_core._load_instructions()


# --------------------------------------------------------------------------------------------
# _call_llm: all three attempts fail; loop exits via natural exhaustion
# --------------------------------------------------------------------------------------------


class _Resp:
    def __init__(self, payload: Any, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)

    def json(self) -> Any:
        return self._payload


def test_call_llm_raises_after_three_consecutive_failures(monkeypatch):
    """All three attempts fail; the loop exhausts and the post-loop raise fires."""
    calls: List[int] = []

    def _post(*args, **kwargs):
        calls.append(1)
        return _Resp({"error": "boom"}, status_code=500)

    monkeypatch.setattr("src.agents.scheduler_core.requests.post", _post)
    monkeypatch.setattr("src.agents.scheduler_core.time.sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="LLM call failed after retries"):
        scheduler_core._call_llm("prompt", "model", "key", "openrouter")
    assert len(calls) == 3


# --------------------------------------------------------------------------------------------
# _parse_json_like: ast.literal_eval succeeds with a dict/list
# --------------------------------------------------------------------------------------------


def test_parse_json_like_accepts_python_dict_literal():
    """Single-quoted dict is invalid JSON but valid Python; the second branch
    parses it via `ast.literal_eval` and returns the dict."""
    result = scheduler_core._parse_json_like("{'x': 1, 'y': 2}")
    assert result == {"x": 1, "y": 2}


def test_parse_json_like_accepts_python_list_literal():
    """Single-quoted list literal also succeeds via `ast.literal_eval`."""
    result = scheduler_core._parse_json_like("['a', 'b']")
    assert result == ["a", "b"]


def test_parse_json_like_returns_none_for_python_only_scalar():
    """Python `True` is invalid JSON and parses to a bool, not a dict/list."""
    assert scheduler_core._parse_json_like("True") is None


# --------------------------------------------------------------------------------------------
# _extract_json_candidate: fenced loop continues when a block parses to None
# --------------------------------------------------------------------------------------------


def test_extract_json_candidate_continues_when_first_fenced_block_is_unparseable():
    """A first fenced block that parses to None must not short-circuit; the
    second block (valid JSON) wins."""
    text = '``\nbroken { not valid }\n``\n``json\n{"hello": "world"}\n``'
    result = scheduler_core._extract_json_candidate(text)
    assert result == {"hello": "world"}


def test_extract_json_candidate_continues_to_brackets_when_braces_yield_none():
    """An unparseable `{...}` substring lets the brace iteration produce
    `None` so the loop continues to `[...]`, which parses cleanly."""
    text = "preface { not valid python or json } trailing [1, 2, 3]"
    result = scheduler_core._extract_json_candidate(text)
    assert result == [1, 2, 3]


# --------------------------------------------------------------------------------------------
# _normalize_llm_schedule: schedule_value is a string parsing to a list
# --------------------------------------------------------------------------------------------


def test_normalize_llm_schedule_wraps_list_extracted_from_text():
    """A string whose extracted JSON candidate is a list is wrapped into the
    standard `{"assignments": [...]}` envelope.

    The leading `{}` makes the brace-extraction substring unparseable, so
    `_extract_json_candidate` falls through to the bracket extraction and
    returns the list of assignment dicts.
    """
    raw = 'Result {} [{"task_id": "g1", "date": "2026-04-15", "start": "09:00"}]'
    result = scheduler_core._normalize_llm_schedule(raw)
    assert result is not None
    assert result["assignments"][0]["task_id"] == "g1"


def test_normalize_llm_schedule_unwraps_first_envelope_then_continues_when_second_yields_none():
    """When `schedule` wrapper recurses to None (no usable assignments
    inside), the loop continues to the next wrapper key."""
    raw = {
        "schedule": {"assignments": "not-a-list"},  # invalid; recursive call returns None
        "result": {"assignments": [{"task_id": "g1", "date": "2026-04-15", "start": "09:00"}]},
    }
    result = scheduler_core._normalize_llm_schedule(raw)
    assert result is not None
    assert result["assignments"][0]["task_id"] == "g1"


def test_normalize_llm_schedule_promotes_endtime_alias():
    """An assignment using the legacy `endtime` field gets normalised to `end`."""
    raw = {"assignments": [{"task_id": "g1", "date": "2026-04-15", "start": "09:00", "endtime": "10:00"}]}
    result = scheduler_core._normalize_llm_schedule(raw)
    assert result is not None
    assert result["assignments"][0]["end"] == "10:00"


# --------------------------------------------------------------------------------------------
# SchedulingAgent.schedule_gamebus: input handling edges
# --------------------------------------------------------------------------------------------


def test_schedule_gamebus_rejects_non_dict_task_sources():
    """A task_sources payload that decodes to a list (not a dict) is rejected."""
    agent = scheduler_core.SchedulingAgent(llm_api_key="sk-or-v1-test")
    with pytest.raises(RuntimeError, match="must be a valid JSON object"):
        agent.schedule_gamebus(
            scheduler_task_sources=[1, 2, 3],
            scheduling_window={
                "scheduleStartDate": "2026-04-13",
                "scheduleEndDate": "2026-04-19",
                "schedulingPeriodWeeks": 1,
                "currentDate": "2026-04-13",
                "currentDayname": "mon",
            },
        )


def test_schedule_gamebus_uses_explicit_routine_lists_when_provided(monkeypatch):
    """When the task_sources payload supplies pre-computed
    routine_schedulable_tasks, routine_context_only_tasks, and
    non_schedulable_task_ids, those values are used as-is rather than derived."""
    monkeypatch.setattr(
        scheduler_core,
        "_call_llm",
        lambda *a, **kw: json.dumps(
            {"assignments": [{"task_id": "g1", "date": "2026-04-15", "start": "09:00"}]}
        ),
    )

    routine_task = {"task_id": "r1", "source_type": "routine", "scheduling_required": True}
    explicit_schedulable = [routine_task]
    explicit_context_only = [{"task_id": "r2", "source_type": "routine"}]
    explicit_non_schedulable = ["r2"]

    agent = scheduler_core.SchedulingAgent(llm_api_key="sk-or-v1-test")
    result = agent.schedule_gamebus(
        scheduler_task_sources={
            "all_tasks": [routine_task, {"task_id": "g1", "source_type": "gamebus"}],
            "routine_tasks": [routine_task],
            "gamebus_tasks": [{"task_id": "g1", "source_type": "gamebus"}],
            "routine_schedulable_tasks": explicit_schedulable,
            "routine_context_only_tasks": explicit_context_only,
            "non_schedulable_task_ids": explicit_non_schedulable,
        },
        scheduling_window={
            "scheduleStartDate": "2026-04-13",
            "scheduleEndDate": "2026-04-19",
            "schedulingPeriodWeeks": 1,
            "currentDate": "2026-04-13",
            "currentDayname": "mon",
        },
    )
    assert result["routine_schedulable_tasks"] == 1
    assert result["routine_context_only_tasks"] == 1


def test_schedule_gamebus_marks_retries_exhausted_when_all_attempts_fail(monkeypatch):
    """Three consecutive invalid LLM responses exit the loop via natural
    exhaustion; the agent records `scheduler_retries_exhausted` and raises."""
    monkeypatch.setattr(scheduler_core, "_call_llm", lambda *a, **kw: "totally not json")
    monkeypatch.setattr(scheduler_core.time, "sleep", lambda _s: None)

    agent = scheduler_core.SchedulingAgent(llm_api_key="sk-or-v1-test")
    with pytest.raises(RuntimeError, match="did not produce a valid schedule"):
        agent.schedule_gamebus(
            scheduler_task_sources={
                "all_tasks": [{"task_id": "g1", "source_type": "gamebus"}],
                "gamebus_tasks": [{"task_id": "g1", "source_type": "gamebus"}],
                "routine_tasks": [],
            },
            scheduling_window={
                "scheduleStartDate": "2026-04-13",
                "scheduleEndDate": "2026-04-19",
                "schedulingPeriodWeeks": 1,
                "currentDate": "2026-04-13",
                "currentDayname": "mon",
            },
        )


# --------------------------------------------------------------------------------------------
# _normalize_busy_intervals: `allDay` carried through when not None
# --------------------------------------------------------------------------------------------


def test_normalize_busy_intervals_includes_alldays_when_provided():
    busy = scheduler_core._normalize_busy_intervals(
        [{"start": "2026-04-15", "end": "2026-04-16", "allDay": True}]
    )
    assert busy[0]["allDay"] is True


def test_normalize_busy_intervals_includes_alldays_when_false():
    busy = scheduler_core._normalize_busy_intervals(
        [{"start": "2026-04-15", "end": "2026-04-16", "allDay": False}]
    )
    assert busy[0]["allDay"] is False


# --------------------------------------------------------------------------------------------
# schedule_from_inputs: feedback parsing branches
# --------------------------------------------------------------------------------------------


def test_schedule_from_inputs_normalizes_list_feedback(monkeypatch):
    """A list of feedback strings is forwarded to the agent unchanged."""
    captured: Dict[str, Any] = {}

    def _capture(self, scheduler_task_sources, **kwargs):
        captured["feedback"] = kwargs.get("controller_feedback")
        return {"schedule_json": {"assignments": []}, "assignments": []}

    monkeypatch.setattr(scheduler_core.SchedulingAgent, "schedule_gamebus", _capture)

    scheduler_core.schedule_from_inputs(
        scheduler_task_sources_input={"all_tasks": [{"task_id": "g1"}]},
        controller_feedback=["routine missing", None, "bad slot"],
        scheduling_window={
            "scheduleStartDate": "2026-04-13",
            "scheduleEndDate": "2026-04-19",
            "schedulingPeriodWeeks": 1,
            "currentDate": "2026-04-13",
            "currentDayname": "mon",
        },
        llm_api_key="sk-or-v1-test",
    )
    # Drops None entries and stringifies the rest.
    assert captured["feedback"] == ["routine missing", "bad slot"]


def test_schedule_from_inputs_wraps_non_list_json_feedback(monkeypatch):
    """A JSON-encoded dict (i.e. not a list) falls into the wrap-as-string
    branch so the original text is preserved verbatim."""
    captured: Dict[str, Any] = {}

    def _capture(self, scheduler_task_sources, **kwargs):
        captured["feedback"] = kwargs.get("controller_feedback")
        return {"schedule_json": {"assignments": []}, "assignments": []}

    monkeypatch.setattr(scheduler_core.SchedulingAgent, "schedule_gamebus", _capture)

    scheduler_core.schedule_from_inputs(
        scheduler_task_sources_input={"all_tasks": [{"task_id": "g1"}]},
        controller_feedback='{"violations": ["routine missing"]}',
        scheduling_window={
            "scheduleStartDate": "2026-04-13",
            "scheduleEndDate": "2026-04-19",
            "schedulingPeriodWeeks": 1,
            "currentDate": "2026-04-13",
            "currentDayname": "mon",
        },
        llm_api_key="sk-or-v1-test",
    )
    assert captured["feedback"] == ['{"violations": ["routine missing"]}']
