"""Provider routing, retry behavior, enrich, and summarization in `scheduler_core`."""

from __future__ import annotations

import json
from typing import List

import pytest

from src.agents import scheduler_core


def test_build_llm_request_config_openai_uses_authorization_header(monkeypatch):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    config = scheduler_core._build_llm_request_config("sk-key", "openai")
    assert config["url"].endswith("/chat/completions")
    assert "Authorization" in config["headers"]
    assert config["headers"]["Authorization"] == "Bearer sk-key"
    # OpenAI does not need OpenRouter-specific headers.
    assert "HTTP-Referer" not in config["headers"]


def test_build_llm_request_config_openrouter_includes_referer():
    config = scheduler_core._build_llm_request_config("sk-key", "openrouter")
    assert "HTTP-Referer" in config["headers"]
    assert "X-Title" in config["headers"]


def test_summarize_assignments_returns_count_for_dict_with_list():
    schedule = {"assignments": [{"task_id": "g1", "type": "gamebus"}, "junk", {"task_id": "g2"}]}
    summary = scheduler_core._summarize_assignments_for_logs(schedule)
    assert summary["count"] == 3  # raw count, including non-dict
    assert all(isinstance(p, dict) for p in summary["preview"])


def test_normalize_llm_schedule_returns_none_for_garbage():
    assert scheduler_core._normalize_llm_schedule(None) is None
    assert scheduler_core._normalize_llm_schedule(42) is None
    assert scheduler_core._normalize_llm_schedule("totally not json") is None


def test_normalize_llm_schedule_drops_assignments_missing_required_fields():
    raw = {
        "assignments": [
            {"task_id": "a", "date": "2026-04-15", "start": "09:00"},
            {"task_id": "b", "date": "2026-04-15"},  # missing start; dropped
            {"task_id": "c", "starttime": "10:00", "startdate": "2026-04-15"},  # legacy field aliases
        ]
    }
    parsed = scheduler_core._normalize_llm_schedule(raw)
    assert parsed is not None
    ids = [a["task_id"] for a in parsed["assignments"]]
    assert "a" in ids
    assert "b" not in ids
    assert "c" in ids


def test_normalize_llm_schedule_promotes_scheduled_tasks_alias():
    raw = {"scheduled_tasks": [{"task_id": "x", "date": "2026-04-15", "start": "09:00"}]}
    parsed = scheduler_core._normalize_llm_schedule(raw)
    assert parsed is not None
    assert parsed["assignments"][0]["task_id"] == "x"


def test_enrich_assignments_with_no_matching_task_id_does_nothing():
    schedule = {"assignments": [{"task_id": "unknown", "date": "2026-04-15"}]}
    enriched = scheduler_core._enrich_assignments_with_task_metadata(schedule, [])
    assert enriched["assignments"][0]["task_id"] == "unknown"


def test_enrich_assignments_returns_payload_when_assignments_not_list():
    payload = {"assignments": "not a list"}
    enriched = scheduler_core._enrich_assignments_with_task_metadata(payload, [])
    assert enriched is payload


def test_extract_json_candidate_handles_array_format():
    result = scheduler_core._extract_json_candidate('[{"task_id": "a"}]')
    assert isinstance(result, list)


def test_compact_controller_feedback_skips_non_strings():
    feedback = ["valid", 42, None, "  "]
    assert scheduler_core._compact_controller_feedback(feedback) == ["valid"]


def test_get_scheduling_period_weeks_reads_env(monkeypatch):
    monkeypatch.setenv("SCHEDULING_PERIOD_WEEKS", "3")
    assert scheduler_core._get_scheduling_period_weeks() == 3


def test_schedule_gamebus_retries_when_llm_response_invalid(monkeypatch):
    """First two LLM calls return junk; third returns a valid schedule."""
    call_log: List[str] = []

    def _call_llm(prompt, model, api_key, provider):
        call_log.append("called")
        if len(call_log) < 3:
            return "not valid json at all"
        return json.dumps(
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
        )

    monkeypatch.setattr(scheduler_core, "_call_llm", _call_llm)
    monkeypatch.setattr(scheduler_core.time, "sleep", lambda _s: None)

    agent = scheduler_core.SchedulingAgent(llm_api_key="sk-or-v1-test")
    result = agent.schedule_gamebus(
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
    assert result["scheduler_attempts_used"] == 3
    assert result["scheduler_retries_exhausted"] is False
    assert len(call_log) == 3


def test_schedule_gamebus_marks_retries_exhausted_when_all_attempts_invalid(monkeypatch):
    monkeypatch.setattr(scheduler_core, "_call_llm", lambda *a, **kw: "still not json")
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


def test_schedule_from_inputs_normalizes_string_feedback(monkeypatch):
    monkeypatch.setattr(
        scheduler_core,
        "_call_llm",
        lambda *a, **kw: json.dumps(
            {"assignments": [{"task_id": "g1", "date": "2026-04-15", "start": "09:00"}]}
        ),
    )
    result = scheduler_core.schedule_from_inputs(
        scheduler_task_sources_input={
            "all_tasks": [{"task_id": "g1", "source_type": "gamebus"}],
            "gamebus_tasks": [{"task_id": "g1", "source_type": "gamebus"}],
            "routine_tasks": [],
        },
        controller_feedback="single string feedback",
        scheduling_window={
            "scheduleStartDate": "2026-04-13",
            "scheduleEndDate": "2026-04-19",
            "schedulingPeriodWeeks": 1,
            "currentDate": "2026-04-13",
            "currentDayname": "mon",
        },
        llm_api_key="sk-or-v1-test",
    )
    assert result["scheduled_assignments"] == 1


def test_schedule_from_inputs_parses_json_array_feedback(monkeypatch):
    monkeypatch.setattr(
        scheduler_core,
        "_call_llm",
        lambda *a, **kw: json.dumps(
            {"assignments": [{"task_id": "g1", "date": "2026-04-15", "start": "09:00"}]}
        ),
    )
    result = scheduler_core.schedule_from_inputs(
        scheduler_task_sources_input={
            "all_tasks": [{"task_id": "g1", "source_type": "gamebus"}],
            "gamebus_tasks": [{"task_id": "g1", "source_type": "gamebus"}],
            "routine_tasks": [],
        },
        controller_feedback='["v1", "v2"]',
        scheduling_window={
            "scheduleStartDate": "2026-04-13",
            "scheduleEndDate": "2026-04-19",
            "schedulingPeriodWeeks": 1,
            "currentDate": "2026-04-13",
            "currentDayname": "mon",
        },
        llm_api_key="sk-or-v1-test",
    )
    assert result is not None


def _stub_capture_feedback(captured):
    def _stub(self, **kwargs):
        captured["feedback"] = kwargs.get("controller_feedback")
        return {"schedule_json": {"assignments": []}}

    return _stub


def test_schedule_from_inputs_normalizes_feedback_string_with_json_list(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "src.agents.scheduler_core.SchedulingAgent.schedule_gamebus", _stub_capture_feedback(captured)
    )
    scheduler_core.schedule_from_inputs(
        scheduler_task_sources_input={"all_tasks": [{"task_id": "x"}]},
        controller_feedback='["a", "b"]',
        scheduling_window={"scheduleStartDate": "2026-04-13", "scheduleEndDate": "2026-04-19"},
    )
    assert captured["feedback"] == ["a", "b"]


def test_schedule_from_inputs_normalizes_feedback_string_with_non_list_json(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "src.agents.scheduler_core.SchedulingAgent.schedule_gamebus", _stub_capture_feedback(captured)
    )
    scheduler_core.schedule_from_inputs(
        scheduler_task_sources_input={"all_tasks": [{"task_id": "x"}]},
        controller_feedback='{"not": "a list"}',
        scheduling_window={"scheduleStartDate": "2026-04-13", "scheduleEndDate": "2026-04-19"},
    )
    assert captured["feedback"] == ['{"not": "a list"}']


def test_schedule_from_inputs_feedback_string_with_unparseable_json(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "src.agents.scheduler_core.SchedulingAgent.schedule_gamebus", _stub_capture_feedback(captured)
    )
    scheduler_core.schedule_from_inputs(
        scheduler_task_sources_input={"all_tasks": [{"task_id": "x"}]},
        controller_feedback="not json {",
        scheduling_window={"scheduleStartDate": "2026-04-13", "scheduleEndDate": "2026-04-19"},
    )
    assert captured["feedback"] == ["not json {"]


def test_scheduling_agent_uses_openai_when_env_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    agent = scheduler_core.SchedulingAgent()
    assert agent.provider == "openai"
    assert agent.api_key == "sk-openai"
    assert agent.model == "gpt-4o-mini"


def test_scheduling_agent_uses_openrouter_without_openai_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    agent = scheduler_core.SchedulingAgent(model="custom-model")
    assert agent.provider == "openrouter"
    assert agent.api_key == "sk-or-test"
    assert agent.model == "custom-model"
