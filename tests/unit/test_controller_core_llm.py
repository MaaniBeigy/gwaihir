"""LLM-enabled paths in ControllerAgent.validate_schedule."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from src.agents.controller_core import (
    ControllerAgent,
    _build_controller_llm_payload,
    _compact_assignment_for_llm,
    _extract_assignments,
    _truncate_text,
)


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload) if not isinstance(payload, str) else payload

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("non-json payload")
        return self._payload


def _llm_response(content):
    return {"choices": [{"message": {"content": content}}]}


def test_truncate_text_short_string_unchanged():
    assert _truncate_text("hello", max_len=10) == "hello"


def test_truncate_text_long_string_truncated_with_ellipsis():
    result = _truncate_text("x" * 500, max_len=20)
    assert len(result) == 20
    assert result.endswith("…")


def test_truncate_text_handles_none_input():
    assert _truncate_text(None) == ""


def test_compact_assignment_keeps_only_known_fields():
    full = {
        "task_id": "g1",
        "type": "gamebus",
        "name": "x" * 400,
        "description": "y" * 400,
        "date": "2026-04-15",
        "start": "09:00",
        "end": "10:00",
        "duration_min": 60,
        "noise_field": "drop me",
    }
    compact = _compact_assignment_for_llm(full)
    assert "noise_field" not in compact
    assert compact["task_id"] == "g1"
    assert len(compact["name"]) <= 260 + 1
    assert len(compact["description"]) <= 320 + 1


def test_compact_assignment_handles_non_dict():
    assert _compact_assignment_for_llm("not a dict") == {}


def test_extract_assignments_supports_envelope_payload():
    payload = {"schedule_json": {"assignments": [{"task_id": "x"}, {"bad": True}, "junk"]}}
    assignments = _extract_assignments(payload)
    assert len(assignments) == 2  # only dicts


def test_extract_assignments_returns_empty_for_non_dict_payload():
    assert _extract_assignments(None) == []
    assert _extract_assignments([]) == []


def test_build_controller_llm_payload_uses_routine_schedulable_when_present():
    payload = _build_controller_llm_payload(
        schedule_payload={"assignments": [{"task_id": "g1", "type": "gamebus"}]},
        expected_gamebus_tasks=2,
        scheduler_task_sources={
            "routine_schedulable_tasks": [{"task_id": "r1"}, {"task_id": "r2"}],
        },
    )
    assert payload["expected_gamebus_tasks"] == 2
    assert payload["scheduler_task_sources_summary"]["routine_expected"] == 2
    assert payload["scheduler_task_sources_summary"]["routine_checks_enabled"] is True


def test_build_controller_llm_payload_falls_back_to_all_tasks():
    payload = _build_controller_llm_payload(
        schedule_payload={"assignments": []},
        expected_gamebus_tasks=0,
        scheduler_task_sources={
            "all_tasks": [
                {"source_type": "routine", "scheduling_required": True, "task_id": "r1"},
                {"source_type": "routine", "scheduling_required": False, "task_id": "r2"},
                {"source_type": "gamebus", "task_id": "g1"},
            ]
        },
    )
    assert payload["scheduler_task_sources_summary"]["routine_expected"] == 1


def test_validate_schedule_with_llm_returns_violations(monkeypatch):
    captured: List[Dict[str, Any]] = []

    def _post(url, headers=None, json=None, timeout=None):
        captured.append({"url": url, "json": json})
        return _FakeResponse(
            _llm_response(
                json_lib_dumps(
                    {
                        "is_valid": False,
                        "violations": ["routine task r1 is not scheduled"],
                        "fix_advice": ["assign r1 on Monday"],
                    }
                )
            )
        )

    monkeypatch.setattr("src.agents.controller_core.requests.post", _post)
    agent = ControllerAgent(enable_llm=True)
    result = agent.validate_schedule(
        schedule_payload={
            "assignments": [
                {"task_id": "g1", "type": "gamebus", "date": "2026-04-15", "start": "09:00", "end": "10:00"}
            ]
        },
        expected_gamebus_tasks=1,
    )
    assert result["is_valid"] is False
    assert "r1" in " ".join(result["violations"])
    assert any("Monday" in advice for advice in result["fix_advice"])
    assert captured  # the LLM was called


def test_validate_schedule_with_llm_unwraps_fenced_json(monkeypatch):
    fenced = "``json\n" + json.dumps({"is_valid": True, "violations": [], "fix_advice": []}) + "\n``"
    monkeypatch.setattr(
        "src.agents.controller_core.requests.post",
        lambda *args, **kwargs: _FakeResponse(_llm_response(fenced)),
    )
    agent = ControllerAgent(enable_llm=True)
    result = agent.validate_schedule(
        schedule_payload={"assignments": []},
        expected_gamebus_tasks=0,
    )
    assert result["is_valid"] is True


def test_validate_schedule_falls_back_when_llm_throws(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("openrouter exploded")

    monkeypatch.setattr("src.agents.controller_core.requests.post", _boom)
    monkeypatch.setattr("src.agents.controller_core.time.sleep", lambda _s: None)

    agent = ControllerAgent(enable_llm=True)
    result = agent.validate_schedule(
        schedule_payload={"assignments": []},
        expected_gamebus_tasks=0,
    )
    # Degraded fallback: still returns a structured result, marks valid.
    assert result["is_valid"] is True
    assert any("Controller" in v for v in result["violations"])


def test_validate_schedule_retries_on_http_failure(monkeypatch):
    calls = []

    def _post(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            return _FakeResponse({"error": "boom"}, status_code=500)
        return _FakeResponse(
            _llm_response(json.dumps({"is_valid": True, "violations": [], "fix_advice": []}))
        )

    monkeypatch.setattr("src.agents.controller_core.requests.post", _post)
    monkeypatch.setattr("src.agents.controller_core.time.sleep", lambda _s: None)

    agent = ControllerAgent(enable_llm=True)
    result = agent.validate_schedule(
        schedule_payload={"assignments": []},
        expected_gamebus_tasks=0,
    )
    assert result["is_valid"] is True
    assert len(calls) == 3


def test_validate_schedule_requires_dict_payload():
    agent = ControllerAgent(enable_llm=False)
    with pytest.raises(RuntimeError, match="schedule_payload"):
        agent.validate_schedule(schedule_payload="not a dict")


def test_call_llm_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = ControllerAgent(enable_llm=True)
    agent.api_key = None
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        agent._call_llm("prompt")


def test_extract_json_candidate_finds_brace_substring():
    agent = ControllerAgent(enable_llm=False)
    parsed = agent._extract_json_candidate('preamble {"is_valid": true} epilogue')
    assert parsed == {"is_valid": True}


def test_extract_json_candidate_returns_none_for_empty():
    agent = ControllerAgent(enable_llm=False)
    assert agent._extract_json_candidate("") is None


def test_parse_json_like_handles_python_literal():
    agent = ControllerAgent(enable_llm=False)
    parsed = agent._parse_json_like("{'is_valid': True, 'violations': []}")
    assert parsed == {"is_valid": True, "violations": []}


def test_parse_json_like_returns_none_for_garbage():
    agent = ControllerAgent(enable_llm=False)
    assert agent._parse_json_like("not parseable") is None


# Helper to keep json.dumps out of the response builder line for readability.
def json_lib_dumps(value):
    return json.dumps(value)
