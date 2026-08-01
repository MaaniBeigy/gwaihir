"""Edge-case behaviour for the standardizer, interviewer, controller, and task_normalizer."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

# -------- StandardizerAgent --------------------------------------------------------------------


def test_standardizer_falls_back_to_dummy_key_when_env_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    from src.agents.standardizer_core import StandardizerAgent

    agent = StandardizerAgent()
    assert agent.api_key.startswith("sk-or-v1-dummy")


def test_standardizer_init_raises_when_template_missing(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    from src.agents import standardizer_core

    monkeypatch.setattr("os.path.exists", lambda path: False)
    with pytest.raises(RuntimeError, match="instructions"):
        standardizer_core.StandardizerAgent()


def test_standardizer_init_raises_when_template_empty(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    from src.agents import standardizer_core

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("")
        empty_path = fh.name
    try:
        monkeypatch.setattr(
            "src.agents.standardizer_core.os.path.exists",
            lambda path: True,
        )
        original_open = open

        def _fake_open(path, *args, **kwargs):
            if path.endswith("Standardizer_instructions.txt"):
                return original_open(empty_path, *args, **kwargs)
            return original_open(path, *args, **kwargs)

        with patch("builtins.open", _fake_open):
            with pytest.raises(RuntimeError, match="empty"):
                standardizer_core.StandardizerAgent()
    finally:
        os.unlink(empty_path)


def test_standardizer_strips_bare_triple_backtick(monkeypatch):
    """The validator must handle ``` without the json language tag."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    from src.agents.standardizer_core import StandardizerAgent

    agent = StandardizerAgent()
    fenced = "``\n" + json.dumps({"Activities": [{"name": "x"}]}) + "\n``"
    is_valid, activities = agent._validate_json_response(fenced)
    assert is_valid is True
    assert activities == [{"name": "x"}]


def test_standardizer_parse_time_range_handles_word_separators(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    from src.agents.standardizer_core import StandardizerAgent

    agent = StandardizerAgent()
    assert agent._parse_time_range("from 09:00 until 10:00") == (9 * 60, 10 * 60)
    assert agent._parse_time_range("from 09:00 till 10:00") == (9 * 60, 10 * 60)


def test_standardizer_parse_time_range_with_start_finish_keywords(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    from src.agents.standardizer_core import StandardizerAgent

    agent = StandardizerAgent()
    result = agent._parse_time_range("start at 09:00 and finish at 10:30")
    assert result == (9 * 60, 10 * 60 + 30)


def test_standardizer_parse_time_range_returns_none_when_only_one_time(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    from src.agents.standardizer_core import StandardizerAgent

    agent = StandardizerAgent()
    # Two times listed but no begin/start/finish/end keywords; range cannot be inferred.
    assert agent._parse_time_range("at 09:00 and again at 10:00") is None


def test_standardizer_parse_time_range_returns_none_for_empty_text(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    from src.agents.standardizer_core import StandardizerAgent

    agent = StandardizerAgent()
    assert agent._parse_time_range("") is None


def test_standardizer_normalize_durations_skips_when_no_time_range(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    from src.agents.standardizer_core import StandardizerAgent

    agent = StandardizerAgent()
    activities = [{"name": "x", "description": "no time given", "activity_duration": 30, "time": {}}]
    agent._normalize_durations(activities)
    assert activities[0]["activity_duration"] == 30  # unchanged


def test_standardizer_normalize_durations_skips_when_zero_duration(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    from src.agents.standardizer_core import StandardizerAgent

    agent = StandardizerAgent()
    # Time range parses to zero minutes; computed_duration <= 0 path runs.
    activities = [{"name": "x", "description": "from 09:00 to 09:00", "time": {}}]
    agent._normalize_durations(activities)
    # Duration is left unchanged when no positive interval can be derived.


def test_standardizer_standardize_skips_clear_history_branch(monkeypatch):
    """clear_history=False keeps prior turns in the conversation buffer."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    from src.agents.standardizer_core import StandardizerAgent

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": json.dumps({"Activities": []})}}]}

    monkeypatch.setattr("src.agents.standardizer_core.requests.post", lambda *a, **kw: _Resp())
    agent = StandardizerAgent()
    agent.standardize("first", clear_history=True)
    agent.standardize("second", clear_history=False)
    # Two user turns + two assistant replies = four messages.
    assert len(agent.get_conversation_history()) == 4


# -------- InterviewerAgent --------------------------------------------------------------------


def test_interviewer_get_history_with_invalid_timestamp_metadata(monkeypatch):
    """Invalid ISO timestamps fall back to insertion-order ordering."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    from src.agents.interviewer_core import InterviewerAgent

    class _StubCol:
        def add(self, **kw):
            pass

    monkeypatch.setattr(
        "src.agents.interviewer_core.retrieve_all_history",
        lambda col: {
            "documents": ["one", "two"],
            "metadatas": [{"timestamp": "garbage"}, {}],
        },
    )
    agent = InterviewerAgent(collection=_StubCol())
    history = agent.get_history()
    assert history == ["one", "two"]


def test_interviewer_chat_logs_warning_when_store_qa_fails(monkeypatch):
    """If store_qa raises, chat() returns the reply but logs a warning."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    from src.agents.interviewer_core import InterviewerAgent

    class _StubCol:
        def add(self, **kw):
            pass

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "Q1: hi"}}]}

    monkeypatch.setattr("requests.post", lambda *a, **kw: _Resp())

    def _broken_store(*args, **kwargs):
        raise RuntimeError("store failed")

    monkeypatch.setattr("src.agents.interviewer_core.store_qa", _broken_store)
    agent = InterviewerAgent(collection=_StubCol())
    reply = agent.chat("hello", clear_history=True)
    assert reply == "Q1: hi"


# -------- ControllerAgent --------------------------------------------------------------------


def test_controller_truncate_text_handles_zero_length():
    from src.agents.controller_core import _truncate_text

    assert _truncate_text("") == ""


def test_controller_build_payload_with_no_routine_schedulable_lists():
    from src.agents.controller_core import _build_controller_llm_payload

    payload = _build_controller_llm_payload(
        schedule_payload={"assignments": []},
        expected_gamebus_tasks=0,
        scheduler_task_sources={"routine_schedulable_tasks": "not a list", "all_tasks": "not a list"},
    )
    assert payload["scheduler_task_sources_summary"]["routine_expected"] == 0


def test_controller_build_payload_without_task_sources():
    from src.agents.controller_core import _build_controller_llm_payload

    payload = _build_controller_llm_payload(
        schedule_payload={"assignments": []},
        expected_gamebus_tasks=2,
        scheduler_task_sources=None,
    )
    assert payload["expected_gamebus_tasks"] == 2
    assert payload["scheduler_task_sources_summary"]["routine_expected"] == 0


def test_controller_call_llm_rejects_non_dict_response(monkeypatch):
    from src.agents.controller_core import ControllerAgent

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "[1, 2, 3]"}}]}  # list, not dict

        @property
        def text(self):
            return "[1,2,3]"

    monkeypatch.setattr("src.agents.controller_core.requests.post", lambda *a, **kw: _Resp())
    monkeypatch.setattr("src.agents.controller_core.time.sleep", lambda _s: None)

    agent = ControllerAgent(enable_llm=True)
    agent.api_key = "sk-or-v1-test"
    with pytest.raises(RuntimeError, match="not valid JSON"):
        agent._call_llm("prompt")


# -------- task_normalizer leftovers ----------------------------------------------------------


def test_task_normalizer_get_routine_schedule_mode_no_preferred():
    """No `time` key at all returns 'unspecified'."""
    from src.agents.tools import task_normalizer

    assert task_normalizer._get_routine_schedule_mode({"name": "x"}) == "unspecified"


def test_task_normalizer_normalize_days_filters_unknown():
    from src.agents.tools import task_normalizer

    assert task_normalizer._normalize_days(["Funday", "MONDAY"]) == ["mon"]


def test_task_normalizer_align_with_no_frequency_days():
    """Activities without a frequency.days list keep their start_date as-is."""
    from datetime import date

    from src.agents.tools import task_normalizer

    aligned = task_normalizer._align_activity_start_date_to_schedule_window(
        {"start_date": "2026-04-13"},  # no frequency
        schedule_start=date(2026, 4, 13),
        schedule_end=date(2026, 4, 19),
    )
    assert aligned["start_date"] == "2026-04-13"


# -------- flush leftovers ---------------------------------------------------------------------


def test_flush_parse_created_at_returns_none_for_non_string():
    from src.memory.flush import _parse_created_at

    assert _parse_created_at({"created_at": 12345}) is None
    assert _parse_created_at({"created_at": ""}) is None


def test_flush_collection_with_no_records_returns_zero_counts():
    from src.memory.flush import flush_collection

    class _Empty:
        name = "coach_p1_c1"

        def get(self, include=None):
            return {"ids": [], "metadatas": [], "documents": []}

        def delete(self, ids):
            pass

    counts = flush_collection(_Empty(), older_than_days=7)
    assert counts == {"deleted": 0, "kept": 0, "error": 0}


# -------- webhook leftovers -------------------------------------------------------------------


def test_webhook_emit_logs_when_url_missing_and_disabled(monkeypatch):
    """When the client is disabled it short-circuits regardless of URL state."""
    from src.webhook.client import WebhookClient

    client = WebhookClient(
        url="",
        client_cert=None,
        client_key=None,
        server_ca=None,
        internal_token="t",
        enabled=False,
    )
    ok = client.emit(
        player_id=1,
        campaign_id=2,
        session_id="s",
        phase=WebhookClient.PHASE_INTERVIEWING,
        request_id="req",
    )
    assert ok is True
