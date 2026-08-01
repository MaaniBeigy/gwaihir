"""Module-init, persistent Chroma client, and assignment-filter paths in `src.agents.app`."""

from __future__ import annotations

import importlib
import logging
from datetime import date
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# --------------------------------------------------------------------------------------------
# Module-level logger init: `if not logger.handlers`
# --------------------------------------------------------------------------------------------


def test_app_module_skips_basic_config_when_logger_already_has_handlers(monkeypatch):
    """When the `coach.app` logger already has a handler attached, the
    module skips `logging.basicConfig` on import."""
    # Pre-attach a handler to the named logger before app.py imports/reloads.
    logger = logging.getLogger("coach.app")
    sentinel = logging.NullHandler()
    logger.addHandler(sentinel)
    try:
        called = {"basic_config": False}

        def _fake_basic_config(**_kwargs):
            called["basic_config"] = True

        monkeypatch.setattr("logging.basicConfig", _fake_basic_config)

        import src.agents.app as app_module

        importlib.reload(app_module)

        assert called["basic_config"] is False
    finally:
        logger.removeHandler(sentinel)


# --------------------------------------------------------------------------------------------
# _build_chroma_client persistent (path) branch
# --------------------------------------------------------------------------------------------


def test_build_chroma_client_uses_persistent_path_when_chroma_host_unset(monkeypatch):
    """An empty `CHROMA_HOST` falls through to the persistent client at
    `CHROMA_PATH`."""
    monkeypatch.delenv("CHROMA_HOST", raising=False)
    monkeypatch.setenv("CHROMA_PATH", "/tmp/test-chroma-persistent")

    captured: Dict[str, Any] = {}

    def _stub_get_client(host=None, port=None, path=None):
        captured["host"] = host
        captured["port"] = port
        captured["path"] = path
        return MagicMock()

    import src.agents.app as app_module

    app_module = importlib.reload(app_module)
    monkeypatch.setattr(app_module, "get_client", _stub_get_client)

    app_module._build_chroma_client()

    assert captured["host"] is None
    assert captured["path"] == "/tmp/test-chroma-persistent"


# --------------------------------------------------------------------------------------------
# _assignment_to_planned_activity: date earlier than schedule_start
# --------------------------------------------------------------------------------------------


def test_assignment_before_schedule_start_returns_none():
    """An assignment date earlier than `schedule_start` is dropped."""
    from src.agents.app import _assignment_to_planned_activity

    planned = _assignment_to_planned_activity(
        {
            "type": "gamebus",
            "task_id": "g1",
            "name": "WALK",
            "date": "2026-04-10",
            "start": "09:00",
            "end": "10:00",
        },
        schedule_start=date(2026, 4, 13),
        schedule_end=date(2026, 4, 19),
    )
    assert planned is None


# --------------------------------------------------------------------------------------------
# Pipeline filter: assignments that convert to None continue the loop
# --------------------------------------------------------------------------------------------


class _StubCollection:
    def __init__(self, name: str):
        self.name = name
        self.records: List[Dict[str, Any]] = []

    def add(self, documents, ids=None, metadatas=None):
        for index, doc in enumerate(documents):
            metadata = metadatas[index] if metadatas else {}
            doc_id = ids[index] if ids else f"id-{len(self.records)}"
            self.records.append({"id": doc_id, "document": doc, "metadata": metadata})

    def get(self, where=None, include=None):
        return {"ids": [], "documents": [], "metadatas": []}

    def delete(self, ids=None, where=None):
        pass


class _StubChromaClient:
    def __init__(self) -> None:
        self.collections: Dict[str, _StubCollection] = {}

    def get_or_create_collection(self, name, embedding_function=None):
        self.collections.setdefault(name, _StubCollection(name))
        return self.collections[name]

    def get_collection(self, name):
        return self.collections.setdefault(name, _StubCollection(name))

    def delete_collection(self, name):
        self.collections.pop(name, None)

    def list_collections(self):
        return [type("Desc", (), {"name": n})() for n in self.collections]


def _chat_body() -> Dict[str, Any]:
    return {
        "playerId": 1,
        "campaignId": 1,
        "sessionId": "s-filter",
        "message": "hi",
        "context": {
            "challenges": [],
            "scheduledActivities": [],
            "schedulingWindow": {
                "scheduleStartDate": "2026-04-13",
                "scheduleEndDate": "2026-04-19",
                "schedulingPeriodWeeks": 1,
                "currentDate": "2026-04-13",
                "currentDayname": "mon",
            },
        },
    }


def test_pipeline_skips_assignments_that_convert_to_none(monkeypatch):
    """An assignment outside the schedule window converts to `None` and the
    pipeline loop continues to the next item without appending it."""
    monkeypatch.setenv("COACH_REQUIRE_MTLS", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")

    import src.memory.embeddings as embeddings_module

    importlib.reload(embeddings_module)
    embeddings_module._embedding_function = object()
    embeddings_module._ready = True
    embeddings_module._loading = False

    chroma_client = _StubChromaClient()

    import src.agents.app as app_module

    app_module = importlib.reload(app_module)
    monkeypatch.setattr(app_module, "_build_chroma_client", lambda: chroma_client)
    monkeypatch.setattr(app_module.flush, "start_flush_scheduler", lambda *a, **k: None)
    monkeypatch.setattr(app_module.embeddings, "warmup_embedding_function", lambda *a, **k: None)
    monkeypatch.setattr(
        "src.agents.interviewer_core.InterviewerAgent._send_to_llm",
        lambda self, messages: "Thank you for your time. The interview is complete.",
    )
    monkeypatch.setattr(
        "src.agents.standardizer_core.StandardizerAgent.standardize",
        lambda self, message, clear_history=False: '{"Activities": []}',
    )
    monkeypatch.setattr(
        "src.agents.scheduler_core.SchedulingAgent.schedule_gamebus",
        lambda self, scheduler_task_sources, **kwargs: {
            "schedule_json": {
                "assignments": [
                    # Inside the window: converts cleanly.
                    {
                        "type": "gamebus",
                        "task_id": "g-inside",
                        "name": "WALK",
                        "date": "2026-04-15",
                        "start": "09:00",
                        "end": "10:00",
                    },
                    # Outside the window: converts to None and is dropped.
                    {
                        "type": "gamebus",
                        "task_id": "g-outside",
                        "name": "RUN",
                        "date": "2026-05-01",
                        "start": "09:00",
                        "end": "10:00",
                    },
                ]
            },
            "assignments": [],
        },
    )
    monkeypatch.setattr(
        "src.agents.controller_core.ControllerAgent.validate_schedule",
        lambda self, **kwargs: {
            "is_valid": True,
            "violations": [],
            "fix_advice": [],
            "summary": {},
            "llm_used": False,
        },
    )
    monkeypatch.setattr("src.webhook.client.WebhookClient.emit", lambda self, **kwargs: True)

    response = TestClient(app_module.app).post("/chat", json=_chat_body())
    assert response.status_code == 200
    payload = response.json()
    assert payload["interviewComplete"] is True
    # Only the in-window assignment survives the filter.
    assert len(payload["plannedActivities"]) == 1
    assert payload["plannedActivities"][0]["activityType"] in {"WALK", "g-inside"}
