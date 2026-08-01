"""Lifespan, HTTP Chroma client, and pipeline error-fallback paths in `src.agents.app`."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# -------- shared stub infrastructure --------------------------------------------------------------


class _StubCollection:
    def __init__(self, name: str, fail_on: set[str] | None = None):
        self.name = name
        self.records: List[Dict[str, Any]] = []
        self.fail_on = fail_on or set()

    def add(self, documents, ids=None, metadatas=None):
        meta_type = metadatas[0].get("type") if metadatas and isinstance(metadatas[0], dict) else None
        if meta_type in self.fail_on:
            raise RuntimeError(f"add failed for type={meta_type}")
        for index, doc in enumerate(documents):
            metadata = metadatas[index] if metadatas else {}
            doc_id = ids[index] if ids else f"id-{len(self.records)}"
            self.records.append({"id": doc_id, "document": doc, "metadata": metadata})

    def get(self, where=None, include=None):
        return {"ids": [], "documents": [], "metadatas": []}

    def delete(self, ids=None, where=None):
        pass


class _StubChromaClient:
    def __init__(self, fail_on: set[str] | None = None):
        self._fail_on = fail_on or set()
        self.collections: Dict[str, _StubCollection] = {}

    def get_or_create_collection(self, name, embedding_function=None):
        self.collections.setdefault(name, _StubCollection(name, fail_on=self._fail_on))
        return self.collections[name]

    def get_collection(self, name):
        return self.collections.setdefault(name, _StubCollection(name, fail_on=self._fail_on))

    def delete_collection(self, name):
        self.collections.pop(name, None)

    def list_collections(self):
        return [type("Desc", (), {"name": n})() for n in self.collections]


def _chat_body():
    return {
        "playerId": 1,
        "campaignId": 1,
        "sessionId": "s1",
        "message": "hi",
        "llmKey": "sk-or-v1-override",
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


@pytest.fixture
def coach_app_with_failing_storage(monkeypatch):
    """A coach-api stack where every Chroma `add` raises. The pipeline must
    log each storage failure as a warning and still complete the chat flow."""
    monkeypatch.setenv("COACH_REQUIRE_MTLS", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")

    import src.memory.embeddings as embeddings_module

    importlib.reload(embeddings_module)
    embeddings_module._embedding_function = object()
    embeddings_module._ready = True
    embeddings_module._loading = False

    chroma_client = _StubChromaClient(
        fail_on={
            "cleaned_conversation",
            "standardized_activities",
            "scheduler_task_sources",
            "scheduled_gamebus",
        }
    )

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
                    {
                        "type": "gamebus",
                        "task_id": "g1",
                        "name": "WALK",
                        "date": "2026-04-15",
                        "start": "09:00",
                        "end": "10:00",
                    }
                ]
            },
            "assignments": [],
        },
    )

    def _boom_validate(self, **kwargs):
        raise RuntimeError("controller down")

    monkeypatch.setattr("src.agents.controller_core.ControllerAgent.validate_schedule", _boom_validate)
    monkeypatch.setattr("src.webhook.client.WebhookClient.emit", lambda self, **kwargs: True)

    return app_module


# -------- tests -----------------------------------------------------------------------------------


def test_chat_pipeline_swallows_storage_and_controller_failures(coach_app_with_failing_storage):
    """All four `store_*` calls fail and `validate_schedule` raises; pipeline keeps going."""
    app_module = coach_app_with_failing_storage
    response = TestClient(app_module.app).post("/chat", json=_chat_body())
    assert response.status_code == 200
    payload = response.json()
    assert payload["interviewComplete"] is True
    assert payload["plannedActivities"]


def test_chat_handles_scheduler_returning_non_dict(monkeypatch):
    """When the scheduler agent returns something that isn't a dict, the pipeline
    falls back to the empty-assignments branch and still returns a 200."""
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
    # Scheduler returns a non-dict; pipeline falls back to an empty assignments list.
    monkeypatch.setattr(
        "src.agents.scheduler_core.SchedulingAgent.schedule_gamebus",
        lambda self, scheduler_task_sources, **kwargs: "definitely-not-a-dict",
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
    assert payload["plannedActivities"] == []  # fallback empty list


def test_build_chroma_client_uses_http_when_chroma_host_is_set(monkeypatch):
    """Setting CHROMA_HOST forces _build_chroma_client to use the HTTP variant."""
    monkeypatch.setenv("CHROMA_HOST", "remote-chroma")
    monkeypatch.setenv("CHROMA_PORT", "8765")

    captured = {}

    def _get_client(host=None, port=None, path=None):
        captured["host"] = host
        captured["port"] = port
        captured["path"] = path
        return MagicMock()

    monkeypatch.setattr("src.agents.app.get_client", _get_client)

    import src.agents.app as app_module

    importlib.reload(app_module)
    monkeypatch.setattr(app_module, "get_client", _get_client)

    app_module._build_chroma_client()
    assert captured["host"] == "remote-chroma"
    assert captured["port"] == 8765
    assert captured["path"] is None


def test_lifespan_starts_and_shuts_down_cleanly(monkeypatch):
    """Entering `TestClient` as a context manager runs the FastAPI lifespan,
    so embedding warmup, flush scheduler startup, and shutdown all fire."""
    monkeypatch.setenv("COACH_REQUIRE_MTLS", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")

    import src.memory.embeddings as embeddings_module

    importlib.reload(embeddings_module)
    embeddings_module._embedding_function = object()
    embeddings_module._ready = True

    import src.agents.app as app_module

    app_module = importlib.reload(app_module)

    started = {"warmed": False, "scheduled": False, "shutdown": False, "closed": False}

    def _warmup():
        started["warmed"] = True

    class _StubScheduler:
        def shutdown(self, wait=False):
            started["shutdown"] = True

    def _start_flush(get_client_fn):
        started["scheduled"] = True
        return _StubScheduler()

    monkeypatch.setattr(app_module.embeddings, "warmup_embedding_function", _warmup)
    monkeypatch.setattr(app_module.flush, "start_flush_scheduler", _start_flush)

    def _close():
        started["closed"] = True

    monkeypatch.setattr(app_module._webhook_client, "close", _close)

    with TestClient(app_module.app) as client:
        client.get("/health")
    # Wait briefly for the daemon thread to register.
    assert started["scheduled"] is True
    assert started["shutdown"] is True
    assert started["closed"] is True


def test_lifespan_swallows_scheduler_start_failure(monkeypatch):
    """If start_flush_scheduler raises, lifespan logs it and proceeds; close() still runs."""
    monkeypatch.setenv("COACH_REQUIRE_MTLS", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")

    import src.memory.embeddings as embeddings_module

    importlib.reload(embeddings_module)
    embeddings_module._embedding_function = object()
    embeddings_module._ready = True

    import src.agents.app as app_module

    app_module = importlib.reload(app_module)

    def _broken_start(_get_client_fn):
        raise RuntimeError("scheduler refused to start")

    closed = {"called": False}

    def _close():
        closed["called"] = True

    monkeypatch.setattr(app_module.embeddings, "warmup_embedding_function", lambda *a, **k: None)
    monkeypatch.setattr(app_module.flush, "start_flush_scheduler", _broken_start)
    monkeypatch.setattr(app_module._webhook_client, "close", _close)

    with TestClient(app_module.app) as client:
        client.get("/health")
    assert closed["called"] is True


def test_lifespan_swallows_shutdown_failures(monkeypatch):
    """Both scheduler.shutdown and webhook.close raising must not bubble out."""
    monkeypatch.setenv("COACH_REQUIRE_MTLS", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")

    import src.memory.embeddings as embeddings_module

    importlib.reload(embeddings_module)
    embeddings_module._embedding_function = object()
    embeddings_module._ready = True

    import src.agents.app as app_module

    app_module = importlib.reload(app_module)

    class _Boom:
        def shutdown(self, wait=False):
            raise RuntimeError("shutdown failed")

    monkeypatch.setattr(app_module.embeddings, "warmup_embedding_function", lambda *a, **k: None)
    monkeypatch.setattr(app_module.flush, "start_flush_scheduler", lambda *_a, **_kw: _Boom())

    def _broken_close():
        raise RuntimeError("close failed")

    monkeypatch.setattr(app_module._webhook_client, "close", _broken_close)

    # Lifespan must not raise even though both shutdown paths throw.
    with TestClient(app_module.app) as client:
        client.get("/health")


def test_chat_skips_non_dict_assignments(monkeypatch):
    """Scheduler returns a list with mixed assignment types; non-dict items are dropped."""
    monkeypatch.setenv("COACH_REQUIRE_MTLS", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")

    import src.memory.embeddings as embeddings_module

    importlib.reload(embeddings_module)
    embeddings_module._embedding_function = object()
    embeddings_module._ready = True

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
                    "not-a-dict",
                    {
                        "type": "gamebus",
                        "task_id": "g1",
                        "name": "W",
                        "date": "2026-04-15",
                        "start": "09:00",
                        "end": "10:00",
                    },
                ]
            }
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
    assert len(response.json()["plannedActivities"]) == 1


def test_parse_iso_date_helpers_in_app():
    from src.agents.app import _parse_iso_date

    assert _parse_iso_date("") is None
    assert _parse_iso_date("not-a-date") is None
    assert _parse_iso_date("2026-04-15") is not None


def test_build_chroma_client_with_host(monkeypatch):
    monkeypatch.setenv("COACH_REQUIRE_MTLS", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setenv("CHROMA_HOST", "chroma-db")
    monkeypatch.setenv("CHROMA_PORT", "8080")

    import src.agents.app as app_module

    app_module = importlib.reload(app_module)
    called = {}

    def _fake_get_client(*args, **kwargs):
        called.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(app_module, "get_client", _fake_get_client)
    app_module._build_chroma_client()
    assert called["host"] == "chroma-db"
    assert called["port"] == 8080


def _stub_app(monkeypatch, **scheduler_kwargs):
    monkeypatch.setenv("COACH_REQUIRE_MTLS", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")

    import src.memory.embeddings as embeddings_module

    importlib.reload(embeddings_module)
    embeddings_module._embedding_function = object()
    embeddings_module._ready = True
    embeddings_module._loading = False

    import src.agents.app as app_module

    app_module = importlib.reload(app_module)

    class _Coll:
        def add(self, *a, **k):
            pass

        def get(self, *a, **k):
            return {"ids": [], "documents": [], "metadatas": []}

        def delete(self, *a, **k):
            pass

    class _Client:
        def get_or_create_collection(self, name, embedding_function=None):
            return _Coll()

        def get_collection(self, name):
            return _Coll()

        def delete_collection(self, name):
            pass

        def list_collections(self):
            return []

    monkeypatch.setattr(app_module, "_build_chroma_client", lambda: _Client())
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
        lambda self, scheduler_task_sources, **kwargs: scheduler_kwargs.get(
            "result", {"schedule_json": {"assignments": []}}
        ),
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
    return app_module


def _body():
    return {
        "playerId": 1,
        "campaignId": 1,
        "sessionId": "s1",
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


def test_two_chat_turns_on_same_session_return_200(monkeypatch):
    app_module = _stub_app(monkeypatch)
    client = TestClient(app_module.app)
    assert client.post("/chat", json=_body()).status_code == 200
    # The same thread is resumed on the next turn from the in-process checkpointer.
    assert client.post("/chat", json=_body()).status_code == 200


def test_pipeline_skips_expansion_when_schedule_dates_unparseable(monkeypatch):
    app_module = _stub_app(monkeypatch)
    # With unparseable window dates the build step skips routine expansion.
    monkeypatch.setattr("src.agents.coach_graph._parse_iso_date", lambda _value: None)
    assert TestClient(app_module.app).post("/chat", json=_body()).status_code == 200


def test_pipeline_swallows_normalizer_failure(monkeypatch):
    app_module = _stub_app(monkeypatch)

    def _boom(payload):
        raise RuntimeError("normalizer down")

    monkeypatch.setattr("src.agents.coach_graph.normalize_standardized_activities", _boom)
    resp = TestClient(app_module.app).post("/chat", json=_body())
    assert resp.status_code == 200
    assert resp.json()["interviewComplete"] is True


def test_pipeline_swallows_expander_failure(monkeypatch):
    app_module = _stub_app(monkeypatch)

    def _boom(payload, *, schedule_start, schedule_end):
        raise RuntimeError("expander down")

    monkeypatch.setattr("src.agents.coach_graph.expand_routines_to_assignments", _boom)
    assert TestClient(app_module.app).post("/chat", json=_body()).status_code == 200


def test_pipeline_skips_non_dict_merged_items(monkeypatch):
    app_module = _stub_app(
        monkeypatch,
        result={
            "schedule_json": {
                "assignments": [
                    "not-a-dict",
                    {
                        "type": "gamebus",
                        "task_id": "g1",
                        "name": "WALK",
                        "date": "2026-04-15",
                        "start": "09:00",
                        "end": "10:00",
                    },
                ]
            }
        },
    )
    monkeypatch.setattr(
        "src.agents.coach_graph.merge_assignments",
        lambda r, s: [
            "skip-me",
            {
                "type": "gamebus",
                "task_id": "g1",
                "name": "WALK",
                "date": "2026-04-15",
                "start": "09:00",
                "end": "10:00",
            },
        ],
    )
    resp = TestClient(app_module.app).post("/chat", json=_body())
    assert resp.status_code == 200
    assert len(resp.json()["plannedActivities"]) == 1
