"""Unit tests for the FastAPI endpoints in `src.agents.app`.

The lifespan is bypassed (no embedding warmup, no flush scheduler); all
external collaborators are mocked. `/chat`, `/reset`, `/admin/flush`,
`/health`, and `/metrics` are driven through `TestClient`.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_with_stubs(monkeypatch):
    monkeypatch.setenv("COACH_REQUIRE_MTLS", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setenv("CHROMA_PATH", "/tmp/coach-chroma-tests")

    import src.memory.embeddings as embeddings_module

    importlib.reload(embeddings_module)
    embeddings_module._embedding_function = object()
    embeddings_module._ready = True
    embeddings_module._loading = False

    class _StubCollection:
        def __init__(self, name):
            self.name = name
            self.records: List[Dict[str, Any]] = []

        def add(self, documents, ids=None, metadatas=None):
            for index, doc in enumerate(documents):
                metadata = metadatas[index] if metadatas else {}
                doc_id = ids[index] if ids else f"id-{len(self.records)}"
                self.records.append({"id": doc_id, "document": doc, "metadata": metadata})

        def get(self, where=None, include=None):
            results = self.records
            if where:
                key = next(iter(where))
                results = [r for r in results if r["metadata"].get(key) == where[key]]
            return {
                "ids": [r["id"] for r in results],
                "documents": [r["document"] for r in results],
                "metadatas": [r["metadata"] for r in results],
            }

        def delete(self, ids=None, where=None):
            if ids:
                self.records = [r for r in self.records if r["id"] not in ids]

    class _StubChromaClient:
        def __init__(self) -> None:
            self.collections: Dict[str, _StubCollection] = {}
            self.deleted: List[str] = []

        def get_or_create_collection(self, name, embedding_function=None):
            self.collections.setdefault(name, _StubCollection(name))
            return self.collections[name]

        def get_collection(self, name):
            return self.collections.setdefault(name, _StubCollection(name))

        def delete_collection(self, name):
            self.deleted.append(name)
            self.collections.pop(name, None)

        def list_collections(self):
            return [type("Descriptor", (), {"name": n})() for n in self.collections]

    chroma_client = _StubChromaClient()

    import src.agents.app as app_module

    app_module = importlib.reload(app_module)

    monkeypatch.setattr(app_module, "_build_chroma_client", lambda: chroma_client)
    monkeypatch.setattr(app_module.flush, "start_flush_scheduler", lambda *a, **k: None)
    monkeypatch.setattr(app_module.embeddings, "warmup_embedding_function", lambda *a, **k: None)
    monkeypatch.setattr(
        "src.agents.interviewer_core.InterviewerAgent._send_to_llm",
        lambda self, messages: "Q1: Pretend question",
    )
    monkeypatch.setattr("src.webhook.client.WebhookClient.emit", lambda self, **kwargs: True)

    return app_module, chroma_client


@pytest.fixture
def client(app_with_stubs):
    app_module, _ = app_with_stubs
    return TestClient(app_module.app)


def _chat_body(message: str = "hello", session_id: str = "s1"):
    return {
        "playerId": 1,
        "campaignId": 1,
        "sessionId": session_id,
        "message": message,
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


def test_health_returns_200_when_embedding_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "embedding_ready": True}


def test_health_returns_503_when_embedding_not_ready(app_with_stubs, client):
    import src.memory.embeddings as embeddings_module

    embeddings_module._ready = False
    try:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["embedding_ready"] is False
    finally:
        embeddings_module._ready = True


def test_chat_when_embedding_not_ready_returns_503(app_with_stubs, client):
    import src.memory.embeddings as embeddings_module

    embeddings_module._ready = False
    try:
        response = client.post("/chat", json=_chat_body())
        assert response.status_code == 503
    finally:
        embeddings_module._ready = True


def test_chat_returns_interviewing_until_completion_marker(client):
    response = client.post("/chat", json=_chat_body())
    assert response.status_code == 200
    payload = response.json()
    assert payload["interviewComplete"] is False
    assert payload["plannedActivities"] is None
    assert "Pretend" in payload["output"]


def test_chat_runs_post_interview_pipeline_on_completion(app_with_stubs, monkeypatch):
    """When the interviewer LLM signals completion, /chat triggers the pipeline."""
    app_module, _client = app_with_stubs

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
                        "challengeRuleId": 555,
                    }
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

    body = _chat_body()
    body["context"]["challenges"] = [
        {
            "challenge_id": 555,
            "name": "Walk",
            "type": "TASKS_COLLECTION",
            "linkToChallengeRules": [{"id": 555, "name": "Walk", "duration": 60}],
        }
    ]

    response = TestClient(app_module.app).post("/chat", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["interviewComplete"] is True
    assert payload["plannedActivities"]
    assert payload["plannedActivities"][0]["challengeRuleId"] == 555


def test_chat_pipeline_errors_become_500(app_with_stubs, monkeypatch):
    app_module, _ = app_with_stubs
    monkeypatch.setattr(
        "src.agents.interviewer_core.InterviewerAgent._send_to_llm",
        lambda self, messages: "Thank you for your time. The interview is complete.",
    )

    def _boom(self, message, clear_history=False):
        raise RuntimeError("standardizer down")

    monkeypatch.setattr("src.agents.standardizer_core.StandardizerAgent.standardize", _boom)

    response = TestClient(app_module.app).post("/chat", json=_chat_body())
    assert response.status_code == 500
    assert "pipeline failed" in response.json()["detail"]


def test_chat_validates_request_body_shape(client):
    response = client.post("/chat", json={"playerId": 1})
    assert response.status_code == 422


def test_reset_deletes_collection_and_session_state(app_with_stubs, client):
    app_module, chroma_client = app_with_stubs
    client.post("/chat", json=_chat_body(session_id="reset-me"))

    response = client.post(
        "/reset",
        json={"playerId": 1, "campaignId": 1, "sessionId": "reset-me"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["deleted_collection"] == "coach_p1_c1"
    assert "coach_p1_c1" in chroma_client.deleted


def test_reset_is_resilient_to_delete_failure(app_with_stubs, client, monkeypatch):
    app_module, chroma_client = app_with_stubs

    def _boom(name):
        raise RuntimeError("missing")

    monkeypatch.setattr(chroma_client, "delete_collection", _boom)
    response = client.post(
        "/reset",
        json={"playerId": 1, "campaignId": 1, "sessionId": "missing-session"},
    )
    assert response.status_code == 200
    assert response.json()["deleted_collection"] == "coach_p1_c1"


def test_admin_flush_returns_totals(app_with_stubs, client, monkeypatch):
    app_module, _ = app_with_stubs
    monkeypatch.setattr(
        app_module.flush,
        "flush_all",
        lambda client, older_than_days: {"deleted": 4, "kept": 2, "error": 0, "collections": 1},
    )

    response = client.post("/admin/flush?older_than_days=14")
    assert response.status_code == 200
    payload = response.json()
    assert payload["older_than_days"] == 14
    assert payload["totals"]["deleted"] == 4


def test_admin_flush_rejects_bad_input(client):
    response = client.post("/admin/flush?older_than_days=0")
    assert response.status_code == 400


def test_metrics_endpoint_serves_prometheus_text(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_authorise_blocks_when_mtls_required(monkeypatch):
    monkeypatch.setenv("COACH_REQUIRE_MTLS", "true")
    monkeypatch.setenv("COACH_INTERNAL_TOKEN", "secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")

    import src.memory.embeddings as embeddings_module

    embeddings_module._ready = True

    import src.agents.app as app_module

    app_module = importlib.reload(app_module)
    monkeypatch.setattr(app_module.flush, "start_flush_scheduler", lambda *a, **k: None)
    monkeypatch.setattr(app_module.embeddings, "warmup_embedding_function", lambda *a, **k: None)

    client = TestClient(app_module.app)
    response = client.post("/chat", json=_chat_body())
    assert response.status_code == 403


def test_assignment_to_planned_activity_handles_iso_timestamps():
    from datetime import date

    from src.agents.app import _assignment_to_planned_activity

    planned = _assignment_to_planned_activity(
        {
            "type": "gamebus",
            "task_id": "g1",
            "name": "WALK",
            "date": "2026-04-15",
            "start": "09:00:00",
            "end": "10:00:00",
        },
        schedule_start=date(2026, 4, 13),
        schedule_end=date(2026, 4, 19),
    )
    assert planned is not None
    assert planned.startDate == "2026-04-15T09:00:00"


def test_assignment_to_planned_activity_drops_unparseable_date():
    from src.agents.app import _assignment_to_planned_activity

    planned = _assignment_to_planned_activity(
        {"task_id": "g1", "date": "not-a-date", "start": "09:00"},
        schedule_start=None,
        schedule_end=None,
    )
    assert planned is None


def test_assignment_to_planned_activity_handles_missing_end():
    from datetime import date

    from src.agents.app import _assignment_to_planned_activity

    planned = _assignment_to_planned_activity(
        {"task_id": "g1", "date": "2026-04-15", "start": "09:00"},
        schedule_start=date(2026, 4, 13),
        schedule_end=date(2026, 4, 19),
    )
    assert planned is not None
    assert planned.startDate == planned.endDate
