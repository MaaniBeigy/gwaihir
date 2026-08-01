"""End-to-end controller loop through the FastAPI app and the LangGraph flow.

Drives `/chat` to completion with a controller that rejects the first schedule
and accepts the second, and asserts the scheduler is re-run with the feedback.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def coach_app(monkeypatch):
    monkeypatch.setenv("COACH_REQUIRE_MTLS", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setenv("CHROMA_PATH", "/tmp/coach-chroma-loop")

    import src.memory.embeddings as embeddings_module

    importlib.reload(embeddings_module)
    embeddings_module._embedding_function = object()
    embeddings_module._ready = True
    embeddings_module._loading = False

    class _StubCollection:
        def __init__(self, name):
            self.name = name

        def add(self, documents, ids=None, metadatas=None):
            pass

        def get(self, where=None, include=None):
            return {"ids": [], "documents": [], "metadatas": []}

        def delete(self, ids=None, where=None):
            pass

    class _StubChromaClient:
        def get_or_create_collection(self, name, embedding_function=None):
            return _StubCollection(name)

        def delete_collection(self, name):
            pass

    import src.agents.app as app_module

    app_module = importlib.reload(app_module)
    monkeypatch.setattr(app_module, "_build_chroma_client", lambda: _StubChromaClient())
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

    schedule_calls: List[Dict[str, Any]] = []

    def _schedule(self, scheduler_task_sources, **kwargs):
        schedule_calls.append(kwargs)
        return {
            "schedule_json": {
                "assignments": [
                    {
                        "type": "gamebus",
                        "task_id": "g1",
                        "name": "WALK",
                        "date": "2026-06-01",
                        "start": "09:00",
                        "end": "10:00",
                    }
                ]
            }
        }

    monkeypatch.setattr("src.agents.scheduler_core.SchedulingAgent.schedule_gamebus", _schedule)

    validate_calls = {"n": 0}

    def _validate(self, **kwargs):
        validate_calls["n"] += 1
        return {
            "is_valid": validate_calls["n"] >= 2,
            "violations": ["task_id g1 starts before wake_up_end"],
            "fix_advice": ["move g1 to 10:00 on 2026-06-01"],
        }

    monkeypatch.setattr("src.agents.controller_core.ControllerAgent.validate_schedule", _validate)

    webhook_phases: List[str] = []

    def _emit(self, **kwargs):
        webhook_phases.append(kwargs["phase"])
        return True

    monkeypatch.setattr("src.webhook.client.WebhookClient.emit", _emit)

    return app_module, schedule_calls, webhook_phases


def _body():
    return {
        "playerId": 7,
        "campaignId": 3,
        "sessionId": "sess-loop",
        "message": "",
        "context": {
            "challenges": [],
            "scheduledActivities": [],
            "schedulingWindow": {
                "scheduleStartDate": "2026-06-01",
                "scheduleEndDate": "2026-06-07",
                "schedulingPeriodWeeks": 1,
                "currentDate": "2026-06-01",
                "currentDayname": "mon",
            },
        },
    }


def test_controller_loop_reschedules_with_feedback(coach_app):
    from fastapi.testclient import TestClient

    app_module, schedule_calls, webhook_phases = coach_app
    response = TestClient(app_module.app).post("/chat", json=_body())

    assert response.status_code == 200
    payload = response.json()
    assert payload["interviewComplete"] is True
    assert payload["plannedActivities"]

    # The controller rejected the first plan, so the scheduler ran twice.
    assert len(schedule_calls) == 2
    # The second run received the controller violations and fix advice as feedback.
    assert schedule_calls[1]["controller_feedback"]

    assert "INTERVIEWING" in webhook_phases
    assert "SCHEDULING" in webhook_phases
    assert "COMPLETE" in webhook_phases
