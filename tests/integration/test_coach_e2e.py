"""End-to-end coach pipeline test with stubbed LLM and webhook target.

Walks a multi-turn interview through the FastAPI app, asserts that the
post-interview pipeline returns plannedActivities and that progress webhook
events are emitted in the right order.
"""

from __future__ import annotations

import importlib
import json
from typing import Any, Dict, List

import pytest


@pytest.fixture
def coach_app(monkeypatch):
    monkeypatch.setenv("COACH_REQUIRE_MTLS", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setenv("CHROMA_PATH", "/tmp/coach-chroma-e2e")

    import src.memory.embeddings as embeddings_module

    embeddings_module = importlib.reload(embeddings_module)

    class _StubEmbeddingFunction:
        def __call__(self, texts: List[str]) -> List[List[float]]:
            return [[0.0] for _ in texts]

        def name(self) -> str:
            return "stub"

    embeddings_module._embedding_function = _StubEmbeddingFunction()
    embeddings_module._ready = True
    embeddings_module._loading = False

    class _StubCollection:
        def __init__(self, name: str):
            self.name = name
            self.records: List[Dict[str, Any]] = []

        def add(self, documents, ids=None, metadatas=None):
            for index, doc in enumerate(documents):
                metadata = metadatas[index] if metadatas and index < len(metadatas) else {}
                doc_id = ids[index] if ids and index < len(ids) else f"id-{len(self.records)}"
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
            elif where:
                key = next(iter(where))
                self.records = [r for r in self.records if r["metadata"].get(key) != where[key]]

    class _StubChromaClient:
        def __init__(self) -> None:
            self.collections: Dict[str, _StubCollection] = {}

        def get_or_create_collection(self, name, embedding_function=None):
            self.collections.setdefault(name, _StubCollection(name))
            return self.collections[name]

        def get_collection(self, name):
            return self.collections[name]

        def delete_collection(self, name):
            self.collections.pop(name, None)

        def list_collections(self):
            return [type("Descriptor", (), {"name": n})() for n in self.collections]

    chroma_client = _StubChromaClient()

    import src.agents.app as app_module

    app_module = importlib.reload(app_module)

    monkeypatch.setattr(app_module, "_build_chroma_client", lambda: chroma_client)
    monkeypatch.setattr(app_module.flush, "start_flush_scheduler", lambda *_args, **_kwargs: None)

    interviewer_replies = iter(
        [
            "Q1: What time do you wake up?",
            "Q2: Any morning workouts?",
            "Thank you for your time. The interview is complete.",
        ]
    )

    def _stub_send_to_llm(self, messages):
        return next(interviewer_replies)

    monkeypatch.setattr(
        "src.agents.interviewer_core.InterviewerAgent._send_to_llm",
        _stub_send_to_llm,
    )

    standardized_payload = {
        "Activities": [
            {
                "name": "Morning walk",
                "description": "Morning walk every weekday at 07:00 for 30 minutes",
                "IsInstantaneous": False,
                "IsRegular": True,
                "frequency": {
                    "type": "specific_days",
                    "days": ["mon", "tue", "wed", "thu", "fri"],
                    "count": 1,
                },
                "IsRepetitive": True,
                "activity_duration": 30,
                "start_date": "2026-04-13",
                "IsArbitrary": False,
                "time": {"preferred": "07:00"},
                "notes": None,
            }
        ]
    }

    def _stub_standardize(self, message, clear_history=False):
        return json.dumps(standardized_payload)

    monkeypatch.setattr(
        "src.agents.standardizer_core.StandardizerAgent.standardize",
        _stub_standardize,
    )

    def _stub_schedule(self, scheduler_task_sources, **kwargs):
        return {
            "schedule_json": {
                "assignments": [
                    {
                        "type": "gamebus",
                        "task_id": "g1",
                        "name": "WALK",
                        "date": "2026-04-15",
                        "start": "09:00",
                        "end": "10:00",
                        "challengeRuleId": 123,
                    }
                ]
            },
            "assignments": [],
        }

    monkeypatch.setattr(
        "src.agents.scheduler_core.SchedulingAgent.schedule_gamebus",
        _stub_schedule,
    )

    monkeypatch.setattr(
        "src.agents.controller_core.ControllerAgent.validate_schedule",
        lambda self, **_kwargs: {
            "is_valid": True,
            "violations": [],
            "fix_advice": [],
            "summary": {},
            "llm_used": False,
        },
    )

    webhook_calls: List[Dict[str, Any]] = []

    def _stub_emit(self, **kwargs):
        webhook_calls.append(kwargs)
        return True

    monkeypatch.setattr("src.webhook.client.WebhookClient.emit", _stub_emit)

    return app_module, webhook_calls


def test_e2e_interview_returns_planned_activities(coach_app):
    app_module, webhook_calls = coach_app
    from fastapi.testclient import TestClient

    client = TestClient(app_module.app)

    body = {
        "playerId": 1,
        "campaignId": 2,
        "sessionId": "sess-1",
        "message": "",
        "context": {
            "challenges": [
                {
                    "challenge_id": 99,
                    "name": "Walk W1",
                    "description": "Walk this week",
                    "type": "TASKS_COLLECTION",
                    "linkToChallengeRules": [{"id": 123, "name": "Take a 30-minute walk", "duration": 30}],
                }
            ],
            "scheduledActivities": [
                {
                    "startDate": "2026-04-15T08:00:00Z",
                    "endDate": "2026-04-15T09:00:00Z",
                    "activityType": "WORK_MEETING",
                    "source": "USER",
                }
            ],
            "schedulingWindow": {
                "scheduleStartDate": "2026-04-13",
                "scheduleEndDate": "2026-04-19",
                "schedulingPeriodWeeks": 1,
                "currentDate": "2026-04-13",
                "currentDayname": "mon",
            },
        },
    }

    # Turn 1: kick off the interview.
    response = client.post("/chat", json=body)
    assert response.status_code == 200
    assert response.json()["interviewComplete"] is False

    # Turn 2: continue.
    body["message"] = "I wake up at 7"
    response = client.post("/chat", json=body)
    assert response.status_code == 200
    assert response.json()["interviewComplete"] is False

    # Turn 3: completion marker triggers the post-interview pipeline.
    body["message"] = "Yes I do morning walks"
    response = client.post("/chat", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["interviewComplete"] is True
    assert payload["plannedActivities"]
    for activity in payload["plannedActivities"]:
        assert activity["gameDescriptorTK"] == "SCHEDULE_ACTIVITY"
    # The pipeline now combines deterministic routine expansion (Morning walk
    # on Mon-Fri at 07:00) with the scheduler's gamebus entry (2026-04-15 at 09:00).
    starts = [a["startDate"] for a in payload["plannedActivities"]]
    assert any(s.startswith("2026-04-15T09:00") for s in starts)
    assert any(s.startswith("2026-04-13T07:00") for s in starts)

    phases = [call["phase"] for call in webhook_calls]
    assert "INTERVIEWING" in phases
    assert "SCHEDULING" in phases
    assert "COMPLETE" in phases
