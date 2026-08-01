"""A work routine collapsed to two days is restored to the full work week.

Guards the standardize, normalize, expand, and merge chain against a regression
where a "Monday to Friday" work block reached the calendar as only Mon and Fri.
"""

from __future__ import annotations

import importlib
import json
from datetime import date, datetime
from typing import Any, Dict, List

import pytest

from src.agents.tools.routine_expander import expand_routines_to_assignments, merge_assignments
from src.agents.tools.standardized_normalizer import normalize_standardized_activities

pytestmark = pytest.mark.integration

_WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _weekday(iso_date: str) -> str:
    return _WEEKDAYS[datetime.strptime(iso_date, "%Y-%m-%d").weekday()]


def _buggy_standardized() -> Dict[str, Any]:
    """Standardizer output with work truncated to ["mon","fri"] and a single-day gym."""
    return {
        "Activities": [
            {
                "name": "work",
                "description": "I usually start working at 08:00 and finish at 17:00 on workdays.",
                "IsInstantaneous": False,
                "IsRegular": True,
                "frequency": {"type": "specific_days", "days": ["mon", "fri"], "count": 1},
                "IsRepetitive": True,
                "activity_duration": 540,
                "start_date": "2026-06-01",
                "IsArbitrary": False,
                "time": {"preferred": "08:00"},
                "notes": "workday schedule applies Monday to Friday",
            },
            {
                "name": "go to gym",
                "description": "I go to the gym on Tuesdays at 19:00 for 2 hours.",
                "IsInstantaneous": False,
                "IsRegular": True,
                "frequency": {"type": "specific_days", "days": ["tue"], "count": 1},
                "IsRepetitive": True,
                "activity_duration": 120,
                "start_date": "2026-06-02",
                "IsArbitrary": False,
                "time": {"preferred": "19:00"},
                "notes": None,
            },
        ]
    }


def test_deterministic_chain_restores_full_work_week() -> None:
    normalized = normalize_standardized_activities(_buggy_standardized())

    routine = expand_routines_to_assignments(
        normalized, schedule_start=date(2026, 6, 1), schedule_end=date(2026, 6, 7)
    )
    # A scheduler that only placed work on Mon and Fri must not constrain the result.
    scheduler = [
        {"name": "work", "type": "routine", "date": "2026-06-01", "start": "08:00", "end": "17:00"},
        {"name": "work", "type": "routine", "date": "2026-06-05", "start": "08:00", "end": "17:00"},
    ]
    merged = merge_assignments(routine, scheduler)

    work_days = sorted({_weekday(a["date"]) for a in merged if a["name"] == "work"})
    assert work_days == ["fri", "mon", "thu", "tue", "wed"]

    gym_days = sorted({_weekday(a["date"]) for a in merged if a["name"] == "go to gym"})
    assert gym_days == ["tue"]


@pytest.fixture
def coach_app(monkeypatch):
    monkeypatch.setenv("COACH_REQUIRE_MTLS", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setenv("CHROMA_PATH", "/tmp/coach-chroma-workdays")

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

    class _StubChromaClient:
        def __init__(self) -> None:
            self.collections: Dict[str, _StubCollection] = {}

        def get_or_create_collection(self, name, embedding_function=None):
            self.collections.setdefault(name, _StubCollection(name))
            return self.collections[name]

        def delete_collection(self, name):
            self.collections.pop(name, None)

    chroma_client = _StubChromaClient()

    import src.agents.app as app_module

    app_module = importlib.reload(app_module)

    monkeypatch.setattr(app_module, "_build_chroma_client", lambda: chroma_client)
    monkeypatch.setattr(app_module.flush, "start_flush_scheduler", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "src.agents.interviewer_core.InterviewerAgent._send_to_llm",
        lambda self, messages: "Interview completed.",
    )
    monkeypatch.setattr(
        "src.agents.standardizer_core.StandardizerAgent.standardize",
        lambda self, message, clear_history=False: json.dumps(_buggy_standardized()),
    )
    # The scheduler only places work on Mon and Fri; the fix must still yield five days.
    monkeypatch.setattr(
        "src.agents.scheduler_core.SchedulingAgent.schedule_gamebus",
        lambda self, scheduler_task_sources, **kwargs: {
            "schedule_json": {
                "assignments": [
                    {
                        "type": "routine",
                        "name": "work",
                        "date": "2026-06-01",
                        "start": "08:00",
                        "end": "17:00",
                    },
                    {
                        "type": "routine",
                        "name": "work",
                        "date": "2026-06-05",
                        "start": "08:00",
                        "end": "17:00",
                    },
                ]
            },
            "assignments": [],
        },
    )
    monkeypatch.setattr(
        "src.agents.controller_core.ControllerAgent.validate_schedule",
        lambda self, **_k: {
            "is_valid": True,
            "violations": [],
            "fix_advice": [],
            "summary": {},
            "llm_used": False,
        },
    )
    monkeypatch.setattr("src.webhook.client.WebhookClient.emit", lambda self, **_k: True)

    return app_module


def test_app_pipeline_returns_work_on_all_weekdays(coach_app):
    from fastapi.testclient import TestClient

    client = TestClient(coach_app.app)
    body = {
        "playerId": 1881,
        "campaignId": 1,
        "sessionId": "sess-workdays",
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

    response = client.post("/chat", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["interviewComplete"] is True

    work_dates = sorted(
        a["startDate"][:10] for a in payload["plannedActivities"] if a["activityType"] == "work"
    )
    assert work_dates == ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05"]
