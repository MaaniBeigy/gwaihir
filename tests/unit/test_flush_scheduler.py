"""Edge-case behaviour for `flush`: env parsing, error handling, scheduler startup."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from src.memory import flush


def _utc_iso(days_old: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()


class _BoomCollection:
    name = "coach_p1_c1"

    def get(self, include=None):
        raise RuntimeError("read failed")

    def delete(self, ids):
        raise AssertionError("should not be called")


class _DeleteFailsCollection:
    name = "coach_p1_c1"

    def __init__(self, records):
        self._records = records

    def get(self, include=None):
        return {
            "ids": [r["id"] for r in self._records],
            "metadatas": [r["metadata"] for r in self._records],
            "documents": ["x"] * len(self._records),
        }

    def delete(self, ids):
        raise RuntimeError("delete failed")


def test_retention_days_default_when_unset(monkeypatch):
    monkeypatch.delenv("COACH_MEMORY_RETENTION_DAYS", raising=False)
    assert flush._retention_days() == 7


def test_retention_days_clamps_to_minimum_one(monkeypatch):
    monkeypatch.setenv("COACH_MEMORY_RETENTION_DAYS", "-5")
    assert flush._retention_days() == 1


def test_retention_days_falls_back_on_invalid_value(monkeypatch):
    monkeypatch.setenv("COACH_MEMORY_RETENTION_DAYS", "not-a-number")
    assert flush._retention_days() == 7


def test_parse_created_at_handles_zulu_timestamp():
    parsed = flush._parse_created_at({"created_at": "2026-04-15T10:00:00Z"})
    assert parsed is not None
    assert parsed.year == 2026


def test_parse_created_at_handles_missing_field():
    assert flush._parse_created_at({"unrelated": "value"}) is None


def test_parse_created_at_returns_none_for_invalid_format():
    assert flush._parse_created_at({"created_at": "not-a-timestamp"}) is None


def test_parse_created_at_falls_back_to_timestamp_field():
    parsed = flush._parse_created_at({"timestamp": "2026-04-15T10:00:00+00:00"})
    assert parsed is not None


def test_flush_collection_records_error_when_get_fails():
    counts = flush.flush_collection(_BoomCollection(), older_than_days=7)
    assert counts["error"] == 1
    assert counts["deleted"] == 0


def test_flush_collection_records_error_when_delete_fails():
    expired = [{"id": "old", "metadata": {"created_at": _utc_iso(30)}}]
    col = _DeleteFailsCollection(expired)
    counts = flush.flush_collection(col, older_than_days=7)
    assert counts["error"] == 1


def test_flush_all_skips_non_coach_collections_and_errors_gracefully(monkeypatch):
    class _GoodCollection:
        def __init__(self, name):
            self.name = name
            self.deleted = []

        def get(self, include=None):
            return {"ids": ["a"], "metadatas": [{"created_at": _utc_iso(30)}], "documents": ["x"]}

        def delete(self, ids):
            self.deleted.extend(ids)

    class _Client:
        def __init__(self):
            self.coach = _GoodCollection("coach_p9_c9")
            self.legacy = _GoodCollection("interviewer")

        def list_collections(self):
            return [
                type("Desc", (), {"name": self.coach.name})(),
                type("Desc", (), {"name": self.legacy.name})(),
            ]

        def get_collection(self, name):
            if name == self.coach.name:
                return self.coach
            raise RuntimeError(f"unknown {name}")

    client = _Client()
    totals = flush.flush_all(client, older_than_days=7)
    assert totals["collections"] == 1
    assert totals["deleted"] == 1
    assert client.legacy.deleted == []


def test_flush_all_handles_get_collection_failure():
    class _Client:
        def list_collections(self):
            return [type("Desc", (), {"name": "coach_p1_c1"})()]

        def get_collection(self, name):
            raise RuntimeError("missing")

    totals = flush.flush_all(_Client(), older_than_days=7)
    assert totals["error"] >= 1


def test_flush_all_handles_dict_descriptor_format():
    class _Col:
        name = "coach_p1_c1"

        def get(self, include=None):
            return {"ids": [], "metadatas": [], "documents": []}

        def delete(self, ids):
            pass

    class _Client:
        def list_collections(self):
            return [{"name": "coach_p1_c1"}]

        def get_collection(self, name):
            return _Col()

    totals = flush.flush_all(_Client(), older_than_days=7)
    assert totals["collections"] == 1


def test_start_flush_scheduler_invokes_background_scheduler(monkeypatch):
    started = {"started": False}
    added_jobs = []

    class _StubScheduler:
        def __init__(self, timezone=None):
            self.timezone = timezone

        def add_job(self, fn, **kwargs):
            added_jobs.append({"fn": fn, **kwargs})

        def start(self):
            started["started"] = True

        def shutdown(self, wait=False):
            started["started"] = False

    monkeypatch.setattr(flush, "BackgroundScheduler", _StubScheduler)

    fake_client = object()
    scheduler = flush.start_flush_scheduler(lambda: fake_client)
    assert started["started"] is True
    assert added_jobs and added_jobs[0]["id"] == "coach_memory_flush"


def test_start_flush_scheduler_tick_handles_exceptions(monkeypatch):
    """The interval-job tick swallows exceptions and bumps the error counter."""
    captured = {"job": None}

    class _StubScheduler:
        def __init__(self, timezone=None):
            pass

        def add_job(self, fn, **kwargs):
            captured["job"] = fn

        def start(self):
            pass

        def shutdown(self, wait=False):
            pass

    monkeypatch.setattr(flush, "BackgroundScheduler", _StubScheduler)

    def _broken_client():
        raise RuntimeError("client unavailable")

    flush.start_flush_scheduler(_broken_client)
    # Manually invoke the tick; it must not raise even when the client provider does.
    captured["job"]()  # must not raise
