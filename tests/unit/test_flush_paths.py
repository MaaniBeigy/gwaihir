"""Defensive paths in `src.memory.flush`: non-dict metadata, naive timestamps,
and the successful scheduler tick."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from src.memory import flush

# --------------------------------------------------------------------------------------------
# flush_collection: non-dict metadata entries are counted as "kept"
# --------------------------------------------------------------------------------------------


class _StubCollection:
    name = "coach_p1_c1"

    def __init__(self, ids: List[str], metadatas: List[Any]):
        self._ids = list(ids)
        self._metadatas = list(metadatas)
        self.deleted_ids: List[str] = []

    def get(self, include=None):
        return {"ids": list(self._ids), "metadatas": list(self._metadatas), "documents": []}

    def delete(self, ids):
        self.deleted_ids.extend(ids)


def test_flush_collection_keeps_records_whose_metadata_is_not_a_dict():
    """Records whose metadata entry isn't a dict are counted in `kept` and
    never considered for deletion."""
    col = _StubCollection(
        ids=["a", "b", "c"],
        metadatas=["not-a-dict", None, 42],
    )
    counts = flush.flush_collection(col, older_than_days=7)
    assert counts == {"deleted": 0, "kept": 3, "error": 0}
    assert col.deleted_ids == []


# --------------------------------------------------------------------------------------------
# flush_collection: naive timestamps get treated as UTC before comparison
# --------------------------------------------------------------------------------------------


def test_flush_collection_treats_naive_timestamps_as_utc():
    """A `created_at` value without timezone information is interpreted as
    UTC; an entry old enough to be expired gets deleted."""
    naive_old = (datetime.now() - timedelta(days=30)).replace(microsecond=0).isoformat()
    col = _StubCollection(
        ids=["old-naive"],
        metadatas=[{"created_at": naive_old}],  # no Z suffix, no offset
    )
    counts = flush.flush_collection(col, older_than_days=7)
    assert counts == {"deleted": 1, "kept": 0, "error": 0}
    assert col.deleted_ids == ["old-naive"]


def test_flush_collection_keeps_naive_timestamp_inside_retention_window():
    """A naive timestamp inside the retention window is kept."""
    naive_recent = (datetime.now() - timedelta(hours=2)).replace(microsecond=0).isoformat()
    col = _StubCollection(
        ids=["recent-naive"],
        metadatas=[{"created_at": naive_recent}],
    )
    counts = flush.flush_collection(col, older_than_days=7)
    assert counts == {"deleted": 0, "kept": 1, "error": 0}
    assert col.deleted_ids == []


# --------------------------------------------------------------------------------------------
# start_flush_scheduler: successful tick invokes flush_all with the live client
# --------------------------------------------------------------------------------------------


class _StubScheduler:
    def __init__(self, timezone=None):
        self.timezone = timezone
        self.added_jobs: List[Dict[str, Any]] = []
        self.started = False

    def add_job(self, fn, **kwargs):
        self.added_jobs.append({"fn": fn, **kwargs})

    def start(self):
        self.started = True

    def shutdown(self, wait=False):
        self.started = False


def test_start_flush_scheduler_tick_invokes_flush_all_on_success(monkeypatch):
    """The interval-job tick resolves the client via `get_client_fn` and
    forwards it to `flush_all` together with the configured retention."""
    monkeypatch.setattr(flush, "BackgroundScheduler", _StubScheduler)
    monkeypatch.setenv("COACH_MEMORY_RETENTION_DAYS", "5")

    flush_calls: List[Dict[str, Any]] = []

    def _record_flush_all(client, *, older_than_days):
        flush_calls.append({"client": client, "older_than_days": older_than_days})
        return {"deleted": 0, "kept": 0, "error": 0, "collections": 0}

    monkeypatch.setattr(flush, "flush_all", _record_flush_all)

    fake_client = object()
    scheduler = flush.start_flush_scheduler(lambda: fake_client)

    # Drive the tick directly to exercise the success branch.
    tick_fn = scheduler.added_jobs[0]["fn"]
    tick_fn()

    assert flush_calls == [{"client": fake_client, "older_than_days": 5}]
