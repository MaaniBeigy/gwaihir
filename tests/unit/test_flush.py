"""Periodic memory flush behaviour."""

from datetime import datetime, timedelta, timezone

from src.memory.collections import is_coach_collection_name
from src.memory.flush import flush_all, flush_collection


class _StubCollection:
    def __init__(self, name: str, records):
        self.name = name
        self._records = list(records)
        self.deleted_ids: list[str] = []

    def get(self, include=None):
        ids = [rec["id"] for rec in self._records]
        metadatas = [rec["metadata"] for rec in self._records]
        return {"ids": ids, "metadatas": metadatas, "documents": ["x"] * len(ids)}

    def delete(self, ids):
        self.deleted_ids.extend(ids)
        self._records = [rec for rec in self._records if rec["id"] not in ids]


class _StubClient:
    def __init__(self, collections):
        self._collections = collections

    def list_collections(self):
        return [type("Descriptor", (), {"name": col.name})() for col in self._collections]

    def get_collection(self, name):
        for col in self._collections:
            if col.name == name:
                return col
        raise KeyError(name)


def _record(doc_id: str, age_days: float):
    created = datetime.now(timezone.utc) - timedelta(days=age_days)
    return {"id": doc_id, "metadata": {"created_at": created.isoformat()}}


def test_flush_collection_deletes_only_expired_records():
    col = _StubCollection(
        "coach_p1_c1",
        [
            _record("old", 30),
            _record("recent", 1),
            _record("borderline", 5),
        ],
    )
    counts = flush_collection(col, older_than_days=7)
    assert counts == {"deleted": 1, "kept": 2, "error": 0}
    assert col.deleted_ids == ["old"]


def test_flush_collection_keeps_records_without_created_at():
    col = _StubCollection(
        "coach_p1_c1",
        [
            {"id": "no-meta", "metadata": {"foo": "bar"}},
            _record("old", 30),
        ],
    )
    counts = flush_collection(col, older_than_days=7)
    assert counts["deleted"] == 1
    assert counts["kept"] == 1
    assert col.deleted_ids == ["old"]


def test_flush_all_only_walks_coach_collections():
    coach_col = _StubCollection("coach_p1_c1", [_record("expired", 30)])
    other_col = _StubCollection("interviewer", [_record("expired", 30)])
    client = _StubClient([coach_col, other_col])

    totals = flush_all(client, older_than_days=7)
    assert totals["collections"] == 1
    assert totals["deleted"] == 1
    assert other_col.deleted_ids == []  # untouched
    assert is_coach_collection_name(coach_col.name)
