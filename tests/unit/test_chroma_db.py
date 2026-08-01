"""Unit tests for the thin ChromaDB helpers."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.agents.ChromaDB import chroma_db


class _RecordingCollection:
    def __init__(self):
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self._stored_records: List[Dict[str, Any]] = []
        self._where = None
        self._include = None

    def add(self, documents, metadatas, ids):
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            self._stored_records.append({"id": doc_id, "document": doc, "metadata": meta})

    def get(self, where=None, include=None):
        self._where = where
        self._include = include
        records = self._stored_records
        if where:
            key = next(iter(where))
            records = [r for r in records if r["metadata"].get(key) == where[key]]
        return {
            "ids": [r["id"] for r in records],
            "documents": [r["document"] for r in records],
            "metadatas": [r["metadata"] for r in records],
        }


def test_store_qa_includes_question_number_and_section():
    col = _RecordingCollection()
    doc_id = chroma_db.store_qa(
        col,
        question="When do you wake up?",
        answer="07:00",
        question_number="Q1",
        section="sleep_eat_routines",
    )
    assert doc_id.startswith("qa_")
    metadata = col.metadatas[0]
    assert metadata["question"] == "When do you wake up?"
    assert metadata["answer"] == "07:00"
    assert metadata["question_number"] == "Q1"
    assert metadata["section"] == "sleep_eat_routines"
    assert metadata["type"] == "qa"
    assert "created_at" in metadata


def test_store_qa_omits_optional_fields_when_unset():
    col = _RecordingCollection()
    chroma_db.store_qa(col, question="q", answer="a")
    metadata = col.metadatas[0]
    assert "question_number" not in metadata
    assert "section" not in metadata


def test_store_cleaned_conversation_merges_metadata():
    col = _RecordingCollection()
    chroma_db.store_cleaned_conversation(col, "cleaned text", metadata={"session_id": "s1"})
    metadata = col.metadatas[0]
    assert metadata["type"] == "cleaned_conversation"
    assert metadata["session_id"] == "s1"


def test_store_standardized_activities_merges_metadata():
    col = _RecordingCollection()
    chroma_db.store_standardized_activities(col, '{"Activities":[]}', metadata={"foo": "bar"})
    assert col.metadatas[0]["type"] == "standardized_activities"
    assert col.metadatas[0]["foo"] == "bar"


def test_store_scheduled_gamebus_merges_metadata():
    col = _RecordingCollection()
    chroma_db.store_scheduled_gamebus(col, '{"assignments":[]}', metadata={"x": 1})
    assert col.metadatas[0]["type"] == "scheduled_gamebus"
    assert col.metadatas[0]["x"] == 1


def test_store_scheduled_gamebus_without_metadata_uses_only_base_fields():
    """Without an explicit `metadata` argument the merge step is skipped and
    only the base metadata (type/timestamp/created_at) is written."""
    col = _RecordingCollection()
    chroma_db.store_scheduled_gamebus(col, '{"assignments":[]}')
    metadata = col.metadatas[0]
    assert metadata["type"] == "scheduled_gamebus"
    assert set(metadata.keys()) == {"type", "timestamp", "created_at"}


def test_store_scheduler_task_sources_merges_metadata():
    col = _RecordingCollection()
    chroma_db.store_scheduler_task_sources(col, '{"all_tasks":[]}', metadata={"x": 1})
    assert col.metadatas[0]["type"] == "scheduler_task_sources"


def test_store_scheduler_task_sources_without_metadata_uses_only_base_fields():
    """Without an explicit `metadata` argument the merge step is skipped and
    only the base metadata (type/timestamp/created_at) is written."""
    col = _RecordingCollection()
    chroma_db.store_scheduler_task_sources(col, '{"all_tasks":[]}')
    metadata = col.metadatas[0]
    assert metadata["type"] == "scheduler_task_sources"
    assert set(metadata.keys()) == {"type", "timestamp", "created_at"}


def test_get_processed_data_filters_by_type():
    col = _RecordingCollection()
    chroma_db.store_cleaned_conversation(col, "a")
    chroma_db.store_standardized_activities(col, '{"Activities":[]}')
    result = chroma_db.get_processed_data(col, data_type="standardized_activities")
    assert col._where == {"type": "standardized_activities"}
    assert len(result["documents"]) == 1


def test_get_processed_data_without_type_returns_all():
    col = _RecordingCollection()
    chroma_db.store_cleaned_conversation(col, "a")
    chroma_db.store_standardized_activities(col, '{"Activities":[]}')
    result = chroma_db.get_processed_data(col)
    assert col._where is None
    assert len(result["documents"]) == 2


def test_get_or_create_collection_requires_name():
    class _StubClient:
        def get_or_create_collection(self, **_):
            raise AssertionError("should not be called")

    with pytest.raises(RuntimeError, match="collection name"):
        chroma_db.get_or_create_collection(_StubClient(), "")


def test_get_or_create_collection_passes_embedding_function():
    captured: Dict[str, Any] = {}

    class _StubClient:
        def get_or_create_collection(self, **kwargs):
            captured.update(kwargs)
            return "stub"

    embedding_function = object()
    result = chroma_db.get_or_create_collection(
        _StubClient(), "coach_p1_c1", embedding_function=embedding_function
    )
    assert result == "stub"
    assert captured["name"] == "coach_p1_c1"
    assert captured["embedding_function"] is embedding_function


def test_get_or_create_collection_skips_embedding_when_none():
    captured: Dict[str, Any] = {}

    class _StubClient:
        def get_or_create_collection(self, **kwargs):
            captured.update(kwargs)
            return "stub"

    chroma_db.get_or_create_collection(_StubClient(), "coach_p1_c1")
    assert "embedding_function" not in captured


def test_get_client_requires_host_or_path():
    with pytest.raises(RuntimeError, match="host=... or path=..."):
        chroma_db.get_client()


def test_get_client_uses_http_when_host_provided(monkeypatch):
    captured = {}

    class _StubHttp:
        def __init__(self, host, port):
            captured["http"] = (host, port)

    class _StubPersistent:
        def __init__(self, path):
            captured["persistent"] = path

    monkeypatch.setattr(chroma_db.chromadb, "HttpClient", _StubHttp)
    monkeypatch.setattr(chroma_db.chromadb, "PersistentClient", _StubPersistent)

    chroma_db.get_client(host="chroma-db", port=8000)
    assert captured["http"] == ("chroma-db", 8000)
    assert "persistent" not in captured


def test_get_client_uses_persistent_when_path_provided(monkeypatch):
    captured = {}

    class _StubHttp:
        def __init__(self, host, port):
            captured["http"] = (host, port)

    class _StubPersistent:
        def __init__(self, path):
            captured["persistent"] = path

    monkeypatch.setattr(chroma_db.chromadb, "HttpClient", _StubHttp)
    monkeypatch.setattr(chroma_db.chromadb, "PersistentClient", _StubPersistent)

    chroma_db.get_client(path="/tmp/db")
    assert captured["persistent"] == "/tmp/db"
    assert "http" not in captured


def test_retrieve_all_history_proxies_to_collection_get():
    class _StubCol:
        def get(self):
            return {"documents": ["doc"], "metadatas": [{}]}

    result = chroma_db.retrieve_all_history(_StubCol())
    assert result["documents"] == ["doc"]
