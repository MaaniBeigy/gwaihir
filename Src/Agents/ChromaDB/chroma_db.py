"""Thin ChromaDB helpers used by the coach agents."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import chromadb


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_client(host: Optional[str] = None, port: Optional[int] = None, path: Optional[str] = None):
    """Return an HTTP Chroma client when `host` is set; otherwise a persistent client at `path`."""
    if host:
        return chromadb.HttpClient(host=host, port=int(port or 8000))
    if not path:
        raise RuntimeError("get_client requires either host=... or path=...")
    return chromadb.PersistentClient(path=path)


def get_or_create_collection(client, name: str, embedding_function=None):
    """Return a Chroma collection by name, creating it if needed."""
    if not name:
        raise RuntimeError("collection name is required")
    kwargs = {"name": name}
    if embedding_function is not None:
        kwargs["embedding_function"] = embedding_function
    return client.get_or_create_collection(**kwargs)


def store_qa(
    collection,
    question: str,
    answer: str,
    question_number: Optional[str] = None,
    section: Optional[str] = None,
) -> str:
    timestamp = _utcnow_iso()
    doc_id = f"qa_{timestamp}"
    document = f"Q: {question}\nA: {answer}"
    metadata: Dict[str, Any] = {
        "type": "qa",
        "timestamp": timestamp,
        "created_at": timestamp,
        "question": question,
        "answer": answer,
    }
    if question_number:
        metadata["question_number"] = question_number
    if section:
        metadata["section"] = section
    collection.add(documents=[document], metadatas=[metadata], ids=[doc_id])
    return doc_id


def retrieve_all_history(collection):
    return collection.get()


def store_cleaned_conversation(
    collection,
    cleaned_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    timestamp = _utcnow_iso()
    doc_id = f"cleaned_{timestamp}"
    base_metadata: Dict[str, Any] = {
        "type": "cleaned_conversation",
        "timestamp": timestamp,
        "created_at": timestamp,
    }
    if metadata:
        base_metadata.update(metadata)
    collection.add(documents=[cleaned_text], metadatas=[base_metadata], ids=[doc_id])
    return doc_id


def store_standardized_activities(
    collection,
    activities_json: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    timestamp = _utcnow_iso()
    doc_id = f"standardized_{timestamp}"
    base_metadata: Dict[str, Any] = {
        "type": "standardized_activities",
        "timestamp": timestamp,
        "created_at": timestamp,
    }
    if metadata:
        base_metadata.update(metadata)
    collection.add(documents=[activities_json], metadatas=[base_metadata], ids=[doc_id])
    return doc_id


def store_scheduled_gamebus(
    collection,
    schedule_json: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    timestamp = _utcnow_iso()
    doc_id = f"scheduled_gamebus_{timestamp}"
    base_metadata: Dict[str, Any] = {
        "type": "scheduled_gamebus",
        "timestamp": timestamp,
        "created_at": timestamp,
    }
    if metadata:
        base_metadata.update(metadata)
    collection.add(documents=[schedule_json], metadatas=[base_metadata], ids=[doc_id])
    return doc_id


def store_scheduler_task_sources(
    collection,
    scheduler_task_sources_json: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    timestamp = _utcnow_iso()
    doc_id = f"scheduler_task_sources_{timestamp}"
    base_metadata: Dict[str, Any] = {
        "type": "scheduler_task_sources",
        "timestamp": timestamp,
        "created_at": timestamp,
    }
    if metadata:
        base_metadata.update(metadata)
    collection.add(documents=[scheduler_task_sources_json], metadatas=[base_metadata], ids=[doc_id])
    return doc_id


def get_processed_data(collection, data_type: Optional[str] = None) -> Dict[str, Any]:
    """Return documents from the collection, optionally filtered by metadata `type`."""
    where = {"type": data_type} if data_type else None
    if where:
        return collection.get(where=where, include=["documents", "metadatas"])
    return collection.get(include=["documents", "metadatas"])
