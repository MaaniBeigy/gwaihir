"""Periodic vector flush for coach Chroma collections."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from prometheus_client import Counter

from src.memory.collections import is_coach_collection_name

logger = logging.getLogger("coach.flush")

flush_vectors_total = Counter(
    "coach_memory_flush_vectors_total",
    "Vectors processed by the periodic coach memory flush.",
    labelnames=("status",),
)


_FLUSH_INTERVAL_HOURS = 24


def _retention_days() -> int:
    raw = os.getenv("COACH_MEMORY_RETENTION_DAYS", "7")
    try:
        days = int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid COACH_MEMORY_RETENTION_DAYS=%r; defaulting to 7", raw)
        days = 7
    return max(1, days)


def _parse_created_at(metadata: Dict[str, Any]) -> Optional[datetime]:
    raw = metadata.get("created_at") or metadata.get("timestamp")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def flush_collection(collection, *, older_than_days: int) -> Dict[str, int]:
    """Delete vectors older than `older_than_days`. Return `{deleted, kept, error}` counts."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    counts = {"deleted": 0, "kept": 0, "error": 0}

    try:
        result = collection.get(include=["metadatas"])
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "flush.collection_read_failed name=%s error=%s", getattr(collection, "name", "?"), exc
        )
        counts["error"] += 1
        flush_vectors_total.labels(status="error").inc()
        return counts

    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []
    expired_ids: list[str] = []

    for doc_id, metadata in zip(ids, metadatas):
        if not isinstance(metadata, dict):
            counts["kept"] += 1
            continue
        created_at = _parse_created_at(metadata)
        if created_at is None:
            counts["kept"] += 1
            continue
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if created_at < cutoff:
            expired_ids.append(doc_id)
        else:
            counts["kept"] += 1

    if expired_ids:
        try:
            collection.delete(ids=expired_ids)
            counts["deleted"] += len(expired_ids)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "flush.collection_delete_failed name=%s ids=%s error=%s",
                getattr(collection, "name", "?"),
                len(expired_ids),
                exc,
            )
            counts["error"] += len(expired_ids)

    flush_vectors_total.labels(status="deleted").inc(counts["deleted"])
    flush_vectors_total.labels(status="kept").inc(counts["kept"])
    flush_vectors_total.labels(status="error").inc(counts["error"])
    return counts


def flush_all(client, *, older_than_days: int) -> Dict[str, int]:
    """Walk every coach collection on `client` and flush expired vectors."""
    totals = {"deleted": 0, "kept": 0, "error": 0, "collections": 0}
    for col_descriptor in client.list_collections():
        name = getattr(col_descriptor, "name", None) or (
            col_descriptor.get("name") if isinstance(col_descriptor, dict) else None
        )
        if not name or not is_coach_collection_name(name):
            continue
        totals["collections"] += 1
        try:
            collection = client.get_collection(name=name)
        except Exception as exc:  # noqa: BLE001
            logger.exception("flush.get_collection_failed name=%s error=%s", name, exc)
            totals["error"] += 1
            flush_vectors_total.labels(status="error").inc()
            continue
        counts = flush_collection(collection, older_than_days=older_than_days)
        totals["deleted"] += counts["deleted"]
        totals["kept"] += counts["kept"]
        totals["error"] += counts["error"]
    logger.info("flush.completed totals=%s older_than_days=%s", totals, older_than_days)
    return totals


def start_flush_scheduler(get_client_fn) -> BackgroundScheduler:
    """Start a background APScheduler that runs `flush_all` every 24 hours.

    `get_client_fn` is a zero-arg callable returning a Chroma client; it is invoked on each tick.
    """
    scheduler = BackgroundScheduler(timezone="UTC")

    def _tick() -> None:
        try:
            client = get_client_fn()
            flush_all(client, older_than_days=_retention_days())
        except Exception as exc:  # noqa: BLE001
            logger.exception("flush.tick_failed error=%s", exc)
            flush_vectors_total.labels(status="error").inc()

    scheduler.add_job(
        _tick,
        trigger="interval",
        hours=_FLUSH_INTERVAL_HOURS,
        id="coach_memory_flush",
        replace_existing=True,
        next_run_time=datetime.now(timezone.utc) + timedelta(hours=_FLUSH_INTERVAL_HOURS),
    )
    scheduler.start()
    logger.info("flush.scheduler_started interval_hours=%s", _FLUSH_INTERVAL_HOURS)
    return scheduler
