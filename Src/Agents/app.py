"""coach-api FastAPI app exposing `/chat`, `/reset`, `/health`, `/admin/flush`."""

from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.agents.ChromaDB.chroma_db import get_client, get_or_create_collection
from src.agents.coach_graph import PlannedActivity
from src.agents.coach_graph import _assignment_to_planned_activity as _assignment_to_planned_activity
from src.agents.coach_graph import _build_gamebus_task_index as _build_gamebus_task_index
from src.agents.coach_graph import _parse_iso_date as _parse_iso_date
from src.agents.coach_graph import build_coach_graph, clear_thread
from src.memory import embeddings, flush
from src.memory.collections import collection_for
from src.webhook.client import WebhookClient

load_dotenv()
logger = logging.getLogger("coach.app")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


REQUIRE_MTLS = os.getenv("COACH_REQUIRE_MTLS", "false").strip().lower() in {"1", "true", "yes", "on"}
EXPECTED_PEER_CN = os.getenv("COACH_EXPECTED_PEER_CN", "gamebus-api-v2")
INTERNAL_TOKEN = os.getenv("COACH_INTERNAL_TOKEN", "")
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
CHROMA_PATH = os.getenv("CHROMA_PATH", "/app/chroma_db")


def _build_chroma_client():
    if CHROMA_HOST:
        return get_client(host=CHROMA_HOST, port=int(CHROMA_PORT))
    return get_client(path=CHROMA_PATH)


def _resolve_collection(player_id: int, campaign_id: int):
    name = collection_for(player_id, campaign_id)
    client = _build_chroma_client()
    embedding_function = embeddings.get_embedding_function()
    return client, get_or_create_collection(client, name, embedding_function=embedding_function)


_webhook_client = WebhookClient()
_flush_scheduler = None
_graph = build_coach_graph()


class SchedulingWindow(BaseModel):
    scheduleStartDate: str
    scheduleEndDate: str
    schedulingPeriodWeeks: int = 1
    currentDate: str
    currentDayname: str = ""


class ChatContext(BaseModel):
    challenges: List[Dict[str, Any]] = Field(default_factory=list)
    scheduledActivities: List[Dict[str, Any]] = Field(default_factory=list)
    schedulingWindow: SchedulingWindow


class ChatRequest(BaseModel):
    playerId: int
    campaignId: int
    sessionId: str
    message: str
    llmKey: Optional[str] = None
    context: ChatContext


class ChatResponse(BaseModel):
    output: str
    interviewComplete: bool = False
    plannedActivities: Optional[List[PlannedActivity]] = None


class ResetRequest(BaseModel):
    playerId: int
    campaignId: int
    sessionId: str


def _request_id_from(request: Request) -> str:
    incoming = request.headers.get("X-Coach-Request-Id")
    if incoming and incoming.strip():
        return incoming.strip()
    generated = uuid.uuid4().hex
    logger.warning("request.missing_request_id generated=%s path=%s", generated, request.url.path)
    return generated


async def authorise(request: Request) -> str:
    """Validate mTLS headers and return the request id."""
    request_id = _request_id_from(request)
    request.state.request_id = request_id

    if not REQUIRE_MTLS:
        return request_id

    cn = request.headers.get("X-Client-CN") or ""
    if cn != EXPECTED_PEER_CN:
        logger.warning("auth.cn_mismatch expected=%s got=%r request_id=%s", EXPECTED_PEER_CN, cn, request_id)
        raise HTTPException(status_code=403, detail="forbidden")

    token = request.headers.get("X-Coach-Internal-Token") or ""
    if not INTERNAL_TOKEN or not hmac.compare_digest(token, INTERNAL_TOKEN):
        logger.warning("auth.token_mismatch request_id=%s", request_id)
        raise HTTPException(status_code=403, detail="forbidden")

    return request_id


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _flush_scheduler

    # Eager-load embedding model in a background thread so /health flips to 200 once ready.
    threading.Thread(target=embeddings.warmup_embedding_function, name="bge-warmup", daemon=True).start()

    try:
        _flush_scheduler = flush.start_flush_scheduler(_build_chroma_client)
    except Exception as exc:  # noqa: BLE001
        logger.exception("flush.scheduler_start_failed error=%s", exc)
        _flush_scheduler = None

    try:
        yield
    finally:
        if _flush_scheduler is not None:
            try:
                _flush_scheduler.shutdown(wait=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("flush.scheduler_shutdown_failed error=%s", exc)
        try:
            _webhook_client.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("webhook.close_failed error=%s", exc)


app = FastAPI(title="coach-api", lifespan=lifespan)


@app.get("/health")
def health() -> Response:
    if not embeddings.is_ready():
        body = {"status": "loading", "embedding_ready": False}
        return Response(content=json.dumps(body), media_type="application/json", status_code=503)
    body = {"status": "healthy", "embedding_ready": True}
    return Response(content=json.dumps(body), media_type="application/json", status_code=200)


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request_id: str = Depends(authorise)) -> ChatResponse:
    if not embeddings.is_ready():
        raise HTTPException(status_code=503, detail="embedding model is loading")

    name = collection_for(req.playerId, req.campaignId)
    _, collection = _resolve_collection(req.playerId, req.campaignId)
    config = {
        "configurable": {
            "thread_id": f"{name}:{req.sessionId}",
            "collection": collection,
            "llm_key": req.llmKey,
            "request_id": request_id,
            "player_id": req.playerId,
            "campaign_id": req.campaignId,
            "session_id": req.sessionId,
            "webhook": _webhook_client,
        }
    }

    try:
        result = _graph.invoke({"message": req.message, "context": req.context.model_dump()}, config)
    except Exception as exc:
        logger.exception("chat.pipeline_failed request_id=%s error=%s", request_id, exc)
        _webhook_client.emit(
            player_id=req.playerId,
            campaign_id=req.campaignId,
            session_id=req.sessionId,
            phase=WebhookClient.PHASE_ERROR,
            request_id=request_id,
            message=str(exc),
        )
        raise HTTPException(status_code=500, detail=f"pipeline failed: {exc}")

    reply = result.get("output") or ""
    complete = bool(result.get("interview_complete"))
    planned = result.get("planned_activities") if complete else None
    return ChatResponse(output=reply, interviewComplete=complete, plannedActivities=planned)


@app.post("/reset")
def reset(req: ResetRequest, request_id: str = Depends(authorise)) -> Dict[str, Any]:
    name = collection_for(req.playerId, req.campaignId)
    client = _build_chroma_client()
    try:
        client.delete_collection(name=name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reset.delete_collection_failed name=%s error=%s", name, exc)
    clear_thread(_graph, f"{name}:{req.sessionId}")
    return {"deleted_collection": name, "request_id": request_id}


@app.post("/admin/flush")
def admin_flush(older_than_days: int = 7, request_id: str = Depends(authorise)) -> Dict[str, Any]:
    if older_than_days < 1:
        raise HTTPException(status_code=400, detail="older_than_days must be >= 1")
    client = _build_chroma_client()
    totals = flush.flush_all(client, older_than_days=older_than_days)
    return {"older_than_days": older_than_days, "totals": totals, "request_id": request_id}
