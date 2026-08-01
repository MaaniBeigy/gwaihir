"""LangGraph flow that drives the coach interview and the scheduling loop."""

from __future__ import annotations

import json
import logging
import os
import warnings
from datetime import date, datetime
from typing import Annotated, Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel
from typing_extensions import TypedDict

from src.agents.ChromaDB.chroma_db import (
    store_cleaned_conversation,
    store_scheduled_gamebus,
    store_scheduler_task_sources,
    store_standardized_activities,
)
from src.agents.controller_core import ControllerAgent
from src.agents.interviewer_core import InterviewerAgent
from src.agents.scheduler_core import _waking_window_minutes, schedule_from_inputs
from src.agents.standardizer_core import StandardizerAgent
from src.agents.tools.conversation_parser import parse_conversation_history
from src.agents.tools.routine_expander import expand_routines_to_assignments, merge_assignments
from src.agents.tools.standardized_normalizer import normalize_standardized_activities
from src.agents.tools.task_normalizer import build_unified_scheduler_tasks
from src.webhook.client import WebhookClient


# langgraph 0.2.x triggers a langchain-core pending-deprecation about the Reviver
# `allowed_objects` default at import time, and warn_deprecated forces it past warning
# filters, so quiet warnings only while importing langgraph and then restore.
def _silence_import_warning(*args, **kwargs):
    pass


_prev_showwarning = warnings.showwarning
warnings.showwarning = _silence_import_warning
try:
    from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
    from langgraph.graph import END, START, StateGraph  # noqa: E402
finally:
    warnings.showwarning = _prev_showwarning

logger = logging.getLogger("coach.graph")

DEFAULT_TIMEZONE = os.getenv("COACH_DEFAULT_TIMEZONE", "Europe/Amsterdam")
START_PROMPT = "Start interview about habits and routines."

INTERVIEW_COMPLETION_MARKERS = (
    "interview is over",
    "interview complete",
    "interview completed",
    "all questions answered",
    "thank you for your time",
    "that concludes our interview",
    "this concludes our interview",
    "interview is complete",
    "we have completed the interview",
    "interview is completed",
)


def _max_iterations() -> int:
    """Return the controller-loop cap from `COACH_MAX_SCHEDULE_ITERATIONS`, default 3."""
    try:
        return max(1, int(os.getenv("COACH_MAX_SCHEDULE_ITERATIONS", "3")))
    except ValueError:
        return 3


class PlannedActivity(BaseModel):
    """One scheduled slot returned to studio for the player to confirm."""

    gameDescriptorTK: str = "SCHEDULE_ACTIVITY"
    activityType: Optional[str] = None
    startDate: str
    endDate: str
    allDay: bool = False
    isHealthActivity: bool = False
    challengeRuleId: Optional[int] = None


def _parse_iso_date(value: Any) -> Optional[date]:
    """Return a date from a `YYYY-MM-DD` string, or None when it does not parse."""
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None


def _is_interview_complete(reply: str) -> bool:
    """Return True when the interviewer reply contains a completion marker."""
    text = (reply or "").lower()
    return any(marker in text for marker in INTERVIEW_COMPLETION_MARKERS)


def _assignment_to_planned_activity(
    item: Dict[str, Any],
    *,
    schedule_start: Optional[date],
    schedule_end: Optional[date],
) -> Optional[PlannedActivity]:
    """Convert one scheduler assignment to a planned activity, or None when out of window."""
    raw_date = item.get("date") or item.get("startdate")
    raw_start = item.get("start") or item.get("starttime")
    raw_end = item.get("end") or item.get("endtime")
    if not raw_date or not raw_start:
        return None

    assignment_date = _parse_iso_date(str(raw_date))
    if assignment_date is None:
        return None
    if schedule_start and assignment_date < schedule_start:
        return None
    if schedule_end and assignment_date > schedule_end:
        return None

    # Emit naive ISO (YYYY-MM-DDTHH:MM:SS); gamebus stores wall-clock values without timezone.
    start_iso = (
        f"{assignment_date.isoformat()}T{raw_start}:00"
        if len(str(raw_start)) == 5
        else f"{assignment_date.isoformat()}T{raw_start}"
    )
    end_iso = (
        f"{assignment_date.isoformat()}T{raw_end}:00"
        if raw_end and len(str(raw_end)) == 5
        else (f"{assignment_date.isoformat()}T{raw_end}" if raw_end else start_iso)
    )

    challenge_rule_id = item.get("challengeRuleId") or item.get("challenge_rule_id") or item.get("task_id")
    try:
        challenge_rule_id_int = int(challenge_rule_id) if challenge_rule_id is not None else None
    except (TypeError, ValueError):
        challenge_rule_id_int = None

    return PlannedActivity(
        gameDescriptorTK="SCHEDULE_ACTIVITY",
        activityType=str(item.get("activityType") or item.get("name") or "ACTIVITY"),
        startDate=start_iso,
        endDate=end_iso,
        allDay=bool(item.get("allDay", False)),
        isHealthActivity=bool(item.get("isHealthActivity", item.get("type") == "gamebus")),
        challengeRuleId=challenge_rule_id_int,
    )


def _routines_as_busy_intervals(routine_assignments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn routine occurrences into immutable busy intervals for the scheduler."""
    busy: List[Dict[str, Any]] = []
    for item in routine_assignments or []:
        if not isinstance(item, dict):
            continue
        day = item.get("date")
        start = item.get("start")
        end = item.get("end")
        if not day or not start or not end:
            continue
        busy.append(
            {
                "startDate": f"{day}T{start}",
                "endDate": f"{day}T{end}",
                "source": "ROUTINE",
                "activityType": item.get("name") or item.get("activityType"),
            }
        )
    return busy


def _build_gamebus_task_index(task_sources: Any) -> Dict[str, Dict[str, Any]]:
    """Map each gamebus task uuid to its extracted task (rule id, game descriptor key, conditions)."""
    store: Dict[str, Any] = {}
    if isinstance(task_sources, dict):
        store = task_sources.get("source_stores", {}).get("gamebus_tasks", {}) or {}
    index: Dict[str, Dict[str, Any]] = {}
    for source_uuid, info in store.items():
        source_task = info.get("task") if isinstance(info, dict) else None
        if isinstance(source_task, dict):
            index[str(source_uuid)] = source_task
    return index


def _append_messages(left: Any, right: Any) -> List[Dict[str, str]]:
    """Append new transcript turns to the running message list."""
    return list(left or []) + list(right or [])


class CoachState(TypedDict, total=False):
    """Graph state holding the transcript, request context, and pipeline payloads."""

    message: str
    context: Dict[str, Any]
    messages: Annotated[List[Dict[str, str]], _append_messages]
    output: str
    interview_complete: bool
    cleaned_conversation: str
    standardized_json: Any
    task_sources: Dict[str, Any]
    scheduler_task_sources: Dict[str, Any]
    gamebus_scheduler_sources: Dict[str, Any]
    busy_intervals: List[Dict[str, Any]]
    routine_assignments: List[Dict[str, Any]]
    gamebus_task_count: int
    schedule_payload: Dict[str, Any]
    controller_feedback: List[str]
    controller_valid: bool
    iteration: int
    planned_activities: List[PlannedActivity]


def _config_value(config: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    """Read one value from the graph `configurable` config, or return a default."""
    if not isinstance(config, dict):
        return default
    return (config.get("configurable") or {}).get(key, default)


def _window_dates(context: Dict[str, Any]) -> Tuple[Optional[date], Optional[date]]:
    """Return the schedule start and end dates parsed from the request context."""
    window = (context or {}).get("schedulingWindow") or {}
    return _parse_iso_date(window.get("scheduleStartDate")), _parse_iso_date(window.get("scheduleEndDate"))


def _emit(
    config: Optional[Dict[str, Any]],
    phase: str,
    *,
    percent: Optional[int] = None,
    message: Optional[str] = None,
) -> None:
    """Post a progress webhook for the current phase when a client is configured."""
    webhook = _config_value(config, "webhook")
    if webhook is None:
        return
    webhook.emit(
        player_id=_config_value(config, "player_id", 0),
        campaign_id=_config_value(config, "campaign_id", 0),
        session_id=_config_value(config, "session_id", "default"),
        phase=phase,
        request_id=_config_value(config, "request_id", "-"),
        percent_complete=percent,
        message=message,
    )


def _store(
    store_fn: Callable[..., Any],
    collection: Any,
    document: str,
    session_id: str,
    request_id: str,
) -> None:
    """Store one document in Chroma and swallow storage errors as a warning."""
    if collection is None:
        return
    try:
        store_fn(collection, document, metadata={"session_id": session_id, "request_id": request_id})
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "graph.store_failed fn=%s request_id=%s error=%s",
            getattr(store_fn, "__name__", "store"),
            request_id,
            exc,
        )


def interview_node(state: CoachState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run one interview turn and flag completion from the reply."""
    collection = _config_value(config, "collection")
    llm_key = _config_value(config, "llm_key")
    prior = list(state.get("messages") or [])
    user_message = START_PROMPT if not prior else str(state.get("message") or "")

    agent = InterviewerAgent(collection=collection, llm_api_key=llm_key)
    agent.conversation_history = [dict(turn) for turn in prior]
    reply = agent.chat(message=user_message, clear_history=False)

    _emit(config, WebhookClient.PHASE_INTERVIEWING, message=reply[:200])
    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply},
        ],
        "output": reply,
        "interview_complete": _is_interview_complete(reply),
    }


def standardize_node(state: CoachState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Clean the transcript and turn it into standardized activities."""
    collection = _config_value(config, "collection")
    llm_key = _config_value(config, "llm_key")
    request_id = _config_value(config, "request_id", "-")
    session_id = _config_value(config, "session_id", "default")

    messages = state.get("messages") or []
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages if m.get("content"))
    cleaned = parse_conversation_history(transcript)
    _store(store_cleaned_conversation, collection, cleaned, session_id, request_id)

    standardizer = StandardizerAgent()
    if llm_key:
        standardizer.api_key = llm_key
    standardized = standardizer.standardize(message=cleaned, clear_history=True)
    try:
        standardized = normalize_standardized_activities(standardized)
    except Exception as exc:  # noqa: BLE001
        logger.warning("graph.standardized_normalizer_failed request_id=%s error=%s", request_id, exc)
    _store(store_standardized_activities, collection, standardized, session_id, request_id)

    _emit(
        config,
        WebhookClient.PHASE_SCHEDULING,
        percent=10,
        message="Interview complete; standardizing routines.",
    )
    return {"cleaned_conversation": cleaned, "standardized_json": standardized}


def build_tasks_node(state: CoachState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the unified task sources, expand routines, and stage the gamebus-only inputs."""
    collection = _config_value(config, "collection")
    request_id = _config_value(config, "request_id", "-")
    session_id = _config_value(config, "session_id", "default")

    context = state.get("context") or {}
    standardized = state.get("standardized_json")
    schedule_start, schedule_end = _window_dates(context)
    challenges = context.get("challenges") or []
    gamebus_payload = {"challenges": challenges, "total_challenges": len(challenges)}

    task_sources = build_unified_scheduler_tasks(
        standardized_activities_payload=standardized,
        gamebus_payload=gamebus_payload,
        schedule_start=schedule_start,
        schedule_end=schedule_end,
    )
    all_tasks = task_sources.get("tasks", []) if isinstance(task_sources, dict) else []
    routine_tasks = [t for t in all_tasks if isinstance(t, dict) and t.get("source_type") == "routine"]
    gamebus_tasks = [t for t in all_tasks if isinstance(t, dict) and t.get("source_type") == "gamebus"]
    routine_schedulable = [t for t in routine_tasks if bool(t.get("scheduling_required"))]
    routine_context_only = [t for t in routine_tasks if not bool(t.get("scheduling_required"))]
    non_schedulable_ids = [
        str(t.get("task_id")) for t in routine_context_only if t.get("task_id") is not None
    ]

    scheduler_task_sources = {
        "generated_at": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "timezone": DEFAULT_TIMEZONE,
        "all_tasks": all_tasks,
        "routine_tasks": routine_tasks,
        "gamebus_tasks": gamebus_tasks,
        "routine_schedulable_tasks": routine_schedulable,
        "routine_context_only_tasks": routine_context_only,
        "non_schedulable_task_ids": non_schedulable_ids,
        "unscheduled_routines": routine_schedulable,
        "unscheduled_gamebus": gamebus_tasks,
    }
    _store(
        store_scheduler_task_sources,
        collection,
        json.dumps(scheduler_task_sources, ensure_ascii=False),
        session_id,
        request_id,
    )

    routine_assignments: List[Dict[str, Any]] = []
    if schedule_start is not None and schedule_end is not None:
        try:
            routine_assignments = expand_routines_to_assignments(
                standardized, schedule_start=schedule_start, schedule_end=schedule_end
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("graph.routine_expansion_failed request_id=%s error=%s", request_id, exc)

    busy_intervals = list(context.get("scheduledActivities") or []) + _routines_as_busy_intervals(
        routine_assignments
    )

    gamebus_sources = dict(scheduler_task_sources)
    gamebus_sources["all_tasks"] = gamebus_tasks
    gamebus_sources["routine_tasks"] = []
    gamebus_sources["routine_schedulable_tasks"] = []
    gamebus_sources["routine_context_only_tasks"] = []
    gamebus_sources["unscheduled_routines"] = []
    gamebus_sources["unscheduled_gamebus"] = gamebus_tasks

    return {
        "task_sources": task_sources,
        "scheduler_task_sources": scheduler_task_sources,
        "gamebus_scheduler_sources": gamebus_sources,
        "busy_intervals": busy_intervals,
        "routine_assignments": routine_assignments,
        "gamebus_task_count": len(gamebus_tasks),
        "iteration": 0,
        "controller_feedback": [],
    }


def schedule_node(state: CoachState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Place the gamebus tasks, passing back any controller feedback from the prior round."""
    collection = _config_value(config, "collection")
    llm_key = _config_value(config, "llm_key")
    request_id = _config_value(config, "request_id", "-")
    session_id = _config_value(config, "session_id", "default")

    context = state.get("context") or {}
    sources = state.get("gamebus_scheduler_sources") or {}
    busy = state.get("busy_intervals") or []
    feedback = state.get("controller_feedback") or []
    iteration = int(state.get("iteration") or 0) + 1

    schedule_payload: Dict[str, Any] = {"assignments": []}
    try:
        result = schedule_from_inputs(
            scheduler_task_sources_input=sources,
            controller_feedback=feedback,
            timezone_str=DEFAULT_TIMEZONE,
            session_id=session_id,
            scheduling_window=context.get("schedulingWindow"),
            busy_intervals=busy,
            llm_api_key=llm_key,
            model=os.getenv("SCHEDULER_MODEL") or None,
        )
        payload = result.get("schedule_json") if isinstance(result, dict) else None
        if not isinstance(payload, dict):
            payload = {"assignments": result.get("assignments", []) if isinstance(result, dict) else []}
        schedule_payload = payload
    except RuntimeError as exc:
        logger.info("graph.scheduler_skipped request_id=%s reason=%s", request_id, exc)

    _store(
        store_scheduled_gamebus,
        collection,
        json.dumps(schedule_payload, ensure_ascii=False),
        session_id,
        request_id,
    )
    return {"schedule_payload": schedule_payload, "iteration": iteration}


def validate_node(state: CoachState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the schedule with the controller and collect feedback for a retry."""
    request_id = _config_value(config, "request_id", "-")
    session_id = _config_value(config, "session_id", "default")

    schedule_payload = state.get("schedule_payload") or {"assignments": []}
    assignments = schedule_payload.get("assignments") if isinstance(schedule_payload, dict) else []
    if not assignments:
        # Nothing was scheduled, so there is nothing for the controller to reject.
        return {"controller_valid": True, "controller_feedback": []}

    try:
        busy = state.get("busy_intervals") or []
        # When the calendar is available the deterministic checks are authoritative,
        # so skip the LLM controller call (it only rubber-stamped placements and
        # doubled latency). Fall back to the LLM controller when there is no calendar.
        controller = ControllerAgent(enable_llm=not bool(busy))
        result = controller.validate_schedule(
            schedule_payload=schedule_payload,
            expected_gamebus_tasks=int(state.get("gamebus_task_count") or 0),
            scheduler_task_sources=state.get("gamebus_scheduler_sources") or {},
            session_id=session_id,
            trace_id=request_id,
            # Give the controller the calendar so it can catch real overlaps.
            busy_intervals=busy,
            waking_window=_waking_window_minutes(),
        )
    except Exception as exc:  # noqa: BLE001
        # A controller failure must not block the schedule already produced.
        logger.warning("graph.controller_validation_failed request_id=%s error=%s", request_id, exc)
        return {"controller_valid": True, "controller_feedback": []}

    is_valid = bool(result.get("is_valid", True))
    # Violations lead (they now include the concrete calendar fixes); fix_advice
    # follows, matching the established contract.
    feedback = [str(v) for v in (result.get("violations") or [])]
    feedback += [str(v) for v in (result.get("fix_advice") or [])]
    return {"controller_valid": is_valid, "controller_feedback": feedback}


def finalize_node(state: CoachState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge routines with the scheduled gamebus tasks and build the planned activities."""
    context = state.get("context") or {}
    schedule_start, schedule_end = _window_dates(context)
    routine_assignments = state.get("routine_assignments") or []
    schedule_payload = state.get("schedule_payload") or {"assignments": []}
    assignments = schedule_payload.get("assignments", []) if isinstance(schedule_payload, dict) else []

    merged = merge_assignments(routine_assignments, assignments)
    gamebus_index = _build_gamebus_task_index(state.get("task_sources"))

    planned: List[PlannedActivity] = []
    for item in merged:
        if not isinstance(item, dict):
            continue
        source_task = gamebus_index.get(str(item.get("task_id")))
        if source_task is not None:
            # Link to the challenge rule and set activityType to the rule game descriptor key
            # (e.g. WALK), so studio routes "mark done" into the normal scoring path.
            if source_task.get("task_id") is not None:
                item["challengeRuleId"] = source_task.get("task_id")
            game_descriptor_key = source_task.get("game_descriptor_key")
            if game_descriptor_key:
                item["activityType"] = game_descriptor_key
        activity = _assignment_to_planned_activity(
            item, schedule_start=schedule_start, schedule_end=schedule_end
        )
        if activity is not None:
            planned.append(activity)

    _emit(config, WebhookClient.PHASE_COMPLETE, percent=100, message=f"Scheduled {len(planned)} activities.")
    return {"planned_activities": planned}


def _route_after_interview(state: CoachState) -> str:
    """Send a completed interview into scheduling, or end the turn to wait for more input."""
    return "standardize" if state.get("interview_complete") else END


def _route_after_validate(state: CoachState) -> str:
    """Loop back to the scheduler while the controller rejects and the cap allows."""
    if state.get("controller_valid") or int(state.get("iteration") or 0) >= _max_iterations():
        return "finalize"
    return "schedule"


def build_coach_graph(checkpointer: Any = None):
    """Compile the interview and scheduling graph with an in-process checkpointer."""
    builder = StateGraph(CoachState)
    builder.add_node("interview", interview_node)
    builder.add_node("standardize", standardize_node)
    builder.add_node("build_tasks", build_tasks_node)
    builder.add_node("schedule", schedule_node)
    builder.add_node("validate", validate_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "interview")
    builder.add_conditional_edges(
        "interview", _route_after_interview, {"standardize": "standardize", END: END}
    )
    builder.add_edge("standardize", "build_tasks")
    builder.add_edge("build_tasks", "schedule")
    builder.add_edge("schedule", "validate")
    builder.add_conditional_edges(
        "validate", _route_after_validate, {"schedule": "schedule", "finalize": "finalize"}
    )
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def clear_thread(graph: Any, thread_id: str) -> None:
    """Drop one thread's in-memory checkpoints so a reset starts clean."""
    saver = getattr(graph, "checkpointer", None)
    storage = getattr(saver, "storage", None)
    if isinstance(storage, dict):
        storage.pop(thread_id, None)
