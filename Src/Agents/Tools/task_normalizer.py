from __future__ import annotations

import json
import os
import re
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.agents.tools import gamebus_task_extractor

TIME_ALIAS = {
    "morning": "08:00",
    "afternoon": "13:00",
    "evening": "18:00",
    "night": "21:00",
}


def _get_scheduling_period_days(default_weeks: int = 1) -> int:
    raw_value = os.getenv("SCHEDULING_PERIOD_WEEKS", str(default_weeks))
    try:
        weeks = int(raw_value)
    except Exception:
        weeks = default_weeks
    weeks = max(1, min(52, weeks))
    return weeks * 7


def _load_json(value_or_path: Any) -> Any:
    if value_or_path is None:
        return None
    if isinstance(value_or_path, (dict, list)):
        return value_or_path
    if isinstance(value_or_path, str):
        return json.loads(value_or_path)
    return None


def _extract_activities(activities_payload: Any) -> List[Dict[str, Any]]:
    if activities_payload is None:
        return []
    if isinstance(activities_payload, str):
        activities_payload = _load_json(activities_payload)
    if isinstance(activities_payload, dict) and "Activities" in activities_payload:
        return activities_payload.get("Activities") or []
    if isinstance(activities_payload, list):
        return activities_payload
    return []


def _normalize_activity_time_preference(preferred: Optional[str]) -> Optional[str]:
    if not preferred:
        return None
    normalized = preferred.strip().lower()
    if normalized in TIME_ALIAS:
        return normalized
    if re.match(r"^\d{1,2}:\d{2}$", normalized):
        return normalized
    return None


def _normalize_days(days: Any) -> List[str]:
    mapping = {
        "monday": "mon",
        "mon": "mon",
        "tuesday": "tue",
        "tue": "tue",
        "tues": "tue",
        "wednesday": "wed",
        "wed": "wed",
        "thursday": "thu",
        "thu": "thu",
        "thur": "thu",
        "thurs": "thu",
        "friday": "fri",
        "fri": "fri",
        "saturday": "sat",
        "sat": "sat",
        "sunday": "sun",
        "sun": "sun",
    }
    normalized: List[str] = []
    for day in days or []:
        key = str(day).strip().lower()
        value = mapping.get(key)
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _safe_duration_minutes(activity: Dict[str, Any]) -> int:
    duration = activity.get("activity_duration")
    if activity.get("IsInstantaneous"):
        return 5
    try:
        parsed = int(duration)
        return parsed if parsed > 0 else 5
    except Exception:
        return 5


def _compute_preferred_end_time(preferred_start_time: Optional[str], duration_min: int) -> Optional[str]:
    if not preferred_start_time:
        return None
    try:
        start_dt = datetime.strptime(preferred_start_time, "%H:%M")
        end_dt = start_dt + timedelta(minutes=max(0, int(duration_min)))
        return end_dt.strftime("%H:%M")
    except Exception:
        return None


def _get_routine_schedule_mode(activity: Dict[str, Any]) -> str:
    preferred = _normalize_activity_time_preference((activity.get("time") or {}).get("preferred"))
    if preferred in TIME_ALIAS:
        return "window"
    if preferred:
        return "fixed"
    return "unspecified"


def _parse_iso_date(value: Any) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None
    if "T" in text:
        text = text.split("T", 1)[0]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except Exception:
        return None


def _date_to_iso(value: Optional[date]) -> Optional[str]:
    return value.isoformat() if value else None


def _get_next_monday(base: date) -> date:
    days_until_monday = (7 - base.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    return base + timedelta(days=days_until_monday)


def _iter_matching_dates(start: date, end: date, allowed_days: List[str]) -> List[date]:
    if end < start or not allowed_days:
        return []
    allowed = set(allowed_days)
    current = start
    matches: List[date] = []
    while current <= end:
        weekday = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"][current.weekday()]
        if weekday in allowed:
            matches.append(current)
        current += timedelta(days=1)
    return matches


def _align_activity_start_date_to_schedule_window(
    activity: Dict[str, Any],
    schedule_start: date,
    schedule_end: date,
) -> Dict[str, Any]:
    aligned = dict(activity)
    raw_start = _parse_iso_date(aligned.get("start_date")) or schedule_start
    if raw_start < schedule_start:
        raw_start = schedule_start

    frequency = aligned.get("frequency") if isinstance(aligned.get("frequency"), dict) else {}
    allowed_days = _normalize_days(frequency.get("days"))
    if allowed_days and raw_start <= schedule_end:
        next_dates = _iter_matching_dates(raw_start, schedule_end, allowed_days)
        if next_dates:
            raw_start = next_dates[0]

    aligned["start_date"] = _date_to_iso(raw_start)
    return aligned


def _activity_to_unified_task(activity: Dict[str, Any], source_uuid: str, index: int) -> Dict[str, Any]:
    name = activity.get("name") or f"routine-{index}"
    normalized_pref = _normalize_activity_time_preference((activity.get("time") or {}).get("preferred"))
    schedule_mode = _get_routine_schedule_mode(activity)
    preferred_window = normalized_pref if normalized_pref in TIME_ALIAS else None
    preferred_start_time = normalized_pref if normalized_pref and normalized_pref not in TIME_ALIAS else None
    est_duration_min = _safe_duration_minutes(activity)
    preferred_end_time = _compute_preferred_end_time(preferred_start_time, est_duration_min)

    freq = activity.get("frequency") or {}
    start_date = activity.get("start_date")
    end_date = activity.get("end_date")
    return {
        "task_id": source_uuid,
        "source_type": "routine",
        "source_ref": {
            "store": "standardized_activities",
            "source_uuid": source_uuid,
        },
        "name": name,
        "description": activity.get("description") or "",
        "est_duration_min": est_duration_min,
        "summary_prefix": "Routine",
        "routine_schedule_mode": schedule_mode,
        "scheduling_required": True,
        "start_date": start_date,
        "end_date": end_date,
        "frequency": {
            "type": freq.get("type"),
            "days": _normalize_days(freq.get("days")),
            "count": freq.get("count"),
        },
        "preferred_window": preferred_window,
        "preferred_start_time": preferred_start_time,
        "preferred_end_time": preferred_end_time,
        "notes": activity.get("notes"),
    }


def _expand_activity_to_unified_tasks(
    activity: Dict[str, Any],
    source_uuid: str,
    index: int,
    schedule_start: Optional[date] = None,
    schedule_end: Optional[date] = None,
) -> List[Dict[str, Any]]:
    base_task = _activity_to_unified_task(activity, source_uuid, index)

    if schedule_start is not None:
        raw_start = _parse_iso_date(base_task.get("start_date"))
        if raw_start is None or raw_start < schedule_start:
            base_task["start_date"] = _date_to_iso(schedule_start)

    freq = base_task.get("frequency") or {}
    days = _normalize_days(freq.get("days"))
    if (
        base_task.get("routine_schedule_mode") != "fixed"
        or str(freq.get("type") or "").strip().lower() != "specific_days"
        or not days
    ):
        return [base_task]

    start = _parse_iso_date(base_task.get("start_date")) or (schedule_start or datetime.now().date())
    configured_end = _parse_iso_date(base_task.get("end_date"))
    horizon_days = _get_scheduling_period_days()
    horizon_end = configured_end or (start + timedelta(days=horizon_days - 1))
    if schedule_end is not None:
        horizon_end = min(horizon_end, schedule_end)
    dates = _iter_matching_dates(start, horizon_end, days)
    if not dates:
        return [base_task]

    expanded: List[Dict[str, Any]] = []
    for occurrence in dates:
        item = dict(base_task)
        occurrence_date = _date_to_iso(occurrence)
        task_id = str(uuid.uuid4())
        item["task_id"] = task_id
        item["start_date"] = occurrence_date
        item["end_date"] = occurrence_date
        item["frequency"] = {
            "type": "specific_days",
            "days": [["mon", "tue", "wed", "thu", "fri", "sat", "sun"][occurrence.weekday()]],
            "count": 1,
        }
        # Pinned to a specific date+time; the scheduler treats these as context only.
        item["scheduling_required"] = False
        expanded.append(item)
    return expanded


def _flatten_simplified_challenges_with_refs(
    slim_json: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    tasks: List[Dict[str, Any]] = []
    source_store: Dict[str, Dict[str, Any]] = {}
    for challenge in slim_json.get("challenges", []):
        for task in challenge.get("tasks", []):
            source_uuid = str(uuid.uuid4())
            source_store[source_uuid] = {
                "challenge": challenge,
                "task": task,
            }
            task_copy = {
                "task_id": source_uuid,
                "source_type": "gamebus",
                "name": task.get("name") or "gamebus-task",
                "description": task.get("description") or "",
                "est_duration_min": (
                    task.get("est_duration_min")
                    if isinstance(task.get("est_duration_min"), (int, float))
                    else 5
                ),
                "max_repeats": (
                    task.get("max_repeats") if isinstance(task.get("max_repeats"), (int, float)) else None
                ),
                "cooldown_days": (
                    task.get("cooldown_days") if isinstance(task.get("cooldown_days"), (int, float)) else None
                ),
            }
            tasks.append(task_copy)
    return tasks, source_store


def _normalize_simplified_challenge(challenge: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "challenge_id": challenge.get("challenge_id"),
        "name": challenge.get("name") or "",
        "description": challenge.get("description") or "",
        "type": challenge.get("type") or "TASKS_COLLECTION",
        "start_date": challenge.get("start_date"),
        "end_date": challenge.get("end_date"),
        "target": challenge.get("target"),
        "success_next": challenge.get("success_next"),
        "failure_next": challenge.get("failure_next"),
        "tasks": challenge.get("tasks") if isinstance(challenge.get("tasks"), list) else [],
    }


def _coerce_to_simplified_challenges(gamebus_json: Any) -> List[Dict[str, Any]]:
    if gamebus_json is None:
        return []

    if isinstance(gamebus_json, dict) and isinstance(gamebus_json.get("challenges"), list):
        candidates = gamebus_json.get("challenges", [])
    elif isinstance(gamebus_json, list):
        candidates = gamebus_json
    else:
        candidates = [gamebus_json]

    simplified: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        if "id" in candidate:
            simplified.append(gamebus_task_extractor.simplify_challenge(candidate))
            continue

        if "challenge_id" in candidate and isinstance(candidate.get("tasks"), list):
            simplified.append(_normalize_simplified_challenge(candidate))

    return simplified


def _build_simplified_gamebus(gamebus_json: Any) -> Dict[str, Any]:
    simplified = _coerce_to_simplified_challenges(gamebus_json)

    return {
        "simplified_at": datetime.now().isoformat(),
        "total_challenges": len(simplified),
        "challenges": simplified,
    }


def _extract_gamebus_with_control_loop(gamebus_payload: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Try common wrapper formats and report whether GameBus tasks were extracted."""
    attempts: List[Tuple[str, Any]] = [("direct", gamebus_payload)]

    if isinstance(gamebus_payload, dict):
        if "simplified_gamebus" in gamebus_payload:
            attempts.append(("simplified_gamebus", gamebus_payload.get("simplified_gamebus")))
        if "challenges" in gamebus_payload:
            attempts.append(("challenges", {"challenges": gamebus_payload.get("challenges")}))
        if "data" in gamebus_payload:
            attempts.append(("data", gamebus_payload.get("data")))

    for source_name, candidate in attempts:
        simplified_gamebus = _build_simplified_gamebus(candidate)
        gamebus_tasks, gamebus_source_store = _flatten_simplified_challenges_with_refs(simplified_gamebus)
        if gamebus_tasks:
            return simplified_gamebus, {
                "gamebus_tasks": gamebus_tasks,
                "gamebus_source_store": gamebus_source_store,
                "control": {
                    "input_present": gamebus_payload is not None,
                    "source_used": source_name,
                    "is_available": True,
                    "tasks_found": len(gamebus_tasks),
                    "challenges_found": simplified_gamebus.get("total_challenges", 0),
                },
            }

    simplified_gamebus = _build_simplified_gamebus(gamebus_payload)
    gamebus_tasks, gamebus_source_store = _flatten_simplified_challenges_with_refs(simplified_gamebus)
    return simplified_gamebus, {
        "gamebus_tasks": gamebus_tasks,
        "gamebus_source_store": gamebus_source_store,
        "control": {
            "input_present": gamebus_payload is not None,
            "source_used": "none",
            "is_available": False,
            "tasks_found": len(gamebus_tasks),
            "challenges_found": simplified_gamebus.get("total_challenges", 0),
        },
    }


def build_unified_scheduler_tasks(
    standardized_activities_payload: Any = None,
    gamebus_payload: Any = None,
    schedule_start: Optional[date] = None,
    schedule_end: Optional[date] = None,
) -> Dict[str, Any]:
    activities = _extract_activities(standardized_activities_payload)

    if schedule_start is None:
        schedule_start = _get_next_monday(datetime.now().date())
    if schedule_end is None:
        schedule_end = schedule_start + timedelta(days=_get_scheduling_period_days() - 1)

    activity_store: Dict[str, Dict[str, Any]] = {}
    routine_tasks: List[Dict[str, Any]] = []
    for index, activity in enumerate(activities, start=1):
        source_uuid = str(uuid.uuid4())
        aligned_activity = _align_activity_start_date_to_schedule_window(
            activity, schedule_start, schedule_end
        )
        activity_store[source_uuid] = aligned_activity
        routine_tasks.extend(
            _expand_activity_to_unified_tasks(
                aligned_activity, source_uuid, index, schedule_start, schedule_end
            )
        )

    simplified_gamebus, gamebus_result = _extract_gamebus_with_control_loop(gamebus_payload)
    gamebus_tasks = gamebus_result.get("gamebus_tasks", [])
    gamebus_source_store = gamebus_result.get("gamebus_source_store", {})

    all_tasks = gamebus_tasks + routine_tasks

    return {
        "normalized_at": datetime.now().isoformat(),
        "total_tasks": len(all_tasks),
        "gamebus_tasks": len(gamebus_tasks),
        "routine_tasks": len(routine_tasks),
        "tasks": all_tasks,
        "source_stores": {
            "standardized_activities": activity_store,
            "gamebus_tasks": gamebus_source_store,
        },
        "simplified_gamebus": simplified_gamebus,
        "gamebus_control": gamebus_result.get("control", {}),
    }
