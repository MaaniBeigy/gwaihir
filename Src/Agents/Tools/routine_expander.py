"""Deterministic expansion of standardized routines into one assignment per occurrence."""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger("coach.routine_expander")

# Clamp absurd durations (e.g. a hallucinated 1260 min block) to keep entries within a day.
_MAX_REASONABLE_DURATION_MIN = 16 * 60

# Default morning slot used when frequency is known but `time.preferred` is null.
_DEFAULT_FALLBACK_TIME = (10, 0)


_WEEKDAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

_DAY_ALIASES = {
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

_HHMM_PATTERN = re.compile(r"^\d{1,2}:\d{2}$")

# Coarse time-of-day aliases the standardizer may emit instead of HH:MM.
_TIME_OF_DAY_ALIASES = {
    "morning": (9, 0),
    "noon": (12, 0),
    "afternoon": (14, 0),
    "evening": (19, 0),
    "night": (21, 0),
}


def _normalize_days(days: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for d in days or []:
        norm = _DAY_ALIASES.get(str(d).strip().lower())
        if norm and norm not in out:
            out.append(norm)
    return out


def _parse_hhmm(value: Any) -> Optional[Tuple[int, int]]:
    """Parse a literal HH:MM string. Aliases are handled in `_resolve_time_preference`."""
    text = str(value or "").strip()
    if not _HHMM_PATTERN.match(text):
        return None
    h, m = text.split(":")
    try:
        return int(h), int(m)
    except ValueError:
        return None


def _resolve_time_preference(value: Any) -> Tuple[Optional[Tuple[int, int]], str]:
    """Return `(hhmm_or_None, source)` for `time.preferred`. source is `hhmm`, `alias:<k>`, or `none`."""
    direct = _parse_hhmm(value)
    if direct is not None:
        return direct, "hhmm"
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _TIME_OF_DAY_ALIASES:
            return _TIME_OF_DAY_ALIASES[key], f"alias:{key}"
    return None, "none"


def _add_minutes(hhmm: Tuple[int, int], minutes: int) -> Tuple[int, int]:
    total = hhmm[0] * 60 + hhmm[1] + max(0, int(minutes or 0))
    total %= 24 * 60
    return total // 60, total % 60


def _format_hhmm(hhmm: Tuple[int, int]) -> str:
    return f"{hhmm[0]:02d}:{hhmm[1]:02d}"


def _safe_duration_minutes(activity: Dict[str, Any]) -> int:
    duration = activity.get("activity_duration")
    if activity.get("IsInstantaneous"):
        return 5
    try:
        parsed = int(duration)
    except (TypeError, ValueError):
        return 5
    if parsed <= 0:
        return 5
    if parsed > _MAX_REASONABLE_DURATION_MIN:
        logger.warning(
            "routine_expander.duration_clamped name=%r raw=%s clamped=%s",
            activity.get("name"),
            parsed,
            _MAX_REASONABLE_DURATION_MIN,
        )
        return _MAX_REASONABLE_DURATION_MIN
    return parsed


def _parse_iso_date(value: Any) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None
    if "T" in text:
        text = text.split("T", 1)[0]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _dates_in_window(start: date, end: date) -> List[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _matching_dates(start: date, end: date, allowed_days: List[str]) -> List[date]:
    if not allowed_days or end < start:
        return []
    allowed = set(allowed_days)
    return [d for d in _dates_in_window(start, end) if _WEEKDAY_NAMES[d.weekday()] in allowed]


def _load_activities(standardized: Any) -> List[Dict[str, Any]]:
    if standardized is None:
        return []
    if isinstance(standardized, str):
        try:
            standardized = json.loads(standardized)
        except json.JSONDecodeError:
            return []
    if isinstance(standardized, dict):
        activities = standardized.get("Activities")
        if isinstance(activities, list):
            return [a for a in activities if isinstance(a, dict)]
    if isinstance(standardized, list):
        return [a for a in standardized if isinstance(a, dict)]
    return []


def _activity_target_dates(
    activity: Dict[str, Any],
    schedule_start: date,
    schedule_end: date,
) -> List[date]:
    """Pick the calendar dates this routine occupies inside the window.

    When `frequency.days` is non-empty, `activity.start_date` is ignored to avoid weekday mismatches.
    """
    freq = activity.get("frequency") if isinstance(activity.get("frequency"), dict) else {}
    ftype = str(freq.get("type") or "").strip().lower()
    days = _normalize_days(freq.get("days") or [])

    raw_end = _parse_iso_date(activity.get("end_date")) or schedule_end
    if raw_end > schedule_end:
        raw_end = schedule_end
    if raw_end < schedule_start:
        return []

    if days:
        # specific_days / weekly with explicit days: bound by the window only.
        return _matching_dates(schedule_start, raw_end, days)

    # No days specified: fall back to start_date as the lower bound.
    raw_start = _parse_iso_date(activity.get("start_date")) or schedule_start
    if raw_start < schedule_start:
        raw_start = schedule_start
    if raw_end < raw_start:
        return []

    if ftype == "daily" or (not ftype and not days):
        return _dates_in_window(raw_start, raw_end)
    if ftype == "weekly":
        weekday = _WEEKDAY_NAMES[raw_start.weekday()]
        return _matching_dates(raw_start, raw_end, [weekday])
    return []


def expand_routines_to_assignments(
    standardized_json: Any,
    *,
    schedule_start: date,
    schedule_end: date,
) -> List[Dict[str, Any]]:
    """Return one assignment dict per routine occurrence in the window."""
    assignments: List[Dict[str, Any]] = []
    for activity in _load_activities(standardized_json):
        preferred = (activity.get("time") or {}).get("preferred")
        start_hhmm, source = _resolve_time_preference(preferred)
        used_fallback_time = False
        if start_hhmm is None:
            # When days or daily is set but no time is given, pin to a default morning slot
            # so the routine still lands on the calendar.
            freq = activity.get("frequency") if isinstance(activity.get("frequency"), dict) else {}
            ftype = str(freq.get("type") or "").strip().lower()
            days = _normalize_days(freq.get("days") or [])
            if days or ftype == "daily":
                start_hhmm = _DEFAULT_FALLBACK_TIME
                used_fallback_time = True
                logger.info(
                    "routine_expander.fallback_time_used name=%r days=%s type=%s default=%s",
                    activity.get("name"),
                    days,
                    ftype or None,
                    _format_hhmm(_DEFAULT_FALLBACK_TIME),
                )
            else:
                continue
        elif source != "hhmm":
            used_fallback_time = True
            logger.info(
                "routine_expander.alias_resolved name=%r preferred=%r resolved=%s source=%s",
                activity.get("name"),
                preferred,
                _format_hhmm(start_hhmm),
                source,
            )
        duration = _safe_duration_minutes(activity)
        end_hhmm = _add_minutes(start_hhmm, duration)
        name = str(activity.get("name") or "routine")
        for occurrence in _activity_target_dates(activity, schedule_start, schedule_end):
            assignments.append(
                {
                    "date": occurrence.isoformat(),
                    "start": _format_hhmm(start_hhmm),
                    "end": _format_hhmm(end_hhmm),
                    "name": name,
                    "activityType": name,
                    "type": "routine",
                    "isHealthActivity": False,
                    "fallback_time": used_fallback_time,
                }
            )
    return assignments


def _entry_name_key(item: Dict[str, Any]) -> str:
    return str(item.get("name") or item.get("activityType") or "").strip().lower()


def merge_assignments(
    routine_assignments: List[Dict[str, Any]],
    scheduler_assignments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Combine deterministic routine expansion with LLM scheduler output.

    Routine entries win; scheduler entries sharing a routine name are dropped.
    Remaining scheduler entries are deduped by `(date, start, name)`.
    """
    seen: Set[Tuple[str, str, str]] = set()
    expanded_names: Set[str] = set()
    merged: List[Dict[str, Any]] = []

    for item in routine_assignments or []:
        if not isinstance(item, dict):
            continue
        date_part = str(item.get("date") or item.get("startdate") or "").strip()
        start_part = str(item.get("start") or item.get("starttime") or "").strip()
        name_part = _entry_name_key(item)
        if not date_part or not start_part:
            continue
        key = (date_part, start_part, name_part)
        if key in seen:
            continue
        seen.add(key)
        if name_part:
            expanded_names.add(name_part)
        merged.append(item)

    dropped_by_name = 0
    for item in scheduler_assignments or []:
        if not isinstance(item, dict):
            continue
        date_part = str(item.get("date") or item.get("startdate") or "").strip()
        start_part = str(item.get("start") or item.get("starttime") or "").strip()
        name_part = _entry_name_key(item)
        if not date_part or not start_part:
            continue
        if name_part and name_part in expanded_names:
            dropped_by_name += 1
            continue
        key = (date_part, start_part, name_part)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)

    if dropped_by_name:
        logger.info(
            "routine_expander.scheduler_routine_entries_dropped count=%d names=%s",
            dropped_by_name,
            sorted(expanded_names),
        )
    return merged
