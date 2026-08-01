"""Deterministic clean-up pass on the standardizer output.

Patches `frequency.{type,days}` when the description implies workdays, weekends,
or explicit weekdays, including day lists the standardizer left incomplete.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("coach.standardized_normalizer")


_WEEKDAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_WORKDAYS = ["mon", "tue", "wed", "thu", "fri"]
_WEEKEND = ["sat", "sun"]

# Phrases implying Mon-Fri.
_WORKDAY_PHRASE = re.compile(
    r"\b("
    r"workdays?|"
    r"work\s+days?|"
    r"weekdays?|"
    r"week\s+days?|"
    r"mon(?:day)?\s*(?:-|to|–)\s*fri(?:day)?|"
    r"mon-fri"
    r")\b",
    re.IGNORECASE,
)

# Phrases implying Sat+Sun.
_WEEKEND_PHRASE = re.compile(r"\bweekends?\b", re.IGNORECASE)

# Map weekday tokens to short codes.
_DAY_TOKEN = re.compile(
    r"\b("
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun"
    r")s?\b",
    re.IGNORECASE,
)
_DAY_ALIAS = {
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


def _explicit_days_in_text(text: str) -> List[str]:
    """Return ordered weekday short codes mentioned in text, deduped."""
    out: List[str] = []
    for match in _DAY_TOKEN.finditer(text or ""):
        code = _DAY_ALIAS.get(match.group(1).lower())
        if code and code not in out:
            out.append(code)
    return out


def _set_frequency(
    activity: Dict[str, Any],
    *,
    days: List[str],
    reason: str,
) -> None:
    """Update `activity.frequency` in place and log the correction."""
    freq = activity.get("frequency") if isinstance(activity.get("frequency"), dict) else {}
    before = {
        "type": freq.get("type"),
        "days": freq.get("days"),
    }
    freq["type"] = "specific_days"
    freq["days"] = days
    if freq.get("count") in (None, 0):
        freq["count"] = 1
    activity["frequency"] = freq
    logger.info(
        "standardized_normalizer.frequency_corrected name=%r reason=%s before=%s after=%s",
        activity.get("name"),
        reason,
        before,
        {"type": freq["type"], "days": freq["days"]},
    )


def _normalize_one(activity: Dict[str, Any]) -> None:
    if not isinstance(activity, dict):
        return
    freq = activity.get("frequency") if isinstance(activity.get("frequency"), dict) else {}
    ftype = str(freq.get("type") or "").strip().lower()
    days = [d for d in (freq.get("days") or []) if isinstance(d, str)]
    current = {_DAY_ALIAS.get(d.strip().lower(), d.strip().lower()) for d in days}

    description = str(activity.get("description") or "")
    notes = str(activity.get("notes") or "")
    haystack = f"{description} {notes}"

    workday_match = _WORKDAY_PHRASE.search(haystack)
    weekend_match = _WEEKEND_PHRASE.search(haystack)

    # A workdays or weekend phrase implies the full day set. Apply it when a canonical
    # day is missing, which also repairs a list collapsed to its endpoints like `["mon", "fri"]`.
    # Checked before single-day tokens so a "Monday to Friday" range is not read as just two days.
    if workday_match and not current.issuperset(_WORKDAYS):
        _set_frequency(activity, days=list(_WORKDAYS), reason="workdays_phrase")
        return
    if weekend_match and not current.issuperset(_WEEKEND):
        _set_frequency(activity, days=list(_WEEKEND), reason="weekend_phrase")
        return

    # Otherwise only fill in days when the frequency is still ambiguous.
    if ftype == "specific_days" and days:
        return

    explicit_days = _explicit_days_in_text(haystack)
    if explicit_days:
        _set_frequency(activity, days=explicit_days, reason="explicit_days_in_description")


def normalize_standardized_activities(payload: Any) -> Any:
    """Correct `frequency.{type,days}` from description text.

    Accepts a dict with `Activities`, a bare list, or a JSON string, and returns the same shape.
    """
    if payload is None:
        return payload

    obj = payload
    was_string = False
    if isinstance(payload, str):
        try:
            obj = json.loads(payload)
            was_string = True
        except json.JSONDecodeError:
            logger.warning("standardized_normalizer.unparseable_string_input ignored")
            return payload

    if isinstance(obj, dict) and isinstance(obj.get("Activities"), list):
        for a in obj["Activities"]:
            _normalize_one(a)
    elif isinstance(obj, list):
        for a in obj:
            _normalize_one(a)
    else:
        return payload

    if was_string:
        return json.dumps(obj, ensure_ascii=False)
    return obj
