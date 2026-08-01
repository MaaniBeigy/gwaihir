"""Scheduling agent that assigns GameBus tasks into free time slots."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytz
import requests

logger = logging.getLogger("scheduler_core")


def _waking_window_minutes() -> tuple:
    """Waking bracket for task placement, minutes from midnight.

    Default 06:00-22:00 to match the calendar-bench evaluator's daily_window;
    override with COACH_WAKING_START_MIN / COACH_WAKING_END_MIN.
    """
    try:
        wake = int(os.getenv("COACH_WAKING_START_MIN", "360"))
        sleep = int(os.getenv("COACH_WAKING_END_MIN", "1320"))
    except ValueError:
        wake, sleep = 360, 1320
    if not (0 <= wake < sleep <= 1440):
        wake, sleep = 360, 1320
    return wake, sleep


def _fmt_min(minutes: int) -> str:
    """Minutes from midnight to HH:MM; 1440 renders as 23:59 like the benchmark."""
    m = int(minutes)
    if m >= 1440:
        return "23:59"
    if m < 0:
        m = 0
    return "%02d:%02d" % (m // 60, m % 60)


def _parse_iso_minute(value: Any) -> Optional[tuple]:
    """Return (date_str, minute_of_day) from 'YYYY-MM-DDTHH:MM', else None."""
    text = str(value or "")
    if "T" not in text:
        return None
    day, clock = text.split("T", 1)
    try:
        return day, int(clock[0:2]) * 60 + int(clock[3:5])
    except (ValueError, IndexError):
        return None


def _merge_intervals(intervals: List[tuple]) -> List[tuple]:
    """Sort-and-sweep merge of [start, end) minute intervals."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _build_day_timelines(
    normalized_busy: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
    wake: int,
    sleep: int,
) -> str:
    """Per-day busy/free timelines within the waking window, as ground truth.

    Free gaps are the waking window minus the merged busy intervals; each free
    line carries its width so the model can pick a gap without subtracting the
    events itself. Busy lines are shown unmerged with their labels, so the model
    can reference a host label verbatim for a concurrent placement.
    """
    try:
        cursor = datetime.strptime(start_date, "%Y-%m-%d").date()
        last = datetime.strptime(end_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return ""

    by_date: Dict[str, List[tuple]] = {}
    for item in normalized_busy or []:
        start = _parse_iso_minute(item.get("start"))
        end = _parse_iso_minute(item.get("end"))
        if not start or not end:
            continue
        (start_day, start_min), (end_day, end_min) = start, end
        label = item.get("activityType") or item.get("source") or "busy"
        concurrent = bool(item.get("concurrent"))
        if start_day == end_day and end_min > start_min:
            by_date.setdefault(start_day, []).append((start_min, end_min, label, concurrent))
        elif end_day != start_day:
            by_date.setdefault(start_day, []).append((start_min, 1440, label, concurrent))
            if end_min > 0:
                by_date.setdefault(end_day, []).append((0, end_min, label, concurrent))

    lines: List[str] = []
    while cursor <= last:
        day_str = cursor.isoformat()
        events = sorted(by_date.get(day_str, []), key=lambda x: (x[0], x[1]))
        # Only hard events block free time; concurrent-able events do not, so a
        # task may run alongside them.
        hard = [(a, b) for a, b, _, conc in events if not conc]
        clipped = [(max(a, wake), min(b, sleep)) for a, b in hard if min(b, sleep) > max(a, wake)]
        merged = _merge_intervals(clipped)
        free: List[tuple] = []
        prev = wake
        for a, b in merged:
            if a > prev:
                free.append((prev, a))
            prev = max(prev, b)
        if prev < sleep:
            free.append((prev, sleep))

        rows: List[tuple] = []
        for a, b, label, conc in events:
            kind = "concurrent_host" if conc else "busy"
            rows.append(
                (a, '{"type":"%s","label":"%s","start":"%s","end":"%s"}' % (kind, label, _fmt_min(a), _fmt_min(b)))
            )
        for a, b in free:
            rows.append(
                (a, '{"type":"free","label":"free","start":"%s","end":"%s","duration_min":%d}' % (_fmt_min(a), _fmt_min(b), b - a))
            )
        lines.append("%s TIMELINE:" % cursor.strftime("%a %d %b %Y"))
        for _, row in sorted(rows, key=lambda x: x[0]):
            lines.append("  " + row)
        cursor += timedelta(days=1)
    return "\n".join(lines)


def _summarize_assignments_for_logs(schedule: Dict[str, Any], max_items: int = 8) -> Dict[str, Any]:
    assignments = schedule.get("assignments", []) if isinstance(schedule, dict) else []
    if not isinstance(assignments, list):
        return {"count": 0, "preview": []}

    preview: List[Dict[str, Any]] = []
    for item in assignments[:max_items]:
        if not isinstance(item, dict):
            continue
        preview.append(
            {
                "task_id": item.get("task_id"),
                "type": item.get("type"),
                "date": item.get("date") or item.get("startdate"),
                "start": item.get("start") or item.get("starttime"),
                "end": item.get("end") or item.get("endtime"),
            }
        )

    return {
        "count": len(assignments),
        "preview": preview,
    }


def _compact_controller_feedback(feedback: Optional[List[str]], max_items: int = 8) -> List[str]:
    if not isinstance(feedback, list):
        return []
    normalized: List[str] = []
    for item in feedback:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        if len(text) > 220:
            text = f"{text[:219].rstrip()}…"
        normalized.append(text)
    if len(normalized) > max_items:
        normalized = normalized[-max_items:]
    return normalized


def _slim_task_sources_for_llm(task_sources: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `task_sources` containing only what the LLM needs to schedule."""
    unscheduled_gamebus: List[Dict[str, Any]] = task_sources.get("unscheduled_gamebus") or []
    routine_tasks: List[Dict[str, Any]] = task_sources.get("routine_tasks") or []

    slimmed = {k: v for k, v in task_sources.items() if k not in {"gamebus_tasks", "all_tasks"}}
    slimmed["gamebus_tasks"] = unscheduled_gamebus
    slimmed["all_tasks"] = routine_tasks + unscheduled_gamebus
    return slimmed


def _build_scheduler_llm_payload(
    *,
    scheduler_task_sources: Dict[str, Any],
    controller_feedback: Optional[List[str]],
    current_date: str,
    scheduling_period_weeks: int,
    schedule_start_date: str,
    schedule_end_date: str,
    today_date: str,
    today_dayname: str,
    busy_intervals: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "scheduler_task_sources": _slim_task_sources_for_llm(scheduler_task_sources),
        "controller_feedback": _compact_controller_feedback(controller_feedback),
        "busy_intervals": busy_intervals or [],
        "current_date": current_date,
        "schedule_start_date": schedule_start_date,
        "schedule_end_date": schedule_end_date,
        "scheduling_period_weeks": scheduling_period_weeks,
        "scheduling_period_days": scheduling_period_weeks * 7,
        "today_date": today_date,
        "today_dayname": today_dayname,
    }


def _get_scheduling_period_weeks() -> int:
    return int(os.getenv("SCHEDULING_PERIOD_WEEKS", "1"))


def _get_next_monday(base_date) -> Any:
    days_until_monday = (7 - base_date.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    return base_date + timedelta(days=days_until_monday)


def _build_schedule_window(start_date_str: str, period_weeks: int) -> Dict[str, str]:
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = start_date + timedelta(days=(period_weeks * 7) - 1)
    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }


def _weekday_short(date_str: str) -> Optional[str]:
    try:
        return ["mon", "tue", "wed", "thu", "fri", "sat", "sun"][
            datetime.strptime(date_str, "%Y-%m-%d").weekday()
        ]
    except Exception:
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


def _enrich_assignments_with_task_metadata(
    schedule: Dict[str, Any],
    all_tasks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for task in all_tasks:
        task_id = task.get("task_id")
        if task_id is None:
            continue
        by_id[str(task_id)] = task

    assignments = schedule.get("assignments")
    if not isinstance(assignments, list):
        return schedule

    for item in assignments:
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id")
        if task_id is None:
            continue
        source_task = by_id.get(str(task_id))
        if not source_task:
            continue

        source_type = source_task.get("source_type")
        if source_type and not item.get("type"):
            item["type"] = source_type

        if source_type == "routine":
            # Keep routine identity stable from the normalized task sources.
            if source_task.get("name"):
                item["name"] = source_task.get("name")
            if source_task.get("description"):
                item["description"] = source_task.get("description")
            freq = source_task.get("frequency") or {}
            item["frequency"] = {
                "type": freq.get("type"),
                "days": _normalize_days(freq.get("days")),
                "count": freq.get("count"),
            }
            item["scheduling_required"] = bool(source_task.get("scheduling_required"))

    return schedule


def _load_json(value_or_path: Any) -> Any:
    if value_or_path is None:
        return None
    if isinstance(value_or_path, (dict, list)):
        return value_or_path
    if isinstance(value_or_path, str):
        path = value_or_path
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return json.load(f)
        return json.loads(value_or_path)
    return None


def _normalize_task_sources_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload

    if isinstance(payload.get("all_tasks"), list):
        return payload

    nested_payload = payload.get("task_sources_json")
    if isinstance(nested_payload, dict):
        if isinstance(nested_payload.get("all_tasks"), list):
            return nested_payload

    tasks_alias = payload.get("tasks")
    if isinstance(tasks_alias, list):
        normalized = dict(payload)
        normalized["all_tasks"] = tasks_alias
        return normalized

    return payload


def _load_instructions() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    instructions_path = os.path.join(current_dir, "..", "memory", "templates", "Scheduler_instructions.txt")
    if not os.path.exists(instructions_path):
        raise RuntimeError(f"Scheduler instructions not found at: {instructions_path}")
    with open(instructions_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    if not content:
        raise RuntimeError("Scheduler instructions file is empty")
    return content


def _normalize_llm_provider(provider: Optional[str]) -> str:
    normalized = str(provider or "openrouter").strip().lower()
    if normalized in {"openai", "openrouter"}:
        return normalized
    raise RuntimeError("Scheduler only supports the OpenAI-compatible or OpenRouter providers")


def _build_llm_request_config(api_key: str, provider: str) -> Dict[str, Any]:
    if provider == "openrouter":
        api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1").rstrip("/")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/habit-agent",
            "X-Title": "Scheduler Agent",
        }
    else:
        api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    return {
        "url": f"{api_base}/chat/completions",
        "headers": headers,
    }


def _call_llm(
    prompt: str,
    model: str,
    api_key: str,
    provider: Optional[str] = "openrouter",
) -> str:
    normalized_provider = _normalize_llm_provider(provider)

    if not api_key:
        missing_key_name = "OPENROUTER_API_KEY" if normalized_provider == "openrouter" else "OPENAI_API_KEY"
        raise RuntimeError(f"{missing_key_name} environment variable not set")

    request_config = _build_llm_request_config(api_key, normalized_provider)
    headers = request_config["headers"]
    url = request_config["url"]

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, 4):  # pragma: no branch
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(20, 180),
            )
            if resp.status_code != 200:
                provider_name = "OpenRouter" if normalized_provider == "openrouter" else "OpenAI"
                raise RuntimeError(f"{provider_name} API error {resp.status_code}: {resp.text}")

            data = resp.json()
            choices = data.get("choices") if isinstance(data, dict) else None
            if not choices:
                provider_name = "OpenRouter" if normalized_provider == "openrouter" else "OpenAI"
                raise RuntimeError(f"{provider_name} API returned no choices")
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                provider_name = "OpenRouter" if normalized_provider == "openrouter" else "OpenAI"
                raise RuntimeError(f"{provider_name} API returned empty content")
            return content.strip()

        except (requests.exceptions.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            is_last_attempt = attempt == 3
            if is_last_attempt:
                break
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"Scheduler LLM call failed after retries: {str(last_error)}")


def _parse_json_like(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass
    return None


def _extract_json_candidate(text: str) -> Optional[Any]:
    cleaned = text.strip()
    if not cleaned:
        return None

    direct = _parse_json_like(cleaned)
    if direct is not None:
        return direct

    fenced_blocks = re.findall(r"``(?:json)?\s*([\s\S]*?)\s*``", cleaned, flags=re.IGNORECASE)
    for block in fenced_blocks:
        parsed = _parse_json_like(block.strip())
        if parsed is not None:
            return parsed

    for start_char, end_char in (("{", "}"), ("[", "]")):
        start_idx = cleaned.find(start_char)
        end_idx = cleaned.rfind(end_char)
        if start_idx == -1 or end_idx <= start_idx:
            continue
        candidate = cleaned[start_idx : end_idx + 1]
        parsed = _parse_json_like(candidate)
        if parsed is not None:
            return parsed

    return None


def _normalize_llm_schedule(schedule_value: Any) -> Optional[Dict[str, Any]]:
    if schedule_value is None:
        return None

    parsed: Any = None
    if isinstance(schedule_value, dict):
        parsed = schedule_value
    elif isinstance(schedule_value, list):
        parsed = {"assignments": schedule_value}
    else:
        raw_text = str(schedule_value)
        candidate = _extract_json_candidate(raw_text)
        if not candidate:
            return None
        parsed = candidate

    if isinstance(parsed, list):
        parsed = {"assignments": parsed}
    if not isinstance(parsed, dict):
        return None

    for wrapper_key in ("schedule", "result", "output", "data"):
        wrapped = parsed.get(wrapper_key)
        if isinstance(wrapped, (dict, list)):
            nested = _normalize_llm_schedule(wrapped)
            if nested is not None:
                return nested

    if "assignments" not in parsed and isinstance(parsed.get("scheduled_tasks"), list):
        parsed["assignments"] = parsed.get("scheduled_tasks")

    assignments = parsed.get("assignments")
    if not isinstance(assignments, list):
        return None

    normalized_assignments: List[Dict[str, Any]] = []
    for item in assignments:
        if not isinstance(item, dict):
            continue
        if not item.get("date") and item.get("startdate"):
            item["date"] = item.get("startdate")
        if not item.get("start") and item.get("starttime"):
            item["start"] = item.get("starttime")
        if not item.get("end") and item.get("endtime"):
            item["end"] = item.get("endtime")
        if not item.get("date") or not item.get("start"):
            continue
        normalized_assignments.append(item)

    parsed["assignments"] = normalized_assignments
    return parsed


class SchedulingAgent:
    def __init__(
        self,
        *,
        llm_api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.instructions = _load_instructions()
        # When OPENAI_API_KEY+OPENAI_MODEL are set, ignore the OpenRouter-flavored `model` arg
        # and route via OpenAI. Otherwise fall back to OpenRouter + the per-agent SCHEDULER_MODEL.
        openai_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL")
        if openai_key and openai_model:
            self.provider = _normalize_llm_provider("openai")
            self.api_key = openai_key
            self.model = openai_model
        else:
            self.provider = _normalize_llm_provider("openrouter")
            self.api_key = llm_api_key or os.getenv("OPENROUTER_API_KEY")
            self.model = model or os.getenv("SCHEDULER_MODEL") or "minimax/minimax-m2.7:fireworks"

    def schedule_gamebus(
        self,
        scheduler_task_sources: Any,
        controller_feedback: Optional[List[str]] = None,
        timezone_str: str = "Europe/Amsterdam",
        session_id: str = "default",
        scheduling_window: Optional[Dict[str, Any]] = None,
        busy_intervals: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        tz = pytz.timezone(timezone_str)
        now = datetime.now(tz)

        if isinstance(scheduling_window, dict):
            schedule_start_date = str(
                scheduling_window.get("scheduleStartDate") or scheduling_window.get("start_date") or ""
            )
            schedule_end_date = str(
                scheduling_window.get("scheduleEndDate") or scheduling_window.get("end_date") or ""
            )
            current_date = str(
                scheduling_window.get("currentDate")
                or scheduling_window.get("current_date")
                or now.date().isoformat()
            )
            current_dayname = str(
                scheduling_window.get("currentDayname")
                or scheduling_window.get("current_dayname")
                or _weekday_short(current_date)
                or ""
            )
            scheduling_period_weeks = int(
                scheduling_window.get("schedulingPeriodWeeks")
                or scheduling_window.get("scheduling_period_weeks")
                or _get_scheduling_period_weeks()
            )
            if not schedule_start_date or not schedule_end_date:
                raise RuntimeError(
                    "scheduling_window must include scheduleStartDate and scheduleEndDate (YYYY-MM-DD)."
                )
        else:
            current_date = now.date().isoformat()
            schedule_start_date = current_date
            scheduling_period_weeks = _get_scheduling_period_weeks()
            schedule_end_date = (now.date() + timedelta(days=(scheduling_period_weeks * 7) - 1)).isoformat()
            current_dayname = _weekday_short(current_date) or ""

        task_sources_payload = (
            _load_json(scheduler_task_sources) if scheduler_task_sources is not None else None
        )
        task_sources_payload = _normalize_task_sources_payload(task_sources_payload)

        if not isinstance(task_sources_payload, dict):
            raise RuntimeError("scheduler_task_sources is required and must be a valid JSON object.")

        all_tasks = task_sources_payload.get("all_tasks", [])
        if not all_tasks:
            raise RuntimeError("No schedulable tasks found in scheduler_task_sources.")

        routine_tasks = task_sources_payload.get("routine_tasks")
        if not isinstance(routine_tasks, list):
            routine_tasks = [task for task in all_tasks if task.get("source_type") == "routine"]

        gamebus_tasks = task_sources_payload.get("gamebus_tasks")
        if not isinstance(gamebus_tasks, list):
            gamebus_tasks = [task for task in all_tasks if task.get("source_type") == "gamebus"]

        routine_schedulable_tasks = task_sources_payload.get("routine_schedulable_tasks")
        if not isinstance(routine_schedulable_tasks, list):
            routine_schedulable_tasks = [
                task for task in routine_tasks if bool(task.get("scheduling_required"))
            ]

        routine_context_only_tasks = task_sources_payload.get("routine_context_only_tasks")
        if not isinstance(routine_context_only_tasks, list):
            routine_context_only_tasks = [
                task for task in routine_tasks if not bool(task.get("scheduling_required"))
            ]

        non_schedulable_task_ids = task_sources_payload.get("non_schedulable_task_ids")
        if not isinstance(non_schedulable_task_ids, list):
            non_schedulable_task_ids = [
                str(task.get("task_id"))
                for task in routine_context_only_tasks
                if task.get("task_id") is not None
            ]

        normalized_busy = _normalize_busy_intervals(busy_intervals)
        wake_min, sleep_min = _waking_window_minutes()
        day_timelines = _build_day_timelines(
            normalized_busy, schedule_start_date, schedule_end_date, wake_min, sleep_min
        )
        # The day timelines now carry the busy events, so drop the duplicate
        # busy_intervals JSON from the payload to save tokens and avoid two views.
        llm_payload = _build_scheduler_llm_payload(
            scheduler_task_sources=task_sources_payload,
            controller_feedback=controller_feedback,
            current_date=current_date,
            scheduling_period_weeks=scheduling_period_weeks,
            schedule_start_date=schedule_start_date,
            schedule_end_date=schedule_end_date,
            today_date=current_date,
            today_dayname=current_dayname,
            busy_intervals=[],
        )
        llm_prompt = (
            f"{self.instructions}\n\n"
            f"## WAKING WINDOW\n"
            f"Place every task inside {_fmt_min(wake_min)}-{_fmt_min(sleep_min)}. "
            f"Never start a task before {_fmt_min(wake_min)} or end it after {_fmt_min(sleep_min)}.\n\n"
            "## DAY TIMELINES (GROUND TRUTH - use these exact intervals)\n"
            "Each day lists the user's `busy` events, the `concurrent_host` events, and the open "
            "`free` gaps between busy events inside the waking window. A `free` interval's "
            "`duration_min` is how long that gap is. Place a STANDALONE task inside a `free` gap "
            "whose duration_min is at least the task's duration; never place a standalone task over "
            "a `busy` interval. A `concurrent_host` event (e.g. a snack or water break) does not "
            "block time: you may place a CONCURRENT task inside it when the task is a proper fit, "
            "which is the best way to earn a merge. For any concurrent placement set "
            "`concurrent_with_event` to the host's `label` verbatim.\n\n"
            f"{day_timelines}\n\n"
            "Return ONLY a valid JSON object matching the required output schema.\n\n"
            f"Scheduling window: {schedule_start_date} to {schedule_end_date} (inclusive). "
            "Every assignment must stay inside this date window and the waking window.\n\n"
            f"Payload:\n{json.dumps(llm_payload, ensure_ascii=False, separators=(',', ':'))}"
        )
        logger.info(
            "scheduler.controller_feedback session_id=%s feedback=%s", session_id, controller_feedback or []
        )
        max_schedule_attempts = 3
        llm_schedule: Optional[Dict[str, Any]] = None
        all_violations: List[str] = []
        current_prompt = llm_prompt
        degraded_fallback_used = False
        schedule_attempts_used = 0

        for attempt_index in range(1, max_schedule_attempts + 1):  # pragma: no branch
            schedule_attempts_used = attempt_index
            llm_response = _call_llm(current_prompt, self.model, self.api_key, self.provider)
            llm_schedule = _normalize_llm_schedule(llm_response)

            if not llm_schedule:
                all_violations = ["LLM response did not include a valid schedule JSON payload."]
            else:
                all_violations = []

            if not all_violations:
                if attempt_index > 1:
                    logger.info("scheduler.retry_valid session_id=%s attempt=%s", session_id, attempt_index)
                break

            attempt_summary = _summarize_assignments_for_logs(llm_schedule or {})
            log_template = (
                "scheduler.attempt1_invalid session_id=%s violations=%s assignments=%s"
                if attempt_index == 1
                else "scheduler.retry_invalid session_id=%s attempt=%s violations=%s assignments=%s"
            )
            if attempt_index == 1:
                logger.warning(log_template, session_id, all_violations[:5], attempt_summary)
            else:
                logger.warning(log_template, session_id, attempt_index, all_violations[:5], attempt_summary)

            if attempt_index == max_schedule_attempts:
                degraded_fallback_used = True
                logger.error(
                    "scheduler.max_retries_reached_using_last_schedule session_id=%s attempts=%s violations=%s",
                    session_id,
                    max_schedule_attempts,
                    all_violations[:5],
                )
                break

            current_prompt = (
                f"{llm_prompt}\n\n"
                f"Attempt {attempt_index} was invalid. Fix these violations and return ONLY corrected JSON:\n"
                + "\n".join(f"- {v}" for v in all_violations)
            )

        if not llm_schedule:
            raise RuntimeError("Scheduler did not produce a valid schedule JSON payload.")

        llm_schedule = _enrich_assignments_with_task_metadata(llm_schedule, all_tasks)

        assignments = llm_schedule.get("assignments", []) if isinstance(llm_schedule, dict) else []

        result = {
            "generated_at": datetime.now(tz).isoformat(),
            "timezone": timezone_str,
            "total_tasks": len(all_tasks),
            "gamebus_tasks": len(gamebus_tasks),
            "routine_tasks": len(routine_tasks),
            "routine_schedulable_tasks": len(routine_schedulable_tasks),
            "routine_context_only_tasks": len(routine_context_only_tasks),
            "scheduled_assignments": len(assignments),
            "scheduling_period_weeks": scheduling_period_weeks,
            "schedule_start_date": schedule_start_date,
            "schedule_end_date": schedule_end_date,
            "scheduler_attempts_used": schedule_attempts_used,
            "scheduler_retries_exhausted": degraded_fallback_used,
            "scheduler_unresolved_violations": all_violations,
        }

        result["schedule_json"] = llm_schedule
        result["assignments"] = assignments

        result["task_schema"] = {
            "normalized_at": task_sources_payload.get("generated_at"),
            "gamebus_tasks": len(gamebus_tasks),
            "routine_tasks": len(routine_tasks),
            "routine_schedulable_tasks": len(routine_schedulable_tasks),
            "routine_context_only_tasks": len(routine_context_only_tasks),
        }

        return result


def _normalize_busy_intervals(busy_intervals: Any) -> List[Dict[str, Any]]:
    """Coerce inbound `scheduledActivities` to a compact busy-interval list."""
    if not isinstance(busy_intervals, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in busy_intervals:
        if not isinstance(item, dict):
            continue
        start = item.get("startDate") or item.get("start") or item.get("starttime")
        end = item.get("endDate") or item.get("end") or item.get("endtime")
        if not start or not end:
            continue
        compact = {
            "start": str(start),
            "end": str(end),
            "source": str(item.get("source") or "USER").upper(),
        }
        if item.get("activityType"):
            compact["activityType"] = item.get("activityType")
        if item.get("allDay") is not None:
            compact["allDay"] = bool(item.get("allDay"))
        if item.get("concurrent") is not None:
            compact["concurrent"] = bool(item.get("concurrent"))
        normalized.append(compact)
    return normalized


def schedule_from_inputs(
    scheduler_task_sources_input: Any,
    controller_feedback: Any = None,
    timezone_str: str = "Europe/Amsterdam",
    session_id: str = "default",
    scheduling_window: Optional[Dict[str, Any]] = None,
    busy_intervals: Optional[List[Dict[str, Any]]] = None,
    llm_api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_feedback: List[str] = []
    if isinstance(controller_feedback, list):
        normalized_feedback = [str(item) for item in controller_feedback if item is not None]
    elif isinstance(controller_feedback, str) and controller_feedback.strip():
        try:
            parsed_feedback = json.loads(controller_feedback)
            if isinstance(parsed_feedback, list):
                normalized_feedback = [str(item) for item in parsed_feedback if item is not None]
            else:
                normalized_feedback = [str(controller_feedback)]
        except Exception:
            normalized_feedback = [str(controller_feedback)]

    agent = SchedulingAgent(llm_api_key=llm_api_key, model=model)
    return agent.schedule_gamebus(
        scheduler_task_sources=scheduler_task_sources_input,
        controller_feedback=normalized_feedback,
        timezone_str=timezone_str,
        session_id=session_id,
        scheduling_window=scheduling_window,
        busy_intervals=busy_intervals,
    )
