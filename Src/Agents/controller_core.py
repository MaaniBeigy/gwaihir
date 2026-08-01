"""Controller agent that validates scheduler output and flags re-scheduling."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

from src.agents.llm_config import resolve_llm_config

logger = logging.getLogger("controller_core")


def _truncate_text(value: Any, max_len: int = 240) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 1].rstrip()}…"


def _compact_assignment_for_llm(item: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {}

    keep_fields = [
        "date",
        "type",
        "task_id",
        "name",
        "description",
        "start",
        "end",
        "duration_min",
        "slot_used",
        "allen_relations",
    ]
    compact: Dict[str, Any] = {}
    for key in keep_fields:
        value = item.get(key)
        if value is None:
            continue
        if key in {"name", "description"}:
            value = _truncate_text(value, 260 if key == "name" else 320)
        compact[key] = value
    return compact


def _extract_assignments(schedule_payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the assignments list from either a scheduler envelope or a raw schedule payload."""
    if not isinstance(schedule_payload, dict):
        return []

    assignments = schedule_payload.get("assignments")
    if isinstance(assignments, list):
        return [item for item in assignments if isinstance(item, dict)]

    nested = schedule_payload.get("schedule_json")
    if isinstance(nested, dict):
        nested_assignments = nested.get("assignments")
        if isinstance(nested_assignments, list):
            return [item for item in nested_assignments if isinstance(item, dict)]

    return []


def _clock_to_min(value: Any) -> Optional[int]:
    """Parse 'HH:MM' (optionally an ISO 'YYYY-MM-DDTHH:MM') to minutes from midnight."""
    text = str(value or "")
    if "T" in text:
        text = text.split("T", 1)[1]
    match = re.match(r"^(\d{1,2}):(\d{2})", text)
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm_label(text: Any) -> str:
    return _NORM_RE.sub("_", str(text or "").lower()).strip("_")


def _busy_by_date(busy_intervals: Optional[List[Dict[str, Any]]]) -> Dict[str, List[tuple]]:
    """Group intervals by date as (start_min, end_min, label, concurrent), splitting midnight."""
    index: Dict[str, List[tuple]] = {}
    for item in busy_intervals or []:
        if not isinstance(item, dict):
            continue
        start_raw = str(item.get("start") or item.get("startDate") or "")
        end_raw = str(item.get("end") or item.get("endDate") or "")
        start_day = start_raw.split("T", 1)[0]
        end_day = end_raw.split("T", 1)[0] or start_day
        start = _clock_to_min(start_raw)
        end = _clock_to_min(end_raw)
        label = item.get("activityType") or item.get("source") or "event"
        concurrent = bool(item.get("concurrent"))
        if start is None or end is None or not start_day:
            continue
        if end_day == start_day and end > start:
            index.setdefault(start_day, []).append((start, end, label, concurrent))
        elif end_day != start_day:
            index.setdefault(start_day, []).append((start, 1440, label, concurrent))
            if end > 0:
                index.setdefault(end_day, []).append((0, end, label, concurrent))
    return index


def _calendar_violations(
    assignments: List[Dict[str, Any]],
    busy_by_date: Dict[str, List[tuple]],
    wake: int,
    sleep: int,
    expected_tasks: int,
    max_items: int = 8,
) -> List[str]:
    """Deterministic calendar checks: overlap, containment, waking window, coverage.

    The coach controller was previously blind to the calendar and rubber-stamped
    every schedule. These checks give it the busy intervals so it can catch the
    real placement mistakes and feed concrete fixes back to the scheduler.
    """
    violations: List[str] = []
    placed_by_date: Dict[str, List[tuple]] = {}
    for item in assignments:
        date = item.get("date")
        start = _clock_to_min(item.get("start"))
        end = _clock_to_min(item.get("end"))
        host = item.get("concurrent_with_event")
        name = item.get("name") or item.get("task_id") or "task"
        if not date or start is None:
            continue
        if end is None or end <= start:
            end = start + 15
        if start < wake or end > sleep:
            violations.append(
                'Move "%s" on %s into %02d:%02d-%02d:%02d: it runs outside waking hours.'
                % (name, date, wake // 60, wake % 60, sleep // 60, sleep % 60)
            )
            continue
        if host:
            hosts = [(s, e) for s, e, label, _ in busy_by_date.get(date, []) if _norm_label(label) == _norm_label(host)]
            if not hosts:
                violations.append(
                    'Place "%s" standalone on %s: there is no "%s" event that day to be concurrent with.'
                    % (name, date, host)
                )
            elif not any(s <= start and end <= e for s, e in hosts):
                violations.append('Keep "%s" fully inside "%s" on %s.' % (name, host, date))
        else:
            # Only hard (non-concurrent-able) events are real conflicts for a
            # standalone task; overlapping a snack or water break is fine.
            hit = [label for s, e, label, conc in busy_by_date.get(date, []) if not conc and start < e and s < end]
            if hit:
                violations.append(
                    'Move "%s" off %s %s-%s: it overlaps "%s". Use a free gap that day.'
                    % (name, date, item.get("start"), item.get("end") or "?", hit[0])
                )
            else:
                for ps, pe in placed_by_date.get(date, []):
                    if start < pe and ps < end:
                        violations.append('Move "%s": it overlaps another task you placed on %s.' % (name, date))
                        break
                placed_by_date.setdefault(date, []).append((start, end))
        if len(violations) >= max_items:
            break

    distinct = len({item.get("task_id") for item in assignments if item.get("task_id") is not None})
    if expected_tasks and distinct < expected_tasks and len(violations) < max_items:
        violations.append(
            "Only %d of %d tasks are scheduled; place the remaining ones in free gaps within the window."
            % (distinct, expected_tasks)
        )
    return violations[:max_items]


def _build_controller_llm_payload(
    schedule_payload: Dict[str, Any],
    expected_gamebus_tasks: int,
    scheduler_task_sources: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_assignments = _extract_assignments(schedule_payload)
    compact_assignments = [
        _compact_assignment_for_llm(item) for item in normalized_assignments if isinstance(item, dict)
    ]

    routine_expected = 0
    if isinstance(scheduler_task_sources, dict):
        routine_schedulable_tasks = scheduler_task_sources.get("routine_schedulable_tasks")
        if isinstance(routine_schedulable_tasks, list):
            routine_expected = len([t for t in routine_schedulable_tasks if isinstance(t, dict)])
        else:
            all_tasks = scheduler_task_sources.get("all_tasks") or []
            routine_expected = len(
                [
                    t
                    for t in all_tasks
                    if isinstance(t, dict)
                    and str(t.get("source_type") or "").strip().lower() == "routine"
                    and bool(t.get("scheduling_required"))
                ]
            )

    return {
        "expected_gamebus_tasks": int(expected_gamebus_tasks or 0),
        "scheduler_task_sources_summary": {
            "routine_expected": int(routine_expected),
            "routine_checks_enabled": bool(routine_expected),
        },
        "schedule_payload": {
            "assignments": compact_assignments,
            "summary": {
                "assignment_count": len(compact_assignments),
                "gamebus_count": len(
                    [
                        item
                        for item in compact_assignments
                        if str(item.get("type") or "").strip().lower() == "gamebus"
                    ]
                ),
            },
        },
    }


def _trace_log(event: str, trace_id: str = "-", **fields: Any) -> None:
    payload = {"trace_id": trace_id, **fields}
    parts = " ".join(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in payload.items())
    logger.info("event=%s %s", event, parts)


class ControllerAgent:
    def __init__(self, enable_llm: bool = True) -> None:
        self.enable_llm = enable_llm
        # Lazy resolve so a missing key does not blow up construction.
        self._llm = None
        try:
            self._llm = resolve_llm_config(
                "CONTROLLER_MODEL",
                fallback_model="openai/gpt-oss-120b:free",
            )
            self.api_key = self._llm.api_key
            self.model = self._llm.model
        except RuntimeError:
            self.api_key = None
            self.model = os.getenv("CONTROLLER_MODEL", "openai/gpt-oss-120b:free")
        self.instructions = self._load_instructions()

    def _load_instructions(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        instructions_path = os.path.join(
            current_dir, "..", "memory", "templates", "Controller_instructions.txt"
        )
        if not os.path.exists(instructions_path):
            raise RuntimeError(f"Controller instructions not found at: {instructions_path}")
        with open(instructions_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if not content:
            raise RuntimeError("Controller instructions file is empty")
        return content

    def _parse_json_like(self, text: str) -> Optional[Any]:
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

    def _extract_json_candidate(self, text: str) -> Optional[Dict[str, Any]]:
        cleaned = (text or "").strip()
        if not cleaned:
            return None

        direct = self._parse_json_like(cleaned)
        if isinstance(direct, dict):
            return direct

        fenced_blocks = re.findall(r"``(?:json)?\s*([\s\S]*?)\s*``", cleaned, flags=re.IGNORECASE)
        for block in fenced_blocks:
            parsed = self._parse_json_like(block.strip())
            if isinstance(parsed, dict):
                return parsed

        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx > start_idx:
            parsed = self._parse_json_like(cleaned[start_idx : end_idx + 1])
            if isinstance(parsed, dict):
                return parsed

        return None

    def _call_llm(self, prompt: str, trace_id: str = "-") -> Dict[str, Any]:
        if not self.enable_llm:
            return {"is_valid": True, "violations": [], "fix_advice": []}
        if not self._llm or not self.api_key:
            raise RuntimeError("No LLM configured (set OPENAI_API_KEY+OPENAI_MODEL or OPENROUTER_API_KEY)")

        headers = self._llm.headers(title="Controller Agent")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                attempt_started_at = time.perf_counter()
                _trace_log(
                    "controller.llm.attempt.start", trace_id=trace_id, model=self.model, attempt=attempt
                )
                resp = requests.post(
                    self._llm.chat_completions_url,
                    headers=headers,
                    json=payload,
                    timeout=(20, 120),
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"{self._llm.provider} API error {resp.status_code}: {resp.text}")

                data = resp.json()
                choices = data.get("choices") if isinstance(data, dict) else None
                if not choices:
                    raise RuntimeError("OpenRouter API returned no choices")
                message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("OpenRouter API returned empty content")

                parsed = self._extract_json_candidate(content)
                if not isinstance(parsed, dict):
                    raise RuntimeError("Controller LLM response was not valid JSON")
                _trace_log(
                    "controller.llm.attempt.success",
                    trace_id=trace_id,
                    model=self.model,
                    attempt=attempt,
                    duration_ms=round((time.perf_counter() - attempt_started_at) * 1000, 2),
                )
                return parsed

            except (requests.exceptions.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
                _trace_log(
                    "controller.llm.attempt.error",
                    trace_id=trace_id,
                    model=self.model,
                    attempt=attempt,
                    is_last_attempt=attempt == 3,
                    error=str(exc),
                )
                if attempt < 3:
                    time.sleep(1.5 * attempt)

        raise RuntimeError(f"Controller LLM call failed after retries: {str(last_error)}")

    def validate_schedule(
        self,
        schedule_payload: Dict[str, Any],
        expected_gamebus_tasks: int = 0,
        scheduler_task_sources: Optional[Dict[str, Any]] = None,
        session_id: str = "default",
        trace_id: str = "-",
        busy_intervals: Optional[List[Dict[str, Any]]] = None,
        waking_window: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        if not isinstance(schedule_payload, dict):
            raise RuntimeError("schedule_payload is required and must be a dict")

        validate_started_at = time.perf_counter()
        _trace_log(
            "controller.validate.start",
            trace_id=trace_id,
            session_id=session_id,
            expected_gamebus_tasks=int(expected_gamebus_tasks or 0),
            assignments=len(_extract_assignments(schedule_payload)),
        )
        try:
            llm_payload = _build_controller_llm_payload(
                schedule_payload=schedule_payload,
                expected_gamebus_tasks=expected_gamebus_tasks,
                scheduler_task_sources=scheduler_task_sources,
            )
            prompt = (
                f"{self.instructions}\n\n"
                "Validate this schedule payload and return ONLY JSON in the required format.\n"
                "If payload.scheduler_task_sources_summary.routine_checks_enabled is false, do not report missing routine-task violations.\n"
                "For recurring routine tasks, treat them as assigned when at least one assignment with the same task_id appears on an allowed date; do not require exactly one matching row.\n"
                "Use lenient routine coverage: allow up to 1 unassigned schedulable routine task when routine_expected >= 3; otherwise allow 0 missing.\n"
                "Use lenient gamebus minimum: minimum_required = max(1, min(2, ceil(total_gamebus_available/3))).\n"
                f"payload={json.dumps(llm_payload, ensure_ascii=False, separators=(',', ':'))}"
            )
            llm_candidate = self._call_llm(prompt, trace_id=trace_id)
            llm_result = {
                "is_valid": bool(llm_candidate.get("is_valid", True)),
                "violations": [str(v) for v in (llm_candidate.get("violations") or [])],
                "fix_advice": [str(v) for v in (llm_candidate.get("fix_advice") or [])],
            }
        except Exception as exc:
            # Safe fallback: keep the pipeline alive with a degraded controller result.
            fallback_message = f"Controller LLM unavailable during validation: {str(exc)}"
            _trace_log(
                "controller.validate.degraded_fallback",
                trace_id=trace_id,
                reason=fallback_message,
            )
            llm_result = {
                "is_valid": True,
                "violations": [
                    "Controller validation could not be completed because the LLM response was unavailable.",
                    fallback_message,
                ],
                "fix_advice": [
                    "Retry controller validation for the same session.",
                    "If this persists, switch to a more stable model/provider and re-run scheduling validation.",
                    "Continue with degraded fallback path only after max retries, and report unresolved validation status.",
                ],
            }

        # Deterministic calendar checks. These are authoritative: the LLM
        # controller cannot see the calendar, so the busy intervals are what let
        # the loop catch real overlaps and drive a re-schedule.
        calendar_violations: List[str] = []
        if busy_intervals:
            wake, sleep = waking_window if waking_window else (360, 1320)
            calendar_violations = _calendar_violations(
                _extract_assignments(schedule_payload),
                _busy_by_date(busy_intervals),
                int(wake),
                int(sleep),
                int(expected_gamebus_tasks or 0),
            )

        merged_violations = [str(v) for v in (llm_result.get("violations") or [])]
        merged_violations = calendar_violations + merged_violations
        merged_violations = list(dict.fromkeys(merged_violations))

        merged_fix_advice = list(
            dict.fromkeys(
                [
                    *[str(v) for v in (llm_result.get("fix_advice") or [])],
                ]
            )
        )

        # Any deterministic calendar violation makes the schedule invalid so the
        # graph loops and the scheduler fixes it, regardless of the LLM verdict.
        is_valid = bool(llm_result.get("is_valid", True)) and not calendar_violations

        result = {
            "is_valid": is_valid,
            "violations": merged_violations,
            "fix_advice": merged_fix_advice,
            "summary": {
                "assignments": len(_extract_assignments(schedule_payload)),
                "expected_gamebus_tasks": int(expected_gamebus_tasks or 0),
            },
            "llm_used": bool(self.enable_llm),
        }

        _trace_log(
            "controller.validate.completed",
            trace_id=trace_id,
            is_valid=bool(result.get("is_valid")),
            duration_ms=round((time.perf_counter() - validate_started_at) * 1000, 2),
        )
        return result
