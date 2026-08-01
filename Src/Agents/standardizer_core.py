"""Standardizer agent that converts natural language habits into standardized JSON activities."""

import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import json_repair
import requests

from .llm_config import LlmConfig, resolve_llm_config


class StandardizerAgent:
    """Convert natural language descriptions to standardized JSON activities."""

    def __init__(self):
        self.logger = logging.getLogger("standardizer")
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
        try:
            self._llm = resolve_llm_config(
                "STANDARDIZER_MODEL",
                fallback_model="openai/gpt-oss-120b:free",
            )
        except RuntimeError:
            # Test fallback: seed a dummy OpenRouter key so importing the module never crashes.
            self._llm = LlmConfig(
                api_base="https://openrouter.ai/api/v1",
                api_key="sk-or-v1-dummy-for-tests",
                model=os.getenv("STANDARDIZER_MODEL", "openai/gpt-oss-120b:free"),
                provider="openrouter",
            )
        self.api_key = self._llm.api_key
        self.api_base = self._llm.api_base
        self.model = self._llm.model
        self.instructions = self._load_instructions()
        self.conversation_history: List[Dict[str, str]] = []

    def _load_instructions(self) -> str:
        """Load standardizer instructions from template."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        instructions_path = os.path.join(
            current_dir, "..", "memory", "templates", "Standardizer_instructions.txt"
        )

        if not os.path.exists(instructions_path):
            raise RuntimeError(f"System instructions file not found at: {instructions_path}")

        with open(instructions_path, "r", encoding="utf-8") as f:
            content = f.read()
            if not content:
                raise RuntimeError("System instructions file is empty")
            return content

    def _get_today_date_string(self) -> str:
        """Return today as YYYY-MM-DD."""
        return datetime.now().strftime("%Y-%m-%d")

    def _send_to_openrouter(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to the configured LLM endpoint and return the reply content."""
        headers = self._llm.headers(title="Habit Standardizer")

        today = self._get_today_date_string()
        system_with_date = f"{self.instructions}\n\n**Today's date (for reference): {today}**"

        full_messages = [{"role": "system", "content": system_with_date}] + messages

        payload = {
            "model": self.model,
            "messages": full_messages,
            "temperature": 0.1,
            "max_tokens": 4000,
            "stop": None,
            "response_format": {"type": "json_object"},
        }

        # Retry on null `message.content`: some free OpenRouter models return HTTP 200 with empty content.
        last_body: Optional[str] = None
        for attempt in range(1, 4):
            response = requests.post(
                self._llm.chat_completions_url,
                headers=headers,
                json=payload,
                timeout=120,
            )
            if response.status_code != 200:
                raise RuntimeError(f"{self._llm.provider} API error {response.status_code}: {response.text}")
            try:
                content = response.json().get("choices", [{}])[0].get("message", {}).get("content")
            except (ValueError, IndexError, AttributeError):
                content = None
            if isinstance(content, str) and content.strip():
                return content
            last_body = response.text[:500] if response.text else "<empty>"
            self.logger.warning(
                "Empty content from %s on attempt %d/3 (model=%s, body=%s)",
                self._llm.provider,
                attempt,
                self.model,
                last_body,
            )
        raise RuntimeError(
            f"{self._llm.provider} returned empty content after 3 attempts "
            f"(model={self.model}, last body preview={last_body})"
        )

    def _validate_json_response(self, raw_response: str) -> Tuple[bool, Optional[List[Dict]]]:
        """Validate JSON response, strip fences, and extract Activities list."""
        if not isinstance(raw_response, str):
            return False, None
        text = raw_response.strip() if raw_response is not None else ""

        # Strip markdown code fences.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()

        # json_repair recovers from malformed JSON (truncation, control chars, missing commas).
        try:
            data = json_repair.loads(text)
        except (ValueError, TypeError, json.JSONDecodeError):
            return False, None
        if data == "":
            return False, None

        if isinstance(data, str):
            try:
                data = json_repair.loads(data)
            except (ValueError, TypeError, json.JSONDecodeError):
                return False, None

        if not isinstance(data, dict):
            return False, None

        activities = data.get("Activities")
        if not isinstance(activities, list):
            return False, None

        return True, activities

    def _parse_time_range(self, text: str) -> Optional[Tuple[int, int]]:
        """Extract a time range from text and return `(start_minutes, end_minutes)`."""
        if not text:
            return None
        patterns = [
            r"(\d{1,2}:\d{2})\s*(?:-|to|until|till)\s*(\d{1,2}:\d{2})",
            r"from\s+(\d{1,2}:\d{2})\s+(?:to|until|till)\s+(\d{1,2}:\d{2})",
            r"start\w*\s+(?:at\s+)?(\d{1,2}:\d{2}).*?(?:finish|end)\w*\s+(?:at\s+)?(\d{1,2}:\d{2})",
        ]

        match = None
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                break

        if match:
            start_str, end_str = match.groups()
        else:
            times = re.findall(r"\b\d{1,2}:\d{2}\b", text)
            if len(times) >= 2 and re.search(r"start|begin|finish|end", text, re.IGNORECASE):
                start_str, end_str = times[0], times[1]
            else:
                return None
        try:
            start_h, start_m = [int(part) for part in start_str.split(":")]
            end_h, end_m = [int(part) for part in end_str.split(":")]
        except ValueError:
            return None
        if not (0 <= start_h <= 23 and 0 <= end_h <= 23 and 0 <= start_m <= 59 and 0 <= end_m <= 59):
            return None
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        if end_minutes <= start_minutes:
            end_minutes += 24 * 60
        return start_minutes, end_minutes

    def _normalize_durations(self, activities: List[Dict[str, any]]) -> None:
        """Recompute durations when an explicit time range is in the description."""
        for activity in activities:
            if activity.get("IsInstantaneous"):
                continue
            description = activity.get("description") or ""
            notes = activity.get("notes") or ""
            time_range = self._parse_time_range(f"{description} {notes}")
            if not time_range:
                continue
            start_minutes, end_minutes = time_range
            computed_duration = end_minutes - start_minutes
            if computed_duration <= 0:
                continue
            existing_duration = activity.get("activity_duration")
            if existing_duration is None or abs(int(existing_duration) - computed_duration) >= 5:
                activity["activity_duration"] = computed_duration
            if not ((activity.get("time") or {}).get("preferred")):
                activity.setdefault("time", {})
                activity["time"]["preferred"] = f"{start_minutes // 60:02d}:{start_minutes % 60:02d}"

    def standardize(self, message: str, clear_history: bool = False) -> str:
        """Standardize natural language habits to JSON.

        Args:
            message: user message, typically the interview history.
            clear_history: clear conversation history before processing.

        Returns:
            JSON string with standardized activities or the raw response if validation fails.
        """
        if clear_history:
            self.conversation_history = []

        self.conversation_history.append({"role": "user", "content": message})

        raw_model_reply = self._send_to_openrouter(self.conversation_history)
        is_valid, activities = self._validate_json_response(raw_model_reply)

        if is_valid and activities is not None:
            self._normalize_durations(activities)
            clean_reply = json.dumps({"Activities": activities}, ensure_ascii=False)
        else:
            safe_len = len(raw_model_reply) if raw_model_reply is not None else 0
            safe_preview = raw_model_reply[:500] if isinstance(raw_model_reply, str) else str(raw_model_reply)
            self.logger.warning("Invalid JSON response from model. Length: %s", safe_len)
            self.logger.warning("First 500 chars: %s", safe_preview)
            clean_reply = raw_model_reply

        self.conversation_history.append({"role": "assistant", "content": clean_reply})

        return clean_reply

    def reset(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Return a copy of the conversation history."""
        return self.conversation_history.copy()
