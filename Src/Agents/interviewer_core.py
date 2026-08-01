"""Interviewer agent that drives the structured habits/routines interview."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from src.agents.ChromaDB.chroma_db import retrieve_all_history, store_qa
from src.agents.llm_config import resolve_llm_config

logger = logging.getLogger("interviewer_core")


class InterviewerAgent:
    EMPTY_RESPONSE_RETRY_ATTEMPTS = 3
    EMPTY_RESPONSE_RETRY_DELAY_SECONDS = 1
    EMPTY_RESPONSE_FALLBACK = (
        "I didn't get a usable response on my side. "
        "Please answer the previous question again so I can continue the interview."
    )

    def __init__(
        self,
        *,
        collection,
        llm_api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        # `llm_api_key` is the player's per-request key forwarded by gamebus-api-v2.
        # The shared resolver routes through OpenAI when OPENAI_API_KEY+OPENAI_MODEL are set.
        self._llm = resolve_llm_config(
            "INTERVIEWER_MODEL",
            fallback_model="openai/gpt-oss-120b:free",
            explicit_api_key=llm_api_key,
            explicit_model=model,
        )
        self.api_key = self._llm.api_key
        self.api_base = self._llm.api_base
        self.model = self._llm.model
        self.instructions = self._load_instructions()
        self.conversation_history: List[Dict[str, str]] = []
        self.collection = collection

    def _load_instructions(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        instructions_path = os.path.join(
            current_dir, "..", "memory", "templates", "Interviewer_instructions.txt"
        )
        if not os.path.exists(instructions_path):
            raise RuntimeError(f"Interviewer instructions not found: {instructions_path}")
        with open(instructions_path, "r", encoding="utf-8") as f:
            return f.read()

    def _send_to_llm(self, messages: List[Dict[str, str]]) -> str:
        headers = self._llm.headers(title="Interviewer Agent")
        history_summary = "\n".join(f"{m['role']}: {m['content']}" for m in messages[-80:])
        today = datetime.now().strftime("%Y-%m-%d")
        system_prompt = f"{self.instructions}\n\nrecent history:\n{history_summary}\n\nToday's date: {today}"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "temperature": 0.2,
        }

        max_http_retries = 5
        http_retry_base_delay = 2

        for http_attempt in range(1, max_http_retries + 1):
            try:
                for attempt in range(1, self.EMPTY_RESPONSE_RETRY_ATTEMPTS + 1):
                    response = requests.post(
                        self._llm.chat_completions_url,
                        headers=headers,
                        json=payload,
                        timeout=60,
                    )
                    response.raise_for_status()
                    reply = response.json()["choices"][0]["message"]["content"] or ""
                    if reply.strip():
                        return reply
                    logger.warning(
                        "Interviewer LLM returned an empty message attempt=%s/%s",
                        attempt,
                        self.EMPTY_RESPONSE_RETRY_ATTEMPTS,
                    )
                    if attempt < self.EMPTY_RESPONSE_RETRY_ATTEMPTS:
                        time.sleep(self.EMPTY_RESPONSE_RETRY_DELAY_SECONDS)
                return self.EMPTY_RESPONSE_FALLBACK
            except requests.exceptions.HTTPError as exc:
                status_code = getattr(exc.response, "status_code", None)
                if status_code == 429:
                    delay = http_retry_base_delay**http_attempt
                    logger.warning(
                        "HTTP 429 from interviewer LLM; retrying in %ss (attempt %s/%s)",
                        delay,
                        http_attempt,
                        max_http_retries,
                    )
                    time.sleep(delay)
                    continue
                logger.error("HTTP error from interviewer LLM: %s", exc)
                raise

        return self.EMPTY_RESPONSE_FALLBACK

    @staticmethod
    def _extract_question_number(reply: str) -> Optional[str]:
        import re

        match = re.search(r"(?:^|\n|\*\*)\s*(?:Q)?(\d+)[.:]", reply)
        if match:
            return f"Q{match.group(1)}"
        return None

    @staticmethod
    def _extract_section(reply: str) -> Optional[str]:
        text = reply.lower()
        if "section 1" in text or "sleep" in text or "eat" in text:
            return "sleep_eat_routines"
        if "section 2" in text or "work" in text or "study" in text:
            return "work_study"
        if "section 3" in text or "hobbies" in text or "physical activities" in text or "sports" in text:
            return "hobbies_sports"
        if "commute" in text or "travel to work" in text:
            return "commute"
        if "weekend" in text:
            return "weekend"
        return None

    def chat(self, message: str, clear_history: bool = False) -> str:
        if clear_history:
            self.conversation_history = []
        self.conversation_history.append({"role": "user", "content": message})
        reply = self._send_to_llm(self.conversation_history)
        self.conversation_history.append({"role": "assistant", "content": reply})

        if self.collection is not None:
            try:
                store_qa(
                    self.collection,
                    question=message,
                    answer=reply,
                    question_number=self._extract_question_number(reply),
                    section=self._extract_section(reply),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to store interviewer QA: %s", exc)

        return reply

    def reset(self) -> None:
        self.conversation_history = []

    def get_history(self) -> List[str]:
        if self.collection is None:
            return []
        try:
            results = retrieve_all_history(self.collection)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to retrieve interviewer history: %s", exc)
            return []

        documents = results.get("documents", []) if isinstance(results, dict) else []
        metadatas = results.get("metadatas", []) if isinstance(results, dict) else []
        if not isinstance(documents, list):
            return []

        entries = []
        for idx, doc in enumerate(documents):
            if doc is None:
                continue
            text = str(doc).strip()
            if not text or text == "startup":
                continue
            parsed_ts = None
            if idx < len(metadatas) and isinstance(metadatas[idx], dict):
                raw_ts = metadatas[idx].get("timestamp") or metadatas[idx].get("created_at")
                if isinstance(raw_ts, str) and raw_ts.strip():
                    try:
                        parsed_ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                    except ValueError:
                        parsed_ts = None
            entries.append((parsed_ts, idx, text))

        if any(ts is not None for ts, _, _ in entries):
            entries.sort(key=lambda item: (item[0] is None, item[0] or datetime.min, item[1]))

        return [text for _, _, text in entries]

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return list(self.conversation_history)
