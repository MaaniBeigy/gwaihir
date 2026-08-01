"""Coach to gateway progress webhook client over mTLS."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("coach.webhook")


class WebhookClient:
    """httpx-based mTLS client with bounded retry and a global timeout."""

    PHASE_INTERVIEWING = "INTERVIEWING"
    PHASE_SCHEDULING = "SCHEDULING"
    PHASE_COMPLETE = "COMPLETE"
    PHASE_ERROR = "ERROR"

    _RETRY_DELAYS_SECONDS = (1, 4, 15)

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        client_cert: Optional[str] = None,
        client_key: Optional[str] = None,
        server_ca: Optional[str] = None,
        internal_token: Optional[str] = None,
        enabled: Optional[bool] = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.url = url or os.getenv("COACH_WEBHOOK_URL", "")
        self.client_cert = client_cert or os.getenv("COACH_WEBHOOK_CLIENT_CERT")
        self.client_key = client_key or os.getenv("COACH_WEBHOOK_CLIENT_KEY")
        self.server_ca = server_ca or os.getenv("COACH_WEBHOOK_SERVER_CA")
        self.internal_token = internal_token or os.getenv("COACH_INTERNAL_TOKEN", "")
        self.timeout_seconds = timeout_seconds

        if enabled is None:
            require_mtls = os.getenv("COACH_REQUIRE_MTLS", "false").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            enabled = bool(self.url) and require_mtls
        self.enabled = enabled
        self._client_lock = threading.Lock()
        self._client: Optional[httpx.Client] = None

    def _build_client(self) -> httpx.Client:
        if not self.client_cert or not self.client_key or not self.server_ca:
            raise RuntimeError(
                "Webhook mTLS configuration is incomplete: COACH_WEBHOOK_CLIENT_CERT / "
                "COACH_WEBHOOK_CLIENT_KEY / COACH_WEBHOOK_SERVER_CA must all be set."
            )
        return httpx.Client(
            cert=(self.client_cert, self.client_key),
            verify=self.server_ca,
            timeout=httpx.Timeout(self.timeout_seconds),
            http2=False,
        )

    def _get_client(self) -> httpx.Client:
        with self._client_lock:
            if self._client is None:
                self._client = self._build_client()
            return self._client

    def close(self) -> None:
        with self._client_lock:
            if self._client is not None:
                try:
                    self._client.close()
                finally:
                    self._client = None

    def emit(
        self,
        *,
        player_id: int,
        campaign_id: int,
        session_id: str,
        phase: str,
        request_id: str,
        percent_complete: Optional[int] = None,
        current_summary: Optional[str] = None,
        message: Optional[str] = None,
    ) -> bool:
        """Post a phase event. Return True on success, False on logged failure."""
        payload: Dict[str, Any] = {
            "playerId": int(player_id),
            "campaignId": int(campaign_id),
            "sessionId": session_id,
            "phase": phase,
            "percentComplete": int(percent_complete) if percent_complete is not None else None,
            "currentSummary": current_summary,
            "message": message,
            "ts": _utc_iso_now(),
        }

        if not self.enabled:
            logger.info(
                "webhook.disabled phase=%s playerId=%s campaignId=%s sessionId=%s request_id=%s",
                phase,
                player_id,
                campaign_id,
                session_id,
                request_id,
            )
            return True

        if not self.url:
            logger.warning(
                "webhook.no_url phase=%s playerId=%s campaignId=%s sessionId=%s request_id=%s",
                phase,
                player_id,
                campaign_id,
                session_id,
                request_id,
            )
            return False

        headers = {
            "Content-Type": "application/json",
            "X-Coach-Internal-Token": self.internal_token,
            "X-Coach-Request-Id": request_id,
        }

        last_error: Optional[Exception] = None
        for attempt, delay in enumerate(self._RETRY_DELAYS_SECONDS, start=1):
            try:
                client = self._get_client()
                response = client.post(self.url, json=payload, headers=headers)
                if 200 <= response.status_code < 300:
                    logger.info(
                        "webhook.success phase=%s status=%s attempt=%s request_id=%s",
                        phase,
                        response.status_code,
                        attempt,
                        request_id,
                    )
                    return True
                last_error = RuntimeError(
                    f"webhook returned status={response.status_code} body={response.text[:200]!r}"
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc

            logger.warning(
                "webhook.retry phase=%s attempt=%s/%s delay_s=%s error=%s request_id=%s",
                phase,
                attempt,
                len(self._RETRY_DELAYS_SECONDS),
                delay,
                last_error,
                request_id,
            )
            if attempt < len(self._RETRY_DELAYS_SECONDS):
                time.sleep(delay)

        logger.error(
            "webhook.failed phase=%s attempts=%s error=%s request_id=%s",
            phase,
            len(self._RETRY_DELAYS_SECONDS),
            last_error,
            request_id,
        )
        return False


def _utc_iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
