"""Webhook client mTLS retry behaviour."""

from typing import Any, Dict, List

import pytest

from src.webhook.client import WebhookClient


class _StubResponse:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


class _StubHttpxClient:
    def __init__(self, responses: List[_StubResponse]):
        self._responses = list(responses)
        self.posts: List[Dict[str, Any]] = []
        self.closed = False

    def post(self, url, json, headers):
        self.posts.append({"url": url, "json": json, "headers": dict(headers)})
        if not self._responses:
            raise AssertionError("Stub exhausted")
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        self.closed = True


def _build_client(monkeypatch, responses, enabled=True):
    client = WebhookClient(
        url="https://gateway.example.com:8443/coach/internal/progress",
        client_cert="dummy.crt",
        client_key="dummy.key",
        server_ca="ca.pem",
        internal_token="t",
        enabled=enabled,
    )
    stub = _StubHttpxClient(responses)
    monkeypatch.setattr(client, "_get_client", lambda: stub)
    monkeypatch.setattr("src.webhook.client.time.sleep", lambda _seconds: None)
    return client, stub


def test_emit_succeeds_on_first_attempt(monkeypatch):
    client, stub = _build_client(monkeypatch, [_StubResponse(204)])
    ok = client.emit(
        player_id=1,
        campaign_id=2,
        session_id="s",
        phase=WebhookClient.PHASE_INTERVIEWING,
        request_id="req-1",
        message="hi",
    )
    assert ok is True
    assert len(stub.posts) == 1
    sent = stub.posts[0]
    assert sent["headers"]["X-Coach-Internal-Token"] == "t"
    assert sent["headers"]["X-Coach-Request-Id"] == "req-1"
    assert sent["json"]["phase"] == WebhookClient.PHASE_INTERVIEWING


def test_emit_retries_on_5xx_then_succeeds(monkeypatch):
    client, stub = _build_client(
        monkeypatch,
        [_StubResponse(503, "boom"), _StubResponse(503, "boom"), _StubResponse(200, "ok")],
    )
    ok = client.emit(
        player_id=1,
        campaign_id=2,
        session_id="s",
        phase=WebhookClient.PHASE_COMPLETE,
        request_id="req-2",
    )
    assert ok is True
    assert len(stub.posts) == 3


def test_emit_returns_false_after_three_failures(monkeypatch):
    client, stub = _build_client(
        monkeypatch,
        [_StubResponse(500), _StubResponse(500), _StubResponse(500)],
    )
    ok = client.emit(
        player_id=1,
        campaign_id=2,
        session_id="s",
        phase=WebhookClient.PHASE_ERROR,
        request_id="req-3",
    )
    assert ok is False
    assert len(stub.posts) == 3


def test_emit_disabled_short_circuits(monkeypatch):
    client, stub = _build_client(monkeypatch, [], enabled=False)
    ok = client.emit(
        player_id=1,
        campaign_id=2,
        session_id="s",
        phase=WebhookClient.PHASE_INTERVIEWING,
        request_id="req-4",
    )
    assert ok is True
    assert stub.posts == []
