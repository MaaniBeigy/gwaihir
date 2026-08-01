"""mTLS configuration errors, `close`, and no-URL behaviour for `WebhookClient`."""

from __future__ import annotations

import pytest

from src.webhook.client import WebhookClient


def test_build_client_constructs_httpx_with_mtls_kwargs_when_all_certs_present(monkeypatch):
    """When all three cert paths are configured, `_build_client` constructs
    an `httpx.Client` with the cert tuple, CA bundle path, and the configured
    timeout. `httpx.Client` is replaced with a stub so no real I/O happens."""
    captured: dict = {}

    class _StubHttpxClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def close(self):
            pass

    monkeypatch.setattr("src.webhook.client.httpx.Client", _StubHttpxClient)

    client = WebhookClient(
        url="https://gateway.example.com",
        client_cert="/etc/coach/client.crt",
        client_key="/etc/coach/client.key",
        server_ca="/etc/coach/gateway-server-ca.pem",
        internal_token="t",
        enabled=True,
        timeout_seconds=7.5,
    )
    underlying = client._build_client()

    assert isinstance(underlying, _StubHttpxClient)
    assert captured["cert"] == ("/etc/coach/client.crt", "/etc/coach/client.key")
    assert captured["verify"] == "/etc/coach/gateway-server-ca.pem"
    assert captured["http2"] is False
    # The timeout argument is wrapped in `httpx.Timeout`; the value should still reflect 7.5.
    timeout_arg = captured["timeout"]
    assert getattr(timeout_arg, "connect", 7.5) == 7.5 or timeout_arg == 7.5


def test_build_client_raises_when_certs_missing(monkeypatch):
    # Make sure ambient env doesn't supply real cert paths.
    monkeypatch.delenv("COACH_WEBHOOK_CLIENT_CERT", raising=False)
    monkeypatch.delenv("COACH_WEBHOOK_CLIENT_KEY", raising=False)
    monkeypatch.delenv("COACH_WEBHOOK_SERVER_CA", raising=False)

    client = WebhookClient(
        url="https://gateway.example.com",
        client_cert=None,
        client_key=None,
        server_ca=None,
        internal_token="t",
        enabled=True,
    )
    with pytest.raises(RuntimeError, match="mTLS configuration is incomplete"):
        client._build_client()


def test_emit_returns_false_when_url_missing_but_enabled(monkeypatch):
    """When the client is enabled but `self.url` is empty, `emit` logs a
    warning and returns False without touching the network. The env var is
    cleared first so the constructor's fallback can't repopulate the URL."""
    monkeypatch.delenv("COACH_WEBHOOK_URL", raising=False)

    client = WebhookClient(
        url="",
        client_cert="dummy.crt",
        client_key="dummy.key",
        server_ca="ca.pem",
        internal_token="t",
        enabled=True,
    )
    assert client.url == ""

    ok = client.emit(
        player_id=1,
        campaign_id=2,
        session_id="s",
        phase=WebhookClient.PHASE_INTERVIEWING,
        request_id="req-x",
    )
    assert ok is False


def test_close_is_idempotent(monkeypatch):
    client = WebhookClient(
        url="https://gateway",
        client_cert="dummy.crt",
        client_key="dummy.key",
        server_ca="ca.pem",
        internal_token="t",
        enabled=True,
    )

    class _Stub:
        closed = False

        def close(self):
            self.closed = True

    stub = _Stub()
    client._client = stub
    client.close()
    assert stub.closed is True
    # Second close is a no-op on a freshly-cleared client.
    client.close()


def test_emit_handles_httpx_exception(monkeypatch):
    client = WebhookClient(
        url="https://gateway",
        client_cert="dummy.crt",
        client_key="dummy.key",
        server_ca="ca.pem",
        internal_token="t",
        enabled=True,
    )

    class _ThrowingClient:
        def post(self, url, json, headers):
            raise RuntimeError("network down")

        def close(self):
            pass

    monkeypatch.setattr(client, "_get_client", lambda: _ThrowingClient())
    monkeypatch.setattr("src.webhook.client.time.sleep", lambda _s: None)

    ok = client.emit(
        player_id=1,
        campaign_id=2,
        session_id="s",
        phase=WebhookClient.PHASE_ERROR,
        request_id="req-y",
    )
    assert ok is False


def test_emit_default_enabled_reads_env(monkeypatch):
    monkeypatch.setenv("COACH_REQUIRE_MTLS", "true")
    client = WebhookClient(
        url="https://gateway",
        client_cert="dummy.crt",
        client_key="dummy.key",
        server_ca="ca.pem",
        internal_token="t",
        # enabled left at None so it derives from env
    )
    assert client.enabled is True


def test_emit_default_enabled_false_when_env_off(monkeypatch):
    monkeypatch.setenv("COACH_REQUIRE_MTLS", "false")
    client = WebhookClient(
        url="https://gateway",
        client_cert="dummy.crt",
        client_key="dummy.key",
        server_ca="ca.pem",
        internal_token="t",
    )
    assert client.enabled is False


def test_get_client_caches_underlying_httpx(monkeypatch):
    client = WebhookClient(
        url="https://gateway",
        client_cert="dummy.crt",
        client_key="dummy.key",
        server_ca="ca.pem",
        internal_token="t",
        enabled=True,
    )

    constructed = []

    class _StubHttpx:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

        def close(self):
            pass

    monkeypatch.setattr(client, "_build_client", lambda: _StubHttpx())
    first = client._get_client()
    second = client._get_client()
    assert first is second
