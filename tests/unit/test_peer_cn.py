"""mTLS request-side validation in app.authorise."""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

import pytest


@pytest.fixture
def reload_app(monkeypatch):
    """Reload src.agents.app with a fresh env, returning the module."""

    def _reload(env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("OPENROUTER_API_KEY", env.get("OPENROUTER_API_KEY", "sk-test"))
        import src.agents.app as app_module

        return importlib.reload(app_module)

    return _reload


class _StubRequest:
    def __init__(self, headers):
        self.headers = headers
        self.url = SimpleNamespace(path="/chat")
        self.state = SimpleNamespace()


def _run(coro):
    return (
        asyncio.get_event_loop().run_until_complete(coro)
        if asyncio.get_event_loop().is_running() is False
        else asyncio.run(coro)
    )


def test_authorise_skips_checks_when_mtls_disabled(reload_app):
    app_module = reload_app({"COACH_REQUIRE_MTLS": "false"})
    request = _StubRequest(headers={})
    request_id = asyncio.run(app_module.authorise(request))
    assert request_id  # generated when missing
    assert request.state.request_id == request_id


def test_authorise_uses_incoming_request_id(reload_app):
    app_module = reload_app({"COACH_REQUIRE_MTLS": "false"})
    request = _StubRequest(headers={"X-Coach-Request-Id": "abc-123"})
    request_id = asyncio.run(app_module.authorise(request))
    assert request_id == "abc-123"


def test_authorise_rejects_unexpected_cn(reload_app):
    app_module = reload_app(
        {
            "COACH_REQUIRE_MTLS": "true",
            "COACH_EXPECTED_PEER_CN": "gamebus-api-v2",
            "COACH_INTERNAL_TOKEN": "secret",
        }
    )
    request = _StubRequest(
        headers={
            "X-Client-CN": "evil-impostor",
            "X-Coach-Internal-Token": "secret",
        }
    )
    with pytest.raises(Exception) as excinfo:
        asyncio.run(app_module.authorise(request))
    assert getattr(excinfo.value, "status_code", None) == 403


def test_authorise_rejects_token_mismatch(reload_app):
    app_module = reload_app(
        {
            "COACH_REQUIRE_MTLS": "true",
            "COACH_EXPECTED_PEER_CN": "gamebus-api-v2",
            "COACH_INTERNAL_TOKEN": "secret",
        }
    )
    request = _StubRequest(
        headers={
            "X-Client-CN": "gamebus-api-v2",
            "X-Coach-Internal-Token": "wrong",
        }
    )
    with pytest.raises(Exception) as excinfo:
        asyncio.run(app_module.authorise(request))
    assert getattr(excinfo.value, "status_code", None) == 403


def test_authorise_accepts_correct_cn_and_token(reload_app):
    app_module = reload_app(
        {
            "COACH_REQUIRE_MTLS": "true",
            "COACH_EXPECTED_PEER_CN": "gamebus-api-v2",
            "COACH_INTERNAL_TOKEN": "secret",
        }
    )
    request = _StubRequest(
        headers={
            "X-Client-CN": "gamebus-api-v2",
            "X-Coach-Internal-Token": "secret",
            "X-Coach-Request-Id": "req-1",
        }
    )
    request_id = asyncio.run(app_module.authorise(request))
    assert request_id == "req-1"
