# tests/conftest.py
"""
pytest configuration file - runs before all tests
Handles API key setup and common fixtures
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_openrouter_key():
    """
    Set OPENROUTER_API_KEY before any module imports.
    This is session-scoped and autouse=True so it runs first.
    """
    # Check if running with --integration flag for real API tests
    if "integration" in os.environ.get("PYTEST_MARKERS", ""):
        # Real API tests - try to load from .env
        if not os.getenv("OPENROUTER_API_KEY"):
            from dotenv import load_dotenv

            load_dotenv()

        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set - skipping integration tests")
    else:
        # Unit tests mock the network, so a dummy key suffices. Keep a real key when one is
        # already set so the live-marked tests can reach the provider.
        os.environ.setdefault("OPENROUTER_API_KEY", "sk-or-v1-dummy-key-for-unit-tests")


@pytest.fixture
def mock_openrouter_response():
    """
    Mock OpenRouter API response for unit tests.
    Returns a standard response structure.
    """
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "Activities": [
                                {
                                    "name": "go to the gym",
                                    "description": "go to the gym every Monday at 19:00 for 1 hour",
                                    "IsInstantaneous": False,
                                    "IsRegular": True,
                                    "frequency": {"type": "specific_days", "days": ["mon"], "count": 1},
                                    "IsRepetitive": True,
                                    "activity_duration": 60,
                                    "start_date": "2026-01-06",
                                    "IsArbitrary": False,
                                    "time": {"preferred": "19:00"},
                                    "notes": None,
                                }
                            ]
                        }
                    )
                }
            }
        ]
    }


@pytest.fixture
def mock_requests_post(monkeypatch, mock_openrouter_response):
    """
    Mock requests.post to avoid real API calls in unit tests.
    Use with: def test_something(mock_requests_post):
    """

    def mock_post(url, *args, **kwargs):
        class MockResponse:
            status_code = 200

            def json(self):
                return mock_openrouter_response

        return MockResponse()

    monkeypatch.setattr("requests.post", mock_post)
    return mock_post


# Pytest markers for test organization
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (requires real API key)"
    )
    config.addinivalue_line("markers", "unit: marks tests as unit tests (can use mocked APIs)")
    config.addinivalue_line(
        "markers",
        "live: marks tests that call the real LLM endpoint "
        "(OPENAI_API_KEY+OPENAI_MODEL or OPENROUTER_API_KEY required); "
        "auto-skipped otherwise",
    )
