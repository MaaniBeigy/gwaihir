import pytest

from src.agents.interviewer_core import InterviewerAgent


class _DummyCollection:
    def __init__(self):
        self.calls = []

    def add(self, documents, ids, metadatas):
        self.calls.append(("add", documents, ids, metadatas))


@pytest.mark.unit
def test_interviewer_retries_when_llm_returns_empty_message(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key")
    monkeypatch.setattr("src.agents.interviewer_core.store_qa", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.agents.interviewer_core.time.sleep", lambda seconds: None)

    llm_contents = ["   ", "Q1: What time do you usually wake up?"]
    request_calls = []

    def mock_post(url, *args, **kwargs):
        request_calls.append({"url": url, "json": kwargs.get("json")})
        content = llm_contents[len(request_calls) - 1]

        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": content}}]}

        return MockResponse()

    monkeypatch.setattr("requests.post", mock_post)

    agent = InterviewerAgent(collection=_DummyCollection())
    reply = agent.chat("Start interview about habits and routines.", clear_history=True)

    assert reply == "Q1: What time do you usually wake up?"
    assert len(request_calls) == 2
    assert agent.get_conversation_history() == [
        {"role": "user", "content": "Start interview about habits and routines."},
        {"role": "assistant", "content": "Q1: What time do you usually wake up?"},
    ]


@pytest.mark.unit
def test_interviewer_get_history_returns_full_chronological_entries(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key")
    monkeypatch.setattr(
        "src.agents.interviewer_core.retrieve_all_history",
        lambda collection: {
            "documents": [
                "Q: What do you do after dinner?\nA: I read.",
                "startup",
                "Q: What time do you wake up?\nA: 07:00.",
                None,
            ],
            "metadatas": [
                {"timestamp": "2026-03-17T10:05:00"},
                {"timestamp": "2026-03-17T10:00:00"},
                {"timestamp": "2026-03-17T09:55:00"},
                {},
            ],
        },
    )

    agent = InterviewerAgent(collection=_DummyCollection())
    history = agent.get_history()

    assert history == [
        "Q: What time do you wake up?\nA: 07:00.",
        "Q: What do you do after dinner?\nA: I read.",
    ]
