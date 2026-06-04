"""HTTP smoke tests for the FastAPI surface.

We patch the compiled graph to a stub so the LLM isn't invoked. This
verifies the endpoint contract (status codes, JSON shape) without cost.
"""
from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from agentstore.server import app


def test_health_returns_ok():
    with TestClient(app) as client:
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_chat_returns_answer_and_agent_with_stubbed_graph():
    class _StubGraph:
        def invoke(self, state, config=None):
            return {
                "messages": [AIMessage(content="Total revenue was $50,617.74.")],
                "last_agent": "sales",
                "final_response": "Total revenue was $50,617.74.",
            }

    with patch("agentstore.server.get_graph", return_value=_StubGraph()):
        with TestClient(app) as client:
            r = client.post("/api/chat", json={"query": "how much revenue?"})
            assert r.status_code == 200
            body = r.json()
            assert body["agent"] == "sales"
            assert "50,617.74" in body["answer"]
            assert isinstance(body["latency_ms"], int)


def test_chat_rejects_empty_query():
    with TestClient(app) as client:
        r = client.post("/api/chat", json={"query": ""})
        assert r.status_code == 422
