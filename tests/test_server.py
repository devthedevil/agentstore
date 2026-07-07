"""HTTP smoke tests for the FastAPI surface.

We patch the compiled graph to a stub so the LLM isn't invoked. This
verifies the endpoint contract (status codes, JSON shape) without cost.
"""
from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from agentstore.server import _allow_credentials_for, app


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


def test_chat_handles_recursion_limit_gracefully():
    """A supervisor/worker loop that never reaches FINISH within
    RECURSION_LIMIT should surface as a normal answer, not an opaque 500."""

    class _StubGraph:
        def invoke(self, state, config=None):
            raise GraphRecursionError(
                "Recursion limit of 8 reached without hitting a stop condition."
            )

    with patch("agentstore.server.get_graph", return_value=_StubGraph()):
        with TestClient(app) as client:
            r = client.post("/api/chat", json={"query": "a very complex multi-domain question"})
            assert r.status_code == 200
            body = r.json()
            assert body["agent"] == "supervisor"
            assert "narrower" in body["answer"]
            assert isinstance(body["latency_ms"], int)


# ── CORS ─────────────────────────────────────────────────────────────────────
#
# Wildcard origin + allow_credentials=True is an unsafe combination: browsers
# refuse a literal "*" on a credentialed response, so CORS middlewares
# (including Starlette's) reflect back whatever Origin header the request
# sent instead — meaning any site could get its origin accepted and make
# credentialed requests, defeating CORS entirely. Credentials must only be
# enabled once explicit, non-wildcard origins are configured.


def test_wildcard_origin_disallows_credentials():
    assert _allow_credentials_for(["*"]) is False


def test_explicit_origins_allow_credentials():
    assert _allow_credentials_for(["https://example.com", "http://localhost:5173"]) is True
