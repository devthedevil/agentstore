"""FastAPI HTTP server that exposes the multi-agent graph as a chat endpoint.

Routes:
  POST /api/chat      → run one full graph turn for a user query
  GET  /api/health    → liveness probe
  GET  /              → serve the bundled single-page chat UI
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

from .config import get_settings
from .graph import RECURSION_LIMIT, get_graph

app = FastAPI(
    title="agentstore",
    description="Multi-agent e-commerce assistant powered by LangGraph + Claude.",
    version="0.1.0",
)


# Middleware must be registered before the app receives any request, so we
# resolve CORS origins eagerly here. Falling back to "*" lets the app start
# even when env isn't fully configured (e.g. during tests), which is fine
# because real auth/secret enforcement lives at the route layer.
def _cors_origins() -> list[str]:
    try:
        return get_settings().cors_origins
    except RuntimeError:
        return ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    agent: str
    latency_ms: int


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


RECURSION_LIMIT_MESSAGE = (
    "This question needed more back-and-forth between specialists than we allow per "
    "request. Try asking a narrower question, or split it into separate questions — "
    "one per topic (sales, inventory, reviews, customer)."
)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    try:
        result: dict[str, Any] = get_graph().invoke(
            {"messages": [HumanMessage(content=req.query)]},
            config={"recursion_limit": RECURSION_LIMIT},
        )
    except GraphRecursionError:
        # Not a server error: the supervisor/worker loop hit its hop budget
        # without reaching FINISH (e.g. a question spanning many domains, or
        # the LLM bouncing indecisively between specialists). Surface it as a
        # normal answer so the UI renders a helpful chat bubble instead of a
        # red error toast with an internal LangGraph error message.
        return ChatResponse(
            answer=RECURSION_LIMIT_MESSAGE,
            agent="supervisor",
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
    except Exception as exc:  # noqa: BLE001 — bubble up as 500 with a short msg
        raise HTTPException(status_code=500, detail=f"graph error: {exc}") from exc

    answer = result.get("final_response")
    if not answer:
        ai_messages = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
        answer = ai_messages[-1].content if ai_messages else "(no response)"
        if not isinstance(answer, str):
            answer = str(answer)

    return ChatResponse(
        answer=answer,
        agent=result.get("last_agent", "unknown"),
        latency_ms=int((time.perf_counter() - start) * 1000),
    )


# ── Static UI ─────────────────────────────────────────────────────────────────

_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/")
def index() -> FileResponse:
    target = _FRONTEND_DIR / "index.html"
    if not target.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(target)
