"""The Supervisor: a small LLM call that routes the user's question to the
right specialist (or to FINISH when the question is fully answered).

We use structured output (a Pydantic model with a single enum field) so the
supervisor cannot drift into free-form text. That makes routing reliable
and means the graph never gets stuck in an "unknown next_agent" state.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentName, AgentState

SUPERVISOR_PROMPT = """You are the Supervisor of a multi-agent e-commerce
assistant. Read the user's most recent question and decide which specialist
should answer it next, or FINISH if the latest assistant message has already
answered the question.

Specialists and their lanes:
- sales      : revenue, profit, monthly trends, top products, category performance
- inventory  : stock levels, low-stock / out-of-stock products, restock priority
- reviews    : customer review sentiment, ratings, return reasons, problem SKUs
- customer   : top customers, customer history, demographic / geographic breakdowns
- FINISH     : the question has been fully answered or is unrelated to commerce

Rules:
1. Pick exactly one option.
2. Pick FINISH if the conversation's last message is an assistant answer
   that already addresses the user's question.
3. If a question spans two domains, pick the dominant one — the other
   specialist can pick up on the next turn if the user follows up.
"""


class Route(BaseModel):
    """Routing decision the supervisor must produce."""

    next_agent: AgentName = Field(..., description="Which specialist should run next, or FINISH.")


def supervisor_node(state: AgentState) -> dict[str, Any]:
    """Graph node that picks the next specialist."""
    llm = get_llm().with_structured_output(Route)
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + list(state.get("messages", []))
    if not state.get("messages"):
        # Nothing to route; finish gracefully.
        return {"next_agent": "FINISH"}
    decision: Route = llm.invoke(messages)  # type: ignore[assignment]
    return {"next_agent": decision.next_agent}


def route_from_supervisor(state: AgentState) -> str:
    """Conditional-edge function read by langgraph after supervisor_node."""
    nxt = state.get("next_agent", "FINISH")
    if nxt == "FINISH":
        return "FINISH"
    # Defensive: anything we don't recognize ends the turn rather than loop.
    if nxt not in {"sales", "inventory", "reviews", "customer"}:
        return "FINISH"
    return nxt


def make_default_route(query: str) -> HumanMessage:
    """Helper used by callers to wrap a raw user query into the initial state."""
    return HumanMessage(content=query)
