"""LangGraph state schema.

The state is what flows along the edges of the graph. We keep it minimal:
the conversation history (which langgraph appends to automatically), a
route hint produced by the supervisor, and a `final_response` slot the
worker agents fill in.
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# The four specialist agents the supervisor can route to. Add a new value
# here, a new node in graph.py, and a new tool module to extend.
AgentName = Literal["sales", "inventory", "reviews", "customer", "FINISH"]


class AgentState(TypedDict, total=False):
    """Shared state that every node reads/writes."""

    # Conversation log. `add_messages` is langgraph's built-in reducer that
    # appends new messages without overwriting prior turns.
    messages: Annotated[list[BaseMessage], add_messages]

    # Set by the supervisor each step. "FINISH" tells the graph to stop.
    next_agent: AgentName

    # Where the most recent worker stored its final answer. Surface this
    # in the API response so the UI can show which agent replied.
    last_agent: str

    # Human-readable response produced by the last worker.
    final_response: str
