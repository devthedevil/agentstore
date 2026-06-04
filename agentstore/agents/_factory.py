"""Shared worker-agent builder.

Every specialist is a prebuilt langgraph ReAct agent: an LLM that can call
the tools we give it, looping until it has a final answer. The wrapper here
adapts the ReAct agent into a graph node that:
  1. Runs ReAct on the latest conversation.
  2. Extracts the final assistant message.
  3. Writes it back into the shared AgentState as `final_response`.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from ..llm import get_llm
from ..state import AgentState


def build_worker(name: str, system_prompt: str, tools: list[Any]) -> Callable[[AgentState], dict[str, Any]]:
    """Return a graph node function for the named specialist agent.

    Parameters
    ----------
    name:
        Short identifier used in logs and surfaced to the UI (`last_agent`).
    system_prompt:
        Persona + responsibilities + tool-usage guidance.
    tools:
        The langchain @tool functions this agent may call.
    """
    # Lazy: build the ReAct agent on first call so module import doesn't
    # require an LLM (tests, lint, doc generation, etc.).
    _react: list[Any] = []

    def node(state: AgentState) -> dict[str, Any]:
        if not _react:
            _react.append(create_react_agent(get_llm(), tools=tools))
        react = _react[0]
        # Prepend the system prompt so each invocation re-establishes role.
        messages: list[BaseMessage] = [SystemMessage(content=system_prompt)] + list(
            state.get("messages", [])
        )
        result = react.invoke({"messages": messages})
        # ReAct returns the full transcript; the last AI message is the answer.
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        final = ai_messages[-1] if ai_messages else AIMessage(content="(no response)")
        return {
            "messages": [final],
            "last_agent": name,
            "final_response": final.content if isinstance(final.content, str) else str(final.content),
        }

    node.__name__ = f"{name}_agent_node"
    return node
