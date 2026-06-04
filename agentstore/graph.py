"""LangGraph wiring.

Topology:

    START → supervisor ─┬─► sales      ──┐
                        ├─► inventory  ──┤
                        ├─► reviews    ──┼─► supervisor (loop)
                        └─► customer   ──┘
                        │
                        └─► FINISH ──► END

The supervisor decides where to go next based on the latest message; each
specialist appends its answer and hands control back so the supervisor can
either continue (multi-step queries) or end the turn.
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from .agents import customer_agent, inventory_agent, review_agent, sales_agent
from .state import AgentState
from .supervisor import route_from_supervisor, supervisor_node

# Hard cap on supervisor → worker hops per request to keep latency and cost
# bounded even if the LLM keeps asking to re-route.
RECURSION_LIMIT = 8


def build_graph():
    """Construct and compile the multi-agent graph."""
    g = StateGraph(AgentState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("sales", sales_agent)
    g.add_node("inventory", inventory_agent)
    g.add_node("reviews", review_agent)
    g.add_node("customer", customer_agent)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "sales": "sales",
            "inventory": "inventory",
            "reviews": "reviews",
            "customer": "customer",
            "FINISH": END,
        },
    )
    # Every worker returns to the supervisor — which usually FINISHes after
    # the first answer but may chain for cross-domain follow-ups.
    for worker in ("sales", "inventory", "reviews", "customer"):
        g.add_edge(worker, "supervisor")

    return g.compile()


@lru_cache(maxsize=1)
def get_graph():
    """Cached compiled graph. Tests can call build_graph() directly to bypass."""
    return build_graph()
