"""Graph wiring tests.

These avoid hitting the live LLM by stubbing the supervisor and worker
nodes to deterministic functions. They verify topology, conditional
routing, and that the worker output reaches the final state.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from agentstore.state import AgentState
from agentstore.supervisor import route_from_supervisor


def _make_supervisor(decision: str):
    def fn(state: AgentState):
        return {"next_agent": decision}
    return fn


def _make_worker(name: str, answer: str):
    def fn(state: AgentState):
        return {
            "messages": [AIMessage(content=answer)],
            "last_agent": name,
            "final_response": answer,
        }
    return fn


def _build_test_graph(supervisor_decision: str, answer: str):
    g = StateGraph(AgentState)
    g.add_node("supervisor", _make_supervisor(supervisor_decision))
    for w in ("sales", "inventory", "reviews", "customer"):
        g.add_node(w, _make_worker(w, answer if w == supervisor_decision else f"({w} did not run)"))
    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"sales": "sales", "inventory": "inventory", "reviews": "reviews", "customer": "customer", "FINISH": END},
    )
    # On worker exit, finish so test loops don't infinite-loop the second supervisor turn.
    for w in ("sales", "inventory", "reviews", "customer"):
        g.add_edge(w, END)
    return g.compile()


def test_routes_to_sales_when_supervisor_picks_sales():
    graph = _build_test_graph("sales", "$5k revenue this month")
    out = graph.invoke({"messages": [HumanMessage(content="how much revenue?")]})
    assert out["last_agent"] == "sales"
    assert "5k" in out["final_response"]


def test_routes_to_inventory_when_supervisor_picks_inventory():
    graph = _build_test_graph("inventory", "12 SKUs under threshold")
    out = graph.invoke({"messages": [HumanMessage(content="any low stock?")]})
    assert out["last_agent"] == "inventory"


def test_finish_short_circuits_without_running_workers():
    graph = _build_test_graph("FINISH", "irrelevant")
    out = graph.invoke({"messages": [HumanMessage(content="hi")]})
    # last_agent never set because no worker ran.
    assert "last_agent" not in out


def test_route_function_defaults_unknown_to_finish():
    assert route_from_supervisor({"next_agent": "not_a_real_agent"}) == "FINISH"
    assert route_from_supervisor({"next_agent": "FINISH"}) == "FINISH"
    assert route_from_supervisor({"next_agent": "sales"}) == "sales"
    assert route_from_supervisor({}) == "FINISH"
