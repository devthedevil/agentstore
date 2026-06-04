"""Tools for the customer-analyst agent."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from langchain_core.tools import tool

from ..data import customer_by_id, customers, orders


def _completed_by_customer() -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for o in orders():
        if o["status"] == "completed":
            grouped[o["customer_id"]].append(o)
    return grouped


@tool
def get_customer_overview() -> dict[str, Any]:
    """Return high-level customer KPIs: total customers, number who have
    placed at least one completed order, and average lifetime spend."""
    grouped = _completed_by_customer()
    with_orders = len(grouped)
    total_spend = sum(sum(o["total"] for o in os) for os in grouped.values())
    return {
        "total_customers": len(customers()),
        "active_customers": with_orders,
        "inactive_customers": len(customers()) - with_orders,
        "avg_lifetime_value": round(total_spend / with_orders, 2) if with_orders else 0,
    }


@tool
def get_top_customers_by_spend(limit: int = 5) -> list[dict[str, Any]]:
    """Return the top N customers ranked by total completed-order spend.
    `limit` defaults to 5; cap at 20."""
    limit = max(1, min(int(limit), 20))
    grouped = _completed_by_customer()
    ranked = sorted(
        ((cid, sum(o["total"] for o in os), len(os)) for cid, os in grouped.items()),
        key=lambda r: r[1],
        reverse=True,
    )[:limit]
    out = []
    for cid, spent, n in ranked:
        c = customer_by_id(cid)
        if not c:
            continue
        out.append({
            "customer_id": cid,
            "name": c["name"],
            "city": c["city"],
            "total_spent": round(spent, 2),
            "order_count": n,
        })
    return out


@tool
def get_customer_history(customer_id: int) -> dict[str, Any]:
    """Return a single customer's profile plus their complete order history
    (all statuses). Returns an error dict if the customer doesn't exist."""
    cid = int(customer_id)
    c = customer_by_id(cid)
    if c is None:
        return {"error": f"Customer id={cid} not found"}
    cust_orders = [o for o in orders() if o["customer_id"] == cid]
    completed = [o for o in cust_orders if o["status"] == "completed"]
    total_spent = sum(o["total"] for o in completed)
    return {
        "customer_id": cid,
        "name": c["name"],
        "city": c["city"],
        "state": c["state"],
        "join_date": c["join_date"],
        "order_count": len(cust_orders),
        "completed_orders": len(completed),
        "total_spent": round(total_spent, 2),
        "orders": [
            {
                "order_id": o["id"],
                "date": o["order_date"],
                "status": o["status"],
                "total": o["total"],
                "items": len(o["items"]),
            }
            for o in cust_orders
        ],
    }


@tool
def get_customers_by_state() -> list[dict[str, Any]]:
    """Return a count of customers by US state, sorted by count descending."""
    counts: dict[str, int] = defaultdict(int)
    for c in customers():
        counts[c["state"]] += 1
    return sorted(
        ({"state": s, "customer_count": n} for s, n in counts.items()),
        key=lambda r: r["customer_count"],
        reverse=True,
    )


CUSTOMER_TOOLS = [
    get_customer_overview,
    get_top_customers_by_spend,
    get_customer_history,
    get_customers_by_state,
]
