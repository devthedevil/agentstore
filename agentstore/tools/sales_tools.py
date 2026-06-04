"""Tools for the sales-analyst agent.

Each tool is a small, pure function decorated with @tool. The docstring
becomes the description Claude uses for tool selection — keep it precise
and example-driven.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from langchain_core.tools import tool

from ..data import categories, orders, product_by_id, sales_summary


def _completed(orders_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [o for o in orders_list if o["status"] == "completed"]


@tool
def get_revenue_summary() -> dict[str, Any]:
    """Return total revenue, profit, order count, and average order value
    across the entire dataset. Use this for high-level "how is the business
    doing" questions."""
    completed = _completed(orders())
    total_rev = sum(o["total"] for o in completed)
    cost = 0.0
    for o in completed:
        for item in o["items"]:
            p = product_by_id(item["product_id"])
            if p:
                cost += p["cost"] * item["quantity"]
    return {
        "total_revenue": round(total_rev, 2),
        "total_cost": round(cost, 2),
        "total_profit": round(total_rev - cost, 2),
        "profit_margin_pct": round((total_rev - cost) / total_rev * 100, 2) if total_rev else 0,
        "completed_orders": len(completed),
        "avg_order_value": round(total_rev / len(completed), 2) if completed else 0,
    }


@tool
def get_monthly_sales() -> list[dict[str, Any]]:
    """Return the monthly sales summary (revenue, profit, orders) for every
    month in the dataset. Use this for trend, month-over-month, or charting
    questions."""
    return sales_summary()


@tool
def get_top_products_by_revenue(limit: int = 5) -> list[dict[str, Any]]:
    """Return the top N products ranked by total revenue across all completed
    orders. `limit` defaults to 5; cap at 20."""
    limit = max(1, min(int(limit), 20))
    revenue: dict[int, float] = defaultdict(float)
    qty: dict[int, int] = defaultdict(int)
    for o in _completed(orders()):
        for item in o["items"]:
            revenue[item["product_id"]] += item["unit_price"] * item["quantity"]
            qty[item["product_id"]] += item["quantity"]
    ranked = sorted(revenue.items(), key=lambda x: x[1], reverse=True)[:limit]
    out = []
    for pid, rev in ranked:
        p = product_by_id(pid)
        if not p:
            continue
        out.append({
            "product_id": pid,
            "name": p["name"],
            "revenue": round(rev, 2),
            "units_sold": qty[pid],
        })
    return out


@tool
def get_revenue_by_category() -> list[dict[str, Any]]:
    """Return revenue and units sold per product category, ranked highest
    first. Use this for "which category is selling best" questions."""
    cat_rev: dict[int, float] = defaultdict(float)
    cat_qty: dict[int, int] = defaultdict(int)
    for o in _completed(orders()):
        for item in o["items"]:
            p = product_by_id(item["product_id"])
            if p:
                cat_rev[p["category_id"]] += item["unit_price"] * item["quantity"]
                cat_qty[p["category_id"]] += item["quantity"]
    cat_map = {c["id"]: c["name"] for c in categories()}
    rows = [
        {"category": cat_map.get(cid, "Unknown"), "revenue": round(rev, 2), "units_sold": cat_qty[cid]}
        for cid, rev in cat_rev.items()
    ]
    rows.sort(key=lambda r: r["revenue"], reverse=True)
    return rows


SALES_TOOLS = [
    get_revenue_summary,
    get_monthly_sales,
    get_top_products_by_revenue,
    get_revenue_by_category,
]
