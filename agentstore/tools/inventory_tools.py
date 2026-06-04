"""Tools for the inventory-analyst agent."""
from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from ..data import categories, products

DEFAULT_LOW_STOCK_THRESHOLD = 80


@tool
def get_stock_overview() -> dict[str, Any]:
    """Return totals: number of products, out-of-stock count, low-stock count
    (<=80 units), and healthy-stock count. Good first call for inventory
    health-check questions."""
    out_of_stock = [p for p in products() if p["stock_quantity"] == 0]
    low = [p for p in products() if 0 < p["stock_quantity"] <= DEFAULT_LOW_STOCK_THRESHOLD]
    healthy = [p for p in products() if p["stock_quantity"] > DEFAULT_LOW_STOCK_THRESHOLD]
    return {
        "total_products": len(products()),
        "out_of_stock": len(out_of_stock),
        "low_stock": len(low),
        "healthy_stock": len(healthy),
        "low_stock_threshold": DEFAULT_LOW_STOCK_THRESHOLD,
    }


@tool
def list_low_stock_products(threshold: int = DEFAULT_LOW_STOCK_THRESHOLD) -> list[dict[str, Any]]:
    """Return products at or below `threshold` units in stock, sorted by
    urgency (lowest first). `threshold` defaults to 80."""
    threshold = max(0, int(threshold))
    rows = [
        {
            "product_id": p["id"],
            "name": p["name"],
            "stock_quantity": p["stock_quantity"],
            "price": p["price"],
        }
        for p in products()
        if p["stock_quantity"] <= threshold
    ]
    rows.sort(key=lambda r: r["stock_quantity"])
    return rows


@tool
def get_stock_by_category() -> list[dict[str, Any]]:
    """Return total stock units and product count per category."""
    cat_map = {c["id"]: c["name"] for c in categories()}
    agg: dict[int, dict[str, int]] = {}
    for p in products():
        a = agg.setdefault(p["category_id"], {"stock": 0, "count": 0})
        a["stock"] += p["stock_quantity"]
        a["count"] += 1
    return [
        {"category": cat_map.get(cid, "Unknown"), "total_stock": v["stock"], "product_count": v["count"]}
        for cid, v in sorted(agg.items())
    ]


@tool
def get_product_stock(product_id: int) -> dict[str, Any]:
    """Return the current stock level for a specific product by ID. Returns
    an error dict if the product doesn't exist."""
    p = next((p for p in products() if p["id"] == int(product_id)), None)
    if p is None:
        return {"error": f"Product id={product_id} not found"}
    return {
        "product_id": p["id"],
        "name": p["name"],
        "stock_quantity": p["stock_quantity"],
        "rating": p["rating"],
    }


INVENTORY_TOOLS = [
    get_stock_overview,
    list_low_stock_products,
    get_stock_by_category,
    get_product_stock,
]
