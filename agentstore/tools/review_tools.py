"""Tools for the review-analyst agent."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from langchain_core.tools import tool

from ..data import product_by_id, reviews


@tool
def get_review_overview() -> dict[str, Any]:
    """Return review-level KPIs: total reviews, average rating, return-request
    count, and overall return rate."""
    rs = reviews()
    returns = [r for r in rs if r.get("has_return_request")]
    avg = round(sum(r["rating"] for r in rs) / len(rs), 2) if rs else 0
    return {
        "total_reviews": len(rs),
        "average_rating": avg,
        "return_requests": len(returns),
        "return_rate_pct": round(len(returns) / len(rs) * 100, 2) if rs else 0,
    }


@tool
def get_top_return_reasons(limit: int = 5) -> list[dict[str, Any]]:
    """Return the most common return reasons across all reviews with
    return requests, sorted by frequency. `limit` defaults to 5."""
    limit = max(1, min(int(limit), 20))
    reasons = [r["return_reason"] for r in reviews() if r.get("has_return_request") and r.get("return_reason")]
    counts = Counter(reasons).most_common(limit)
    total = len(reasons)
    return [
        {"reason": reason, "count": n, "share_pct": round(n / total * 100, 2) if total else 0}
        for reason, n in counts
    ]


@tool
def get_rating_distribution() -> dict[str, int]:
    """Return how many 1- through 5-star reviews exist. Useful for sentiment
    bucket questions."""
    buckets = {f"{i}_star": 0 for i in range(1, 6)}
    for r in reviews():
        buckets[f"{r['rating']}_star"] += 1
    return buckets


@tool
def get_reviews_for_product(product_id: int, limit: int = 5) -> dict[str, Any]:
    """Return up to `limit` review summaries for a specific product, plus
    the product's aggregate rating and review count."""
    pid = int(product_id)
    matches = [r for r in reviews() if r["product_id"] == pid]
    matches.sort(key=lambda r: r["review_date"], reverse=True)
    p = product_by_id(pid)
    avg = round(sum(r["rating"] for r in matches) / len(matches), 2) if matches else 0
    return {
        "product_id": pid,
        "product_name": p["name"] if p else "Unknown",
        "review_count": len(matches),
        "average_rating": avg,
        "reviews": [
            {
                "rating": r["rating"],
                "title": r["title"],
                "date": r["review_date"],
                "has_return_request": r.get("has_return_request", False),
            }
            for r in matches[: max(1, min(int(limit), 20))]
        ],
    }


@tool
def get_lowest_rated_products(limit: int = 5) -> list[dict[str, Any]]:
    """Return products with the lowest average review rating, requiring at
    least 2 reviews so single outliers don't dominate."""
    agg: dict[int, list[int]] = defaultdict(list)
    for r in reviews():
        agg[r["product_id"]].append(r["rating"])
    rows = []
    for pid, ratings in agg.items():
        if len(ratings) < 2:
            continue
        p = product_by_id(pid)
        if not p:
            continue
        rows.append({
            "product_id": pid,
            "name": p["name"],
            "average_rating": round(sum(ratings) / len(ratings), 2),
            "review_count": len(ratings),
        })
    rows.sort(key=lambda r: r["average_rating"])
    return rows[: max(1, min(int(limit), 20))]


REVIEW_TOOLS = [
    get_review_overview,
    get_top_return_reasons,
    get_rating_distribution,
    get_reviews_for_product,
    get_lowest_rated_products,
]
