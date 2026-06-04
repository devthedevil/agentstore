"""Unit tests for the tool functions.

These run against the bundled JSON dataset, no LLM required, so they're
fast and deterministic — perfect for CI.
"""
from __future__ import annotations

from agentstore.tools.customer_tools import (
    get_customer_history,
    get_customer_overview,
    get_top_customers_by_spend,
)
from agentstore.tools.inventory_tools import (
    get_product_stock,
    get_stock_by_category,
    get_stock_overview,
    list_low_stock_products,
)
from agentstore.tools.review_tools import (
    get_lowest_rated_products,
    get_rating_distribution,
    get_review_overview,
    get_top_return_reasons,
)
from agentstore.tools.sales_tools import (
    get_monthly_sales,
    get_revenue_by_category,
    get_revenue_summary,
    get_top_products_by_revenue,
)


def _invoke(tool, **kwargs):
    """Call a @tool function with its underlying implementation."""
    return tool.invoke(kwargs)


# ── Sales ────────────────────────────────────────────────────────────────────


def test_revenue_summary_has_expected_shape():
    r = _invoke(get_revenue_summary)
    for key in ("total_revenue", "total_profit", "completed_orders", "avg_order_value"):
        assert key in r
    assert r["total_revenue"] > 0
    assert r["completed_orders"] > 0


def test_monthly_sales_returns_chronological_rows():
    rows = _invoke(get_monthly_sales)
    assert len(rows) > 0
    months = [r["month"] for r in rows]
    assert months == sorted(months)


def test_top_products_respects_limit_and_caps_at_20():
    assert len(_invoke(get_top_products_by_revenue, limit=3)) == 3
    assert len(_invoke(get_top_products_by_revenue, limit=999)) <= 20


def test_revenue_by_category_sorted_descending():
    rows = _invoke(get_revenue_by_category)
    revs = [r["revenue"] for r in rows]
    assert revs == sorted(revs, reverse=True)


# ── Inventory ────────────────────────────────────────────────────────────────


def test_stock_overview_sums_consistently():
    r = _invoke(get_stock_overview)
    assert r["total_products"] == r["out_of_stock"] + r["low_stock"] + r["healthy_stock"]


def test_low_stock_sorted_by_urgency():
    rows = _invoke(list_low_stock_products, threshold=80)
    qtys = [r["stock_quantity"] for r in rows]
    assert qtys == sorted(qtys)


def test_get_product_stock_unknown_returns_error():
    r = _invoke(get_product_stock, product_id=999999)
    assert "error" in r


def test_stock_by_category_non_empty():
    assert len(_invoke(get_stock_by_category)) > 0


# ── Reviews ──────────────────────────────────────────────────────────────────


def test_review_overview_rating_bounded():
    r = _invoke(get_review_overview)
    assert 0 <= r["average_rating"] <= 5
    assert 0 <= r["return_rate_pct"] <= 100


def test_top_return_reasons_sums_to_total():
    rows = _invoke(get_top_return_reasons, limit=20)
    shares = sum(r["share_pct"] for r in rows)
    # Allow small rounding error.
    assert abs(shares - 100) < 1 or rows == []


def test_rating_distribution_has_all_buckets():
    d = _invoke(get_rating_distribution)
    for i in range(1, 6):
        assert f"{i}_star" in d


def test_lowest_rated_filters_low_signal():
    # Every returned product must have >=2 reviews.
    for r in _invoke(get_lowest_rated_products, limit=20):
        assert r["review_count"] >= 2


# ── Customer ─────────────────────────────────────────────────────────────────


def test_customer_overview_active_plus_inactive_equals_total():
    r = _invoke(get_customer_overview)
    assert r["active_customers"] + r["inactive_customers"] == r["total_customers"]


def test_top_customers_sorted_desc():
    rows = _invoke(get_top_customers_by_spend, limit=5)
    spent = [r["total_spent"] for r in rows]
    assert spent == sorted(spent, reverse=True)


def test_customer_history_unknown_returns_error():
    r = _invoke(get_customer_history, customer_id=99999)
    assert "error" in r


def test_customer_history_known_returns_orders():
    r = _invoke(get_customer_history, customer_id=1)
    assert "orders" in r
    assert r["customer_id"] == 1
