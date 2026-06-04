"""In-memory store for the demo e-commerce dataset.

Reads the JSON files in `data/` once at import time and exposes typed accessors.
For a real deployment you'd swap this for a database client; the rest of the
codebase only depends on these functions, so the swap is local.
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from .config import get_settings


def _load(name: str) -> list[dict[str, Any]]:
    path: Path = get_settings().data_dir / name
    with path.open() as f:
        return json.load(f)


@cache
def products() -> list[dict[str, Any]]:
    return _load("products.json")


@cache
def categories() -> list[dict[str, Any]]:
    return _load("categories.json")


@cache
def orders() -> list[dict[str, Any]]:
    return _load("orders.json")


@cache
def customers() -> list[dict[str, Any]]:
    return _load("customers.json")


@cache
def reviews() -> list[dict[str, Any]]:
    return _load("reviews.json")


@cache
def sales_summary() -> list[dict[str, Any]]:
    return _load("sales_summary.json")


# ── Lookup helpers ────────────────────────────────────────────────────────────


def product_by_id(pid: int) -> dict[str, Any] | None:
    return next((p for p in products() if p["id"] == pid), None)


def customer_by_id(cid: int) -> dict[str, Any] | None:
    return next((c for c in customers() if c["id"] == cid), None)


def category_by_id(cat_id: int) -> dict[str, Any] | None:
    return next((c for c in categories() if c["id"] == cat_id), None)


def reset_cache() -> None:
    """Clear all cached loads. Useful for tests that mutate fixtures."""
    for fn in (products, categories, orders, customers, reviews, sales_summary):
        fn.cache_clear()
