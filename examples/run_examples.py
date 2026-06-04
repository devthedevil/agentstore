"""Drive the graph with a handful of representative queries.

Run with: python examples/run_examples.py
Requires ANTHROPIC_API_KEY in your environment.
"""
from __future__ import annotations

import json
import sys
import time

from langchain_core.messages import HumanMessage

# Make `agentstore` importable when run from the repo root without installing.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from agentstore.graph import RECURSION_LIMIT, get_graph  # noqa: E402

EXAMPLES = [
    "What were our total sales and profit?",
    "Which products are running low on stock?",
    "What are the top 3 reasons customers return items?",
    "Who are our top 5 customers by spend?",
    "Which category sold the most overall?",
]


def main() -> int:
    graph = get_graph()
    for q in EXAMPLES:
        print(f"\n── {q} ──")
        start = time.perf_counter()
        result = graph.invoke(
            {"messages": [HumanMessage(content=q)]},
            config={"recursion_limit": RECURSION_LIMIT},
        )
        elapsed = (time.perf_counter() - start) * 1000
        print(json.dumps(
            {
                "agent": result.get("last_agent", "unknown"),
                "answer": result.get("final_response", "")[:400],
                "latency_ms": int(elapsed),
            },
            indent=2,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
