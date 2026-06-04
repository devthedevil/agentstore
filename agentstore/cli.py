"""Tiny CLI: `python -m agentstore.cli "your question"`.

Handy for smoke-testing the graph from a terminal without bringing the server up.
"""
from __future__ import annotations

import json
import sys

from langchain_core.messages import HumanMessage

from .graph import RECURSION_LIMIT, get_graph


def main() -> int:
    if len(sys.argv) < 2:
        print('usage: python -m agentstore.cli "your question"', file=sys.stderr)
        return 2
    query = " ".join(sys.argv[1:])
    result = get_graph().invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": RECURSION_LIMIT},
    )
    print(json.dumps(
        {
            "agent": result.get("last_agent", "unknown"),
            "answer": result.get("final_response", ""),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
