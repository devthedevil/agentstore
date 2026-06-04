"""Pytest setup.

Sets a dummy ANTHROPIC_API_KEY so importing agentstore.config doesn't blow up
in CI. Tests that exercise the LLM should mock it explicitly.
"""
from __future__ import annotations

import os

# Force a dummy key. We can't use setdefault here because some shells set
# ANTHROPIC_API_KEY="" (empty string), which would still defeat the default.
if not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-dummy"
