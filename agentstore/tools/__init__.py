"""Tool collections exposed to specialist agents.

Each module groups tools by domain. Agents only see their own toolset,
which keeps prompts focused and reduces tool-selection errors.
"""
from .customer_tools import CUSTOMER_TOOLS
from .inventory_tools import INVENTORY_TOOLS
from .review_tools import REVIEW_TOOLS
from .sales_tools import SALES_TOOLS

__all__ = ["SALES_TOOLS", "INVENTORY_TOOLS", "REVIEW_TOOLS", "CUSTOMER_TOOLS"]
