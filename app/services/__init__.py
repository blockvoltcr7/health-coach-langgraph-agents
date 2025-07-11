"""Service layer for the application."""

from .chatbot_service import (
    SalesAgentService,
    get_sales_agent_service,
    shutdown_sales_agent_service
)

__all__ = [
    "SalesAgentService",
    "get_sales_agent_service",
    "shutdown_sales_agent_service",
]