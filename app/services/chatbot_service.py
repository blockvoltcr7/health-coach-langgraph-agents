"""Limitless OS Sales Agent Service.

This service provides direct access to the specialized sales agent without
session management. All conversation context is handled by mem0 persistent
memory, eliminating the need for in-memory session state.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from app.core.chatbot_factory import create_default_sales_agent
from app.core.chatbot_base import LimitlessOSIntelligentAgent

logger = logging.getLogger(__name__)


class SalesAgentService:
    """Service class for Limitless OS sales agent operations.
    
    This service provides direct access to the specialized sales agent without
    maintaining any session state. All conversation context is managed by mem0,
    making the service stateless and scalable.
    """
    
    def __init__(self):
        """Initialize the sales agent service."""
        self._initialized = False
        self._sales_agent: LimitlessOSIntelligentAgent = None
    
    async def _ensure_initialized(self) -> None:
        """Ensure the service is properly initialized."""
        if not self._initialized:
            self._sales_agent = create_default_sales_agent()
            self._initialized = True
            logger.info("Sales agent service initialized")
    
    async def chat_with_sales_agent(self, message: str, user_id: str, **kwargs) -> str:
        """Process a chat message with the Limitless OS sales agent.
        
        Args:
            message: User message
            user_id: User identifier for mem0 context
            **kwargs: Additional parameters
            
        Returns:
            str: Sales agent response
        """
        await self._ensure_initialized()
        
        try:
            # Process the message with the sales agent
            response = await self._sales_agent.chat(
                message=message,
                user_id=user_id,
                **kwargs
            )
            
            logger.info(f"Sales agent response generated for user: {user_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error in sales agent chat for user {user_id}: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the sales agent service.
        
        Returns:
            Dict[str, Any]: Health check status
        """
        await self._ensure_initialized()
        
        try:
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "agent": "Limitless OS Sales Agent",
                "message": "Sales agent service is running normally",
                "capabilities": ["memory", "web_search", "datetime", "sales_psychology"]
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "message": "Sales agent service is experiencing issues"
            }


# Global service instance
_sales_agent_service: SalesAgentService = None


async def get_sales_agent_service() -> SalesAgentService:
    """Get the global sales agent service instance.
    
    Returns:
        SalesAgentService: The service instance
    """
    global _sales_agent_service
    if _sales_agent_service is None:
        _sales_agent_service = SalesAgentService()
    await _sales_agent_service._ensure_initialized()
    return _sales_agent_service


async def shutdown_sales_agent_service() -> None:
    """Shutdown the sales agent service.
    
    Since the service is now stateless, this is a no-op but kept
    for compatibility with existing shutdown procedures.
    """
    global _sales_agent_service
    if _sales_agent_service:
        logger.info("Sales agent service shutdown completed") 