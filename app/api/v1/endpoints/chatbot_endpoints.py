"""Limitless OS Sales Agent API endpoint.

This module provides the REST API endpoint for the Limitless OS sales agent.
The agent uses complete memory retrieval and advanced sales techniques to
qualify leads and close deals.
"""

from datetime import datetime
from typing import Dict, Any
import logging

from fastapi import APIRouter, HTTPException, Depends

from app.api.v1.schemas.chatbot_schemas import (
    SalesAgentRequest,
    SalesAgentResponse,
    ErrorResponse
)
from app.services.chatbot_service import get_sales_agent_service, SalesAgentService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat/full-memory", response_model=SalesAgentResponse)
async def chat_with_sales_agent(
    request: SalesAgentRequest,
    service: SalesAgentService = Depends(get_sales_agent_service)
) -> SalesAgentResponse:
    """Process a chat message with the Limitless OS sales agent.
    
    This endpoint provides direct access to the specialized sales agent that:
    - Retrieves ALL stored memories for complete context
    - Uses advanced sales techniques and psychology
    - Combines web search and datetime tools for enhanced capabilities
    - Focuses on qualifying leads and closing deals for Limitless OS
    
    Args:
        request: Sales agent request containing message and user_id
        service: Sales agent service dependency
        
    Returns:
        SalesAgentResponse: Agent response with metadata
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Process the chat message with the sales agent
        response = await service.chat_with_sales_agent(
            message=request.message,
            user_id=request.user_id,
            **(request.metadata or {})
        )
        
        # Return structured response
        return SalesAgentResponse(
            response=response,
            user_id=request.user_id,
            agent_name="Limitless OS Sales Agent",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Sales agent processing error: {e}")
        raise HTTPException(status_code=500, detail="Sales agent processing failed")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Perform health check of the sales agent service.
    
    Returns:
        Dict[str, Any]: Service health status
    """
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "agent": "Limitless OS Sales Agent",
            "message": "Sales agent service is running normally",
            "capabilities": ["memory", "web_search", "datetime", "sales_psychology"]
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "message": "Sales agent service is experiencing issues"
        } 