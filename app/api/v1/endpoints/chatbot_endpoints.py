"""Limitless OS Sales Agent API endpoint.

This module provides the REST API endpoint for the Limitless OS sales agent.
The agent uses complete memory retrieval and advanced sales techniques to
qualify leads and close deals.
"""

from datetime import datetime
from typing import Dict, Any, Optional
import logging

from fastapi import APIRouter, HTTPException, Depends

from app.api.v1.schemas.chatbot_schemas import (
    SalesAgentRequest,
    SalesAgentResponse,
    ErrorResponse,
    ConversationHistoryResponse,
    UserConversationsResponse,
    ConversationMessage,
    ConversationSummary
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
        result = await service.chat_with_sales_agent(
            message=request.message,
            user_id=request.user_id,
            channel=request.channel,
            metadata=request.metadata
        )
        
        # Return structured response with conversation data
        return SalesAgentResponse(
            response=result["response"],
            user_id=result["user_id"],
            conversation_id=result["conversation_id"],
            event=result["event"],
            sales_stage=result.get("sales_stage"),
            is_qualified=result.get("is_qualified", False),
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


@router.get("/conversations/{user_id}", response_model=UserConversationsResponse)
async def get_user_conversations(
    user_id: str,
    service: SalesAgentService = Depends(get_sales_agent_service)
) -> UserConversationsResponse:
    """Get all conversations for a user.
    
    Args:
        user_id: User identifier
        service: Sales agent service dependency
        
    Returns:
        UserConversationsResponse: List of user conversations
        
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        result = await service.get_conversation_history(
            user_id=user_id,
            conversation_id=None
        )
        
        return UserConversationsResponse(**result)
        
    except Exception as e:
        logger.error(f"Error getting user conversations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversations")


@router.get("/conversations/{user_id}/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    user_id: str,
    conversation_id: str,
    limit: Optional[int] = None,
    service: SalesAgentService = Depends(get_sales_agent_service)
) -> ConversationHistoryResponse:
    """Get conversation history for a specific conversation.
    
    Args:
        user_id: User identifier
        conversation_id: Conversation identifier
        limit: Optional limit on messages
        service: Sales agent service dependency
        
    Returns:
        ConversationHistoryResponse: Conversation messages and summary
        
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        result = await service.get_conversation_history(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=limit
        )
        
        # Convert to response model
        messages = [
            ConversationMessage(**msg) for msg in result["messages"]
        ]
        
        summary = ConversationSummary(**result["summary"])
        
        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            messages=messages,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation history") 