"""Limitless OS Sales Agent Service.

This service provides direct access to the specialized sales agent without
session management. All conversation context is handled by mem0 persistent
memory, eliminating the need for in-memory session state.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from app.core.chatbot_factory import create_default_sales_agent
from app.core.chatbot_base import LimitlessOSIntelligentAgent
from app.services.conversation_service import get_conversation_service, ConversationService
from app.db.mongodb.schemas.conversation_schema import MessageRole, AgentName

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
        self._conversation_service: Optional[ConversationService] = None
    
    async def _ensure_initialized(self) -> None:
        """Ensure the service is properly initialized."""
        if not self._initialized:
            self._sales_agent = create_default_sales_agent()
            self._conversation_service = await get_conversation_service()
            self._initialized = True
            logger.info("Sales agent service initialized")
    
    async def chat_with_sales_agent(self, message: str, user_id: str, **kwargs) -> Dict[str, Any]:
        """Process a chat message with the Limitless OS sales agent.
        
        Args:
            message: User message
            user_id: User identifier for mem0 context
            **kwargs: Additional parameters (channel, metadata, etc.)
            
        Returns:
            Dict containing response and conversation metadata
        """
        await self._ensure_initialized()
        
        try:
            # Create or resume conversation in MongoDB
            channel = kwargs.get("channel", "web")
            metadata = kwargs.get("metadata", {})
            
            # Ensure mem0_user_id is set
            if "mem0_user_id" not in metadata:
                metadata["mem0_user_id"] = user_id
            
            conversation, event = await self._conversation_service.create_or_resume_conversation(
                user_id=user_id,
                channel=channel,
                metadata=metadata
            )
            
            conversation_id = str(conversation["_id"])
            logger.info(f"Using conversation {conversation_id} for user {user_id} (event: {event})")
            
            # Process the message with the sales agent
            response = await self._sales_agent.chat(
                message=message,
                user_id=user_id,  # This is used by mem0
                **kwargs
            )
            
            # Save the conversation turn to MongoDB
            await self._conversation_service.save_conversation_turn(
                conversation_id=conversation_id,
                user_message=message,
                agent_response=response,
                agent_name=AgentName.SUPERVISOR.value  # Default to supervisor
            )
            
            logger.info(f"Sales agent response generated and saved for user: {user_id}")
            
            # Return response with conversation metadata
            return {
                "response": response,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "event": event.value,
                "sales_stage": conversation.get("sales_stage"),
                "is_qualified": conversation.get("is_qualified", False)
            }
            
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
    
    async def get_conversation_history(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get conversation history for a user.
        
        Args:
            user_id: User identifier
            conversation_id: Optional specific conversation ID
            limit: Optional limit on messages
            
        Returns:
            Dict containing conversation history
        """
        await self._ensure_initialized()
        
        try:
            if conversation_id:
                # Get specific conversation history
                messages = await self._conversation_service.get_conversation_history(
                    conversation_id=conversation_id,
                    limit=limit
                )
                summary = await self._conversation_service.get_conversation_summary(
                    conversation_id=conversation_id
                )
                
                return {
                    "conversation_id": conversation_id,
                    "messages": messages,
                    "summary": summary
                }
            else:
                # Get user's conversations
                conversations = await self._conversation_service.get_user_conversations(
                    user_id=user_id,
                    include_closed=False,
                    limit=10
                )
                
                return {
                    "user_id": user_id,
                    "conversations": [
                        {
                            "conversation_id": str(conv["_id"]),
                            "status": conv["status"],
                            "sales_stage": conv["sales_stage"],
                            "message_count": len(conv.get("messages", [])),
                            "created_at": conv["created_at"],
                            "updated_at": conv["updated_at"]
                        }
                        for conv in conversations
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            raise


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