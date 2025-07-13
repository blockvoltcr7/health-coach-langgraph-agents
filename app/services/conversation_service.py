"""Conversation service for managing chat conversations with MongoDB persistence.

This service provides high-level conversation management, integrating MongoDB
persistence with the chatbot's memory system (mem0).
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from enum import Enum

from app.db.mongodb.async_conversation_repository import AsyncConversationRepository
from app.db.mongodb.async_client import get_async_database
from app.db.mongodb.schemas.conversation_schema import (
    ConversationStatus,
    MessageRole,
    SalesStage,
    AgentName,
    ObjectionType
)

logger = logging.getLogger(__name__)


class ConversationEvent(str, Enum):
    """Conversation event types for tracking."""
    CREATED = "conversation_created"
    RESUMED = "conversation_resumed"
    MESSAGE_ADDED = "message_added"
    STAGE_CHANGED = "stage_changed"
    HANDOFF = "agent_handoff"
    QUALIFIED = "lead_qualified"
    CLOSED = "conversation_closed"
    OBJECTION_RAISED = "objection_raised"
    OBJECTION_RESOLVED = "objection_resolved"
    OBJECTION_DEFERRED = "objection_deferred"


class ConversationService:
    """Service for managing conversations with MongoDB persistence.
    
    This service provides:
    - Conversation lifecycle management
    - Message persistence
    - State tracking
    - Integration with mem0 for memory
    - Error handling and recovery
    """
    
    def __init__(self, repository: Optional[AsyncConversationRepository] = None):
        """Initialize conversation service.
        
        Args:
            repository: Optional conversation repository instance
        """
        self._repository = repository or AsyncConversationRepository()
        self._initialized = False
    
    async def _ensure_initialized(self) -> None:
        """Ensure service is initialized."""
        if not self._initialized:
            # Any initialization logic here
            self._initialized = True
            logger.info("Conversation service initialized")
    
    async def create_or_resume_conversation(
        self,
        user_id: str,
        channel: str = "web",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], ConversationEvent]:
        """Create new or resume existing conversation.
        
        Args:
            user_id: User identifier
            channel: Conversation channel
            metadata: Optional metadata (mem0_user_id, source, etc.)
            
        Returns:
            Tuple of (conversation_document, event_type)
        """
        await self._ensure_initialized()
        
        try:
            # Try to find active conversation
            conversation = await self._repository.find_active_by_user(user_id)
            
            if conversation:
                logger.info(f"Resuming conversation {conversation['_id']} for user {user_id}")
                return conversation, ConversationEvent.RESUMED
            
            # Create new conversation
            logger.info(f"Creating new conversation for user {user_id}")
            
            # Ensure mem0_user_id is in metadata
            if metadata is None:
                metadata = {}
            if "mem0_user_id" not in metadata:
                metadata["mem0_user_id"] = f"mem0_{user_id}"
            
            conversation_id = await self._repository.create_conversation(
                user_id=user_id,
                channel=channel,
                metadata=metadata,
                source=metadata.get("source"),
                campaign=metadata.get("campaign")
            )
            
            conversation = await self._repository.get_conversation_state(conversation_id)
            return conversation, ConversationEvent.CREATED
            
        except Exception as e:
            logger.error(f"Error in create_or_resume_conversation: {e}")
            raise
    
    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        update_metrics: bool = True
    ) -> None:
        """Save a message to the conversation.
        
        Args:
            conversation_id: Conversation ID
            role: Message role (user, assistant, agent name)
            content: Message content
            update_metrics: Whether to update conversation metrics
        """
        await self._ensure_initialized()
        
        try:
            # Save message to MongoDB
            result = await self._repository.add_message_async(
                conversation_id=conversation_id,
                role=role,
                content=content
            )
            
            if result.modified_count == 0:
                logger.warning(f"Message not saved to conversation {conversation_id}")
            else:
                logger.info(f"Message saved to conversation {conversation_id}")
                
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            # Don't raise - allow conversation to continue even if persistence fails
    
    async def save_conversation_turn(
        self,
        conversation_id: str,
        user_message: str,
        agent_response: str,
        agent_name: str = MessageRole.ASSISTANT.value
    ) -> None:
        """Save a complete conversation turn (user message + agent response).
        
        Args:
            conversation_id: Conversation ID
            user_message: User's message
            agent_response: Agent's response
            agent_name: Name of responding agent
        """
        await self._ensure_initialized()
        
        try:
            # Save user message
            await self.save_message(
                conversation_id=conversation_id,
                role=MessageRole.USER.value,
                content=user_message
            )
            
            # Save agent response
            await self.save_message(
                conversation_id=conversation_id,
                role=agent_name,
                content=agent_response
            )
            
        except Exception as e:
            logger.error(f"Error saving conversation turn: {e}")
    
    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        format_for_llm: bool = False
    ) -> List[Dict[str, Any]]:
        """Get conversation history.
        
        Args:
            conversation_id: Conversation ID
            limit: Optional limit on messages
            format_for_llm: Format messages for LLM consumption
            
        Returns:
            List of message dictionaries
        """
        await self._ensure_initialized()
        
        try:
            messages = await self._repository.get_conversation_history(
                conversation_id=conversation_id,
                limit=limit
            )
            
            if format_for_llm:
                # Format messages for LLM (role/content pairs)
                return [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in messages
                ]
            
            return messages
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    async def update_conversation_metadata(
        self,
        conversation_id: str,
        metadata_updates: Dict[str, Any]
    ) -> None:
        """Update conversation metadata.
        
        Args:
            conversation_id: Conversation ID
            metadata_updates: Metadata fields to update
        """
        await self._ensure_initialized()
        
        try:
            update_query = {
                "$set": {
                    f"metadata.{key}": value
                    for key, value in metadata_updates.items()
                }
            }
            
            await self._repository.update_by_id(conversation_id, update_query)
            logger.info(f"Updated metadata for conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error updating conversation metadata: {e}")
    
    async def update_sales_stage(
        self,
        conversation_id: str,
        new_stage: str,
        notes: Optional[str] = None
    ) -> None:
        """Update conversation sales stage.
        
        Args:
            conversation_id: Conversation ID
            new_stage: New sales stage
            notes: Optional notes about the change
        """
        await self._ensure_initialized()
        
        try:
            await self._repository.update_sales_stage_async(
                conversation_id=conversation_id,
                new_stage=new_stage,
                notes=notes
            )
            logger.info(f"Updated sales stage to {new_stage} for conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error updating sales stage: {e}")
    
    async def handle_agent_handoff(
        self,
        conversation_id: str,
        from_agent: str,
        to_agent: str,
        reason: str
    ) -> None:
        """Handle agent handoff.
        
        Args:
            conversation_id: Conversation ID
            from_agent: Current agent
            to_agent: Target agent
            reason: Handoff reason
        """
        await self._ensure_initialized()
        
        try:
            await self._repository.add_handoff_async(
                conversation_id=conversation_id,
                from_agent=from_agent,
                to_agent=to_agent,
                reason=reason
            )
            logger.info(f"Handoff from {from_agent} to {to_agent} for conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error handling agent handoff: {e}")
    
    async def get_user_conversations(
        self,
        user_id: str,
        include_closed: bool = False,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get all conversations for a user.
        
        Args:
            user_id: User ID
            include_closed: Include closed conversations
            limit: Maximum number of conversations
            
        Returns:
            List of conversation documents
        """
        await self._ensure_initialized()
        
        try:
            filter = {"user_id": user_id}
            
            if not include_closed:
                filter["status"] = {"$ne": ConversationStatus.CLOSED.value}
            
            conversations = await self._repository.find_many(
                filter=filter,
                limit=limit,
                sort=[("updated_at", -1)]
            )
            
            return conversations
            
        except Exception as e:
            logger.error(f"Error getting user conversations: {e}")
            return []
    
    async def close_conversation(
        self,
        conversation_id: str,
        reason: Optional[str] = None
    ) -> None:
        """Close a conversation.
        
        Args:
            conversation_id: Conversation ID
            reason: Optional closure reason
        """
        await self._ensure_initialized()
        
        try:
            await self._repository.close_conversation(
                conversation_id=conversation_id,
                reason=reason
            )
            logger.info(f"Closed conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error closing conversation: {e}")
    
    async def get_conversation_summary(
        self,
        conversation_id: str
    ) -> Dict[str, Any]:
        """Get conversation summary with key metrics.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Summary dictionary
        """
        await self._ensure_initialized()
        
        try:
            conversation = await self._repository.get_conversation_state(
                conversation_id,
                raise_on_missing=False
            )
            
            if not conversation:
                return {}
            
            return {
                "conversation_id": str(conversation["_id"]),
                "user_id": conversation["user_id"],
                "status": conversation["status"],
                "sales_stage": conversation["sales_stage"],
                "message_count": len(conversation.get("messages", [])),
                "created_at": conversation["created_at"],
                "updated_at": conversation["updated_at"],
                "is_qualified": conversation.get("is_qualified", False),
                "current_agent": conversation.get("current_agent"),
                "handoff_count": len(conversation.get("handoffs", [])),
                "qualification_score": conversation.get("qualification", {}).get("overall_score", 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation summary: {e}")
            return {}
    
    async def raise_objection(
        self,
        conversation_id: str,
        objection_type: str,
        content: str,
        severity: str = "medium"
    ) -> Optional[str]:
        """Raise an objection in the conversation.
        
        Args:
            conversation_id: Conversation ID
            objection_type: Type of objection (from ObjectionType enum)
            content: Objection content/description
            severity: Objection severity (high, medium, low)
            
        Returns:
            Optional[str]: Objection ID if successful, None otherwise
        """
        await self._ensure_initialized()
        
        try:
            objection_id = await self._repository.add_objection(
                conversation_id=conversation_id,
                objection_type=objection_type,
                content=content,
                severity=severity
            )
            
            logger.info(f"Raised {severity} {objection_type} objection in conversation {conversation_id}")
            return objection_id
            
        except Exception as e:
            logger.error(f"Error raising objection: {e}")
            return None
    
    async def handle_objection(
        self,
        conversation_id: str,
        objection_id: str,
        resolution_method: str,
        resolution_notes: str,
        handled_by: str = AgentName.OBJECTION_HANDLER.value,
        confidence: float = 0.8
    ) -> bool:
        """Handle/resolve an objection.
        
        Args:
            conversation_id: Conversation ID
            objection_id: Objection ID to handle
            resolution_method: How the objection was resolved
            resolution_notes: Detailed notes about the resolution
            handled_by: Agent who handled the objection
            confidence: Confidence level in the resolution (0-1)
            
        Returns:
            bool: True if successful, False otherwise
        """
        await self._ensure_initialized()
        
        try:
            result = await self._repository.mark_objection_handled(
                conversation_id=conversation_id,
                objection_id=objection_id,
                resolution_method=resolution_method,
                resolution_notes=resolution_notes,
                handled_by=handled_by,
                confidence=confidence
            )
            
            if result.modified_count > 0:
                logger.info(f"Resolved objection {objection_id} in conversation {conversation_id}")
                return True
            else:
                logger.warning(f"Failed to resolve objection {objection_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error handling objection: {e}")
            return False
    
    async def defer_objection(
        self,
        conversation_id: str,
        objection_id: str,
        reason: str,
        follow_up_date: Optional[datetime] = None
    ) -> bool:
        """Defer an objection for later handling.
        
        Args:
            conversation_id: Conversation ID
            objection_id: Objection ID to defer
            reason: Reason for deferral
            follow_up_date: Optional date to follow up
            
        Returns:
            bool: True if successful, False otherwise
        """
        await self._ensure_initialized()
        
        try:
            result = await self._repository.defer_objection(
                conversation_id=conversation_id,
                objection_id=objection_id,
                reason=reason,
                follow_up_date=follow_up_date
            )
            
            if result.modified_count > 0:
                logger.info(f"Deferred objection {objection_id} in conversation {conversation_id}")
                return True
            else:
                logger.warning(f"Failed to defer objection {objection_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deferring objection: {e}")
            return False
    
    async def get_active_objections(
        self,
        conversation_id: str
    ) -> List[Dict[str, Any]]:
        """Get all active objections for a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of active objection dictionaries
        """
        await self._ensure_initialized()
        
        try:
            objections = await self._repository.get_active_objections(conversation_id)
            return objections
            
        except Exception as e:
            logger.error(f"Error getting active objections: {e}")
            return []
    
    async def perform_atomic_objection_resolution(
        self,
        conversation_id: str,
        objection_id: str,
        resolution_data: Dict[str, Any],
        move_to_closing: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """Atomically resolve objection and potentially move to closing stage.
        
        Args:
            conversation_id: Conversation ID
            objection_id: Objection to resolve
            resolution_data: Resolution details
            move_to_closing: Whether to move to closing if all objections resolved
            
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        await self._ensure_initialized()
        
        try:
            # Determine next stage if moving forward
            next_stage = SalesStage.CLOSING.value if move_to_closing else None
            
            success, error_msg = await self._repository.perform_atomic_objection_resolution(
                conversation_id=conversation_id,
                objection_id=objection_id,
                resolution_data=resolution_data,
                move_to_next_stage=next_stage
            )
            
            if success:
                logger.info(f"Atomically resolved objection {objection_id}")
            else:
                logger.error(f"Failed to atomically resolve objection: {error_msg}")
                
            return success, error_msg
            
        except Exception as e:
            logger.error(f"Error in atomic objection resolution: {e}")
            return False, str(e)
    
    async def perform_atomic_handoff_with_stage(
        self,
        conversation_id: str,
        from_agent: str,
        to_agent: str,
        new_stage: str,
        handoff_reason: str,
        stage_notes: Optional[str] = None
    ) -> bool:
        """Atomically perform agent handoff with stage update.
        
        Args:
            conversation_id: Conversation ID
            from_agent: Current agent
            to_agent: Target agent
            new_stage: New sales stage
            handoff_reason: Reason for handoff
            stage_notes: Notes about stage change
            
        Returns:
            bool: True if successful
        """
        await self._ensure_initialized()
        
        try:
            success = await self._repository.perform_atomic_handoff_with_stage_update(
                conversation_id=conversation_id,
                from_agent=from_agent,
                to_agent=to_agent,
                new_stage=new_stage,
                handoff_reason=handoff_reason,
                stage_notes=stage_notes
            )
            
            if success:
                logger.info(f"Atomic handoff from {from_agent} to {to_agent} with stage {new_stage}")
            else:
                logger.error("Failed to perform atomic handoff")
                
            return success
            
        except Exception as e:
            logger.error(f"Error in atomic handoff: {e}")
            return False
    
    async def get_objection_analytics(
        self,
        user_id: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get objection analytics.
        
        Args:
            user_id: Optional user ID to filter by
            date_from: Start date for analysis
            date_to: End date for analysis
            
        Returns:
            Dict containing objection analytics
        """
        await self._ensure_initialized()
        
        try:
            analytics = await self._repository.get_objection_analytics(
                user_id=user_id,
                date_from=date_from,
                date_to=date_to
            )
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting objection analytics: {e}")
            return {
                "total_objections": 0,
                "resolved_count": 0,
                "resolution_rate": 0,
                "by_type": []
            }


# Global service instance
_conversation_service: Optional[ConversationService] = None


async def get_conversation_service() -> ConversationService:
    """Get the global conversation service instance.
    
    Returns:
        ConversationService: The service instance
    """
    global _conversation_service
    if _conversation_service is None:
        _conversation_service = ConversationService()
    return _conversation_service