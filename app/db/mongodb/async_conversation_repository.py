"""Async conversation repository for MongoDB operations using motor.

This module provides async conversation-specific database operations
extending the async base repository pattern.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorClientSession
from pymongo.results import UpdateResult
from bson import ObjectId
from pymongo.errors import PyMongoError

from app.db.mongodb.async_base_repository import AsyncBaseRepository
from app.db.mongodb.schemas.conversation_schema import (
    ConversationSchema,
    ConversationStatus,
    MessageRole,
    SalesStage,
    AgentName,
    ObjectionType
)
from app.db.mongodb.validators import ConversationValidator, SalesStageTransitionValidator

logger = logging.getLogger(__name__)


class AsyncConversationRepository(AsyncBaseRepository[Dict[str, Any]]):
    """Async repository for conversation collection operations."""
    
    def __init__(self, database: Optional[AsyncIOMotorDatabase] = None):
        """Initialize async repository.
        
        Args:
            database: Optional async database instance
        """
        super().__init__(database)
    
    @property
    def collection_name(self) -> str:
        """Get the collection name."""
        return "conversations"
    
    async def create_conversation(
        self,
        user_id: str,
        channel: str = "web",
        initial_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        campaign: Optional[str] = None
    ) -> str:
        """Create a new conversation with validation.
        
        Args:
            user_id: User ID as string
            channel: Conversation channel (web, mobile, api)
            initial_message: Optional initial user message
            metadata: Optional metadata dictionary
            source: Traffic source
            campaign: Marketing campaign identifier
            
        Returns:
            str: Created conversation ID
            
        Raises:
            ValueError: If validation fails
        """
        # Create conversation document
        doc = ConversationSchema.create_conversation_document(
            user_id=user_id,
            channel=channel,
            initial_message=initial_message,
            metadata=metadata,
            source=source,
            campaign=campaign
        )
        
        # Validate document
        is_valid, errors = ConversationValidator.validate_conversation_document(doc)
        if not is_valid:
            logger.error(f"Conversation validation failed: {errors}")
            raise ValueError(f"Invalid conversation document: {errors}")
        
        # Insert document
        result = await self.create_one(doc)
        return str(result.inserted_id)
    
    async def get_conversation_state(
        self,
        conversation_id: str,
        raise_on_missing: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get conversation state by ID with error handling.
        
        Args:
            conversation_id: Conversation ID
            raise_on_missing: Whether to raise exception if not found
            
        Returns:
            Optional[Dict[str, Any]]: Conversation document or None
            
        Raises:
            ValueError: If conversation not found and raise_on_missing=True
        """
        conversation = await self.find_by_id(conversation_id)
        
        if conversation is None and raise_on_missing:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        return conversation
    
    async def find_active_by_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Find active conversation for a user.
        
        Args:
            user_id: User ID as string
            
        Returns:
            Optional[Dict[str, Any]]: Active conversation or None
        """
        return await self.find_one({
            "user_id": user_id,
            "status": ConversationStatus.ACTIVE.value
        })
    
    async def find_or_create_conversation(
        self,
        user_id: str,
        channel: str = "web",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Find existing active conversation or create new one.
        
        Args:
            user_id: User ID
            channel: Conversation channel
            metadata: Optional metadata
            
        Returns:
            Dict[str, Any]: Conversation document
        """
        # Try to find active conversation
        conversation = await self.find_active_by_user(user_id)
        
        if conversation:
            logger.info(f"Found active conversation for user {user_id}")
            return conversation
        
        # Create new conversation
        logger.info(f"Creating new conversation for user {user_id}")
        conversation_id = await self.create_conversation(
            user_id=user_id,
            channel=channel,
            metadata=metadata
        )
        
        return await self.get_conversation_state(conversation_id)
    
    async def add_message_async(
        self,
        conversation_id: str,
        role: str,
        content: str
    ) -> UpdateResult:
        """Add a message to a conversation and update metrics.
        
        Args:
            conversation_id: Conversation ID
            role: Message role (user, assistant, agent name)
            content: Message content
            
        Returns:
            UpdateResult: Result of the update operation
        """
        now = datetime.now(timezone.utc)
        message = {
            "role": role,
            "content": content,
            "timestamp": now.isoformat() + "Z"
        }
        
        # Determine which agent metrics to update
        agent_field = None
        if role in [agent.value for agent in AgentName]:
            agent_field = f"agent_metrics.{role}"
        
        # Build update query
        update_query = {
            "$push": {"messages": message},
            "$inc": {
                "agent_metrics.total_messages": 1,
                "agent_context.interaction_count": 1
            },
            "$set": {"updated_at": now.isoformat() + "Z"}
        }
        
        # Add agent-specific metrics update if applicable
        if agent_field:
            update_query["$inc"][f"{agent_field}.messages_sent"] = 1
        
        return await self.update_by_id(conversation_id, update_query)
    
    async def validate_stage_transition(
        self,
        conversation_id: str,
        new_stage: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Validate if a stage transition is allowed.
        
        Args:
            conversation_id: Conversation ID
            new_stage: Target stage
            context: Additional validation context
            
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: 
                (is_valid, current_stage, error_message)
        """
        # Get current conversation state
        conversation = await self.get_conversation_state(
            conversation_id,
            raise_on_missing=False
        )
        
        if not conversation:
            return False, None, f"Conversation {conversation_id} not found"
        
        current_stage = conversation.get("sales_stage")
        if not current_stage:
            return False, None, "Current sales stage not found"
        
        # Check if already at target stage
        if current_stage == new_stage:
            return True, current_stage, None
        
        # Validate transition
        is_valid, error_msg = SalesStageTransitionValidator.validate_transition(
            current_stage,
            new_stage,
            context
        )
        
        return is_valid, current_stage, error_msg
    
    async def update_sales_stage_async(
        self,
        conversation_id: str,
        new_stage: str,
        notes: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        validate: bool = True,
        triggered_by: str = "system",
        session: Optional[AsyncIOMotorClientSession] = None
    ) -> UpdateResult:
        """Update conversation sales stage with validation and history tracking.
        
        Args:
            conversation_id: Conversation ID
            new_stage: New sales stage
            notes: Optional notes about the stage change
            context: Context for validation (e.g., qualification_complete)
            validate: Whether to validate the transition
            triggered_by: Who/what triggered the transition
            
        Returns:
            UpdateResult: Result of the update operation
            
        Raises:
            ValueError: If transition is invalid and validate=True
        """
        # Validate transition if requested
        if validate:
            is_valid, current_stage, error_msg = await self.validate_stage_transition(
                conversation_id,
                new_stage,
                context
            )
            
            if not is_valid:
                raise ValueError(f"Invalid stage transition: {error_msg}")
        else:
            # Get current stage for history
            conversation = await self.get_conversation_state(conversation_id)
            current_stage = conversation.get("sales_stage") if conversation else None
        
        now = datetime.now(timezone.utc)
        
        # Create stage history entry
        stage_entry = {
            "stage": new_stage,
            "timestamp": now.isoformat() + "Z",
            "notes": notes or ""
        }
        
        # Create stage transition record if we have current stage
        update_query = {
            "$set": {
                "sales_stage": new_stage,
                "updated_at": now.isoformat() + "Z"
            },
            "$push": {"stage_history": stage_entry}
        }
        
        # Add stage transition if current stage exists
        if current_stage:
            transition = {
                "from": current_stage,
                "to": new_stage,
                "timestamp": now.isoformat() + "Z",
                "reason": notes or "Stage progression",
                "triggered_by": triggered_by,
                "context": context or {}
            }
            update_query["$push"]["stage_transitions"] = transition
        
        # Update is_qualified flag for qualified stage
        if new_stage == SalesStage.QUALIFIED.value:
            update_query["$set"]["is_qualified"] = True
            update_query["$set"]["qualification.qualified_at"] = now.isoformat() + "Z"
            update_query["$set"]["qualification.qualified_by"] = triggered_by
        
        if session:
            collection = await self._ensure_collection()
            return await collection.update_one(
                {"_id": ObjectId(conversation_id)},
                update_query,
                session=session
            )
        else:
            return await self.update_by_id(conversation_id, update_query)
    
    async def update_qualification(
        self,
        conversation_id: str,
        qualification_data: Optional[Dict[str, Any]] = None,
        budget_info: Optional[Dict[str, Any]] = None,
        authority_info: Optional[Dict[str, Any]] = None,
        need_info: Optional[Dict[str, Any]] = None,
        timeline_info: Optional[Dict[str, Any]] = None,
        session: Optional[AsyncIOMotorClientSession] = None
    ) -> UpdateResult:
        """Update conversation qualification data.
        
        Args:
            conversation_id: Conversation ID
            qualification_data: Complete qualification data (legacy)
            budget_info: Budget qualification data
            authority_info: Authority qualification data
            need_info: Need qualification data
            timeline_info: Timeline qualification data
            session: Optional database session for transactions
            
        Returns:
            UpdateResult: Result of the update operation
        """
        now = datetime.now(timezone.utc)
        update_query = {"$set": {"updated_at": now.isoformat() + "Z"}}
        
        # Handle legacy qualification_data parameter
        if qualification_data:
            for field, value in qualification_data.items():
                update_query["$set"][f"qualification.{field}"] = value
        else:
            # Handle individual BANT components
            if budget_info:
                for k, v in budget_info.items():
                    update_query["$set"][f"qualification.budget.{k}"] = v
            if authority_info:
                for k, v in authority_info.items():
                    update_query["$set"][f"qualification.authority.{k}"] = v
            if need_info:
                for k, v in need_info.items():
                    update_query["$set"][f"qualification.need.{k}"] = v
            if timeline_info:
                for k, v in timeline_info.items():
                    update_query["$set"][f"qualification.timeline.{k}"] = v
        
        # Calculate overall score if all components provided
        if all([budget_info, authority_info, need_info, timeline_info]):
            scores = []
            for info in [budget_info, authority_info, need_info, timeline_info]:
                if info.get("meets_criteria"):
                    scores.append(info.get("confidence", 50))
                else:
                    scores.append(0)
            overall_score = sum(scores) / len(scores)
            update_query["$set"]["qualification.overall_score"] = overall_score
        
        if session:
            collection = await self._ensure_collection()
            return await collection.update_one(
                {"_id": ObjectId(conversation_id)},
                update_query,
                session=session
            )
        else:
            return await self.update_by_id(conversation_id, update_query)
    
    async def add_handoff_async(
        self,
        conversation_id: str,
        from_agent: str,
        to_agent: str,
        reason: str,
        session: Optional[AsyncIOMotorClientSession] = None
    ) -> UpdateResult:
        """Add agent handoff record to conversation.
        
        Args:
            conversation_id: Conversation ID
            from_agent: Agent handing off
            to_agent: Agent receiving
            reason: Reason for handoff
            
        Returns:
            UpdateResult: Result of the update operation
        """
        now = datetime.now(timezone.utc)
        
        handoff = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "reason": reason,
            "timestamp": now.isoformat() + "Z"
        }
        
        update_query = {
            "$push": {"handoffs": handoff},
            "$set": {
                "current_agent": to_agent,
                "agent_context.previous_agent": from_agent,
                "agent_context.handoff_reason": reason,
                "updated_at": now.isoformat() + "Z"
            }
        }
        
        if session:
            collection = await self._ensure_collection()
            return await collection.update_one(
                {"_id": ObjectId(conversation_id)},
                update_query,
                session=session
            )
        else:
            return await self.update_by_id(conversation_id, update_query)
    
    async def close_conversation(
        self,
        conversation_id: str,
        reason: Optional[str] = None
    ) -> UpdateResult:
        """Close a conversation.
        
        Args:
            conversation_id: Conversation ID
            reason: Optional closure reason
            
        Returns:
            UpdateResult: Result of the update operation
        """
        now = datetime.now(timezone.utc)
        
        update_query = {
            "$set": {
                "status": ConversationStatus.CLOSED.value,
                "updated_at": now.isoformat() + "Z"
            }
        }
        
        if reason:
            update_query["$set"]["metadata.closure_reason"] = reason
        
        return await self.update_by_id(conversation_id, update_query)
    
    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation message history.
        
        Args:
            conversation_id: Conversation ID
            limit: Optional limit on number of messages
            
        Returns:
            List[Dict[str, Any]]: List of messages or empty list
        """
        conversation = await self.get_conversation_state(
            conversation_id,
            raise_on_missing=False
        )
        
        if not conversation:
            return []
        
        messages = conversation.get("messages", [])
        
        if limit and limit < len(messages):
            return messages[-limit:]
        
        return messages
    
    async def find_by_sales_stage_async(
        self,
        stage: str,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Find conversations by sales stage.
        
        Args:
            stage: Sales stage
            status: Optional status filter
            limit: Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: List of conversations
        """
        filter = {"sales_stage": stage}
        if status:
            filter["status"] = status
            
        return await self.find_many(
            filter,
            limit=limit,
            sort=[("updated_at", -1)]
        )
    
    async def add_objection(
        self,
        conversation_id: str,
        objection_type: str,
        content: str,
        severity: str = "medium",
        raised_by: str = MessageRole.USER.value
    ) -> str:
        """Add an objection to a conversation.
        
        Args:
            conversation_id: Conversation ID
            objection_type: Type of objection (from ObjectionType enum)
            content: Objection content/description
            severity: Objection severity (high, medium, low)
            raised_by: Who raised the objection
            
        Returns:
            str: Generated objection ID
            
        Raises:
            ValueError: If objection_type is invalid
        """
        # Validate objection type
        valid_types = [obj.value for obj in ObjectionType]
        if objection_type not in valid_types:
            raise ValueError(f"Invalid objection type. Must be one of: {valid_types}")
        
        # Validate severity
        valid_severities = ["high", "medium", "low"]
        if severity not in valid_severities:
            raise ValueError(f"Invalid severity. Must be one of: {valid_severities}")
        
        now = datetime.now(timezone.utc)
        objection_id = str(uuid.uuid4())
        
        objection = {
            "objection_id": objection_id,
            "type": objection_type,
            "content": content,
            "raised_at": now.isoformat() + "Z",
            "severity": severity,
            "status": "active",
            "raised_by": raised_by,
            "handling_attempts": 0,
            "resolution": {
                "resolved": False,
                "method": None,
                "resolved_at": None,
                "resolution_notes": None,
                "confidence": 0.0
            }
        }
        
        update_query = {
            "$push": {"objections": objection},
            "$set": {"updated_at": now.isoformat() + "Z"},
            "$inc": {"agent_metrics.total_objections": 1}
        }
        
        result = await self.update_by_id(conversation_id, update_query)
        
        if result.modified_count == 0:
            raise ValueError(f"Failed to add objection to conversation {conversation_id}")
        
        logger.info(f"Added objection {objection_id} to conversation {conversation_id}")
        return objection_id
    
    async def mark_objection_handled(
        self,
        conversation_id: str,
        objection_id: str,
        resolution_method: str,
        resolution_notes: str,
        handled_by: str,
        confidence: float = 0.8,
        session: Optional[AsyncIOMotorClientSession] = None
    ) -> UpdateResult:
        """Mark an objection as handled/resolved.
        
        Args:
            conversation_id: Conversation ID
            objection_id: Objection ID to mark as handled
            resolution_method: How the objection was resolved
            resolution_notes: Detailed notes about the resolution
            handled_by: Agent who handled the objection
            confidence: Confidence level in the resolution (0-1)
            
        Returns:
            UpdateResult: Result of the update operation
        """
        # Validate confidence
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        now = datetime.now(timezone.utc)
        
        # Update the specific objection
        update_query = {
            "$set": {
                "objections.$[obj].status": "resolved",
                "objections.$[obj].handled_by": handled_by,
                "objections.$[obj].resolution.resolved": True,
                "objections.$[obj].resolution.method": resolution_method,
                "objections.$[obj].resolution.resolved_at": now.isoformat() + "Z",
                "objections.$[obj].resolution.resolution_notes": resolution_notes,
                "objections.$[obj].resolution.confidence": confidence,
                "updated_at": now.isoformat() + "Z"
            },
            "$inc": {
                "objections.$[obj].handling_attempts": 1,
                f"agent_metrics.{handled_by}.objections_resolved": 1
            }
        }
        
        # Array filter to target specific objection
        array_filters = [{"obj.objection_id": objection_id}]
        
        # Execute update with array filters
        collection = await self._ensure_collection()
        if session:
            result = await collection.update_one(
                {"_id": ObjectId(conversation_id)},
                update_query,
                array_filters=array_filters,
                session=session
            )
        else:
            result = await collection.update_one(
                {"_id": ObjectId(conversation_id)},
                update_query,
                array_filters=array_filters
            )
        
        if result.modified_count > 0:
            logger.info(f"Marked objection {objection_id} as resolved in conversation {conversation_id}")
        else:
            logger.warning(f"No objection found with ID {objection_id} in conversation {conversation_id}")
        
        return result
    
    async def defer_objection(
        self,
        conversation_id: str,
        objection_id: str,
        reason: str,
        follow_up_date: Optional[datetime] = None,
        deferred_by: str = AgentName.SUPERVISOR.value
    ) -> UpdateResult:
        """Defer an objection for later handling.
        
        Args:
            conversation_id: Conversation ID
            objection_id: Objection ID to defer
            reason: Reason for deferral
            follow_up_date: Optional date to follow up
            deferred_by: Agent deferring the objection
            
        Returns:
            UpdateResult: Result of the update operation
        """
        now = datetime.now(timezone.utc)
        
        # Update objection status to deferred
        update_query = {
            "$set": {
                "objections.$[obj].status": "deferred",
                "objections.$[obj].deferred_reason": reason,
                "objections.$[obj].deferred_by": deferred_by,
                "objections.$[obj].deferred_at": now.isoformat() + "Z",
                "updated_at": now.isoformat() + "Z"
            }
        }
        
        # If follow-up date provided, update follow-up settings
        if follow_up_date:
            update_query["$set"]["follow_up.required"] = True
            update_query["$set"]["follow_up.scheduled_date"] = follow_up_date.isoformat() + "Z"
            update_query["$set"]["follow_up.type"] = "objection_follow_up"
            update_query["$set"]["follow_up.context.objection_id"] = objection_id
        
        # Array filter to target specific objection
        array_filters = [{"obj.objection_id": objection_id}]
        
        # Execute update
        collection = await self._ensure_collection()
        result = await collection.update_one(
            {"_id": ObjectId(conversation_id)},
            update_query,
            array_filters=array_filters
        )
        
        if result.modified_count > 0:
            logger.info(f"Deferred objection {objection_id} in conversation {conversation_id}")
        
        return result
    
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
        conversation = await self.get_conversation_state(
            conversation_id,
            raise_on_missing=False
        )
        
        if not conversation:
            return []
        
        # Filter for active objections
        objections = conversation.get("objections", [])
        return [obj for obj in objections if obj.get("status") == "active"]
    
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
        # Build match stage
        match_stage = {}
        if user_id:
            match_stage["user_id"] = user_id
        if date_from or date_to:
            date_filter = {}
            if date_from:
                date_filter["$gte"] = date_from.isoformat() + "Z"
            if date_to:
                date_filter["$lte"] = date_to.isoformat() + "Z"
            match_stage["created_at"] = date_filter
        
        pipeline = [
            {"$match": match_stage} if match_stage else {"$match": {}},
            {"$unwind": "$objections"},
            {
                "$group": {
                    "_id": {
                        "type": "$objections.type",
                        "status": "$objections.status"
                    },
                    "count": {"$sum": 1},
                    "avg_confidence": {
                        "$avg": "$objections.resolution.confidence"
                    }
                }
            },
            {
                "$group": {
                    "_id": "$_id.type",
                    "total": {"$sum": "$count"},
                    "by_status": {
                        "$push": {
                            "status": "$_id.status",
                            "count": "$count",
                            "avg_confidence": "$avg_confidence"
                        }
                    }
                }
            },
            {"$sort": {"total": -1}}
        ]
        
        results = await self.aggregate(pipeline)
        
        # Calculate summary statistics
        total_objections = sum(r["total"] for r in results)
        resolved_count = sum(
            s["count"] for r in results 
            for s in r["by_status"] 
            if s["status"] == "resolved"
        )
        
        return {
            "total_objections": total_objections,
            "resolved_count": resolved_count,
            "resolution_rate": resolved_count / total_objections if total_objections > 0 else 0,
            "by_type": results
        }
    
    async def get_next_valid_stages(
        self,
        conversation_id: str
    ) -> List[str]:
        """Get valid next stages for a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List[str]: List of valid next stages
        """
        conversation = await self.get_conversation_state(
            conversation_id,
            raise_on_missing=False
        )
        
        if not conversation:
            return []
        
        current_stage = conversation.get("sales_stage")
        if not current_stage:
            return [SalesStage.LEAD.value]  # Default starting stage
        
        return SalesStageTransitionValidator.get_next_stages(current_stage)
    
    async def check_qualification_complete(
        self,
        conversation_id: str
    ) -> Tuple[bool, Dict[str, bool]]:
        """Check if BANT qualification is complete.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Tuple[bool, Dict[str, bool]]: 
                (is_complete, dict of individual BANT completions)
        """
        conversation = await self.get_conversation_state(
            conversation_id,
            raise_on_missing=False
        )
        
        if not conversation:
            return False, {}
        
        qualification = conversation.get("qualification", {})
        
        # Check each BANT component
        bant_status = {
            "budget": False,
            "authority": False,
            "need": False,
            "timeline": False
        }
        
        # Budget is complete if we have a value and meets_criteria is set
        if qualification.get("budget"):
            budget = qualification["budget"]
            bant_status["budget"] = (
                budget.get("meets_criteria") is not None and
                budget.get("value") is not None
            )
        
        # Authority is complete if we have role and meets_criteria
        if qualification.get("authority"):
            authority = qualification["authority"]
            bant_status["authority"] = (
                authority.get("meets_criteria") is not None and
                authority.get("role") is not None
            )
        
        # Need is complete if we have pain points or use case
        if qualification.get("need"):
            need = qualification["need"]
            bant_status["need"] = (
                need.get("meets_criteria") is not None and
                (bool(need.get("pain_points")) or bool(need.get("use_case")))
            )
        
        # Timeline is complete if we have timeframe
        if qualification.get("timeline"):
            timeline = qualification["timeline"]
            bant_status["timeline"] = (
                timeline.get("meets_criteria") is not None and
                timeline.get("timeframe") is not None
            )
        
        # All must be complete
        is_complete = all(bant_status.values())
        
        return is_complete, bant_status
    
    async def perform_atomic_handoff_with_stage_update(
        self,
        conversation_id: str,
        from_agent: str,
        to_agent: str,
        new_stage: str,
        handoff_reason: str,
        stage_notes: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Perform atomic handoff with stage update in a transaction.
        
        This ensures both handoff and stage transition happen together or not at all.
        
        Args:
            conversation_id: Conversation ID
            from_agent: Current agent
            to_agent: Target agent
            new_stage: New sales stage
            handoff_reason: Reason for handoff
            stage_notes: Notes about stage change
            context: Additional context
            
        Returns:
            bool: True if successful, False otherwise
        """
        async with await self.db.client.start_session() as session:
            try:
                async with session.start_transaction():
                    # Validate stage transition first
                    is_valid, current_stage, error_msg = await self.validate_stage_transition(
                        conversation_id, new_stage, context
                    )
                    if not is_valid:
                        raise ValueError(f"Invalid stage transition: {error_msg}")
                    
                    # Perform handoff
                    await self.add_handoff_async(
                        conversation_id,
                        from_agent,
                        to_agent,
                        handoff_reason,
                        session=session
                    )
                    
                    # Update stage
                    await self.update_sales_stage_async(
                        conversation_id,
                        new_stage,
                        notes=stage_notes or f"Stage updated during handoff from {from_agent} to {to_agent}",
                        context=context,
                        validate=False,  # Already validated
                        triggered_by=to_agent,
                        session=session
                    )
                    
                    # Commit transaction
                    await session.commit_transaction()
                    return True
                    
            except Exception as e:
                logger.error(f"Atomic handoff failed: {e}")
                await session.abort_transaction()
                return False
    
    async def perform_atomic_qualification_and_stage_update(
        self,
        conversation_id: str,
        qualification_data: Dict[str, Any],
        move_to_qualified: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """Atomically update qualification and potentially move to qualified stage.
        
        Args:
            conversation_id: Conversation ID
            qualification_data: Complete BANT qualification data
            move_to_qualified: Whether to auto-move to qualified stage if complete
            
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        async with await self.db.client.start_session() as session:
            try:
                async with session.start_transaction():
                    # Update qualification data
                    await self.update_qualification(
                        conversation_id,
                        budget_info=qualification_data.get("budget"),
                        authority_info=qualification_data.get("authority"),
                        need_info=qualification_data.get("need"),
                        timeline_info=qualification_data.get("timeline"),
                        session=session
                    )
                    
                    # Check if qualification is complete
                    is_complete, bant_status = await self.check_qualification_complete(
                        conversation_id
                    )
                    
                    # Move to qualified stage if requested and complete
                    if move_to_qualified and is_complete:
                        await self.update_sales_stage_async(
                            conversation_id,
                            SalesStage.QUALIFIED.value,
                            notes="BANT qualification completed",
                            context={"qualification_complete": True, "bant_status": bant_status},
                            triggered_by="qualifier",
                            session=session
                        )
                    
                    await session.commit_transaction()
                    return True, None
                    
            except Exception as e:
                logger.error(f"Atomic qualification update failed: {e}")
                await session.abort_transaction()
                return False, str(e)
    
    async def perform_atomic_close_deal(
        self,
        conversation_id: str,
        deal_details: Dict[str, Any],
        close_type: str = "won"
    ) -> Tuple[bool, Optional[str]]:
        """Atomically close a deal with all required updates.
        
        Args:
            conversation_id: Conversation ID
            deal_details: Deal information (value, payment method, etc.)
            close_type: "won" or "lost"
            
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        async with await self.db.client.start_session() as session:
            try:
                async with session.start_transaction():
                    now = datetime.now(timezone.utc)
                    
                    # Determine stage and context
                    if close_type == "won":
                        new_stage = SalesStage.CLOSED_WON.value
                        context = {
                            "deal_value": deal_details.get("monthly_value"),
                            "payment_method": deal_details.get("payment_method")
                        }
                    else:
                        new_stage = SalesStage.CLOSED_LOST.value
                        context = {"loss_reason": deal_details.get("close_reason", "Not specified")}
                    
                    # Update stage
                    await self.update_sales_stage_async(
                        conversation_id,
                        new_stage,
                        notes=deal_details.get("close_reason", f"Deal closed - {close_type}"),
                        context=context,
                        triggered_by="closer",
                        session=session
                    )
                    
                    # Update deal details
                    deal_update = {"$set": {}}
                    for key, value in deal_details.items():
                        deal_update["$set"][f"deal_details.{key}"] = value
                    
                    if close_type == "won":
                        deal_update["$set"]["deal_details.close_date"] = now.isoformat() + "Z"
                    
                    # Execute deal update
                    collection = await self._ensure_collection()
                    await collection.update_one(
                        {"_id": ObjectId(conversation_id)},
                        deal_update,
                        session=session
                    )
                    
                    await session.commit_transaction()
                    return True, None
                    
            except Exception as e:
                logger.error(f"Atomic deal close failed: {e}")
                await session.abort_transaction()
                return False, str(e)
    
    async def perform_atomic_objection_resolution(
        self,
        conversation_id: str,
        objection_id: str,
        resolution_data: Dict[str, Any],
        move_to_next_stage: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """Atomically resolve objection and optionally move to next stage.
        
        Args:
            conversation_id: Conversation ID
            objection_id: Objection to resolve
            resolution_data: Resolution details
            move_to_next_stage: Optional next stage to move to
            
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        async with await self.db.client.start_session() as session:
            try:
                async with session.start_transaction():
                    # Mark objection as handled
                    await self.mark_objection_handled(
                        conversation_id,
                        objection_id,
                        resolution_method=resolution_data.get("method", "addressed"),
                        resolution_notes=resolution_data.get("notes", ""),
                        handled_by=resolution_data.get("handled_by", "objection_handler"),
                        confidence=resolution_data.get("confidence", 0.8),
                        session=session
                    )
                    
                    # Check if all objections are resolved
                    active_objections = await self.get_active_objections(conversation_id)
                    
                    # Move to next stage if requested and no active objections
                    if move_to_next_stage and len(active_objections) == 0:
                        await self.update_sales_stage_async(
                            conversation_id,
                            move_to_next_stage,
                            notes="All objections resolved, moving forward",
                            context={"objections_resolved": True},
                            triggered_by="objection_handler",
                            session=session
                        )
                    
                    await session.commit_transaction()
                    return True, None
                    
            except Exception as e:
                logger.error(f"Atomic objection resolution failed: {e}")
                await session.abort_transaction()
                return False, str(e)
    
    async def search_conversations_by_content(
        self,
        search_query: str,
        user_id: Optional[str] = None,
        limit: int = 20,
        include_score: bool = True
    ) -> List[Dict[str, Any]]:
        """Search conversations by message content using text search.
        
        Args:
            search_query: Text to search for in messages
            user_id: Optional user ID to filter results
            limit: Maximum number of results to return
            include_score: Whether to include text search score
            
        Returns:
            List[Dict[str, Any]]: Matching conversations with search scores
        """
        try:
            # Build the search pipeline
            pipeline = []
            
            # Text search stage
            search_stage = {
                "$match": {
                    "$text": {"$search": search_query}
                }
            }
            
            # Add user filter if provided
            if user_id:
                search_stage["$match"]["user_id"] = user_id
                
            pipeline.append(search_stage)
            
            # Add text score if requested
            if include_score:
                pipeline.append({
                    "$addFields": {
                        "search_score": {"$meta": "textScore"}
                    }
                })
                # Sort by relevance
                pipeline.append({
                    "$sort": {"search_score": -1}
                })
            
            # Limit results
            pipeline.append({"$limit": limit})
            
            # Project relevant fields
            pipeline.append({
                "$project": {
                    "_id": 1,
                    "user_id": 1,
                    "channel": 1,
                    "status": 1,
                    "sales_stage": 1,
                    "created_at": 1,
                    "updated_at": 1,
                    "search_score": 1,
                    "messages": {
                        "$filter": {
                            "input": "$messages",
                            "as": "msg",
                            "cond": {
                                "$regexMatch": {
                                    "input": "$$msg.content",
                                    "regex": search_query,
                                    "options": "i"
                                }
                            }
                        }
                    }
                }
            })
            
            # Execute search
            results = []
            async for doc in self.collection.aggregate(pipeline):
                results.append(doc)
                
            logger.info(f"Found {len(results)} conversations matching '{search_query}'")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search conversations: {e}")
            return []
    
    async def get_conversation_messages_containing(
        self,
        conversation_id: str,
        search_text: str,
        case_sensitive: bool = False
    ) -> List[Dict[str, Any]]:
        """Get messages from a conversation that contain specific text.
        
        Args:
            conversation_id: The conversation ID
            search_text: Text to search for
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List[Dict[str, Any]]: Messages containing the search text
        """
        try:
            # Build regex options
            options = "" if case_sensitive else "i"
            
            # Use aggregation to filter messages
            pipeline = [
                {"$match": {"_id": ObjectId(conversation_id)}},
                {"$unwind": "$messages"},
                {
                    "$match": {
                        "messages.content": {
                            "$regex": search_text,
                            "$options": options
                        }
                    }
                },
                {
                    "$project": {
                        "role": "$messages.role",
                        "content": "$messages.content",
                        "timestamp": "$messages.timestamp",
                        "_id": 0
                    }
                }
            ]
            
            messages = []
            async for msg in self.collection.aggregate(pipeline):
                messages.append(msg)
                
            return messages
            
        except Exception as e:
            logger.error(f"Failed to search messages in conversation: {e}")
            return []