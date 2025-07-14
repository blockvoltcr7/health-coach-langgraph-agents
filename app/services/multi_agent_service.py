"""Multi-Agent Service for orchestrating the sales agent graph.

This service integrates the multi-agent LangGraph with MongoDB and Mem0,
providing seamless state synchronization and conversation persistence.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import json

from app.core.multi_agent_graph import (
    build_sales_agent_graph,
    SalesAgentState,
    NextAgent,
    SalesStage
)
from app.services.conversation_service import (
    get_conversation_service,
    ConversationService,
    ConversationEvent
)
from app.mem0.mem0AsyncClient import (
    get_mem0_client,
    Mem0AsyncClientWrapper,
    add_conversation_memory
)
from app.db.mongodb.schemas.conversation_schema import (
    MessageRole,
    AgentName,
    ConversationStatus
)
from app.db.mongodb.validators import SalesStageTransitionValidator

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)


class MultiAgentService:
    """Service for managing multi-agent sales conversations.
    
    This service provides:
    - Multi-agent graph orchestration
    - MongoDB conversation persistence
    - Mem0 memory integration
    - Workflow state synchronization
    - Agent handoff tracking
    """
    
    def __init__(
        self,
        conversation_service: Optional[ConversationService] = None,
        mem0_client: Optional[Mem0AsyncClientWrapper] = None,
        stage_validator: Optional[SalesStageTransitionValidator] = None
    ):
        """Initialize the multi-agent service.
        
        Args:
            conversation_service: Optional conversation service instance
            mem0_client: Optional Mem0 client instance
            stage_validator: Optional stage transition validator
        """
        self._conversation_service = conversation_service
        self._mem0_client = mem0_client
        self._stage_validator = stage_validator or SalesStageTransitionValidator()
        self._graph = None
        self._initialized = False
    
    async def _ensure_initialized(self) -> None:
        """Ensure all services are initialized."""
        if not self._initialized:
            # Initialize services if not provided
            if not self._conversation_service:
                self._conversation_service = await get_conversation_service()
            
            if not self._mem0_client:
                self._mem0_client = await get_mem0_client()
            
            # Build the multi-agent graph
            self._graph = build_sales_agent_graph(self._stage_validator)
            
            self._initialized = True
            logger.info("Multi-agent service initialized")
    
    async def process_with_multi_agent(
        self,
        message: str,
        user_id: str,
        session_id: Optional[str] = None,
        channel: str = "web",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a message using the multi-agent graph.
        
        Args:
            message: User message
            user_id: User identifier
            session_id: Optional session ID
            channel: Conversation channel
            metadata: Optional metadata
            
        Returns:
            Dict containing response and conversation metadata
        """
        await self._ensure_initialized()
        
        try:
            # Create or resume conversation in MongoDB
            if metadata is None:
                metadata = {}
            if "mem0_user_id" not in metadata:
                metadata["mem0_user_id"] = f"mem0_{user_id}"
            
            conversation, event = await self._conversation_service.create_or_resume_conversation(
                user_id=user_id,
                channel=channel,
                metadata=metadata
            )
            
            conversation_id = str(conversation["_id"])
            logger.info(f"Processing with multi-agent for conversation {conversation_id}")
            
            # Load workflow state from MongoDB
            workflow_state = await self.load_workflow_from_mongodb(conversation_id)
            
            # Get conversation context from Mem0
            context = await self.get_agent_context(
                user_id=metadata["mem0_user_id"],
                agent_name="supervisor"
            )
            
            # Prepare initial state for the graph
            initial_state: SalesAgentState = {
                "messages": [HumanMessage(content=message)],
                "user_id": user_id,
                "session_id": session_id or f"session_{datetime.now(timezone.utc).timestamp()}",
                "conversation_id": conversation_id,
                "workflow": workflow_state or {},
                "next_agent": NextAgent.SUPERVISOR,
                "current_stage": SalesStage(conversation.get("sales_stage", SalesStage.LEAD.value)),
                "metadata": {
                    "mem0_context": context,
                    "mongodb_conversation": conversation
                }
            }
            
            # Process through the multi-agent graph
            result = await self._graph.ainvoke(
                initial_state,
                config={"recursion_limit": 10}
            )
            
            # Extract the agent's response
            agent_response = None
            responding_agent = None
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and "[Supervisor:" not in msg.content:
                    agent_response = msg.content
                    # Check for agent metadata
                    responding_agent = msg.additional_kwargs.get("agent", "assistant")
                    break
            
            if not agent_response:
                agent_response = "I apologize, but I couldn't process your request properly."
                responding_agent = "supervisor"
            
            # Save the conversation turn
            await self._conversation_service.save_conversation_turn(
                conversation_id=conversation_id,
                user_message=message,
                agent_response=agent_response,
                agent_name=responding_agent
            )
            
            # Save to Mem0
            await add_conversation_memory(
                user_message=message,
                assistant_message=agent_response,
                user_id=metadata["mem0_user_id"],
                metadata={
                    "agent": responding_agent,
                    "conversation_id": conversation_id,
                    "sales_stage": result["current_stage"].value
                }
            )
            
            # Sync workflow state to MongoDB
            await self.sync_workflow_to_mongodb(
                conversation_id=conversation_id,
                workflow=result["workflow"],
                current_stage=result["current_stage"]
            )
            
            # Track agent interactions
            if result["workflow"].get("routing_history"):
                latest_routing = result["workflow"]["routing_history"][-1]
                await self.track_agent_interaction(
                    conversation_id=conversation_id,
                    routing_decision=latest_routing,
                    agent_response=agent_response
                )
            
            logger.info(f"Multi-agent processing completed for conversation {conversation_id}")
            
            return {
                "response": agent_response,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "responding_agent": responding_agent,
                "event": event.value,
                "sales_stage": result["current_stage"].value,
                "workflow_status": result["workflow"].get("status", "in_progress"),
                "agents_visited": result["workflow"].get("agents_visited", [])
            }
            
        except Exception as e:
            logger.error(f"Error in multi-agent processing: {e}")
            raise
    
    async def sync_workflow_to_mongodb(
        self,
        conversation_id: str,
        workflow: Dict[str, Any],
        current_stage: SalesStage
    ) -> None:
        """Sync workflow state to MongoDB.
        
        Args:
            conversation_id: Conversation ID
            workflow: Workflow state from LangGraph
            current_stage: Current sales stage
        """
        try:
            # Update conversation metadata with workflow
            metadata_updates = {
                "workflow_id": workflow.get("_id"),
                "workflow_status": workflow.get("status"),
                "last_workflow_update": datetime.now(timezone.utc).isoformat()
            }
            
            await self._conversation_service.update_conversation_metadata(
                conversation_id=conversation_id,
                metadata_updates=metadata_updates
            )
            
            # Update sales stage if changed
            current_db_stage = workflow.get("current_stage")
            if current_db_stage != current_stage.value:
                await self._conversation_service.update_sales_stage(
                    conversation_id=conversation_id,
                    new_stage=current_stage.value,
                    notes=f"Updated by multi-agent workflow"
                )
            
            # Store qualification data if updated
            if workflow.get("qualification_data"):
                qual_data = workflow["qualification_data"]
                # Check if any qualification data has been updated
                if any(qual_data[field]["value"] is not None for field in ["budget", "authority", "need", "timeline"]):
                    # This would need a new method in ConversationService
                    # For now, we'll store in metadata
                    await self._conversation_service.update_conversation_metadata(
                        conversation_id=conversation_id,
                        metadata_updates={"qualification_progress": qual_data}
                    )
            
            logger.info(f"Synced workflow state to MongoDB for conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error syncing workflow to MongoDB: {e}")
            # Don't raise - allow conversation to continue
    
    async def load_workflow_from_mongodb(
        self,
        conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load workflow state from MongoDB.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Workflow state dict or None if not found
        """
        try:
            conversation = await self._conversation_service._repository.get_conversation_state(
                conversation_id,
                raise_on_missing=False
            )
            
            if not conversation:
                return None
            
            # Reconstruct workflow state from MongoDB data
            workflow = {
                "_id": conversation.get("metadata", {}).get("workflow_id"),
                "created_at": conversation.get("created_at"),
                "status": conversation.get("metadata", {}).get("workflow_status", "in_progress"),
                "current_stage": conversation.get("sales_stage", SalesStage.LEAD.value),
                "agents_visited": [],  # Would need to track this separately
                "agent_responses": {},  # Would need to reconstruct from messages
                "routing_history": [],  # Would need to reconstruct from handoffs
                "qualification_data": conversation.get("metadata", {}).get("qualification_progress", {
                    "budget": {"value": None, "confidence": 0},
                    "authority": {"value": None, "confidence": 0},
                    "need": {"value": None, "confidence": 0},
                    "timeline": {"value": None, "confidence": 0}
                }),
                "objections": conversation.get("objections", []),
                "metadata": conversation.get("metadata", {})
            }
            
            # Reconstruct agents visited from handoffs
            for handoff in conversation.get("handoffs", []):
                if handoff["to_agent"] not in workflow["agents_visited"]:
                    workflow["agents_visited"].append(handoff["to_agent"])
            
            logger.info(f"Loaded workflow state from MongoDB for conversation {conversation_id}")
            return workflow
            
        except Exception as e:
            logger.error(f"Error loading workflow from MongoDB: {e}")
            return None
    
    async def track_agent_interaction(
        self,
        conversation_id: str,
        routing_decision: Dict[str, Any],
        agent_response: str
    ) -> None:
        """Track agent interactions and handoffs.
        
        Args:
            conversation_id: Conversation ID
            routing_decision: Routing decision from workflow
            agent_response: Agent's response
        """
        try:
            # Track handoff if agent changed
            if routing_decision["to"] != NextAgent.END.value:
                await self._conversation_service.handle_agent_handoff(
                    conversation_id=conversation_id,
                    from_agent=routing_decision["from"],
                    to_agent=routing_decision["to"],
                    reason=routing_decision["reason"]
                )
            
            logger.info(f"Tracked agent interaction for conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error tracking agent interaction: {e}")
    
    async def get_agent_context(
        self,
        user_id: str,
        agent_name: str,
        limit: int = 10
    ) -> str:
        """Get relevant memory context for a specific agent.
        
        Args:
            user_id: User ID (mem0 format)
            agent_name: Name of the agent requesting context
            limit: Maximum memories to retrieve
            
        Returns:
            Formatted context string
        """
        try:
            # Get all memories for the user
            memories = await self._mem0_client.get_memory_history(user_id, limit)
            
            if not memories:
                return "No previous conversation history found."
            
            # Filter memories relevant to the agent
            # For now, return all memories, but could filter by agent in metadata
            context_lines = [f"Previous conversation context for {agent_name}:"]
            
            for i, memory in enumerate(memories, 1):
                # Check if this memory was created by a specific agent
                memory_agent = memory.metadata.get("agent", "unknown")
                context_lines.append(f"{i}. [{memory_agent}] {memory.memory}")
            
            return "\n".join(context_lines)
            
        except Exception as e:
            logger.error(f"Error getting agent context: {e}")
            return "Unable to retrieve conversation context."
    
    async def get_workflow_summary(
        self,
        conversation_id: str
    ) -> Dict[str, Any]:
        """Get a summary of the workflow state.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Workflow summary dict
        """
        try:
            workflow = await self.load_workflow_from_mongodb(conversation_id)
            
            if not workflow:
                return {"error": "No workflow found"}
            
            # Get conversation summary
            conv_summary = await self._conversation_service.get_conversation_summary(
                conversation_id
            )
            
            return {
                "conversation_id": conversation_id,
                "workflow_id": workflow.get("_id"),
                "status": workflow.get("status"),
                "current_stage": workflow.get("current_stage"),
                "agents_visited": workflow.get("agents_visited", []),
                "routing_count": len(workflow.get("routing_history", [])),
                "qualification_progress": workflow.get("qualification_data", {}),
                "objection_count": len(workflow.get("objections", [])),
                "message_count": conv_summary.get("message_count", 0),
                "created_at": workflow.get("created_at"),
                "last_updated": conv_summary.get("updated_at")
            }
            
        except Exception as e:
            logger.error(f"Error getting workflow summary: {e}")
            return {"error": str(e)}
    
    async def handle_stage_transition(
        self,
        conversation_id: str,
        from_stage: SalesStage,
        to_stage: SalesStage,
        reason: str
    ) -> bool:
        """Handle sales stage transitions with validation.
        
        Args:
            conversation_id: Conversation ID
            from_stage: Current stage
            to_stage: Target stage
            reason: Transition reason
            
        Returns:
            True if transition successful
        """
        try:
            # Validate transition
            is_valid, validation_msg = self._stage_validator.validate_transition(
                from_stage.value,
                to_stage.value
            )
            
            if not is_valid:
                logger.warning(f"Invalid stage transition: {validation_msg}")
                return False
            
            # Update stage in MongoDB
            await self._conversation_service.update_sales_stage(
                conversation_id=conversation_id,
                new_stage=to_stage.value,
                notes=f"Multi-agent transition: {reason}"
            )
            
            logger.info(f"Stage transition from {from_stage.value} to {to_stage.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error handling stage transition: {e}")
            return False


# Global service instance
_multi_agent_service: Optional[MultiAgentService] = None


async def get_multi_agent_service() -> MultiAgentService:
    """Get the global multi-agent service instance.
    
    Returns:
        MultiAgentService: The service instance
    """
    global _multi_agent_service
    if _multi_agent_service is None:
        _multi_agent_service = MultiAgentService()
    await _multi_agent_service._ensure_initialized()
    return _multi_agent_service