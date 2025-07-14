"""
Multi-Agent Sales Graph Orchestrator

This module implements a LangGraph-based multi-agent system for the Sales AI Closer,
with a supervisor that routes conversations to specialized sales agents.

Based on the supervisor pattern demonstrated in the student homework helper example.
"""

import logging
from typing import Annotated, TypedDict, Literal, List, Dict, Any, Optional
from datetime import datetime, timezone
from enum import Enum

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.db.mongodb.schemas.conversation_schema import (
    SalesStage,
    MessageRole,
    AgentName
)
from app.db.mongodb.validators import SalesStageTransitionValidator
from app.models.multi_agent_models import (
    WorkflowState,
    AgentResponse,
    RoutingDecision,
    CurrentTask,
    TaskStatus
)

logger = logging.getLogger(__name__)


class NextAgent(str, Enum):
    """Enum for routing decisions"""
    SUPERVISOR = "supervisor"
    QUALIFIER = "qualifier"
    OBJECTION_HANDLER = "objection_handler"
    CLOSER = "closer"
    END = "end"


class SalesAgentState(TypedDict):
    """Enhanced state that flows through the sales agent pipeline
    
    This extends the basic AgentState with workflow tracking similar to
    the student homework helper example.
    """
    # Core conversation state
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    session_id: str
    
    # MongoDB conversation tracking
    conversation_id: Optional[str]  # MongoDB ObjectId as string
    
    # Workflow tracking (similar to student example)
    workflow: Dict[str, Any]
    # Example workflow structure:
    # {
    #     "_id": "workflow_123",
    #     "created_at": "2024-01-11T10:00:00Z",
    #     "status": "in_progress",  # in_progress, completed, failed
    #     "current_stage": "qualification",  # Maps to SalesStage enum
    #     "current_task": {
    #         "type": "qualification",  # qualification, objection_handling, closing
    #         "assigned_agent": "qualifier",
    #         "status": "processing",  # pending, processing, completed
    #         "context": {},  # Task-specific context
    #         "iterations": 1,
    #         "max_iterations": 5
    #     },
    #     "agents_visited": ["supervisor", "qualifier"],
    #     "agent_responses": {
    #         "qualifier": {
    #             "response": "Let me understand your needs...",
    #             "tools_used": [],
    #             "timestamp": "2024-01-11T10:00:05Z",
    #             "metadata": {}
    #         }
    #     },
    #     "routing_history": [
    #         {
    #             "from": "supervisor",
    #             "to": "qualifier",
    #             "reason": "new lead requires qualification",
    #             "timestamp": "...",
    #             "stage_before": "lead",
    #             "stage_after": "qualification"
    #         }
    #     ],
    #     "qualification_data": {
    #         "budget": {"value": null, "confidence": 0},
    #         "authority": {"value": null, "confidence": 0},
    #         "need": {"value": null, "confidence": 0},
    #         "timeline": {"value": null, "confidence": 0}
    #     },
    #     "objections": [],
    #     "metadata": {}
    # }
    
    # Routing decision
    next_agent: NextAgent
    
    # Current sales stage
    current_stage: SalesStage
    
    # Additional metadata
    metadata: Dict[str, Any]
    
    # Service references (optional, for enhanced agents)
    conversation_service: Optional[Any]  # ConversationService
    mem0_client: Optional[Any]  # Mem0AsyncClientWrapper


def create_supervisor_node(stage_validator: Optional[SalesStageTransitionValidator] = None):
    """Create the supervisor agent node that routes to appropriate sales agents
    
    Args:
        stage_validator: Optional validator for stage transitions
        
    Returns:
        Supervisor node function
    """
    logger.info("🤖 Creating Supervisor Node")
    
    # Use default validator if none provided
    if stage_validator is None:
        stage_validator = SalesStageTransitionValidator()
    
    def supervisor_node(state: SalesAgentState) -> Dict[str, Any]:
        """Supervisor agent that analyzes conversation and routes to specialists"""
        logger.info("🎯 SUPERVISOR AGENT ACTIVATED")
        
        messages = state["messages"]
        workflow = state.get("workflow", {})
        current_stage = state.get("current_stage", SalesStage.LEAD)
        
        # Initialize workflow if needed
        if not workflow:
            workflow = {
                "_id": f"workflow_{datetime.now(timezone.utc).timestamp()}",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "in_progress",
                "current_stage": current_stage.value,
                "current_task": {},
                "agents_visited": [],
                "agent_responses": {},
                "routing_history": [],
                "qualification_data": {
                    "budget": {"value": None, "confidence": 0},
                    "authority": {"value": None, "confidence": 0},
                    "need": {"value": None, "confidence": 0},
                    "timeline": {"value": None, "confidence": 0}
                },
                "objections": [],
                "metadata": {}
            }
        
        # Add supervisor to visited agents
        if "supervisor" not in workflow["agents_visited"]:
            workflow["agents_visited"].append("supervisor")
        
        # Get the latest user message
        latest_message = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                latest_message = msg.content
                break
        
        # Check current task status
        current_task = workflow.get("current_task", {})
        
        # If task is completed, check if we should end or continue
        if current_task.get("status") == "completed":
            logger.info("🎯 SUPERVISOR: Current task completed, evaluating next steps")
            
            # Check if sale is closed
            if current_stage in [SalesStage.CLOSED_WON, SalesStage.CLOSED_LOST]:
                logger.info(f"🎯 SUPERVISOR: Sale closed ({current_stage.value}), ending workflow")
                workflow["status"] = "completed"
                return {
                    "messages": [],
                    "workflow": workflow,
                    "next_agent": NextAgent.END,
                    "current_stage": current_stage
                }
        
        # Check iteration count to prevent infinite loops
        iterations = current_task.get("iterations", 0)
        max_iterations = current_task.get("max_iterations", 5)
        
        if iterations >= max_iterations:
            logger.warning(f"🎯 SUPERVISOR: Max iterations ({max_iterations}) reached")
            workflow["status"] = "failed"
            workflow["current_task"]["status"] = "failed"
            workflow["current_task"]["failure_reason"] = "Max iterations exceeded"
            return {
                "messages": [AIMessage(content="I apologize for the confusion. Let me connect you with a human representative.")],
                "workflow": workflow,
                "next_agent": NextAgent.END,
                "current_stage": current_stage
            }
        
        logger.info(f"🎯 SUPERVISOR analyzing: '{latest_message}' (Stage: {current_stage.value})")
        
        # Update current task
        workflow["current_task"] = {
            "type": "unknown",
            "status": "pending",
            "iterations": iterations + 1,
            "max_iterations": max_iterations,
            "context": {
                "user_message": latest_message,
                "current_stage": current_stage.value
            }
        }
        
        # Determine routing based on sales stage and conversation context
        next_agent = NextAgent.END
        routing_reason = ""
        
        # Stage-based routing logic
        if current_stage == SalesStage.LEAD:
            # New leads go to qualifier
            next_agent = NextAgent.QUALIFIER
            routing_reason = "New lead requires qualification"
            workflow["current_task"]["type"] = "qualification"
            
        elif current_stage == SalesStage.QUALIFICATION:
            # Check if qualification is complete
            qual_data = workflow.get("qualification_data", {})
            total_confidence = sum(
                qual_data.get(field, {}).get("confidence", 0)
                for field in ["budget", "authority", "need", "timeline"]
            )
            
            if total_confidence < 2.0:  # Not fully qualified
                next_agent = NextAgent.QUALIFIER
                routing_reason = "Continue qualification process"
                workflow["current_task"]["type"] = "qualification"
            else:
                # Move to next stage - check for objections
                next_agent = NextAgent.OBJECTION_HANDLER
                routing_reason = "Qualification complete, handling potential objections"
                workflow["current_task"]["type"] = "objection_handling"
                
        elif current_stage == SalesStage.OBJECTION_HANDLING:
            # Check if there are unresolved objections
            unresolved = [
                obj for obj in workflow.get("objections", [])
                if not obj.get("resolved", False)
            ]
            
            if unresolved:
                next_agent = NextAgent.OBJECTION_HANDLER
                routing_reason = f"Handling {len(unresolved)} unresolved objections"
                workflow["current_task"]["type"] = "objection_handling"
            else:
                # Move to closing
                next_agent = NextAgent.CLOSER
                routing_reason = "Objections resolved, moving to close"
                workflow["current_task"]["type"] = "closing"
                
        elif current_stage == SalesStage.CLOSING:
            next_agent = NextAgent.CLOSER
            routing_reason = "Finalizing the sale"
            workflow["current_task"]["type"] = "closing"
            
        else:
            # Default routing
            logger.warning(f"🎯 SUPERVISOR: Unexpected stage {current_stage.value}")
            next_agent = NextAgent.END
            routing_reason = f"Unknown stage: {current_stage.value}"
        
        # Log routing decision
        logger.info("=" * 60)
        logger.info(f"🎯 SUPERVISOR ROUTING DECISION: {current_stage.value} → {next_agent.value}")
        logger.info(f"🎯 REASON: {routing_reason}")
        logger.info("=" * 60)
        
        # Update workflow with routing decision
        if next_agent != NextAgent.END:
            workflow["current_task"]["assigned_agent"] = next_agent.value
            workflow["current_task"]["status"] = "processing"
        
        # Add routing history entry
        routing_entry = {
            "from": "supervisor",
            "to": next_agent.value,
            "reason": routing_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage_before": current_stage.value,
            "stage_after": current_stage.value  # Stage transition happens after agent completes
        }
        workflow["routing_history"].append(routing_entry)
        
        # Create supervisor message
        supervisor_msg = AIMessage(
            content=f"[Supervisor: Routing to {next_agent.value} - {routing_reason}]",
            additional_kwargs={"agent": "supervisor", "internal": True}
        )
        
        return {
            "messages": [supervisor_msg],
            "workflow": workflow,
            "next_agent": next_agent,
            "current_stage": current_stage
        }
    
    return supervisor_node


def create_qualifier_agent():
    """Create the Qualifier Agent for BANT qualification"""
    logger.info("🤖 Creating Qualifier Agent (placeholder)")
    
    def qualifier_node(state: SalesAgentState) -> Dict[str, Any]:
        """Qualifier agent placeholder"""
        logger.info("🔍 QUALIFIER AGENT ACTIVATED (placeholder)")
        
        messages = state["messages"]
        workflow = state.get("workflow", {})
        current_stage = state.get("current_stage", SalesStage.QUALIFICATION)
        
        # Add agent to visited list
        if "qualifier" not in workflow.get("agents_visited", []):
            workflow["agents_visited"].append("qualifier")
        
        # Placeholder response
        response = AIMessage(
            content="I'd love to understand more about your needs. Can you tell me about your current challenges?",
            additional_kwargs={"agent": "qualifier"}
        )
        
        # Update workflow
        workflow["agent_responses"]["qualifier"] = {
            "response": response.content,
            "tools_used": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {}
        }
        
        # For first interaction, don't mark as complete yet
        # This prevents infinite loops in testing
        if len(messages) <= 2:  # Initial message + supervisor routing
            workflow["current_task"]["status"] = "pending"
        else:
            # Mark task as complete after response (placeholder)
            workflow["current_task"]["status"] = "completed"
        
        return {
            "messages": [response],
            "workflow": workflow,
            "current_stage": current_stage
        }
    
    return qualifier_node


def create_objection_handler_agent():
    """Create the Objection Handler Agent"""
    logger.info("🤖 Creating Objection Handler Agent (placeholder)")
    
    def objection_handler_node(state: SalesAgentState) -> Dict[str, Any]:
        """Objection handler agent placeholder"""
        logger.info("🛡️ OBJECTION HANDLER AGENT ACTIVATED (placeholder)")
        
        messages = state["messages"]
        workflow = state.get("workflow", {})
        current_stage = state.get("current_stage", SalesStage.OBJECTION_HANDLING)
        
        # Add agent to visited list
        if "objection_handler" not in workflow.get("agents_visited", []):
            workflow["agents_visited"].append("objection_handler")
        
        # Placeholder response
        response = AIMessage(
            content="I understand your concerns. Let me address those for you...",
            additional_kwargs={"agent": "objection_handler"}
        )
        
        # Update workflow
        workflow["agent_responses"]["objection_handler"] = {
            "response": response.content,
            "tools_used": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {}
        }
        
        # Mark task as complete for now (placeholder)
        workflow["current_task"]["status"] = "completed"
        
        return {
            "messages": [response],
            "workflow": workflow,
            "current_stage": current_stage
        }
    
    return objection_handler_node


def create_closer_agent():
    """Create the Closer Agent"""
    logger.info("🤖 Creating Closer Agent (placeholder)")
    
    def closer_node(state: SalesAgentState) -> Dict[str, Any]:
        """Closer agent placeholder"""
        logger.info("💰 CLOSER AGENT ACTIVATED (placeholder)")
        
        messages = state["messages"]
        workflow = state.get("workflow", {})
        current_stage = state.get("current_stage", SalesStage.CLOSING)
        
        # Add agent to visited list
        if "closer" not in workflow.get("agents_visited", []):
            workflow["agents_visited"].append("closer")
        
        # Placeholder response
        response = AIMessage(
            content="Great! Let's move forward with getting you started...",
            additional_kwargs={"agent": "closer"}
        )
        
        # Update workflow
        workflow["agent_responses"]["closer"] = {
            "response": response.content,
            "tools_used": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {}
        }
        
        # Mark task as complete and move to closed_won (placeholder)
        workflow["current_task"]["status"] = "completed"
        
        return {
            "messages": [response],
            "workflow": workflow,
            "current_stage": SalesStage.CLOSED_WON  # Placeholder success
        }
    
    return closer_node


def build_sales_agent_graph(
    stage_validator: Optional[SalesStageTransitionValidator] = None,
    with_services: bool = False
):
    """Build the multi-agent sales graph
    
    Args:
        stage_validator: Optional validator for stage transitions
        with_services: Whether to enable service integration
        
    Returns:
        Compiled LangGraph workflow
    """
    logger.info("🏗️ Building Multi-Agent Sales Graph")
    if with_services:
        logger.info("📦 Service integration enabled")
    
    # Create the graph
    workflow = StateGraph(SalesAgentState)
    
    # Create all agent nodes
    supervisor = create_supervisor_node(stage_validator)
    qualifier = create_qualifier_agent()
    objection_handler = create_objection_handler_agent()
    closer = create_closer_agent()
    
    # Add all nodes to the graph
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("qualifier", qualifier)
    workflow.add_node("objection_handler", objection_handler)
    workflow.add_node("closer", closer)
    
    # Define routing function
    def route_from_supervisor(state: SalesAgentState) -> str:
        """Route based on supervisor's decision"""
        next_agent = state.get("next_agent", NextAgent.END)
        logger.info(f"🔀 Routing from supervisor to: {next_agent}")
        
        if next_agent == NextAgent.END:
            return END
        return next_agent.value
    
    def route_to_supervisor(state: SalesAgentState) -> str:
        """All agents route back to supervisor for next decision"""
        return "supervisor"
    
    # Add edges
    # Start with supervisor
    workflow.add_edge(START, "supervisor")
    
    # Supervisor routes to agents or END
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            NextAgent.QUALIFIER.value: "qualifier",
            NextAgent.OBJECTION_HANDLER.value: "objection_handler",
            NextAgent.CLOSER.value: "closer",
            END: END
        }
    )
    
    # All agents route back to supervisor
    workflow.add_edge("qualifier", "supervisor")
    workflow.add_edge("objection_handler", "supervisor")
    workflow.add_edge("closer", "supervisor")
    
    # Compile and return the graph
    return workflow.compile()


# Example usage and testing
if __name__ == "__main__":
    # This is just for testing the graph structure
    from langchain_core.messages import HumanMessage
    
    # Build the graph
    sales_graph = build_sales_agent_graph()
    
    # Test state
    test_state = {
        "messages": [HumanMessage(content="I'm interested in your product")],
        "user_id": "test_user",
        "session_id": "test_session",
        "conversation_id": None,
        "workflow": {},
        "next_agent": NextAgent.SUPERVISOR,
        "current_stage": SalesStage.LEAD,
        "metadata": {}
    }
    
    # Run a test with limited recursion
    print("Testing multi-agent graph structure...")
    try:
        # Test with just 3 steps to see the routing
        result = sales_graph.invoke(test_state, {"recursion_limit": 3})
        print(f"✅ Graph executed with recursion limit")
        print(f"   Agents visited: {result['workflow'].get('agents_visited', [])}")
        print(f"   Routing history:")
        for route in result['workflow'].get('routing_history', []):
            print(f"     - {route['from']} → {route['to']} ({route['reason']})")
        print(f"   Final stage: {result.get('current_stage', 'unknown')}")
        print(f"   Messages exchanged: {len(result['messages'])}")
    except Exception as e:
        if "recursion_limit" in str(e).lower():
            print(f"⚠️  Hit recursion limit as expected (testing with limited steps)")
            print(f"   This is normal for the placeholder implementation")
        else:
            print(f"❌ Unexpected error: {e}")