"""Tests for multi-agent workflow models."""

import pytest
from datetime import datetime

from app.models.multi_agent_models import (
    WorkflowState,
    WorkflowStatus,
    TaskStatus,
    CurrentTask,
    AgentResponse,
    RoutingDecision,
    QualificationData,
    QualificationField,
    AgentInteraction
)


def test_qualification_field():
    """Test QualificationField model."""
    field = QualificationField(
        value=50000,
        confidence=0.8,
        source="qualifier"
    )
    
    assert field.value == 50000
    assert field.confidence == 0.8
    assert field.source == "qualifier"


def test_qualification_data():
    """Test QualificationData model and properties."""
    qual_data = QualificationData()
    
    # Initially unqualified
    assert qual_data.overall_confidence == 0.0
    assert qual_data.is_qualified is False
    
    # Update fields
    qual_data.budget = QualificationField(value=100000, confidence=0.9)
    qual_data.authority = QualificationField(value="CEO", confidence=0.8)
    qual_data.need = QualificationField(value="automation", confidence=0.7)
    qual_data.timeline = QualificationField(value="Q2 2024", confidence=0.6)
    
    # Check properties
    assert qual_data.overall_confidence == 0.75  # (0.9 + 0.8 + 0.7 + 0.6) / 4
    assert qual_data.is_qualified is True


def test_current_task():
    """Test CurrentTask model."""
    task = CurrentTask(
        type="qualification",
        assigned_agent="qualifier",
        status=TaskStatus.PROCESSING,
        context={"user_message": "I need help with automation"},
        iterations=1
    )
    
    assert task.type == "qualification"
    assert task.assigned_agent == "qualifier"
    assert task.status == TaskStatus.PROCESSING
    assert task.iterations == 1
    assert task.max_iterations == 5


def test_agent_response():
    """Test AgentResponse model."""
    response = AgentResponse(
        response="Let me help you with that",
        tools_used=["calculator", "web_search"],
        metadata={"confidence": 0.9}
    )
    
    assert response.response == "Let me help you with that"
    assert len(response.tools_used) == 2
    assert response.metadata["confidence"] == 0.9
    assert isinstance(response.timestamp, datetime)


def test_routing_decision():
    """Test RoutingDecision model."""
    decision = RoutingDecision(
        from_agent="supervisor",
        to_agent="qualifier",
        reason="New lead requires qualification",
        stage_before="lead",
        stage_after="qualification"
    )
    
    assert decision.from_agent == "supervisor"
    assert decision.to_agent == "qualifier"
    assert decision.stage_before == "lead"
    assert decision.stage_after == "qualification"


def test_workflow_state():
    """Test WorkflowState model and methods."""
    workflow = WorkflowState(
        id="workflow123",
        current_stage="lead"
    )
    
    # Check initial state
    assert workflow.id == "workflow123"
    assert workflow.status == WorkflowStatus.IN_PROGRESS
    assert workflow.current_stage == "lead"
    assert len(workflow.agents_visited) == 0
    
    # Add routing decision
    decision = RoutingDecision(
        from_agent="supervisor",
        to_agent="qualifier",
        reason="New lead",
        stage_before="lead",
        stage_after="qualification"
    )
    workflow.add_routing_decision(decision)
    
    assert len(workflow.routing_history) == 1
    assert workflow.routing_history[0].to_agent == "qualifier"
    
    # Record agent response
    response = AgentResponse(
        response="Hello, how can I help?",
        tools_used=[]
    )
    workflow.record_agent_response("qualifier", response)
    
    assert "qualifier" in workflow.agent_responses
    assert "qualifier" in workflow.agents_visited
    
    # Update qualification
    workflow.update_qualification("budget", 50000, 0.8)
    assert workflow.qualification_data.budget.value == 50000
    assert workflow.qualification_data.budget.confidence == 0.8


def test_workflow_state_mongodb_conversion():
    """Test WorkflowState MongoDB conversion."""
    workflow = WorkflowState(
        id="workflow123",
        current_stage="qualification"
    )
    
    # Add some data
    workflow.record_agent_response(
        "qualifier",
        AgentResponse(response="Test response")
    )
    
    # Convert to MongoDB doc
    doc = workflow.to_mongodb_doc()
    
    assert isinstance(doc["created_at"], str)
    assert isinstance(doc["updated_at"], str)
    assert doc["id"] == "workflow123"
    
    # Convert back
    workflow2 = WorkflowState.from_mongodb_doc(doc)
    assert workflow2.id == "workflow123"
    assert isinstance(workflow2.created_at, datetime)


def test_agent_interaction():
    """Test AgentInteraction model."""
    interaction = AgentInteraction(
        conversation_id="conv123",
        agent_name="qualifier",
        user_message="I need automation",
        agent_response="Let me help you with that",
        sales_stage="qualification",
        tools_used=["web_search"],
        objections_raised=["price_concern"]
    )
    
    assert interaction.conversation_id == "conv123"
    assert interaction.agent_name == "qualifier"
    assert len(interaction.objections_raised) == 1
    assert interaction.objections_raised[0] == "price_concern"