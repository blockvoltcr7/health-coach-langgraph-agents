"""Tests for the multi-agent service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.services.multi_agent_service import MultiAgentService, get_multi_agent_service
from app.services.conversation_service import ConversationEvent
from app.core.multi_agent_graph import SalesStage, NextAgent
from app.models.multi_agent_models import WorkflowState, WorkflowStatus


@pytest.fixture
def mock_conversation_service():
    """Create a mock conversation service."""
    service = AsyncMock()
    
    # Mock conversation creation
    service.create_or_resume_conversation.return_value = (
        {
            "_id": "conv123",
            "user_id": "user123",
            "sales_stage": "lead",
            "status": "active",
            "messages": [],
            "metadata": {"mem0_user_id": "mem0_user123"}
        },
        ConversationEvent.CREATED
    )
    
    # Mock other methods
    service.save_conversation_turn = AsyncMock()
    service.update_conversation_metadata = AsyncMock()
    service.update_sales_stage = AsyncMock()
    service.handle_agent_handoff = AsyncMock()
    service.get_conversation_summary = AsyncMock(return_value={
        "message_count": 0,
        "updated_at": datetime.utcnow().isoformat()
    })
    
    # Mock repository
    service._repository = AsyncMock()
    service._repository.get_conversation_state = AsyncMock(return_value=None)
    
    return service


@pytest.fixture
def mock_mem0_client():
    """Create a mock Mem0 client."""
    client = AsyncMock()
    
    # Mock memory operations
    client.get_memory_history = AsyncMock(return_value=[])
    client.add_memory = AsyncMock(return_value={"id": "mem123"})
    
    return client


@pytest.fixture
def multi_agent_service(mock_conversation_service, mock_mem0_client):
    """Create multi-agent service with mocked dependencies."""
    return MultiAgentService(
        conversation_service=mock_conversation_service,
        mem0_client=mock_mem0_client
    )


@pytest.mark.asyncio
async def test_service_initialization(multi_agent_service):
    """Test service initialization."""
    await multi_agent_service._ensure_initialized()
    
    assert multi_agent_service._initialized
    assert multi_agent_service._graph is not None
    assert multi_agent_service._conversation_service is not None
    assert multi_agent_service._mem0_client is not None


@pytest.mark.asyncio
async def test_process_with_multi_agent_basic(multi_agent_service, mock_conversation_service):
    """Test basic multi-agent processing flow."""
    # Mock the graph execution
    mock_graph_result = {
        "messages": [
            MagicMock(content="Hello"),  # User message
            MagicMock(
                content="I'd love to help you. Let me understand your needs.",
                additional_kwargs={"agent": "qualifier"}
            )  # Agent response
        ],
        "workflow": {
            "_id": "workflow123",
            "status": "in_progress",
            "agents_visited": ["supervisor", "qualifier"],
            "routing_history": [{
                "from": "supervisor",
                "to": "qualifier",
                "reason": "New lead requires qualification"
            }]
        },
        "current_stage": SalesStage.QUALIFICATION,
        "next_agent": NextAgent.QUALIFIER
    }
    
    # Patch the graph invoke
    with patch.object(multi_agent_service, '_graph') as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=mock_graph_result)
        
        result = await multi_agent_service.process_with_multi_agent(
            message="I'm interested in your product",
            user_id="user123"
        )
    
    # Verify result
    assert result["response"] == "I'd love to help you. Let me understand your needs."
    assert result["conversation_id"] == "conv123"
    assert result["responding_agent"] == "qualifier"
    assert result["sales_stage"] == "qualification"
    assert result["agents_visited"] == ["supervisor", "qualifier"]
    
    # Verify service calls
    mock_conversation_service.create_or_resume_conversation.assert_called_once()
    mock_conversation_service.save_conversation_turn.assert_called_once()


@pytest.mark.asyncio
async def test_workflow_sync_to_mongodb(multi_agent_service, mock_conversation_service):
    """Test workflow synchronization to MongoDB."""
    workflow = {
        "_id": "workflow123",
        "status": "in_progress",
        "current_stage": "qualification",
        "qualification_data": {
            "budget": {"value": 50000, "confidence": 0.8},
            "authority": {"value": None, "confidence": 0},
            "need": {"value": "automation", "confidence": 0.9},
            "timeline": {"value": None, "confidence": 0}
        }
    }
    
    await multi_agent_service.sync_workflow_to_mongodb(
        conversation_id="conv123",
        workflow=workflow,
        current_stage=SalesStage.QUALIFICATION
    )
    
    # Verify metadata update was called
    mock_conversation_service.update_conversation_metadata.assert_called()
    
    # Verify qualification data was stored
    call_args = mock_conversation_service.update_conversation_metadata.call_args_list[-1]
    assert "qualification_progress" in call_args[1]["metadata_updates"]


@pytest.mark.asyncio
async def test_load_workflow_from_mongodb(multi_agent_service, mock_conversation_service):
    """Test loading workflow from MongoDB."""
    # Mock conversation with workflow data
    mock_conversation_service._repository.get_conversation_state.return_value = {
        "_id": "conv123",
        "sales_stage": "qualification",
        "created_at": datetime.utcnow().isoformat(),
        "metadata": {
            "workflow_id": "workflow123",
            "workflow_status": "in_progress",
            "qualification_progress": {
                "budget": {"value": 50000, "confidence": 0.8}
            }
        },
        "handoffs": [
            {"to_agent": "qualifier"}
        ],
        "objections": []
    }
    
    workflow = await multi_agent_service.load_workflow_from_mongodb("conv123")
    
    assert workflow is not None
    assert workflow["_id"] == "workflow123"
    assert workflow["status"] == "in_progress"
    assert workflow["current_stage"] == "qualification"
    assert "qualifier" in workflow["agents_visited"]


@pytest.mark.asyncio
async def test_get_agent_context(multi_agent_service, mock_mem0_client):
    """Test getting agent context from memories."""
    # Mock memory entries
    mock_memories = [
        MagicMock(
            memory="User is interested in automation solutions",
            metadata={"agent": "qualifier"}
        ),
        MagicMock(
            memory="Budget range is 50-100k",
            metadata={"agent": "qualifier"}
        )
    ]
    mock_mem0_client.get_memory_history.return_value = mock_memories
    
    context = await multi_agent_service.get_agent_context(
        user_id="mem0_user123",
        agent_name="qualifier"
    )
    
    assert "Previous conversation context" in context
    assert "automation solutions" in context
    assert "Budget range" in context


@pytest.mark.asyncio
async def test_handle_stage_transition(multi_agent_service, mock_conversation_service):
    """Test handling stage transitions with validation."""
    # Test valid transition
    result = await multi_agent_service.handle_stage_transition(
        conversation_id="conv123",
        from_stage=SalesStage.LEAD,
        to_stage=SalesStage.QUALIFICATION,
        reason="User expressed interest"
    )
    
    assert result is True
    mock_conversation_service.update_sales_stage.assert_called_once()
    
    # Test invalid transition
    result = await multi_agent_service.handle_stage_transition(
        conversation_id="conv123",
        from_stage=SalesStage.CLOSING,
        to_stage=SalesStage.LEAD,
        reason="Invalid backwards transition"
    )
    
    assert result is False


@pytest.mark.asyncio
async def test_get_workflow_summary(multi_agent_service, mock_conversation_service):
    """Test getting workflow summary."""
    # Mock workflow data
    mock_conversation_service._repository.get_conversation_state.return_value = {
        "_id": "conv123",
        "sales_stage": "qualification",
        "created_at": datetime.utcnow().isoformat(),
        "metadata": {
            "workflow_id": "workflow123",
            "workflow_status": "in_progress"
        },
        "handoffs": [{"to_agent": "qualifier"}],
        "objections": []
    }
    
    summary = await multi_agent_service.get_workflow_summary("conv123")
    
    assert summary["workflow_id"] == "workflow123"
    assert summary["status"] == "in_progress"
    assert summary["current_stage"] == "qualification"
    assert "qualifier" in summary["agents_visited"]


@pytest.mark.asyncio
async def test_get_multi_agent_service():
    """Test getting the global service instance."""
    with patch('app.services.multi_agent_service.get_conversation_service') as mock_get_conv:
        with patch('app.services.multi_agent_service.get_mem0_client') as mock_get_mem0:
            mock_get_conv.return_value = AsyncMock()
            mock_get_mem0.return_value = AsyncMock()
            
            service = await get_multi_agent_service()
            assert service is not None
            assert service._initialized