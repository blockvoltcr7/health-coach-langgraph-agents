"""Async integration tests for MongoDB implementation.

These tests require a real MongoDB connection and test the async
conversation management functionality.
"""

import pytest
import asyncio
from datetime import datetime
from bson import ObjectId

from app.db.mongodb.async_client import (
    get_async_mongodb_client,
    get_async_database,
    close_async_mongodb_connection
)
from app.db.mongodb.async_conversation_repository import AsyncConversationRepository
from app.db.mongodb.schemas.conversation_schema import (
    MessageRole,
    SalesStage,
    AgentName,
    ConversationStatus
)
from app.services.conversation_service import (
    ConversationService,
    ConversationEvent
)


@pytest.mark.asyncio
@pytest.mark.integration
class TestAsyncMongoDBIntegration:
    """Async integration tests that require real MongoDB connection."""
    
    @pytest.fixture
    async def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Setup - use test database
        self.db = await get_async_database("test_limitless_os_sales_async")
        self.repository = AsyncConversationRepository(self.db)
        self.service = ConversationService(self.repository)
        
        yield
        
        # Teardown - clean up test data
        collections = await self.db.list_collection_names()
        if "conversations" in collections:
            await self.db.drop_collection("conversations")
        await close_async_mongodb_connection()
    
    async def test_async_client_connection(self):
        """Test async MongoDB client connection."""
        client = await get_async_mongodb_client()
        
        # Test health check
        health = await client.health_check()
        
        assert health['status'] == 'healthy'
        assert health['connected'] is True
        assert 'version' in health
    
    async def test_create_conversation_async(self, setup_and_teardown):
        """Test creating a conversation asynchronously."""
        # Create conversation
        conversation_id = await self.repository.create_conversation(
            user_id="test_user_123",
            channel="web",
            initial_message="Hello, I want to learn about Limitless OS",
            metadata={"source": "test", "mem0_user_id": "mem0_test_123"}
        )
        
        assert conversation_id is not None
        
        # Verify conversation was created
        conversation = await self.repository.find_by_id(conversation_id)
        assert conversation is not None
        assert conversation['user_id'] == "test_user_123"
        assert conversation['channel'] == "web"
        assert len(conversation['messages']) == 1
        assert conversation['messages'][0]['content'] == "Hello, I want to learn about Limitless OS"
    
    async def test_conversation_lifecycle(self, setup_and_teardown):
        """Test complete conversation lifecycle."""
        user_id = "lifecycle_test_user"
        
        # Create or resume conversation
        conversation, event = await self.service.create_or_resume_conversation(
            user_id=user_id,
            channel="api",
            metadata={"test": True}
        )
        
        assert event == ConversationEvent.CREATED
        conversation_id = str(conversation["_id"])
        
        # Add messages
        await self.service.save_conversation_turn(
            conversation_id=conversation_id,
            user_message="Tell me about pricing",
            agent_response="Our pricing starts at $297/month",
            agent_name=AgentName.SUPERVISOR.value
        )
        
        # Update sales stage
        await self.service.update_sales_stage(
            conversation_id=conversation_id,
            new_stage=SalesStage.QUALIFIED.value,
            notes="Budget confirmed"
        )
        
        # Get conversation history
        history = await self.service.get_conversation_history(
            conversation_id=conversation_id
        )
        
        assert len(history) == 3  # Initial + user + agent
        assert history[1]['content'] == "Tell me about pricing"
        assert history[2]['content'] == "Our pricing starts at $297/month"
        
        # Get summary
        summary = await self.service.get_conversation_summary(
            conversation_id=conversation_id
        )
        
        assert summary['sales_stage'] == SalesStage.QUALIFIED.value
        assert summary['message_count'] == 3
        
        # Close conversation
        await self.service.close_conversation(
            conversation_id=conversation_id,
            reason="Test completed"
        )
        
        # Verify closed
        final_state = await self.repository.get_conversation_state(conversation_id)
        assert final_state['status'] == ConversationStatus.CLOSED.value
    
    async def test_concurrent_conversations(self, setup_and_teardown):
        """Test handling multiple concurrent conversations."""
        users = [f"concurrent_user_{i}" for i in range(5)]
        
        # Create conversations concurrently
        tasks = [
            self.service.create_or_resume_conversation(
                user_id=user_id,
                channel="web"
            )
            for user_id in users
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all conversations created
        assert len(results) == 5
        for (conversation, event), user_id in zip(results, users):
            assert event == ConversationEvent.CREATED
            assert conversation['user_id'] == user_id
    
    async def test_error_handling(self, setup_and_teardown):
        """Test error handling in async operations."""
        # Test invalid conversation ID
        with pytest.raises(ValueError, match="not found"):
            await self.repository.get_conversation_state(
                "507f1f77bcf86cd799439999",  # Non-existent ID
                raise_on_missing=True
            )
        
        # Test graceful handling when not raising
        result = await self.repository.get_conversation_state(
            "507f1f77bcf86cd799439999",
            raise_on_missing=False
        )
        assert result is None
        
        # Test invalid ObjectId format
        result = await self.repository.find_by_id("invalid_id")
        assert result is None
    
    async def test_find_or_create_conversation(self, setup_and_teardown):
        """Test find or create conversation logic."""
        user_id = "find_or_create_user"
        
        # First call - should create
        conv1, event1 = await self.service.create_or_resume_conversation(
            user_id=user_id
        )
        assert event1 == ConversationEvent.CREATED
        
        # Second call - should find existing
        conv2, event2 = await self.service.create_or_resume_conversation(
            user_id=user_id
        )
        assert event2 == ConversationEvent.RESUMED
        assert str(conv1["_id"]) == str(conv2["_id"])
    
    async def test_message_persistence(self, setup_and_teardown):
        """Test message saving and retrieval."""
        # Create conversation
        conversation_id = await self.repository.create_conversation(
            user_id="message_test_user"
        )
        
        # Add multiple messages
        messages = [
            (MessageRole.USER.value, "Hello"),
            (AgentName.SUPERVISOR.value, "Hi there!"),
            (MessageRole.USER.value, "Tell me about your product"),
            (AgentName.QUALIFIER.value, "I'd be happy to explain...")
        ]
        
        for role, content in messages:
            await self.repository.add_message_async(
                conversation_id=conversation_id,
                role=role,
                content=content
            )
        
        # Retrieve and verify
        history = await self.repository.get_conversation_history(
            conversation_id=conversation_id
        )
        
        assert len(history) == 4
        for i, (role, content) in enumerate(messages):
            assert history[i]['role'] == role
            assert history[i]['content'] == content
            assert 'timestamp' in history[i]
    
    async def test_sales_stage_progression(self, setup_and_teardown):
        """Test sales stage updates and history."""
        conversation_id = await self.repository.create_conversation(
            user_id="stage_test_user"
        )
        
        # Progress through stages
        stages = [
            (SalesStage.QUALIFIED.value, "BANT criteria met"),
            (SalesStage.PROPOSAL.value, "Sent pricing proposal"),
            (SalesStage.NEGOTIATION.value, "Discussing terms"),
            (SalesStage.CLOSED.value, "Deal closed!")
        ]
        
        for stage, notes in stages:
            await self.repository.update_sales_stage_async(
                conversation_id=conversation_id,
                new_stage=stage,
                notes=notes
            )
        
        # Verify final state
        conversation = await self.repository.get_conversation_state(conversation_id)
        assert conversation['sales_stage'] == SalesStage.CLOSED.value
        assert len(conversation['stage_history']) == 5  # Initial + 4 updates
        
        # Verify history
        for i, (stage, notes) in enumerate(stages):
            history_entry = conversation['stage_history'][i + 1]  # Skip initial
            assert history_entry['stage'] == stage
            assert history_entry['notes'] == notes


@pytest.mark.asyncio
async def test_async_mongodb_transaction(setup_and_teardown):
    """Test async MongoDB transactions."""
    from app.db.mongodb.async_client import async_mongodb_transaction
    
    db = await get_async_database("test_limitless_os_sales_async")
    repository = AsyncConversationRepository(db)
    
    try:
        async with async_mongodb_transaction(db) as session:
            # Create conversation in transaction
            conversation_id = await repository.create_conversation(
                user_id="transaction_test_user",
                channel="web"
            )
            
            # Add message in same transaction
            await repository.add_message_async(
                conversation_id=conversation_id,
                role=MessageRole.USER.value,
                content="Transaction test message"
            )
            
            # Transaction commits here
        
        # Verify data persisted
        conversation = await repository.get_conversation_state(conversation_id)
        assert conversation is not None
        assert len(conversation['messages']) == 1
        
    finally:
        # Cleanup
        if "conversations" in await db.list_collection_names():
            await db.drop_collection("conversations")


if __name__ == "__main__":
    # Run async integration tests manually
    pytest.main([__file__, "-v", "-m", "integration and asyncio"])