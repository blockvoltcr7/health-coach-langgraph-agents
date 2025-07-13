"""Extended async integration tests for AsyncConversationRepository.

These tests cover methods that were missing test coverage in the main test files.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from bson import ObjectId

from app.db.mongodb.async_client import (
    get_async_database,
    close_async_mongodb_connection
)
from app.db.mongodb.async_conversation_repository import AsyncConversationRepository
from app.db.mongodb.schemas.conversation_schema import (
    MessageRole,
    SalesStage,
    AgentName,
    ConversationStatus,
    ObjectionType,
    FollowUpType
)


@pytest.mark.asyncio
@pytest.mark.integration
class TestAsyncRepositoryExtended:
    """Extended tests for AsyncConversationRepository methods."""
    
    @pytest.fixture
    async def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Setup - use test database
        self.db = await get_async_database("test_limitless_os_sales_extended")
        self.repository = AsyncConversationRepository(self.db)
        
        # Create some test data
        self.test_users = []
        self.test_conversations = []
        
        yield
        
        # Teardown - clean up test data
        collections = await self.db.list_collection_names()
        if "conversations" in collections:
            await self.db.drop_collection("conversations")
        await close_async_mongodb_connection()
    
    async def _create_test_conversation(self, user_id: str, **kwargs) -> str:
        """Helper to create a test conversation."""
        defaults = {
            "channel": "web",
            "initial_message": "Test message",
            "metadata": {"test": True}
        }
        defaults.update(kwargs)
        
        conversation_id = await self.repository.create_conversation(
            user_id=user_id,
            **defaults
        )
        
        self.test_users.append(user_id)
        self.test_conversations.append(conversation_id)
        
        return conversation_id
    
    async def test_find_or_create_conversation(self, setup_and_teardown):
        """Test find_or_create_conversation method."""
        user_id = "find_or_create_test_user"
        
        # First call - should create new conversation
        conversation_id1, created1 = await self.repository.find_or_create_conversation(
            user_id=user_id,
            channel="instagram_dm",
            metadata={"source": "instagram_story"}
        )
        
        assert created1 is True
        assert conversation_id1 is not None
        
        # Verify conversation was created correctly
        conv1 = await self.repository.find_by_id(conversation_id1)
        assert conv1['user_id'] == user_id
        assert conv1['channel'] == "instagram_dm"
        assert conv1['metadata']['source'] == "instagram_story"
        
        # Second call - should find existing active conversation
        conversation_id2, created2 = await self.repository.find_or_create_conversation(
            user_id=user_id,
            channel="web"  # Different channel should be ignored
        )
        
        assert created2 is False
        assert conversation_id2 == conversation_id1
        
        # Close the conversation
        await self.repository.update_by_id(
            conversation_id1,
            {"$set": {"status": ConversationStatus.CLOSED.value}}
        )
        
        # Third call - should create new conversation since previous is closed
        conversation_id3, created3 = await self.repository.find_or_create_conversation(
            user_id=user_id,
            channel="api"
        )
        
        assert created3 is True
        assert conversation_id3 != conversation_id1
        
        # Verify new conversation
        conv3 = await self.repository.find_by_id(conversation_id3)
        assert conv3['channel'] == "api"
        assert conv3['status'] == ConversationStatus.ACTIVE.value
    
    async def test_add_handoff_async(self, setup_and_teardown):
        """Test add_handoff_async method."""
        conversation_id = await self._create_test_conversation("handoff_test_user")
        
        # Add initial handoff
        result = await self.repository.add_handoff_async(
            conversation_id=conversation_id,
            from_agent=AgentName.SUPERVISOR.value,
            to_agent=AgentName.QUALIFIER.value,
            reason="User needs qualification",
            trigger_type="stage_complete",
            trigger_details="Initial contact established",
            confidence_score=0.95
        )
        
        assert result.modified_count == 1
        
        # Verify handoff was added
        conversation = await self.repository.get_conversation_state(conversation_id)
        assert len(conversation['handoffs']) == 1
        
        handoff = conversation['handoffs'][0]
        assert handoff['from_agent'] == AgentName.SUPERVISOR.value
        assert handoff['to_agent'] == AgentName.QUALIFIER.value
        assert handoff['reason'] == "User needs qualification"
        assert handoff['trigger']['type'] == "stage_complete"
        assert handoff['trigger']['details'] == "Initial contact established"
        assert handoff['context']['confidence_score'] == 0.95
        assert 'handoff_id' in handoff
        assert 'timestamp' in handoff
        
        # Verify current agent was updated
        assert conversation['current_agent'] == AgentName.QUALIFIER.value
        assert conversation['agent_context']['previous_agent'] == AgentName.SUPERVISOR.value
        
        # Add second handoff
        await self.repository.add_handoff_async(
            conversation_id=conversation_id,
            from_agent=AgentName.QUALIFIER.value,
            to_agent=AgentName.OBJECTION_HANDLER.value,
            reason="Price objection raised",
            trigger_type="user_intent",
            trigger_details="User mentioned price concerns"
        )
        
        # Verify multiple handoffs
        conversation = await self.repository.get_conversation_state(conversation_id)
        assert len(conversation['handoffs']) == 2
        assert conversation['current_agent'] == AgentName.OBJECTION_HANDLER.value
    
    async def test_find_by_sales_stage_async(self, setup_and_teardown):
        """Test find_by_sales_stage_async method."""
        # Create conversations in different stages
        stages_data = [
            (SalesStage.LEAD.value, "user1", ConversationStatus.ACTIVE.value),
            (SalesStage.LEAD.value, "user2", ConversationStatus.ACTIVE.value),
            (SalesStage.QUALIFIED.value, "user3", ConversationStatus.ACTIVE.value),
            (SalesStage.QUALIFIED.value, "user4", ConversationStatus.CLOSED.value),
            (SalesStage.CLOSING.value, "user5", ConversationStatus.ACTIVE.value),
        ]
        
        for stage, user_id, status in stages_data:
            conv_id = await self._create_test_conversation(user_id)
            await self.repository.update_by_id(
                conv_id,
                {"$set": {"sales_stage": stage, "status": status}}
            )
        
        # Test finding all LEAD stage conversations
        lead_convs = await self.repository.find_by_sales_stage_async(
            stage=SalesStage.LEAD.value
        )
        assert len(lead_convs) == 2
        
        # Test finding only active QUALIFIED conversations
        qualified_active = await self.repository.find_by_sales_stage_async(
            stage=SalesStage.QUALIFIED.value,
            status=ConversationStatus.ACTIVE.value
        )
        assert len(qualified_active) == 1
        assert qualified_active[0]['user_id'] == "user3"
        
        # Test with limit
        limited = await self.repository.find_by_sales_stage_async(
            stage=SalesStage.LEAD.value,
            limit=1
        )
        assert len(limited) == 1
        
        # Test non-existent stage
        empty = await self.repository.find_by_sales_stage_async(
            stage=SalesStage.CLOSED_WON.value
        )
        assert len(empty) == 0
    
    async def test_perform_atomic_qualification_and_stage_update(self, setup_and_teardown):
        """Test atomic qualification and stage update method."""
        conversation_id = await self._create_test_conversation("atomic_test_user")
        
        # Perform atomic update
        qualification_data = {
            "budget": {
                "meets_criteria": True,
                "value": "$500/month",
                "confidence": 0.9,
                "captured_at": datetime.utcnow().isoformat() + "Z"
            },
            "authority": {
                "meets_criteria": True,
                "role": "CEO",
                "needs_approval": False,
                "confidence": 1.0
            },
            "need": {
                "meets_criteria": True,
                "pain_points": ["scaling", "automation"],
                "use_case": "Business automation",
                "confidence": 0.85
            },
            "timeline": {
                "meets_criteria": True,
                "timeframe": "This quarter",
                "urgency": "high",
                "confidence": 0.95
            }
        }
        
        result = await self.repository.perform_atomic_qualification_and_stage_update(
            conversation_id=conversation_id,
            qualification_data=qualification_data,
            new_stage=SalesStage.QUALIFIED.value,
            qualified_by=AgentName.QUALIFIER.value
        )
        
        assert result is True
        
        # Verify the update
        conversation = await self.repository.get_conversation_state(conversation_id)
        
        # Check qualification data
        qual = conversation['qualification']
        assert qual['budget']['meets_criteria'] is True
        assert qual['budget']['value'] == "$500/month"
        assert qual['authority']['role'] == "CEO"
        assert qual['need']['pain_points'] == ["scaling", "automation"]
        assert qual['timeline']['urgency'] == "high"
        
        # Check overall score calculation
        assert 'overall_score' in qual
        assert qual['overall_score'] > 0
        
        # Check stage update
        assert conversation['sales_stage'] == SalesStage.QUALIFIED.value
        assert conversation['is_qualified'] is True
        assert qual['qualified_by'] == AgentName.QUALIFIER.value
        assert qual['qualified_at'] is not None
        
        # Verify stage history
        assert len(conversation['stage_history']) >= 2
        latest_stage = conversation['stage_history'][-1]
        assert latest_stage['stage'] == SalesStage.QUALIFIED.value
    
    async def test_search_conversations_by_content(self, setup_and_teardown):
        """Test text search functionality."""
        # Create conversations with different message content
        test_data = [
            ("search_user1", ["I need help with automation", "Looking for AI solutions"]),
            ("search_user2", ["Tell me about pricing", "Is there a discount for annual?"]),
            ("search_user3", ["I want to automate my workflow", "Need integration with Slack"]),
            ("search_user4", ["Just browsing", "Not interested right now"])
        ]
        
        for user_id, messages in test_data:
            conv_id = await self._create_test_conversation(user_id)
            for msg in messages:
                await self.repository.add_message_async(
                    conversation_id=conv_id,
                    role=MessageRole.USER.value,
                    content=msg
                )
        
        # Search for "automation" - should find 2 conversations
        automation_results = await self.repository.search_conversations_by_content(
            search_query="automation"
        )
        assert len(automation_results) == 2
        user_ids = [r['user_id'] for r in automation_results]
        assert "search_user1" in user_ids
        assert "search_user3" in user_ids
        
        # Check text score is included
        if automation_results:
            assert 'score' in automation_results[0]
        
        # Search for "pricing" - should find 1 conversation
        pricing_results = await self.repository.search_conversations_by_content(
            search_query="pricing discount"
        )
        assert len(pricing_results) == 1
        assert pricing_results[0]['user_id'] == "search_user2"
        
        # Search with user filter
        user_filtered = await self.repository.search_conversations_by_content(
            search_query="automation",
            user_id="search_user1"
        )
        assert len(user_filtered) == 1
        assert user_filtered[0]['user_id'] == "search_user1"
        
        # Search for non-existent term
        no_results = await self.repository.search_conversations_by_content(
            search_query="blockchain cryptocurrency"
        )
        assert len(no_results) == 0
    
    async def test_get_conversation_messages_containing(self, setup_and_teardown):
        """Test searching messages within a specific conversation."""
        conversation_id = await self._create_test_conversation("message_search_user")
        
        # Add various messages
        messages = [
            (MessageRole.USER.value, "I need help with automation tools"),
            (AgentName.SUPERVISOR.value, "I'd be happy to help you with automation!"),
            (MessageRole.USER.value, "What about pricing for automation features?"),
            (AgentName.QUALIFIER.value, "Our automation pricing starts at $297/month"),
            (MessageRole.USER.value, "Do you have enterprise plans?"),
            (AgentName.QUALIFIER.value, "Yes, we offer custom enterprise solutions")
        ]
        
        for role, content in messages:
            await self.repository.add_message_async(
                conversation_id=conversation_id,
                role=role,
                content=content
            )
        
        # Search for "automation" in this conversation
        automation_msgs = await self.repository.get_conversation_messages_containing(
            conversation_id=conversation_id,
            search_text="automation"
        )
        
        assert len(automation_msgs) == 3  # 3 messages contain "automation"
        
        # Verify message content
        contents = [msg['content'] for msg in automation_msgs]
        assert any("automation tools" in c for c in contents)
        assert any("help you with automation" in c for c in contents)
        assert any("automation pricing" in c for c in contents)
        
        # Search for "pricing"
        pricing_msgs = await self.repository.get_conversation_messages_containing(
            conversation_id=conversation_id,
            search_text="pricing"
        )
        
        assert len(pricing_msgs) == 2
        
        # Search with limit
        limited = await self.repository.get_conversation_messages_containing(
            conversation_id=conversation_id,
            search_text="automation",
            limit=2
        )
        
        assert len(limited) == 2
        
        # Search for non-existent text
        no_results = await self.repository.get_conversation_messages_containing(
            conversation_id=conversation_id,
            search_text="blockchain"
        )
        
        assert len(no_results) == 0
    
    async def test_concurrent_operations(self, setup_and_teardown):
        """Test concurrent async operations."""
        # Create multiple conversations concurrently
        user_ids = [f"concurrent_user_{i}" for i in range(10)]
        
        create_tasks = [
            self.repository.find_or_create_conversation(
                user_id=user_id,
                channel="api"
            )
            for user_id in user_ids
        ]
        
        results = await asyncio.gather(*create_tasks)
        
        # All should be created
        assert all(created for _, created in results)
        conversation_ids = [conv_id for conv_id, _ in results]
        
        # Perform concurrent updates
        update_tasks = [
            self.repository.add_message_async(
                conversation_id=conv_id,
                role=MessageRole.USER.value,
                content=f"Concurrent message for {user_id}"
            )
            for conv_id, user_id in zip(conversation_ids, user_ids)
        ]
        
        await asyncio.gather(*update_tasks)
        
        # Verify all messages were added
        for conv_id in conversation_ids:
            conv = await self.repository.get_conversation_state(conv_id)
            assert len(conv['messages']) == 1
            assert "Concurrent message" in conv['messages'][0]['content']
    
    async def test_error_handling_extended(self, setup_and_teardown):
        """Test error handling in extended methods."""
        # Test find_or_create with invalid data
        with pytest.raises(ValueError):
            await self.repository.find_or_create_conversation(
                user_id="",  # Empty user ID
                channel="web"
            )
        
        # Test atomic update with non-existent conversation
        fake_id = str(ObjectId())
        result = await self.repository.perform_atomic_qualification_and_stage_update(
            conversation_id=fake_id,
            qualification_data={},
            new_stage=SalesStage.QUALIFIED.value
        )
        assert result is False
        
        # Test search with invalid conversation ID
        messages = await self.repository.get_conversation_messages_containing(
            conversation_id="invalid_id",
            search_text="test"
        )
        assert len(messages) == 0


if __name__ == "__main__":
    # Run extended async tests
    pytest.main([__file__, "-v", "-m", "integration and asyncio"])