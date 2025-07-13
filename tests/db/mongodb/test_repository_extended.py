"""Extended integration tests for ConversationRepository.

These tests cover methods that were missing test coverage in the main test files.
Uses synchronous MongoDB operations due to Motor Python 3.12 compatibility issues.
"""

import pytest
from datetime import datetime, timedelta
from bson import ObjectId

from app.db.mongodb.client import get_database
from app.db.mongodb.schemas.conversation_schema import (
    ConversationRepository,
    ConversationSchema,
    MessageRole,
    SalesStage,
    AgentName,
    ConversationStatus,
    ObjectionType,
    FollowUpType
)


@pytest.mark.integration
class TestRepositoryExtended:
    """Extended tests for ConversationRepository methods."""
    
    @pytest.fixture
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Setup - use test database
        self.db = get_database("test_limitless_os_sales_extended")
        self.repository = ConversationRepository(self.db)
        
        # Create conversations collection if it doesn't exist
        if "conversations" not in self.db.list_collection_names():
            ConversationSchema.create_collection(self.db)
        
        yield
        
        # Teardown - clean up test data
        if "conversations" in self.db.list_collection_names():
            self.db.drop_collection("conversations")
    
    def _create_test_conversation(self, user_id: str, **kwargs) -> str:
        """Helper to create a test conversation."""
        defaults = {
            "channel": "web",
            "initial_message": "Test message",
            "metadata": {"test": True}
        }
        defaults.update(kwargs)
        
        doc = ConversationSchema.create_conversation_document(
            user_id=user_id,
            **defaults
        )
        
        result = self.repository.create_one(doc)
        return str(result.inserted_id)
    
    def test_text_search_functionality(self, setup_and_teardown):
        """Test text search functionality with the repository."""
        # Create conversations with different message content
        test_data = [
            ("search_user1", ["I need help with automation", "Looking for AI solutions"]),
            ("search_user2", ["Tell me about pricing", "Is there a discount for annual?"]),
            ("search_user3", ["I want to automate my workflow", "Need integration with Slack"]),
            ("search_user4", ["Just browsing", "Not interested right now"])
        ]
        
        conv_ids = []
        for user_id, messages in test_data:
            conv_id = self._create_test_conversation(user_id)
            conv_ids.append(conv_id)
            for msg in messages:
                self.repository.add_message(
                    conversation_id=conv_id,
                    role=MessageRole.USER.value,
                    content=msg
                )
        
        # Test basic text search using MongoDB $text operator
        # Search for conversations containing "automation"
        automation_convs = list(self.repository.find_many({
            "$text": {"$search": "automation"}
        }))
        
        # Should find at least 2 conversations
        assert len(automation_convs) >= 2
        found_users = [c['user_id'] for c in automation_convs]
        assert "search_user1" in found_users
        assert "search_user3" in found_users
        
        # Search for "pricing"
        pricing_convs = list(self.repository.find_many({
            "$text": {"$search": "pricing"}
        }))
        
        assert len(pricing_convs) >= 1
        assert any(c['user_id'] == "search_user2" for c in pricing_convs)
        
        # Test compound text search
        compound_search = list(self.repository.find_many({
            "$and": [
                {"$text": {"$search": "automation"}},
                {"user_id": "search_user1"}
            ]
        }))
        
        assert len(compound_search) == 1
        assert compound_search[0]['user_id'] == "search_user1"
    
    def test_sales_stage_filtering(self, setup_and_teardown):
        """Test finding conversations by sales stage."""
        # Create conversations in different stages
        stages_data = [
            (SalesStage.LEAD.value, "stage_user1", ConversationStatus.ACTIVE.value),
            (SalesStage.LEAD.value, "stage_user2", ConversationStatus.ACTIVE.value),
            (SalesStage.QUALIFIED.value, "stage_user3", ConversationStatus.ACTIVE.value),
            (SalesStage.QUALIFIED.value, "stage_user4", ConversationStatus.CLOSED.value),
            (SalesStage.CLOSING.value, "stage_user5", ConversationStatus.ACTIVE.value),
        ]
        
        for stage, user_id, status in stages_data:
            conv_id = self._create_test_conversation(user_id)
            self.repository.update_by_id(
                conv_id,
                {"$set": {"sales_stage": stage, "status": status}}
            )
        
        # Use the existing find_by_sales_stage method
        lead_convs = self.repository.find_by_sales_stage(
            stage=SalesStage.LEAD.value
        )
        assert len(lead_convs) == 2
        
        # Find only active QUALIFIED conversations
        qualified_active = self.repository.find_by_sales_stage(
            stage=SalesStage.QUALIFIED.value,
            status=ConversationStatus.ACTIVE.value
        )
        assert len(qualified_active) == 1
        assert qualified_active[0]['user_id'] == "stage_user3"
        
        # Test with limit
        limited = self.repository.find_by_sales_stage(
            stage=SalesStage.LEAD.value,
            limit=1
        )
        assert len(limited) == 1
    
    def test_handoff_functionality(self, setup_and_teardown):
        """Test handoff functionality."""
        conversation_id = self._create_test_conversation("handoff_test_user")
        
        # Add initial handoff
        result = self.repository.add_handoff(
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
        conversation = self.repository.find_by_id(conversation_id)
        assert len(conversation['handoffs']) == 1
        
        handoff = conversation['handoffs'][0]
        assert handoff['from_agent'] == AgentName.SUPERVISOR.value
        assert handoff['to_agent'] == AgentName.QUALIFIER.value
        assert handoff['reason'] == "User needs qualification"
        assert handoff['trigger']['type'] == "stage_complete"
        assert handoff['context']['confidence_score'] == 0.95
        
        # Add second handoff
        self.repository.add_handoff(
            conversation_id=conversation_id,
            from_agent=AgentName.QUALIFIER.value,
            to_agent=AgentName.OBJECTION_HANDLER.value,
            reason="Price objection raised"
        )
        
        # Verify multiple handoffs
        conversation = self.repository.find_by_id(conversation_id)
        assert len(conversation['handoffs']) == 2
        assert conversation['current_agent'] == AgentName.OBJECTION_HANDLER.value
    
    def test_qualification_update(self, setup_and_teardown):
        """Test qualification update functionality."""
        conversation_id = self._create_test_conversation("qualification_test_user")
        
        # Update qualification data
        result = self.repository.update_qualification(
            conversation_id=conversation_id,
            budget_info={
                "meets_criteria": True,
                "value": "$500/month",
                "confidence": 90,  # Will be converted to 0.9
                "captured_at": datetime.utcnow().isoformat() + "Z"
            },
            authority_info={
                "meets_criteria": True,
                "role": "CEO",
                "needs_approval": False,
                "confidence": 100  # Will be converted to 1.0
            },
            need_info={
                "meets_criteria": True,
                "pain_points": ["scaling", "automation"],
                "use_case": "Business automation",
                "confidence": 85  # Will be converted to 0.85
            },
            timeline_info={
                "meets_criteria": True,
                "timeframe": "This quarter",
                "urgency": "high",
                "confidence": 95  # Will be converted to 0.95
            }
        )
        
        assert result.modified_count == 1
        
        # Verify the update
        conversation = self.repository.find_by_id(conversation_id)
        qual = conversation['qualification']
        
        # Check data was stored correctly
        assert qual['budget']['meets_criteria'] is True
        assert qual['budget']['value'] == "$500/month"
        assert qual['budget']['confidence'] == 0.9  # Converted to 0-1 range
        
        assert qual['authority']['role'] == "CEO"
        assert qual['authority']['confidence'] == 1.0
        
        assert qual['need']['pain_points'] == ["scaling", "automation"]
        assert qual['need']['confidence'] == 0.85
        
        assert qual['timeline']['urgency'] == "high"
        assert qual['timeline']['confidence'] == 0.95
        
        # Check overall score calculation
        assert 'overall_score' in qual
        expected_score = (90 + 100 + 85 + 95) / 4  # Average of all scores
        assert abs(qual['overall_score'] - expected_score) < 0.01
    
    def test_follow_up_scheduling(self, setup_and_teardown):
        """Test follow-up scheduling functionality."""
        conversation_id = self._create_test_conversation("followup_test_user")
        
        # Schedule a follow-up
        scheduled_date = datetime.utcnow() + timedelta(days=3)
        result = self.repository.schedule_follow_up(
            conversation_id=conversation_id,
            scheduled_date=scheduled_date,
            follow_up_type=FollowUpType.RE_ENGAGEMENT.value,
            priority="high",
            assigned_agent=AgentName.QUALIFIER.value,
            context={"last_topic": "pricing discussion"}
        )
        
        assert result.modified_count == 1
        
        # Verify follow-up was scheduled
        conversation = self.repository.find_by_id(conversation_id)
        follow_up = conversation['follow_up']
        
        assert follow_up['required'] is True
        assert follow_up['type'] == FollowUpType.RE_ENGAGEMENT.value
        assert follow_up['priority'] == "high"
        assert follow_up['assigned_agent'] == AgentName.QUALIFIER.value
        assert follow_up['context']['last_topic'] == "pricing discussion"
        
        # Check final attempt date is 7 days after scheduled date
        scheduled_dt = datetime.fromisoformat(follow_up['scheduled_date'].replace('Z', '+00:00'))
        final_dt = datetime.fromisoformat(follow_up['final_attempt_date'].replace('Z', '+00:00'))
        assert (final_dt - scheduled_dt).days == 7
        
        # Test finding follow-ups due
        due_followups = self.repository.find_follow_ups_due(
            before_date=scheduled_date + timedelta(days=1)
        )
        
        assert len(due_followups) == 1
        assert due_followups[0]['_id'] == ObjectId(conversation_id)
    
    def test_active_conversation_finding(self, setup_and_teardown):
        """Test finding active conversations for users."""
        user_id = "active_test_user"
        
        # Create an active conversation
        active_id = self._create_test_conversation(user_id, channel="web")
        
        # Find active conversation
        active_conv = self.repository.find_active_by_user(user_id)
        assert active_conv is not None
        assert str(active_conv['_id']) == active_id
        
        # Close the conversation
        self.repository.update_by_id(
            active_id,
            {"$set": {"status": ConversationStatus.CLOSED.value}}
        )
        
        # Should not find active conversation now
        no_active = self.repository.find_active_by_user(user_id)
        assert no_active is None
        
        # Create new active conversation
        new_active_id = self._create_test_conversation(user_id, channel="api")
        
        # Should find the new active conversation
        new_active = self.repository.find_active_by_user(user_id)
        assert new_active is not None
        assert str(new_active['_id']) == new_active_id
    
    def test_agent_assignment_queries(self, setup_and_teardown):
        """Test finding conversations by current agent."""
        # Create conversations assigned to different agents
        agents_data = [
            (AgentName.SUPERVISOR.value, "agent_user1"),
            (AgentName.SUPERVISOR.value, "agent_user2"),
            (AgentName.QUALIFIER.value, "agent_user3"),
            (AgentName.OBJECTION_HANDLER.value, "agent_user4"),
            (AgentName.CLOSER.value, "agent_user5")
        ]
        
        for agent, user_id in agents_data:
            conv_id = self._create_test_conversation(user_id)
            self.repository.update_by_id(
                conv_id,
                {"$set": {"current_agent": agent}}
            )
        
        # Find conversations by agent
        supervisor_convs = self.repository.find_by_current_agent(
            agent=AgentName.SUPERVISOR.value
        )
        assert len(supervisor_convs) == 2
        
        qualifier_convs = self.repository.find_by_current_agent(
            agent=AgentName.QUALIFIER.value
        )
        assert len(qualifier_convs) == 1
        assert qualifier_convs[0]['user_id'] == "agent_user3"
        
        # Test with limit
        limited_convs = self.repository.find_by_current_agent(
            agent=AgentName.SUPERVISOR.value,
            limit=1
        )
        assert len(limited_convs) == 1
    
    def test_message_metrics_update(self, setup_and_teardown):
        """Test that message metrics are updated correctly."""
        conversation_id = self._create_test_conversation("metrics_test_user")
        
        # Add messages from different agents
        messages = [
            (MessageRole.USER.value, "Hello"),
            (AgentName.SUPERVISOR.value, "Hi there!"),
            (MessageRole.USER.value, "I need help"),
            (AgentName.QUALIFIER.value, "Let me qualify your needs"),
            (MessageRole.USER.value, "I have concerns about price"),
            (AgentName.OBJECTION_HANDLER.value, "Let me address your concerns")
        ]
        
        for role, content in messages:
            self.repository.add_message(
                conversation_id=conversation_id,
                role=role,
                content=content
            )
        
        # Verify metrics
        conversation = self.repository.find_by_id(conversation_id)
        metrics = conversation['agent_metrics']
        
        assert metrics['total_messages'] == 6
        assert metrics['supervisor']['messages_sent'] == 1
        assert metrics['qualifier']['messages_sent'] == 1
        assert metrics['objection_handler']['messages_sent'] == 1
        assert conversation['agent_context']['interaction_count'] == 6


if __name__ == "__main__":
    # Run extended tests
    pytest.main([__file__, "-v", "-m", "integration"])