"""Tests for conversation schema implementation."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, call

from bson import ObjectId
from pymongo.results import UpdateResult

from app.db.mongodb.schemas.conversation_schema import (
    ChannelType,
    ConversationStatus,
    SalesStage,
    MessageRole,
    AgentName,
    ConversationSchema,
    ConversationRepository,
    CONVERSATION_SCHEMA
)


class TestEnums:
    """Test enum definitions."""
    
    def test_channel_type_values(self):
        """Test ChannelType enum values."""
        assert ChannelType.WEB.value == "web"
        assert ChannelType.MOBILE.value == "mobile"
        assert ChannelType.API.value == "api"
    
    def test_conversation_status_values(self):
        """Test ConversationStatus enum values."""
        assert ConversationStatus.ACTIVE.value == "active"
        assert ConversationStatus.INACTIVE.value == "inactive"
        assert ConversationStatus.CLOSED.value == "closed"
    
    def test_sales_stage_values(self):
        """Test SalesStage enum values."""
        assert SalesStage.LEAD.value == "lead"
        assert SalesStage.QUALIFIED.value == "qualified"
        assert SalesStage.PROPOSAL.value == "proposal"
        assert SalesStage.NEGOTIATION.value == "negotiation"
        assert SalesStage.CLOSED.value == "closed"
    
    def test_message_role_values(self):
        """Test MessageRole enum values."""
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SUPERVISOR.value == "supervisor"
        assert MessageRole.QUALIFIER.value == "qualifier"
        assert MessageRole.OBJECTION_HANDLER.value == "objection_handler"
        assert MessageRole.CLOSER.value == "closer"
    
    def test_agent_name_values(self):
        """Test AgentName enum values."""
        assert AgentName.SUPERVISOR.value == "supervisor"
        assert AgentName.QUALIFIER.value == "qualifier"
        assert AgentName.OBJECTION_HANDLER.value == "objection_handler"
        assert AgentName.CLOSER.value == "closer"


class TestConversationSchema:
    """Test ConversationSchema helper class."""
    
    def test_create_collection(self, mock_database):
        """Test creating collection with schema validation."""
        # Setup
        mock_collection = Mock()
        mock_database.list_collection_names.return_value = []
        mock_database.create_collection.return_value = mock_collection
        
        # Create collection
        with patch.object(ConversationSchema, 'create_indexes') as mock_create_indexes:
            collection = ConversationSchema.create_collection(mock_database)
        
        # Assert
        mock_database.create_collection.assert_called_once_with(
            "conversations",
            validator=CONVERSATION_SCHEMA
        )
        mock_create_indexes.assert_called_once_with(mock_collection)
        assert collection == mock_collection
    
    def test_create_collection_drops_existing(self, mock_database):
        """Test that existing collection is dropped."""
        # Setup
        mock_collection = Mock()
        mock_database.list_collection_names.return_value = ["conversations"]
        mock_database.create_collection.return_value = mock_collection
        
        # Create collection
        with patch.object(ConversationSchema, 'create_indexes'):
            ConversationSchema.create_collection(mock_database)
        
        # Assert existing was dropped
        mock_database.drop_collection.assert_called_once_with("conversations")
    
    def test_create_indexes(self, mock_collection):
        """Test index creation."""
        # Create indexes
        ConversationSchema.create_indexes(mock_collection)
        
        # Assert all expected indexes created
        expected_calls = [
            call([("user_id", 1), ("updated_at", -1)]),
            call([("sales_stage", 1), ("status", 1)]),
            call([("channel", 1)]),
            call([("created_at", -1)])
        ]
        
        assert mock_collection.create_index.call_count == 4
        mock_collection.create_index.assert_has_calls(expected_calls)
    
    def test_create_conversation_document_minimal(self):
        """Test creating conversation document with minimal data."""
        # Create document
        doc = ConversationSchema.create_conversation_document(
            user_id="507f1f77bcf86cd799439012",
            channel="web"
        )
        
        # Assert structure
        assert isinstance(doc['user_id'], ObjectId)
        assert doc['channel'] == "web"
        assert doc['status'] == ConversationStatus.ACTIVE.value
        assert doc['sales_stage'] == SalesStage.LEAD.value
        assert len(doc['stage_history']) == 1
        assert doc['stage_history'][0]['stage'] == SalesStage.LEAD.value
        assert isinstance(doc['created_at'], datetime)
        assert isinstance(doc['updated_at'], datetime)
        assert doc['messages'] == []
        assert doc['metadata'] == {}
    
    def test_create_conversation_document_with_message(self):
        """Test creating conversation document with initial message."""
        # Create document
        doc = ConversationSchema.create_conversation_document(
            user_id="507f1f77bcf86cd799439012",
            channel="mobile",
            initial_message="Hello, I'm interested"
        )
        
        # Assert message added
        assert len(doc['messages']) == 1
        assert doc['messages'][0]['role'] == MessageRole.USER.value
        assert doc['messages'][0]['content'] == "Hello, I'm interested"
        assert isinstance(doc['messages'][0]['timestamp'], datetime)
    
    def test_create_conversation_document_with_metadata(self):
        """Test creating conversation document with metadata."""
        # Create document
        metadata = {"mem0_user_id": "mem0_123", "source": "landing_page"}
        doc = ConversationSchema.create_conversation_document(
            user_id="507f1f77bcf86cd799439012",
            channel="api",
            metadata=metadata
        )
        
        # Assert metadata
        assert doc['metadata'] == metadata
    
    def test_conversation_schema_validation(self):
        """Test that schema validation structure is correct."""
        # Check schema has required fields
        assert "$jsonSchema" in CONVERSATION_SCHEMA
        schema = CONVERSATION_SCHEMA["$jsonSchema"]
        
        # Check required fields
        assert "required" in schema
        required_fields = schema["required"]
        assert "user_id" in required_fields
        assert "channel" in required_fields
        assert "status" in required_fields
        assert "sales_stage" in required_fields
        
        # Check enum validations
        properties = schema["properties"]
        assert properties["channel"]["enum"] == [c.value for c in ChannelType]
        assert properties["status"]["enum"] == [s.value for s in ConversationStatus]
        assert properties["sales_stage"]["enum"] == [s.value for s in SalesStage]


class TestConversationRepository:
    """Test ConversationRepository implementation."""
    
    def test_collection_name(self, mock_database):
        """Test collection name property."""
        repo = ConversationRepository(mock_database)
        assert repo.collection_name == "conversations"
    
    def test_find_active_by_user(self, mock_database, mock_collection):
        """Test finding active conversation by user."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        expected_doc = {'_id': ObjectId(), 'status': 'active'}
        mock_collection.find_one.return_value = expected_doc
        
        # Find active conversation
        repo = ConversationRepository(mock_database)
        result = repo.find_active_by_user("507f1f77bcf86cd799439012")
        
        # Assert
        mock_collection.find_one.assert_called_once()
        call_args = mock_collection.find_one.call_args[0][0]
        assert isinstance(call_args['user_id'], ObjectId)
        assert call_args['status'] == ConversationStatus.ACTIVE.value
        assert result == expected_doc
    
    def test_find_by_sales_stage(self, mock_database, mock_collection):
        """Test finding conversations by sales stage."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        mock_cursor = Mock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.__iter__.return_value = iter([{'_id': 1}, {'_id': 2}])
        mock_collection.find.return_value = mock_cursor
        
        # Find by stage
        repo = ConversationRepository(mock_database)
        results = repo.find_by_sales_stage(
            stage=SalesStage.QUALIFIED.value,
            status=ConversationStatus.ACTIVE.value,
            limit=10
        )
        
        # Assert
        mock_collection.find.assert_called_once_with({
            'sales_stage': SalesStage.QUALIFIED.value,
            'status': ConversationStatus.ACTIVE.value
        })
        assert len(results) == 2
    
    def test_add_message(self, mock_database, mock_collection):
        """Test adding message to conversation."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        mock_update_result = Mock(spec=UpdateResult)
        mock_collection.update_one.return_value = mock_update_result
        
        # Add message
        repo = ConversationRepository(mock_database)
        result = repo.add_message(
            conversation_id="507f1f77bcf86cd799439011",
            role=MessageRole.SUPERVISOR.value,
            content="Hello, how can I help?"
        )
        
        # Assert
        mock_collection.update_one.assert_called_once()
        filter_arg = mock_collection.update_one.call_args[0][0]
        update_arg = mock_collection.update_one.call_args[0][1]
        
        assert isinstance(filter_arg['_id'], ObjectId)
        assert '$push' in update_arg
        assert 'messages' in update_arg['$push']
        message = update_arg['$push']['messages']
        assert message['role'] == MessageRole.SUPERVISOR.value
        assert message['content'] == "Hello, how can I help?"
        assert isinstance(message['timestamp'], datetime)
    
    def test_update_sales_stage(self, mock_database, mock_collection):
        """Test updating sales stage."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        mock_update_result = Mock(spec=UpdateResult)
        mock_collection.update_one.return_value = mock_update_result
        
        # Update stage
        repo = ConversationRepository(mock_database)
        result = repo.update_sales_stage(
            conversation_id="507f1f77bcf86cd799439011",
            new_stage=SalesStage.QUALIFIED.value,
            notes="BANT criteria met"
        )
        
        # Assert
        mock_collection.update_one.assert_called_once()
        update_arg = mock_collection.update_one.call_args[0][1]
        
        assert update_arg['$set']['sales_stage'] == SalesStage.QUALIFIED.value
        assert '$push' in update_arg
        stage_history = update_arg['$push']['stage_history']
        assert stage_history['stage'] == SalesStage.QUALIFIED.value
        assert stage_history['notes'] == "BANT criteria met"
        assert isinstance(stage_history['timestamp'], datetime)
    
    def test_add_handoff(self, mock_database, mock_collection):
        """Test adding handoff record."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        mock_update_result = Mock(spec=UpdateResult)
        mock_collection.update_one.return_value = mock_update_result
        
        # Add handoff
        repo = ConversationRepository(mock_database)
        result = repo.add_handoff(
            conversation_id="507f1f77bcf86cd799439011",
            from_agent=AgentName.QUALIFIER.value,
            to_agent=AgentName.CLOSER.value,
            reason="Qualified lead ready for closing"
        )
        
        # Assert
        mock_collection.update_one.assert_called_once()
        update_arg = mock_collection.update_one.call_args[0][1]
        
        assert '$push' in update_arg
        handoff = update_arg['$push']['handoffs']
        assert handoff['from_agent'] == AgentName.QUALIFIER.value
        assert handoff['to_agent'] == AgentName.CLOSER.value
        assert handoff['reason'] == "Qualified lead ready for closing"
        assert isinstance(handoff['timestamp'], datetime)