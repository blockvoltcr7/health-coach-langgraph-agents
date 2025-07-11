"""Integration tests for MongoDB implementation.

These tests require a real MongoDB connection and should be run manually
or in a CI environment with MongoDB available.
"""

import pytest
from datetime import datetime
from bson import ObjectId

from app.db.mongodb.client import get_mongodb_client, get_database, close_mongodb_connection
from app.db.mongodb.schemas.conversation_schema import (
    ConversationSchema,
    ConversationRepository,
    MessageRole,
    SalesStage,
    AgentName
)
from app.db.mongodb.utils import (
    initialize_database,
    validate_collection_schema,
    create_sample_conversation,
    get_database_stats,
    check_connection_health
)


@pytest.mark.integration
class TestMongoDBIntegration:
    """Integration tests that require real MongoDB connection."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Setup - use test database
        self.db = get_database("test_limitless_os_sales")
        
        yield
        
        # Teardown - clean up test data
        if "conversations" in self.db.list_collection_names():
            self.db.drop_collection("conversations")
        close_mongodb_connection()
    
    def test_database_initialization(self):
        """Test initializing the database."""
        # Initialize database
        results = initialize_database(self.db)
        
        # Assert
        assert results['database'] == 'test_limitless_os_sales'
        assert 'conversations' in results['collections_created']
        assert len(results['indexes_created']) >= 3  # At least 3 custom indexes
        assert len(results['errors']) == 0
        
        # Verify collection exists
        assert 'conversations' in self.db.list_collection_names()
    
    def test_connection_health(self):
        """Test health check functionality."""
        health = check_connection_health(self.db)
        
        assert health['status'] == 'healthy'
        assert health['connected'] is True
        assert 'version' in health
        assert health['database'] == 'test_limitless_os_sales'
    
    def test_conversation_crud_operations(self):
        """Test full CRUD operations on conversations."""
        # Initialize database
        initialize_database(self.db)
        
        # Create repository
        repo = ConversationRepository(self.db)
        
        # Create conversation
        doc = ConversationSchema.create_conversation_document(
            user_id="507f1f77bcf86cd799439012",
            channel="web",
            initial_message="I want to learn about Limitless OS",
            metadata={"mem0_user_id": "mem0_test123"}
        )
        
        result = repo.create_one(doc)
        conversation_id = str(result.inserted_id)
        assert conversation_id is not None
        
        # Read conversation
        conversation = repo.find_by_id(conversation_id)
        assert conversation is not None
        assert conversation['channel'] == 'web'
        assert len(conversation['messages']) == 1
        assert conversation['metadata']['mem0_user_id'] == 'mem0_test123'
        
        # Update - add message
        repo.add_message(
            conversation_id,
            MessageRole.SUPERVISOR.value,
            "Hello! I'd be happy to tell you about Limitless OS."
        )
        
        # Verify message added
        updated = repo.find_by_id(conversation_id)
        assert len(updated['messages']) == 2
        assert updated['messages'][1]['role'] == MessageRole.SUPERVISOR.value
        
        # Update sales stage
        repo.update_sales_stage(
            conversation_id,
            SalesStage.QUALIFIED.value,
            "Customer expressed interest and has budget"
        )
        
        # Verify stage updated
        updated = repo.find_by_id(conversation_id)
        assert updated['sales_stage'] == SalesStage.QUALIFIED.value
        assert len(updated['stage_history']) == 2
        
        # Add handoff
        repo.add_handoff(
            conversation_id,
            AgentName.SUPERVISOR.value,
            AgentName.QUALIFIER.value,
            "Initial interest detected, passing to qualifier"
        )
        
        # Verify handoff added
        updated = repo.find_by_id(conversation_id)
        assert len(updated['handoffs']) == 1
        assert updated['handoffs'][0]['to_agent'] == AgentName.QUALIFIER.value
        
        # Delete conversation
        delete_result = repo.delete_by_id(conversation_id)
        assert delete_result.deleted_count == 1
        
        # Verify deleted
        deleted = repo.find_by_id(conversation_id)
        assert deleted is None
    
    def test_find_active_conversations(self):
        """Test finding active conversations for a user."""
        # Initialize and create repository
        initialize_database(self.db)
        repo = ConversationRepository(self.db)
        
        user_id = "507f1f77bcf86cd799439012"
        
        # Create multiple conversations
        for i in range(3):
            doc = ConversationSchema.create_conversation_document(
                user_id=user_id,
                channel="web"
            )
            if i > 0:  # Make first one inactive
                doc['status'] = 'inactive'
            repo.create_one(doc)
        
        # Find active conversation
        active = repo.find_active_by_user(user_id)
        assert active is not None
        assert active['status'] == 'active'
        
        # Find by sales stage
        leads = repo.find_by_sales_stage('lead', limit=10)
        assert len(leads) == 3
    
    def test_sample_conversation_creation(self):
        """Test creating a sample conversation."""
        # Initialize database
        initialize_database(self.db)
        
        # Create sample
        user_id = "507f1f77bcf86cd799439012"
        conversation_id = create_sample_conversation(user_id, self.db)
        
        assert conversation_id is not None
        
        # Verify created
        repo = ConversationRepository(self.db)
        conversation = repo.find_by_id(conversation_id)
        
        assert conversation is not None
        assert str(conversation['user_id']) == user_id
        assert len(conversation['messages']) == 2  # Initial + response
        assert conversation['metadata']['mem0_user_id'] == f"mem0_{user_id}"
    
    def test_database_stats(self):
        """Test getting database statistics."""
        # Initialize and add some data
        initialize_database(self.db)
        repo = ConversationRepository(self.db)
        
        # Create a few conversations
        for i in range(5):
            doc = ConversationSchema.create_conversation_document(
                user_id=f"507f1f77bcf86cd79943901{i}",
                channel="web"
            )
            repo.create_one(doc)
        
        # Get stats
        stats = get_database_stats(self.db)
        
        assert stats['database'] == 'test_limitless_os_sales'
        assert 'conversations' in stats['collections']
        assert stats['collections']['conversations']['count'] == 5
        assert stats['total_documents'] >= 5
    
    def test_schema_validation(self):
        """Test schema validation enforcement."""
        # Initialize database with schema
        initialize_database(self.db)
        
        # Try to insert invalid document directly
        collection = self.db['conversations']
        
        # This should fail due to missing required fields
        with pytest.raises(Exception):  # PyMongo will raise WriteError
            collection.insert_one({
                "invalid_field": "test"
            })
        
        # This should fail due to invalid enum value
        with pytest.raises(Exception):
            collection.insert_one({
                "user_id": ObjectId(),
                "channel": "invalid_channel",  # Not in enum
                "status": "active",
                "sales_stage": "lead",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })


if __name__ == "__main__":
    # Run integration tests manually
    pytest.main([__file__, "-v", "-m", "integration"])