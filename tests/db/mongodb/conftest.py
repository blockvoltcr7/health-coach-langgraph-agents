"""Pytest fixtures for MongoDB tests."""

import pytest
from typing import Generator
from unittest.mock import Mock, patch

from pymongo.database import Database
from pymongo.collection import Collection
from bson import ObjectId

from app.db.mongodb.config import MongoDBConfig
from app.db.mongodb.client import MongoDBClientSingleton


@pytest.fixture
def mock_mongodb_config():
    """Mock MongoDB configuration."""
    config = Mock(spec=MongoDBConfig)
    config.connection_uri = "mongodb://test:test@localhost:27017/"
    config.database_name = "test_limitless_os_sales"
    config.conversations_collection = "conversations"
    config.connection_options = {
        'maxPoolSize': 50,
        'minPoolSize': 10,
        'connectTimeoutMS': 10000,
        'serverSelectionTimeoutMS': 5000,
    }
    return config


@pytest.fixture
def mock_mongodb_client():
    """Mock MongoDB client."""
    client = Mock()
    
    # Mock admin commands
    client.admin.command.return_value = {
        'ok': 1.0,
        'version': '6.0.0'
    }
    
    # Mock server_info
    client.server_info.return_value = {
        'version': '6.0.0',
        'ok': 1.0
    }
    
    return client


@pytest.fixture
def mock_database(mock_mongodb_client):
    """Mock MongoDB database."""
    db = Mock(spec=Database)
    db.name = "test_limitless_os_sales"
    db.client = mock_mongodb_client
    
    # Mock list_collection_names
    db.list_collection_names.return_value = []
    
    # Mock command
    db.command.return_value = {'ok': 1.0}
    
    return db


@pytest.fixture
def mock_collection():
    """Mock MongoDB collection."""
    collection = Mock(spec=Collection)
    collection.name = "conversations"
    
    # Mock insert operations
    collection.insert_one.return_value = Mock(
        inserted_id=ObjectId('507f1f77bcf86cd799439011')
    )
    collection.insert_many.return_value = Mock(
        inserted_ids=[
            ObjectId('507f1f77bcf86cd799439011'),
            ObjectId('507f1f77bcf86cd799439012')
        ]
    )
    
    # Mock find operations
    collection.find_one.return_value = {
        '_id': ObjectId('507f1f77bcf86cd799439011'),
        'user_id': ObjectId('507f1f77bcf86cd799439012'),
        'status': 'active'
    }
    
    # Mock update operations
    collection.update_one.return_value = Mock(
        matched_count=1,
        modified_count=1
    )
    collection.update_many.return_value = Mock(
        matched_count=5,
        modified_count=5
    )
    
    # Mock delete operations
    collection.delete_one.return_value = Mock(deleted_count=1)
    collection.delete_many.return_value = Mock(deleted_count=5)
    
    # Mock count
    collection.count_documents.return_value = 10
    
    # Mock aggregate
    collection.aggregate.return_value = [
        {'_id': 'lead', 'count': 5},
        {'_id': 'qualified', 'count': 3}
    ]
    
    # Mock index operations
    collection.create_index.return_value = 'test_index'
    collection.list_indexes.return_value = [
        {'name': '_id_'},
        {'name': 'user_id_1_updated_at_-1'}
    ]
    
    return collection


@pytest.fixture
def sample_conversation_doc():
    """Sample conversation document for testing."""
    from datetime import datetime
    
    return {
        '_id': ObjectId('507f1f77bcf86cd799439011'),
        'user_id': ObjectId('507f1f77bcf86cd799439012'),
        'channel': 'web',
        'status': 'active',
        'sales_stage': 'lead',
        'stage_history': [{
            'stage': 'lead',
            'timestamp': datetime.utcnow(),
            'notes': 'Conversation initiated'
        }],
        'qualification': {
            'budget': False,
            'authority': False,
            'need': False,
            'timeline': False,
            'score': 0.0,
            'notes': ''
        },
        'messages': [{
            'role': 'user',
            'content': 'Hello, I want to learn about Limitless OS',
            'timestamp': datetime.utcnow()
        }],
        'objections': [],
        'handoffs': [],
        'metadata': {'source': 'website'},
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the MongoDB singleton before each test."""
    MongoDBClientSingleton._instance = None
    MongoDBClientSingleton._client = None
    MongoDBClientSingleton._config = None
    yield
    # Cleanup after test
    MongoDBClientSingleton._instance = None
    MongoDBClientSingleton._client = None
    MongoDBClientSingleton._config = None