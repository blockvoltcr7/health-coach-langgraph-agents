"""Tests for MongoDB client implementation."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from pymongo.errors import ConnectionFailure

from app.db.mongodb.client import (
    MongoDBClientSingleton,
    get_mongodb_client,
    get_database,
    close_mongodb_connection
)


class TestMongoDBClientSingleton:
    """Test MongoDB client singleton implementation."""
    
    @patch('app.db.mongodb.client.get_mongodb_config')
    @patch('app.db.mongodb.client.MongoClient')
    def test_singleton_pattern(self, mock_mongo_client, mock_get_config, mock_mongodb_config):
        """Test that only one instance is created."""
        # Setup
        mock_get_config.return_value = mock_mongodb_config
        mock_client_instance = Mock()
        mock_client_instance.admin.command.return_value = {'ok': 1}
        mock_mongo_client.return_value = mock_client_instance
        
        # Create multiple instances
        instance1 = MongoDBClientSingleton()
        instance2 = MongoDBClientSingleton()
        
        # Assert same instance
        assert instance1 is instance2
        assert mock_mongo_client.call_count == 1
    
    @patch('app.db.mongodb.client.get_mongodb_config')
    @patch('app.db.mongodb.client.MongoClient')
    def test_successful_initialization(self, mock_mongo_client, mock_get_config, mock_mongodb_config):
        """Test successful client initialization."""
        # Setup
        mock_get_config.return_value = mock_mongodb_config
        mock_client_instance = Mock()
        mock_client_instance.admin.command.return_value = {'ok': 1}
        mock_mongo_client.return_value = mock_client_instance
        
        # Initialize
        client_singleton = MongoDBClientSingleton()
        
        # Assert
        assert client_singleton._client is not None
        mock_client_instance.admin.command.assert_called_once_with('ping')
    
    @patch('app.db.mongodb.client.get_mongodb_config')
    @patch('app.db.mongodb.client.MongoClient')
    def test_initialization_failure(self, mock_mongo_client, mock_get_config, mock_mongodb_config):
        """Test handling of initialization failure."""
        # Setup
        mock_get_config.return_value = mock_mongodb_config
        mock_mongo_client.side_effect = Exception("Connection failed")
        
        # Assert initialization fails
        with pytest.raises(ConnectionFailure):
            MongoDBClientSingleton()
    
    @patch('app.db.mongodb.client.get_mongodb_config')
    @patch('app.db.mongodb.client.MongoClient')
    def test_get_database(self, mock_mongo_client, mock_get_config, mock_mongodb_config, mock_database):
        """Test getting database instance."""
        # Setup
        mock_get_config.return_value = mock_mongodb_config
        mock_client_instance = MagicMock()
        mock_client_instance.admin.command.return_value = {'ok': 1}
        mock_client_instance.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client_instance
        
        # Get database
        client_singleton = MongoDBClientSingleton()
        db = client_singleton.get_database()
        
        # Assert
        mock_client_instance.__getitem__.assert_called_with(mock_mongodb_config.database_name)
        assert db == mock_database
    
    @patch('app.db.mongodb.client.get_mongodb_config')
    @patch('app.db.mongodb.client.MongoClient')
    def test_get_database_custom_name(self, mock_mongo_client, mock_get_config, mock_mongodb_config, mock_database):
        """Test getting database with custom name."""
        # Setup
        mock_get_config.return_value = mock_mongodb_config
        mock_client_instance = MagicMock()
        mock_client_instance.admin.command.return_value = {'ok': 1}
        mock_client_instance.__getitem__.return_value = mock_database
        mock_mongo_client.return_value = mock_client_instance
        
        # Get database with custom name
        client_singleton = MongoDBClientSingleton()
        db = client_singleton.get_database("custom_db")
        
        # Assert
        mock_client_instance.__getitem__.assert_called_with("custom_db")
    
    @patch('app.db.mongodb.client.get_mongodb_config')
    @patch('app.db.mongodb.client.MongoClient')
    def test_health_check_success(self, mock_mongo_client, mock_get_config, mock_mongodb_config):
        """Test successful health check."""
        # Setup
        mock_get_config.return_value = mock_mongodb_config
        mock_client_instance = Mock()
        mock_client_instance.admin.command.return_value = {'ok': 1}
        mock_client_instance.server_info.return_value = {'version': '6.0.0'}
        mock_mongo_client.return_value = mock_client_instance
        
        # Health check
        client_singleton = MongoDBClientSingleton()
        health = client_singleton.health_check()
        
        # Assert
        assert health['status'] == 'healthy'
        assert health['connected'] is True
        assert health['version'] == '6.0.0'
    
    @patch('app.db.mongodb.client.get_mongodb_config')
    @patch('app.db.mongodb.client.MongoClient')
    def test_health_check_failure(self, mock_mongo_client, mock_get_config, mock_mongodb_config):
        """Test health check when connection fails."""
        # Setup
        mock_get_config.return_value = mock_mongodb_config
        mock_client_instance = Mock()
        mock_client_instance.admin.command.side_effect = [{'ok': 1}, Exception("Connection lost")]
        mock_mongo_client.return_value = mock_client_instance
        
        # Health check
        client_singleton = MongoDBClientSingleton()
        health = client_singleton.health_check()
        
        # Assert
        assert health['status'] == 'unhealthy'
        assert health['connected'] is False
        assert 'error' in health
    
    @patch('app.db.mongodb.client.get_mongodb_config')
    @patch('app.db.mongodb.client.MongoClient')
    def test_reconnect(self, mock_mongo_client, mock_get_config, mock_mongodb_config):
        """Test reconnection functionality."""
        # Setup
        mock_get_config.return_value = mock_mongodb_config
        mock_client_instance1 = Mock()
        mock_client_instance1.admin.command.return_value = {'ok': 1}
        mock_client_instance2 = Mock()
        mock_client_instance2.admin.command.return_value = {'ok': 1}
        mock_mongo_client.side_effect = [mock_client_instance1, mock_client_instance2]
        
        # Initialize and reconnect
        client_singleton = MongoDBClientSingleton()
        assert client_singleton._client == mock_client_instance1
        
        client_singleton.reconnect()
        assert client_singleton._client == mock_client_instance2
        
        # Assert old connection was closed
        mock_client_instance1.close.assert_called_once()
    
    @patch('app.db.mongodb.client.get_mongodb_config')
    @patch('app.db.mongodb.client.MongoClient')
    def test_close(self, mock_mongo_client, mock_get_config, mock_mongodb_config):
        """Test closing connection."""
        # Setup
        mock_get_config.return_value = mock_mongodb_config
        mock_client_instance = Mock()
        mock_client_instance.admin.command.return_value = {'ok': 1}
        mock_mongo_client.return_value = mock_client_instance
        
        # Initialize and close
        client_singleton = MongoDBClientSingleton()
        client_singleton.close()
        
        # Assert
        mock_client_instance.close.assert_called_once()
        assert client_singleton._client is None


class TestModuleFunctions:
    """Test module-level convenience functions."""
    
    @patch('app.db.mongodb.client.MongoDBClientSingleton')
    def test_get_mongodb_client(self, mock_singleton_class):
        """Test get_mongodb_client function."""
        mock_instance = Mock()
        mock_singleton_class.return_value = mock_instance
        
        client = get_mongodb_client()
        
        assert client == mock_instance
    
    @patch('app.db.mongodb.client.get_mongodb_client')
    def test_get_database(self, mock_get_client):
        """Test get_database function."""
        mock_client = Mock()
        mock_database = Mock()
        mock_client.get_database.return_value = mock_database
        mock_get_client.return_value = mock_client
        
        db = get_database()
        
        mock_client.get_database.assert_called_once_with(None)
        assert db == mock_database
    
    @patch('app.db.mongodb.client.get_mongodb_client')
    def test_get_database_custom_name(self, mock_get_client):
        """Test get_database with custom name."""
        mock_client = Mock()
        mock_database = Mock()
        mock_client.get_database.return_value = mock_database
        mock_get_client.return_value = mock_client
        
        db = get_database("custom_db")
        
        mock_client.get_database.assert_called_once_with("custom_db")
    
    @patch('app.db.mongodb.client._mongodb_singleton')
    def test_close_mongodb_connection(self, mock_singleton):
        """Test closing MongoDB connection."""
        mock_singleton_instance = Mock()
        
        # Set global singleton
        import app.db.mongodb.client
        app.db.mongodb.client._mongodb_singleton = mock_singleton_instance
        
        close_mongodb_connection()
        
        mock_singleton_instance.close.assert_called_once()
        assert app.db.mongodb.client._mongodb_singleton is None