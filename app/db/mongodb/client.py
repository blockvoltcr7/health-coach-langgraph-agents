"""MongoDB client implementation with singleton pattern.

This module provides a thread-safe singleton MongoDB client with automatic
reconnection, health checks, and graceful shutdown capabilities.
"""

import logging
import threading
from typing import Optional, Dict, Any
from contextlib import contextmanager

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pymongo.server_api import ServerApi

from app.db.mongodb.config import MongoDBConfig, get_mongodb_config

logger = logging.getLogger(__name__)


class MongoDBClientSingleton:
    """Thread-safe singleton MongoDB client manager.
    
    This class ensures a single MongoDB client instance is used throughout
    the application lifecycle with proper connection management.
    """
    
    _instance: Optional['MongoDBClientSingleton'] = None
    _lock: threading.Lock = threading.Lock()
    _client: Optional[MongoClient] = None
    _config: Optional[MongoDBConfig] = None
    
    def __new__(cls) -> 'MongoDBClientSingleton':
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the MongoDB client singleton."""
        # Only initialize once
        if self._client is not None:
            return
            
        self._config = get_mongodb_config()
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the MongoDB client with configuration."""
        try:
            logger.info("Initializing MongoDB client...")
            
            # Create MongoClient with all configuration options
            self._client = MongoClient(
                self._config.connection_uri,
                server_api=ServerApi('1'),
                **self._config.connection_options
            )
            
            # Test the connection
            self._client.admin.command('ping')
            logger.info("Successfully connected to MongoDB!")
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB client: {e}")
            raise ConnectionFailure(f"Could not connect to MongoDB: {e}")
    
    @property
    def client(self) -> MongoClient:
        """Get the MongoDB client instance.
        
        Returns:
            MongoClient: The active MongoDB client
            
        Raises:
            ConnectionFailure: If client is not initialized
        """
        if self._client is None:
            raise ConnectionFailure("MongoDB client is not initialized")
        return self._client
    
    def get_database(self, database_name: Optional[str] = None) -> Database:
        """Get a database instance.
        
        Args:
            database_name: Name of the database. Uses config default if not provided.
            
        Returns:
            Database: MongoDB database instance
        """
        db_name = database_name or self._config.database_name
        return self.client[db_name]
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the MongoDB connection.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        try:
            # Ping the server
            result = self.client.admin.command('ping')
            
            # Get server info
            server_info = self.client.server_info()
            
            return {
                'status': 'healthy',
                'ping': result,
                'version': server_info.get('version', 'unknown'),
                'connected': True
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'connected': False
            }
    
    def reconnect(self) -> None:
        """Reconnect to MongoDB.
        
        This method closes the existing connection and creates a new one.
        """
        logger.info("Reconnecting to MongoDB...")
        
        # Close existing connection
        if self._client is not None:
            self._client.close()
            self._client = None
        
        # Initialize new connection
        self._initialize_client()
    
    def close(self) -> None:
        """Close the MongoDB connection gracefully."""
        if self._client is not None:
            logger.info("Closing MongoDB connection...")
            self._client.close()
            self._client = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# Module-level convenience functions

_mongodb_singleton: Optional[MongoDBClientSingleton] = None


def get_mongodb_client() -> MongoDBClientSingleton:
    """Get the MongoDB client singleton instance.
    
    Returns:
        MongoDBClientSingleton: The MongoDB client singleton
    """
    global _mongodb_singleton
    if _mongodb_singleton is None:
        _mongodb_singleton = MongoDBClientSingleton()
    return _mongodb_singleton


def get_database(database_name: Optional[str] = None) -> Database:
    """Get a MongoDB database instance.
    
    Args:
        database_name: Name of the database. Uses config default if not provided.
        
    Returns:
        Database: MongoDB database instance
    """
    client = get_mongodb_client()
    return client.get_database(database_name)


def close_mongodb_connection() -> None:
    """Close the MongoDB connection.
    
    This should be called during application shutdown.
    """
    global _mongodb_singleton
    if _mongodb_singleton is not None:
        _mongodb_singleton.close()
        _mongodb_singleton = None


@contextmanager
def mongodb_transaction(database: Optional[Database] = None):
    """Context manager for MongoDB transactions.
    
    Args:
        database: Database instance. Uses default if not provided.
        
    Yields:
        session: MongoDB session for transaction
    """
    db = database or get_database()
    client = get_mongodb_client().client
    
    with client.start_session() as session:
        with session.start_transaction():
            yield session