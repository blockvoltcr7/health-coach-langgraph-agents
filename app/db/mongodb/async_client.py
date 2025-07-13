"""Async MongoDB client implementation using motor.

This module provides an async MongoDB client with singleton pattern for
non-blocking database operations in FastAPI applications.
"""

import logging
import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import motor.motor_asyncio
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pymongo.server_api import ServerApi

from app.db.mongodb.config import MongoDBConfig, get_mongodb_config

logger = logging.getLogger(__name__)


class AsyncMongoDBClientSingleton:
    """Thread-safe singleton async MongoDB client manager.
    
    This class ensures a single async MongoDB client instance is used throughout
    the application lifecycle with proper connection management.
    """
    
    _instance: Optional['AsyncMongoDBClientSingleton'] = None
    _lock: asyncio.Lock = asyncio.Lock()
    _client: Optional[AsyncIOMotorClient] = None
    _config: Optional[MongoDBConfig] = None
    
    def __new__(cls) -> 'AsyncMongoDBClientSingleton':
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def initialize(self):
        """Initialize the async MongoDB client."""
        async with self._lock:
            # Only initialize once
            if self._client is not None:
                return
                
            self._config = get_mongodb_config()
            await self._initialize_client()
    
    async def _initialize_client(self) -> None:
        """Initialize the async MongoDB client with configuration."""
        try:
            logger.info("Initializing async MongoDB client...")
            
            # Create AsyncIOMotorClient with all configuration options
            self._client = AsyncIOMotorClient(
                self._config.connection_uri,
                server_api=ServerApi('1'),
                **self._config.connection_options
            )
            
            # Test the connection
            await self._client.admin.command('ping')
            logger.info("Successfully connected to MongoDB (async)!")
            
        except Exception as e:
            logger.error(f"Failed to initialize async MongoDB client: {e}")
            raise ConnectionFailure(f"Could not connect to MongoDB: {e}")
    
    @property
    def client(self) -> AsyncIOMotorClient:
        """Get the async MongoDB client instance.
        
        Returns:
            AsyncIOMotorClient: The active async MongoDB client
            
        Raises:
            ConnectionFailure: If client is not initialized
        """
        if self._client is None:
            raise ConnectionFailure("Async MongoDB client is not initialized")
        return self._client
    
    def get_database(self, database_name: Optional[str] = None) -> AsyncIOMotorDatabase:
        """Get an async database instance.
        
        Args:
            database_name: Name of the database. Uses config default if not provided.
            
        Returns:
            AsyncIOMotorDatabase: Async MongoDB database instance
        """
        db_name = database_name or self._config.database_name
        return self.client[db_name]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the MongoDB connection.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        try:
            # Ping the server
            result = await self.client.admin.command('ping')
            
            # Get server info
            server_info = await self.client.server_info()
            
            return {
                'status': 'healthy',
                'ping': result,
                'version': server_info.get('version', 'unknown'),
                'connected': True
            }
        except Exception as e:
            logger.error(f"Async health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'connected': False
            }
    
    async def close(self) -> None:
        """Close the async MongoDB connection gracefully."""
        if self._client is not None:
            logger.info("Closing async MongoDB connection...")
            self._client.close()
            self._client = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Module-level convenience functions

_async_mongodb_singleton: Optional[AsyncMongoDBClientSingleton] = None


async def get_async_mongodb_client() -> AsyncMongoDBClientSingleton:
    """Get the async MongoDB client singleton instance.
    
    Returns:
        AsyncMongoDBClientSingleton: The async MongoDB client singleton
    """
    global _async_mongodb_singleton
    if _async_mongodb_singleton is None:
        _async_mongodb_singleton = AsyncMongoDBClientSingleton()
        await _async_mongodb_singleton.initialize()
    return _async_mongodb_singleton


async def get_async_database(database_name: Optional[str] = None) -> AsyncIOMotorDatabase:
    """Get an async MongoDB database instance.
    
    Args:
        database_name: Name of the database. Uses config default if not provided.
        
    Returns:
        AsyncIOMotorDatabase: Async MongoDB database instance
    """
    client = await get_async_mongodb_client()
    return client.get_database(database_name)


async def close_async_mongodb_connection() -> None:
    """Close the async MongoDB connection.
    
    This should be called during application shutdown.
    """
    global _async_mongodb_singleton
    if _async_mongodb_singleton is not None:
        await _async_mongodb_singleton.close()
        _async_mongodb_singleton = None


@asynccontextmanager
async def async_mongodb_transaction(database: Optional[AsyncIOMotorDatabase] = None):
    """Async context manager for MongoDB transactions.
    
    Args:
        database: Database instance. Uses default if not provided.
        
    Yields:
        session: Async MongoDB session for transaction
    """
    db = database or await get_async_database()
    client = (await get_async_mongodb_client()).client
    
    async with await client.start_session() as session:
        async with session.start_transaction():
            yield session