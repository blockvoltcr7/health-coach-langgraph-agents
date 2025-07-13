"""Async base repository pattern for MongoDB operations using motor.

This module provides a generic async base class for MongoDB CRUD operations
that can be extended by specific repositories.
"""

from typing import TypeVar, Generic, Dict, List, Optional, Any
from datetime import datetime, timezone
from abc import ABC, abstractmethod

from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo.results import (
    InsertOneResult,
    InsertManyResult,
    UpdateResult,
    DeleteResult
)
from bson import ObjectId

from app.db.mongodb.async_client import get_async_database

T = TypeVar('T', bound=Dict[str, Any])


class AsyncBaseRepository(Generic[T], ABC):
    """Abstract async base repository for MongoDB operations.
    
    This class provides common async CRUD operations for MongoDB collections
    with support for pagination, filtering, and aggregation.
    """
    
    def __init__(self, database: Optional[AsyncIOMotorDatabase] = None):
        """Initialize the async repository.
        
        Args:
            database: Async MongoDB database instance. Uses default if not provided.
        """
        self._database = database
        self._collection: Optional[AsyncIOMotorCollection] = None
    
    async def _ensure_database(self) -> AsyncIOMotorDatabase:
        """Ensure database is available."""
        if self._database is None:
            self._database = await get_async_database()
        return self._database
    
    async def _ensure_collection(self) -> AsyncIOMotorCollection:
        """Ensure collection is available."""
        if self._collection is None:
            db = await self._ensure_database()
            self._collection = db[self.collection_name]
        return self._collection
    
    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Get the collection name.
        
        Returns:
            str: Name of the MongoDB collection
        """
        pass
    
    @property
    async def collection(self) -> AsyncIOMotorCollection:
        """Get the async MongoDB collection instance.
        
        Returns:
            AsyncIOMotorCollection: Async MongoDB collection
        """
        return await self._ensure_collection()
    
    async def create_one(self, document: T) -> InsertOneResult:
        """Insert a single document.
        
        Args:
            document: Document to insert
            
        Returns:
            InsertOneResult: Result of the insert operation
        """
        collection = await self._ensure_collection()
        
        # Add timestamps if not present
        now = datetime.now(timezone.utc)
        if 'created_at' not in document:
            document['created_at'] = now.isoformat() + "Z"
        if 'updated_at' not in document:
            document['updated_at'] = now.isoformat() + "Z"
            
        return await collection.insert_one(document)
    
    async def create_many(self, documents: List[T]) -> InsertManyResult:
        """Insert multiple documents.
        
        Args:
            documents: List of documents to insert
            
        Returns:
            InsertManyResult: Result of the insert operation
        """
        collection = await self._ensure_collection()
        
        # Add timestamps to all documents
        now = datetime.now(timezone.utc)
        for doc in documents:
            if 'created_at' not in doc:
                doc['created_at'] = now.isoformat() + "Z"
            if 'updated_at' not in doc:
                doc['updated_at'] = now.isoformat() + "Z"
                
        return await collection.insert_many(documents)
    
    async def find_one(self, filter: Dict[str, Any]) -> Optional[T]:
        """Find a single document.
        
        Args:
            filter: Query filter
            
        Returns:
            Optional[T]: Found document or None
        """
        collection = await self._ensure_collection()
        return await collection.find_one(filter)
    
    async def find_by_id(self, id: str) -> Optional[T]:
        """Find a document by its ID.
        
        Args:
            id: Document ID (string representation)
            
        Returns:
            Optional[T]: Found document or None
        """
        collection = await self._ensure_collection()
        try:
            return await collection.find_one({'_id': ObjectId(id)})
        except Exception:
            # Handle invalid ObjectId format
            return None
    
    async def find_many(
        self,
        filter: Dict[str, Any],
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort: Optional[List[tuple]] = None
    ) -> List[T]:
        """Find multiple documents with pagination support.
        
        Args:
            filter: Query filter
            limit: Maximum number of documents to return
            skip: Number of documents to skip
            sort: List of (field, direction) tuples for sorting
            
        Returns:
            List[T]: List of found documents
        """
        collection = await self._ensure_collection()
        cursor = collection.find(filter)
        
        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)
            
        return await cursor.to_list(length=limit)
    
    async def update_one(
        self,
        filter: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False
    ) -> UpdateResult:
        """Update a single document.
        
        Args:
            filter: Query filter
            update: Update operations
            upsert: Create document if it doesn't exist
            
        Returns:
            UpdateResult: Result of the update operation
        """
        collection = await self._ensure_collection()
        
        # Add updated_at timestamp
        now = datetime.now(timezone.utc)
        if '$set' not in update:
            update['$set'] = {}
        update['$set']['updated_at'] = now.isoformat() + "Z"
        
        return await collection.update_one(filter, update, upsert=upsert)
    
    async def update_by_id(
        self,
        id: str,
        update: Dict[str, Any],
        upsert: bool = False
    ) -> UpdateResult:
        """Update a document by its ID.
        
        Args:
            id: Document ID (string representation)
            update: Update operations
            upsert: Create document if it doesn't exist
            
        Returns:
            UpdateResult: Result of the update operation
        """
        try:
            return await self.update_one({'_id': ObjectId(id)}, update, upsert)
        except Exception:
            # Handle invalid ObjectId format
            return UpdateResult(acknowledged=False, matched_count=0, modified_count=0, upserted_id=None)
    
    async def update_many(
        self,
        filter: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False
    ) -> UpdateResult:
        """Update multiple documents.
        
        Args:
            filter: Query filter
            update: Update operations
            upsert: Create documents if they don't exist
            
        Returns:
            UpdateResult: Result of the update operation
        """
        collection = await self._ensure_collection()
        
        # Add updated_at timestamp
        now = datetime.now(timezone.utc)
        if '$set' not in update:
            update['$set'] = {}
        update['$set']['updated_at'] = now.isoformat() + "Z"
        
        return await collection.update_many(filter, update, upsert=upsert)
    
    async def delete_one(self, filter: Dict[str, Any]) -> DeleteResult:
        """Delete a single document.
        
        Args:
            filter: Query filter
            
        Returns:
            DeleteResult: Result of the delete operation
        """
        collection = await self._ensure_collection()
        return await collection.delete_one(filter)
    
    async def delete_by_id(self, id: str) -> DeleteResult:
        """Delete a document by its ID.
        
        Args:
            id: Document ID (string representation)
            
        Returns:
            DeleteResult: Result of the delete operation
        """
        collection = await self._ensure_collection()
        try:
            return await collection.delete_one({'_id': ObjectId(id)})
        except Exception:
            # Handle invalid ObjectId format
            return DeleteResult(acknowledged=False, deleted_count=0)
    
    async def delete_many(self, filter: Dict[str, Any]) -> DeleteResult:
        """Delete multiple documents.
        
        Args:
            filter: Query filter
            
        Returns:
            DeleteResult: Result of the delete operation
        """
        collection = await self._ensure_collection()
        return await collection.delete_many(filter)
    
    async def count_documents(self, filter: Dict[str, Any]) -> int:
        """Count documents matching the filter.
        
        Args:
            filter: Query filter
            
        Returns:
            int: Number of matching documents
        """
        collection = await self._ensure_collection()
        return await collection.count_documents(filter)
    
    async def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute an aggregation pipeline.
        
        Args:
            pipeline: Aggregation pipeline stages
            
        Returns:
            List[Dict[str, Any]]: Aggregation results
        """
        collection = await self._ensure_collection()
        cursor = collection.aggregate(pipeline)
        return await cursor.to_list(length=None)
    
    async def exists(self, filter: Dict[str, Any]) -> bool:
        """Check if a document exists.
        
        Args:
            filter: Query filter
            
        Returns:
            bool: True if document exists, False otherwise
        """
        collection = await self._ensure_collection()
        count = await collection.count_documents(filter, limit=1)
        return count > 0
    
    async def create_index(self, keys: List[tuple], **kwargs) -> str:
        """Create an index on the collection.
        
        Args:
            keys: List of (field, direction) tuples
            **kwargs: Additional index options
            
        Returns:
            str: Name of the created index
        """
        collection = await self._ensure_collection()
        return await collection.create_index(keys, **kwargs)
    
    async def drop_collection(self) -> None:
        """Drop the collection.
        
        Warning: This will delete all data in the collection!
        """
        collection = await self._ensure_collection()
        await collection.drop()