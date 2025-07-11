"""Base repository pattern for MongoDB operations.

This module provides a generic base class for MongoDB CRUD operations
that can be extended by specific repositories.
"""

from typing import TypeVar, Generic, Dict, List, Optional, Any, Type
from datetime import datetime
from abc import ABC, abstractmethod

from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.results import (
    InsertOneResult,
    InsertManyResult,
    UpdateResult,
    DeleteResult
)
from bson import ObjectId

from app.db.mongodb.client import get_database

T = TypeVar('T', bound=Dict[str, Any])


class BaseRepository(Generic[T], ABC):
    """Abstract base repository for MongoDB operations.
    
    This class provides common CRUD operations for MongoDB collections
    with support for pagination, filtering, and aggregation.
    """
    
    def __init__(self, database: Optional[Database] = None):
        """Initialize the repository.
        
        Args:
            database: MongoDB database instance. Uses default if not provided.
        """
        self._database = database or get_database()
        self._collection: Collection = self._database[self.collection_name]
    
    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Get the collection name.
        
        Returns:
            str: Name of the MongoDB collection
        """
        pass
    
    @property
    def collection(self) -> Collection:
        """Get the MongoDB collection instance.
        
        Returns:
            Collection: MongoDB collection
        """
        return self._collection
    
    def create_one(self, document: T) -> InsertOneResult:
        """Insert a single document.
        
        Args:
            document: Document to insert
            
        Returns:
            InsertOneResult: Result of the insert operation
        """
        # Add timestamps if not present
        if 'created_at' not in document:
            document['created_at'] = datetime.utcnow()
        if 'updated_at' not in document:
            document['updated_at'] = datetime.utcnow()
            
        return self._collection.insert_one(document)
    
    def create_many(self, documents: List[T]) -> InsertManyResult:
        """Insert multiple documents.
        
        Args:
            documents: List of documents to insert
            
        Returns:
            InsertManyResult: Result of the insert operation
        """
        # Add timestamps to all documents
        for doc in documents:
            if 'created_at' not in doc:
                doc['created_at'] = datetime.utcnow()
            if 'updated_at' not in doc:
                doc['updated_at'] = datetime.utcnow()
                
        return self._collection.insert_many(documents)
    
    def find_one(self, filter: Dict[str, Any]) -> Optional[T]:
        """Find a single document.
        
        Args:
            filter: Query filter
            
        Returns:
            Optional[T]: Found document or None
        """
        return self._collection.find_one(filter)
    
    def find_by_id(self, id: str) -> Optional[T]:
        """Find a document by its ID.
        
        Args:
            id: Document ID (string representation)
            
        Returns:
            Optional[T]: Found document or None
        """
        return self._collection.find_one({'_id': ObjectId(id)})
    
    def find_many(
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
        cursor = self._collection.find(filter)
        
        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)
            
        return list(cursor)
    
    def update_one(
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
        # Add updated_at timestamp
        if '$set' not in update:
            update['$set'] = {}
        update['$set']['updated_at'] = datetime.utcnow()
        
        return self._collection.update_one(filter, update, upsert=upsert)
    
    def update_by_id(
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
        return self.update_one({'_id': ObjectId(id)}, update, upsert)
    
    def update_many(
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
        # Add updated_at timestamp
        if '$set' not in update:
            update['$set'] = {}
        update['$set']['updated_at'] = datetime.utcnow()
        
        return self._collection.update_many(filter, update, upsert=upsert)
    
    def delete_one(self, filter: Dict[str, Any]) -> DeleteResult:
        """Delete a single document.
        
        Args:
            filter: Query filter
            
        Returns:
            DeleteResult: Result of the delete operation
        """
        return self._collection.delete_one(filter)
    
    def delete_by_id(self, id: str) -> DeleteResult:
        """Delete a document by its ID.
        
        Args:
            id: Document ID (string representation)
            
        Returns:
            DeleteResult: Result of the delete operation
        """
        return self._collection.delete_one({'_id': ObjectId(id)})
    
    def delete_many(self, filter: Dict[str, Any]) -> DeleteResult:
        """Delete multiple documents.
        
        Args:
            filter: Query filter
            
        Returns:
            DeleteResult: Result of the delete operation
        """
        return self._collection.delete_many(filter)
    
    def count_documents(self, filter: Dict[str, Any]) -> int:
        """Count documents matching the filter.
        
        Args:
            filter: Query filter
            
        Returns:
            int: Number of matching documents
        """
        return self._collection.count_documents(filter)
    
    def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute an aggregation pipeline.
        
        Args:
            pipeline: Aggregation pipeline stages
            
        Returns:
            List[Dict[str, Any]]: Aggregation results
        """
        return list(self._collection.aggregate(pipeline))
    
    def exists(self, filter: Dict[str, Any]) -> bool:
        """Check if a document exists.
        
        Args:
            filter: Query filter
            
        Returns:
            bool: True if document exists, False otherwise
        """
        return self._collection.count_documents(filter, limit=1) > 0
    
    def create_index(self, keys: List[tuple], **kwargs) -> str:
        """Create an index on the collection.
        
        Args:
            keys: List of (field, direction) tuples
            **kwargs: Additional index options
            
        Returns:
            str: Name of the created index
        """
        return self._collection.create_index(keys, **kwargs)
    
    def drop_collection(self) -> None:
        """Drop the collection.
        
        Warning: This will delete all data in the collection!
        """
        self._collection.drop()