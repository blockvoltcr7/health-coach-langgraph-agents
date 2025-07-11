"""Tests for base repository pattern."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from bson import ObjectId
from pymongo.results import (
    InsertOneResult,
    InsertManyResult,
    UpdateResult,
    DeleteResult
)

from app.db.mongodb.base_repository import BaseRepository


class ConcreteRepository(BaseRepository):
    """Concrete implementation for testing."""
    
    @property
    def collection_name(self) -> str:
        return "test_collection"


class TestBaseRepository:
    """Test BaseRepository implementation."""
    
    def test_initialization(self, mock_database, mock_collection):
        """Test repository initialization."""
        mock_database.__getitem__.return_value = mock_collection
        
        repo = ConcreteRepository(mock_database)
        
        assert repo._database == mock_database
        assert repo.collection == mock_collection
        mock_database.__getitem__.assert_called_once_with("test_collection")
    
    def test_initialization_default_database(self, mock_database, mock_collection):
        """Test repository initialization with default database."""
        mock_database.__getitem__.return_value = mock_collection
        
        with patch('app.db.mongodb.base_repository.get_database', return_value=mock_database):
            repo = ConcreteRepository()
        
        assert repo._database == mock_database
    
    def test_create_one(self, mock_database, mock_collection):
        """Test creating single document."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        mock_result = Mock(spec=InsertOneResult)
        mock_result.inserted_id = ObjectId()
        mock_collection.insert_one.return_value = mock_result
        
        # Create document
        repo = ConcreteRepository(mock_database)
        doc = {"name": "test", "value": 123}
        result = repo.create_one(doc)
        
        # Assert
        assert result == mock_result
        mock_collection.insert_one.assert_called_once()
        inserted_doc = mock_collection.insert_one.call_args[0][0]
        assert inserted_doc["name"] == "test"
        assert inserted_doc["value"] == 123
        assert "created_at" in inserted_doc
        assert "updated_at" in inserted_doc
        assert isinstance(inserted_doc["created_at"], datetime)
        assert isinstance(inserted_doc["updated_at"], datetime)
    
    def test_create_one_preserves_timestamps(self, mock_database, mock_collection):
        """Test that existing timestamps are preserved."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        existing_time = datetime(2023, 1, 1)
        
        # Create document with existing timestamps
        repo = ConcreteRepository(mock_database)
        doc = {
            "name": "test",
            "created_at": existing_time,
            "updated_at": existing_time
        }
        repo.create_one(doc)
        
        # Assert timestamps preserved
        inserted_doc = mock_collection.insert_one.call_args[0][0]
        assert inserted_doc["created_at"] == existing_time
        assert inserted_doc["updated_at"] == existing_time
    
    def test_create_many(self, mock_database, mock_collection):
        """Test creating multiple documents."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        mock_result = Mock(spec=InsertManyResult)
        mock_result.inserted_ids = [ObjectId(), ObjectId()]
        mock_collection.insert_many.return_value = mock_result
        
        # Create documents
        repo = ConcreteRepository(mock_database)
        docs = [{"name": "test1"}, {"name": "test2"}]
        result = repo.create_many(docs)
        
        # Assert
        assert result == mock_result
        mock_collection.insert_many.assert_called_once()
        inserted_docs = mock_collection.insert_many.call_args[0][0]
        assert len(inserted_docs) == 2
        for doc in inserted_docs:
            assert "created_at" in doc
            assert "updated_at" in doc
    
    def test_find_one(self, mock_database, mock_collection):
        """Test finding single document."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        expected_doc = {"_id": ObjectId(), "name": "test"}
        mock_collection.find_one.return_value = expected_doc
        
        # Find document
        repo = ConcreteRepository(mock_database)
        result = repo.find_one({"name": "test"})
        
        # Assert
        assert result == expected_doc
        mock_collection.find_one.assert_called_once_with({"name": "test"})
    
    def test_find_by_id(self, mock_database, mock_collection):
        """Test finding document by ID."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        doc_id = "507f1f77bcf86cd799439011"
        expected_doc = {"_id": ObjectId(doc_id), "name": "test"}
        mock_collection.find_one.return_value = expected_doc
        
        # Find by ID
        repo = ConcreteRepository(mock_database)
        result = repo.find_by_id(doc_id)
        
        # Assert
        mock_collection.find_one.assert_called_once()
        filter_arg = mock_collection.find_one.call_args[0][0]
        assert isinstance(filter_arg["_id"], ObjectId)
        assert str(filter_arg["_id"]) == doc_id
    
    def test_find_many(self, mock_database, mock_collection):
        """Test finding multiple documents."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        mock_cursor = Mock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.skip.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.__iter__.return_value = iter([{"_id": 1}, {"_id": 2}])
        mock_collection.find.return_value = mock_cursor
        
        # Find documents
        repo = ConcreteRepository(mock_database)
        results = repo.find_many(
            filter={"status": "active"},
            limit=10,
            skip=5,
            sort=[("created_at", -1)]
        )
        
        # Assert
        assert len(results) == 2
        mock_collection.find.assert_called_once_with({"status": "active"})
        mock_cursor.sort.assert_called_once_with([("created_at", -1)])
        mock_cursor.skip.assert_called_once_with(5)
        mock_cursor.limit.assert_called_once_with(10)
    
    def test_update_one(self, mock_database, mock_collection):
        """Test updating single document."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        mock_result = Mock(spec=UpdateResult)
        mock_collection.update_one.return_value = mock_result
        
        # Update document
        repo = ConcreteRepository(mock_database)
        result = repo.update_one(
            filter={"name": "test"},
            update={"$set": {"value": 456}}
        )
        
        # Assert
        assert result == mock_result
        mock_collection.update_one.assert_called_once()
        filter_arg, update_arg = mock_collection.update_one.call_args[0]
        assert filter_arg == {"name": "test"}
        assert "$set" in update_arg
        assert "updated_at" in update_arg["$set"]
        assert isinstance(update_arg["$set"]["updated_at"], datetime)
    
    def test_update_by_id(self, mock_database, mock_collection):
        """Test updating document by ID."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        mock_result = Mock(spec=UpdateResult)
        mock_collection.update_one.return_value = mock_result
        doc_id = "507f1f77bcf86cd799439011"
        
        # Update by ID
        repo = ConcreteRepository(mock_database)
        result = repo.update_by_id(
            id=doc_id,
            update={"$inc": {"count": 1}}
        )
        
        # Assert
        filter_arg = mock_collection.update_one.call_args[0][0]
        assert isinstance(filter_arg["_id"], ObjectId)
        assert str(filter_arg["_id"]) == doc_id
    
    def test_update_many(self, mock_database, mock_collection):
        """Test updating multiple documents."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        mock_result = Mock(spec=UpdateResult)
        mock_result.matched_count = 5
        mock_result.modified_count = 5
        mock_collection.update_many.return_value = mock_result
        
        # Update documents
        repo = ConcreteRepository(mock_database)
        result = repo.update_many(
            filter={"status": "pending"},
            update={"$set": {"status": "processed"}}
        )
        
        # Assert
        assert result == mock_result
        mock_collection.update_many.assert_called_once()
        update_arg = mock_collection.update_many.call_args[0][1]
        assert "updated_at" in update_arg["$set"]
    
    def test_delete_one(self, mock_database, mock_collection):
        """Test deleting single document."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        mock_result = Mock(spec=DeleteResult)
        mock_result.deleted_count = 1
        mock_collection.delete_one.return_value = mock_result
        
        # Delete document
        repo = ConcreteRepository(mock_database)
        result = repo.delete_one({"name": "test"})
        
        # Assert
        assert result == mock_result
        mock_collection.delete_one.assert_called_once_with({"name": "test"})
    
    def test_delete_by_id(self, mock_database, mock_collection):
        """Test deleting document by ID."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        mock_result = Mock(spec=DeleteResult)
        mock_collection.delete_one.return_value = mock_result
        doc_id = "507f1f77bcf86cd799439011"
        
        # Delete by ID
        repo = ConcreteRepository(mock_database)
        result = repo.delete_by_id(doc_id)
        
        # Assert
        filter_arg = mock_collection.delete_one.call_args[0][0]
        assert isinstance(filter_arg["_id"], ObjectId)
        assert str(filter_arg["_id"]) == doc_id
    
    def test_delete_many(self, mock_database, mock_collection):
        """Test deleting multiple documents."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        mock_result = Mock(spec=DeleteResult)
        mock_result.deleted_count = 10
        mock_collection.delete_many.return_value = mock_result
        
        # Delete documents
        repo = ConcreteRepository(mock_database)
        result = repo.delete_many({"expired": True})
        
        # Assert
        assert result == mock_result
        mock_collection.delete_many.assert_called_once_with({"expired": True})
    
    def test_count_documents(self, mock_database, mock_collection):
        """Test counting documents."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        mock_collection.count_documents.return_value = 42
        
        # Count documents
        repo = ConcreteRepository(mock_database)
        count = repo.count_documents({"status": "active"})
        
        # Assert
        assert count == 42
        mock_collection.count_documents.assert_called_once_with({"status": "active"})
    
    def test_aggregate(self, mock_database, mock_collection):
        """Test aggregation pipeline."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        expected_results = [
            {"_id": "group1", "count": 10},
            {"_id": "group2", "count": 5}
        ]
        mock_collection.aggregate.return_value = expected_results
        
        # Run aggregation
        repo = ConcreteRepository(mock_database)
        pipeline = [
            {"$group": {"_id": "$category", "count": {"$sum": 1}}}
        ]
        results = repo.aggregate(pipeline)
        
        # Assert
        assert results == expected_results
        mock_collection.aggregate.assert_called_once_with(pipeline)
    
    def test_exists(self, mock_database, mock_collection):
        """Test document existence check."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        
        # Test exists
        mock_collection.count_documents.return_value = 1
        repo = ConcreteRepository(mock_database)
        exists = repo.exists({"name": "test"})
        assert exists is True
        mock_collection.count_documents.assert_called_with({"name": "test"}, limit=1)
        
        # Test not exists
        mock_collection.count_documents.return_value = 0
        not_exists = repo.exists({"name": "nonexistent"})
        assert not_exists is False
    
    def test_create_index(self, mock_database, mock_collection):
        """Test index creation."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        mock_collection.create_index.return_value = "test_index_name"
        
        # Create index
        repo = ConcreteRepository(mock_database)
        index_name = repo.create_index(
            [("field1", 1), ("field2", -1)],
            unique=True
        )
        
        # Assert
        assert index_name == "test_index_name"
        mock_collection.create_index.assert_called_once_with(
            [("field1", 1), ("field2", -1)],
            unique=True
        )
    
    def test_drop_collection(self, mock_database, mock_collection):
        """Test dropping collection."""
        # Setup
        mock_database.__getitem__.return_value = mock_collection
        
        # Drop collection
        repo = ConcreteRepository(mock_database)
        repo.drop_collection()
        
        # Assert
        mock_collection.drop.assert_called_once()