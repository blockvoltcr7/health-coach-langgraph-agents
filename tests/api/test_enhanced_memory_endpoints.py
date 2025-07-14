"""Tests for Enhanced Memory API Endpoints.

This module contains comprehensive tests for the enhanced memory endpoints
including categorization, importance scoring, analytics, and MongoDB snapshots.
"""

import pytest
import allure
from typing import Dict, List, Any
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from httpx import AsyncClient
from fastapi import status

from app.mem0.mem0AsyncClient import MemoryCategory, MemoryEntry


@allure.epic("Memory Management API")
@allure.feature("Enhanced Memory Endpoints")
class TestEnhancedMemoryEndpoints:
    """Test class for enhanced memory API endpoints."""
    
    @pytest.fixture
    def mock_mem0_client(self):
        """Create a mock Mem0 client for testing."""
        mock_client = AsyncMock()
        return mock_client
    
    @pytest.fixture
    def mock_memory_entries(self):
        """Create mock memory entries for testing."""
        return [
            MemoryEntry(
                id="mem_1",
                memory="Budget is $50,000",
                user_id="test_user",
                category=MemoryCategory.QUALIFICATION,
                importance_score=8.5,
                access_count=3,
                created_at=datetime.now(timezone.utc),
                metadata={"source": "chat"}
            ),
            MemoryEntry(
                id="mem_2",
                memory="Prefers morning meetings",
                user_id="test_user",
                category=MemoryCategory.PREFERENCE,
                importance_score=5.0,
                access_count=1,
                created_at=datetime.now(timezone.utc),
                metadata={"source": "chat"}
            )
        ]
    
    @allure.story("Add Enhanced Memory")
    @allure.severity(allure.severity_level.CRITICAL)
    async def test_add_enhanced_memory_success(self, client: AsyncClient, mock_mem0_client):
        """Test adding memory with category and importance score."""
        with allure.step("Mock memory client"):
            with patch('app.api.v1.endpoints.memory_endpoints.get_memory_client') as mock_get_client:
                mock_get_client.return_value = mock_mem0_client
                mock_mem0_client.add_memory.return_value = {"id": "mem_enhanced_123"}
        
        with allure.step("Prepare request data"):
            request_data = {
                "messages": [
                    {"role": "user", "content": "My budget is $100,000"},
                    {"role": "assistant", "content": "Budget of $100,000 noted"}
                ],
                "user_id": "test_user",
                "category": "qualification",
                "importance_score": 9.0,
                "metadata": {"source": "sales_call"}
            }
        
        with allure.step("Send request"):
            response = await client.post("/api/v1/memory/add-enhanced", json=request_data)
        
        with allure.step("Verify response"):
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True
            assert data["message"] == "Enhanced memory added successfully"
            assert "memory_id" in data
            
            # Verify the client was called correctly
            mock_mem0_client.add_memory.assert_called_once()
            call_args = mock_mem0_client.add_memory.call_args
            assert call_args.kwargs["category"] == MemoryCategory.QUALIFICATION
            assert call_args.kwargs["importance_score"] == 9.0
    
    @allure.story("Add Enhanced Memory")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_add_enhanced_memory_auto_category(self, client: AsyncClient, mock_mem0_client):
        """Test adding memory with auto-categorization."""
        with allure.step("Mock memory client"):
            with patch('app.api.v1.endpoints.memory_endpoints.get_memory_client') as mock_get_client:
                mock_get_client.return_value = mock_mem0_client
                mock_mem0_client.add_memory.return_value = {"id": "mem_auto_123"}
        
        with allure.step("Prepare request without category"):
            request_data = {
                "messages": [
                    {"role": "user", "content": "I'm worried about the price"},
                    {"role": "assistant", "content": "I understand your price concern"}
                ],
                "user_id": "test_user",
                "importance_score": 7.0
            }
        
        with allure.step("Send request"):
            response = await client.post("/api/v1/memory/add-enhanced", json=request_data)
        
        with allure.step("Verify auto-categorization"):
            assert response.status_code == status.HTTP_200_OK
            # The category should be auto-detected, check if add_memory was called
            mock_mem0_client.add_memory.assert_called_once()
    
    @allure.story("Search Enhanced Memories")
    @allure.severity(allure.severity_level.CRITICAL)
    async def test_search_enhanced_memories_with_filters(self, client: AsyncClient, mock_mem0_client, mock_memory_entries):
        """Test searching memories with category and importance filters."""
        from app.mem0.mem0AsyncClient import MemorySearchResult
        
        with allure.step("Mock memory client"):
            with patch('app.api.v1.endpoints.memory_endpoints.get_memory_client') as mock_get_client:
                mock_get_client.return_value = mock_mem0_client
                mock_search_result = MemorySearchResult(
                    memories=mock_memory_entries[:1],  # Only qualification memory
                    total_count=1,
                    query="budget"
                )
                mock_mem0_client.search_memories.return_value = mock_search_result
        
        with allure.step("Prepare search request"):
            request_data = {
                "query": "budget",
                "user_id": "test_user",
                "limit": 10,
                "category": "qualification",
                "min_importance_score": 7.0
            }
        
        with allure.step("Send request"):
            response = await client.post("/api/v1/memory/search-enhanced", json=request_data)
        
        with allure.step("Verify filtered results"):
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True
            assert data["total_count"] == 1
            assert len(data["memories"]) == 1
            assert data["memories"][0]["category"] == "qualification"
            
            # Verify search was called with filters
            mock_mem0_client.search_memories.assert_called_once_with(
                query="budget",
                user_id="test_user",
                limit=10,
                category=MemoryCategory.QUALIFICATION,
                min_importance_score=7.0
            )
    
    @allure.story("Get Memories by Importance")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_get_memories_by_importance(self, client: AsyncClient, mock_mem0_client, mock_memory_entries):
        """Test retrieving memories sorted by importance score."""
        with allure.step("Mock memory client"):
            with patch('app.api.v1.endpoints.memory_endpoints.get_memory_client') as mock_get_client:
                mock_get_client.return_value = mock_mem0_client
                # Sort by importance score descending
                sorted_memories = sorted(mock_memory_entries, key=lambda m: m.importance_score, reverse=True)
                mock_mem0_client.get_memories_by_importance.return_value = sorted_memories
        
        with allure.step("Send request with filters"):
            response = await client.get(
                "/api/v1/memory/by-importance/test_user",
                params={
                    "limit": 20,
                    "min_score": 5.0,
                    "include_decay": "true"
                }
            )
        
        with allure.step("Verify sorted results"):
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True
            assert len(data["memories"]) == 2
            # First memory should have higher importance score
            assert data["memories"][0]["importance_score"] > data["memories"][1]["importance_score"]
            
            # Verify method was called with correct parameters
            mock_mem0_client.get_memories_by_importance.assert_called_once_with(
                user_id="test_user",
                limit=20,
                min_score=5.0,
                category=None,
                include_decay=True
            )
    
    @allure.story("Get Memories by Importance")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_get_memories_by_importance_with_category(self, client: AsyncClient, mock_mem0_client):
        """Test retrieving memories by importance with category filter."""
        with allure.step("Mock memory client"):
            with patch('app.api.v1.endpoints.memory_endpoints.get_memory_client') as mock_get_client:
                mock_get_client.return_value = mock_mem0_client
                mock_mem0_client.get_memories_by_importance.return_value = []
        
        with allure.step("Send request with category filter"):
            response = await client.get(
                "/api/v1/memory/by-importance/test_user",
                params={
                    "category": "objection",
                    "limit": 10
                }
            )
        
        with allure.step("Verify category filter applied"):
            assert response.status_code == status.HTTP_200_OK
            mock_mem0_client.get_memories_by_importance.assert_called_once_with(
                user_id="test_user",
                limit=10,
                min_score=0.0,
                category=MemoryCategory.OBJECTION,
                include_decay=True
            )
    
    @allure.story("Memory Analytics")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_get_memory_analytics_success(self, client: AsyncClient, mock_mem0_client):
        """Test retrieving memory analytics."""
        with allure.step("Mock analytics data"):
            analytics_data = {
                "total_memories": 50,
                "categories": {
                    "fact": 15,
                    "preference": 10,
                    "objection": 8,
                    "outcome": 5,
                    "context": 10,
                    "qualification": 2
                },
                "avg_importance_score": 6.5,
                "avg_access_count": 3.2,
                "importance_distribution": {
                    "high": 10,
                    "medium": 25,
                    "low": 15
                },
                "most_accessed": [
                    {
                        "memory": "Budget is $50,000",
                        "access_count": 15,
                        "importance_score": 8.5
                    }
                ]
            }
        
        with allure.step("Mock memory client"):
            with patch('app.api.v1.endpoints.memory_endpoints.get_memory_client') as mock_get_client:
                mock_get_client.return_value = mock_mem0_client
                mock_mem0_client.get_memory_analytics.return_value = analytics_data
        
        with allure.step("Send request"):
            response = await client.get("/api/v1/memory/analytics/test_user")
        
        with allure.step("Verify analytics response"):
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total_memories"] == 50
            assert data["categories"]["fact"] == 15
            assert data["avg_importance_score"] == 6.5
            assert data["avg_access_count"] == 3.2
            assert len(data["most_accessed"]) == 1
    
    @allure.story("Memory Analytics")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_get_memory_analytics_error(self, client: AsyncClient, mock_mem0_client):
        """Test analytics error handling."""
        with allure.step("Mock analytics error"):
            with patch('app.api.v1.endpoints.memory_endpoints.get_memory_client') as mock_get_client:
                mock_get_client.return_value = mock_mem0_client
                mock_mem0_client.get_memory_analytics.return_value = {
                    "error": "Database connection failed"
                }
        
        with allure.step("Send request"):
            response = await client.get("/api/v1/memory/analytics/test_user")
        
        with allure.step("Verify error response"):
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            data = response.json()
            assert "Analytics error" in data["detail"]
    
    @allure.story("MongoDB Snapshot")
    @allure.severity(allure.severity_level.CRITICAL)
    async def test_create_memory_snapshot_success(self, client: AsyncClient, mock_mem0_client):
        """Test creating a MongoDB memory snapshot."""
        with allure.step("Mock snapshot creation"):
            with patch('app.api.v1.endpoints.memory_endpoints.get_memory_client') as mock_get_client:
                mock_get_client.return_value = mock_mem0_client
                mock_mem0_client.sync_to_mongodb.return_value = {
                    "success": True,
                    "memory_count": 25,
                    "snapshot_id": "snapshot_123",
                    "categories": {
                        "fact": 10,
                        "preference": 8,
                        "objection": 3,
                        "outcome": 2,
                        "context": 2,
                        "qualification": 0
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
        
        with allure.step("Send request"):
            response = await client.post("/api/v1/memory/snapshot/test_user")
        
        with allure.step("Verify snapshot response"):
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True
            assert data["message"] == "Memory snapshot created successfully"
            assert data["memory_count"] == 25
            assert "data" in data
            assert data["data"]["snapshot_id"] == "snapshot_123"
    
    @allure.story("MongoDB Snapshot")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_create_memory_snapshot_failure(self, client: AsyncClient, mock_mem0_client):
        """Test snapshot creation failure handling."""
        with allure.step("Mock snapshot failure"):
            with patch('app.api.v1.endpoints.memory_endpoints.get_memory_client') as mock_get_client:
                mock_get_client.return_value = mock_mem0_client
                mock_mem0_client.sync_to_mongodb.return_value = {
                    "success": False,
                    "message": "No memories found for user",
                    "memory_count": 0
                }
        
        with allure.step("Send request"):
            response = await client.post("/api/v1/memory/snapshot/test_user")
        
        with allure.step("Verify failure response"):
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is False
            assert data["message"] == "No memories found for user"
            assert data["memory_count"] == 0
    
    @allure.story("Error Handling")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_endpoint_validation_errors(self, client: AsyncClient, mock_mem0_client):
        """Test validation errors for enhanced endpoints."""
        with allure.step("Test invalid importance score"):
            request_data = {
                "messages": [{"role": "user", "content": "test"}],
                "user_id": "test_user",
                "importance_score": 15.0  # Invalid: > 10.0
            }
            response = await client.post("/api/v1/memory/add-enhanced", json=request_data)
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        with allure.step("Test invalid category"):
            request_data = {
                "messages": [{"role": "user", "content": "test"}],
                "user_id": "test_user",
                "category": "invalid_category"
            }
            response = await client.post("/api/v1/memory/add-enhanced", json=request_data)
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        with allure.step("Test negative limit"):
            request_data = {
                "query": "test",
                "user_id": "test_user",
                "limit": -1
            }
            response = await client.post("/api/v1/memory/search-enhanced", json=request_data)
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY