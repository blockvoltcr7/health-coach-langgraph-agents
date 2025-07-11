"""Tests for Memory Management API Endpoints.

This module contains comprehensive tests for the memory management API endpoints
that use the reusable Mem0 async client wrapper.
"""

import pytest
import allure
from typing import Dict, List, Any
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from datetime import datetime

from app.main import app
from app.mem0.mem0AsyncClient import MemoryEntry, MemorySearchResult


@allure.epic("API Endpoints")
@allure.feature("Memory Management Endpoints")
class TestMemoryEndpoints:
    """Test class for memory management API endpoints."""
    
    @pytest.fixture
    def client(self) -> TestClient:
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_memory_client(self):
        """Create a mock memory client."""
        mock_client = AsyncMock()
        return mock_client
    
    @pytest.fixture
    def sample_memory_entries(self) -> List[MemoryEntry]:
        """Create sample memory entries for testing."""
        return [
            MemoryEntry(
                id="mem_1",
                memory="User prefers morning workouts",
                user_id="test_user_123",
                created_at=datetime(2024, 1, 1, 10, 0, 0),
                updated_at=datetime(2024, 1, 1, 10, 0, 0),
                metadata={"source": "chat"}
            ),
            MemoryEntry(
                id="mem_2",
                memory="User is a software developer",
                user_id="test_user_123",
                created_at=datetime(2024, 1, 2, 10, 0, 0),
                updated_at=datetime(2024, 1, 2, 10, 0, 0),
                metadata={"source": "profile"}
            )
        ]
    
    @allure.story("Memory Addition")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_add_memory_success(self, client: TestClient, mock_memory_client: AsyncMock):
        """Test successful memory addition via API."""
        with allure.step("Prepare test data"):
            request_data = {
                "messages": [
                    {"role": "user", "content": "I prefer morning workouts"},
                    {"role": "assistant", "content": "I'll remember you prefer morning workouts"}
                ],
                "user_id": "test_user_123",
                "metadata": {"source": "test"}
            }
            
            expected_response = {"id": "mem_123", "status": "success"}
        
        with allure.step("Mock memory client"):
            mock_memory_client.add_memory.return_value = expected_response
            
            with patch('app.api.v1.endpoints.memory_endpoints.get_mem0_client') as mock_get_client:
                mock_get_client.return_value = mock_memory_client
                
                response = client.post("/api/v1/memory/add", json=request_data)
        
        with allure.step("Verify response"):
            assert response.status_code == 200
            response_data = response.json()
            
            assert response_data["success"] is True
            assert "Memory successfully added" in response_data["message"]
            assert response_data["data"] == expected_response
            
            mock_memory_client.add_memory.assert_called_once_with(
                messages=request_data["messages"],
                user_id=request_data["user_id"],
                metadata=request_data["metadata"]
            )
    
    @allure.story("Memory Addition")
    @allure.severity(allure.severity_level.NORMAL)
    def test_add_memory_validation_error(self, client: TestClient, mock_memory_client: AsyncMock):
        """Test memory addition with validation errors."""
        with allure.step("Prepare invalid test data"):
            request_data = {
                "messages": [],  # Empty messages should cause validation error
                "user_id": "test_user_123"
            }
        
        with allure.step("Mock memory client to raise validation error"):
            mock_memory_client.add_memory.side_effect = ValueError("Messages cannot be empty")
            
            with patch('app.api.v1.endpoints.memory_endpoints.get_mem0_client') as mock_get_client:
                mock_get_client.return_value = mock_memory_client
                
                response = client.post("/api/v1/memory/add", json=request_data)
        
        with allure.step("Verify error response"):
            assert response.status_code == 400
            response_data = response.json()
            assert "Messages cannot be empty" in response_data["detail"]
    
    @allure.story("Conversation Addition")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_add_conversation_success(self, client: TestClient):
        """Test successful conversation addition via API."""
        with allure.step("Prepare test data"):
            request_data = {
                "user_message": "Hello, I'm interested in your services",
                "assistant_message": "Great! I'd be happy to help you learn about our services.",
                "user_id": "test_user_123",
                "metadata": {"source": "website_chat"}
            }
            
            expected_response = {"id": "conv_123", "status": "success"}
        
        with allure.step("Mock convenience function"):
            with patch('app.api.v1.endpoints.memory_endpoints.add_conversation_memory') as mock_add_conv:
                mock_add_conv.return_value = expected_response
                
                response = client.post("/api/v1/memory/add-conversation", json=request_data)
        
        with allure.step("Verify response"):
            assert response.status_code == 200
            response_data = response.json()
            
            assert response_data["success"] is True
            assert "Conversation successfully added" in response_data["message"]
            assert response_data["data"] == expected_response
            
            mock_add_conv.assert_called_once_with(
                user_message=request_data["user_message"],
                assistant_message=request_data["assistant_message"],
                user_id=request_data["user_id"],
                metadata=request_data["metadata"]
            )
    
    @allure.story("Memory Search")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_search_memories_success(
        self, 
        client: TestClient, 
        mock_memory_client: AsyncMock,
        sample_memory_entries: List[MemoryEntry]
    ):
        """Test successful memory search via API."""
        with allure.step("Prepare test data"):
            request_data = {
                "query": "workout preferences",
                "user_id": "test_user_123",
                "limit": 5
            }
            
            search_result = MemorySearchResult(
                memories=sample_memory_entries,
                total_count=2,
                query="workout preferences"
            )
        
        with allure.step("Mock memory client"):
            mock_memory_client.search_memories.return_value = search_result
            
            with patch('app.api.v1.endpoints.memory_endpoints.get_mem0_client') as mock_get_client:
                mock_get_client.return_value = mock_memory_client
                
                response = client.post("/api/v1/memory/search", json=request_data)
        
        with allure.step("Verify response"):
            assert response.status_code == 200
            response_data = response.json()
            
            assert response_data["success"] is True
            assert "Found 2 memories" in response_data["message"]
            assert response_data["total_count"] == 2
            assert len(response_data["memories"]) == 2
            assert response_data["memories"][0]["memory"] == "User prefers morning workouts"
            
            mock_memory_client.search_memories.assert_called_once_with(
                query=request_data["query"],
                user_id=request_data["user_id"],
                limit=request_data["limit"]
            )
    
    @allure.story("Memory Retrieval")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_get_all_user_memories_success(
        self, 
        client: TestClient, 
        mock_memory_client: AsyncMock,
        sample_memory_entries: List[MemoryEntry]
    ):
        """Test successful retrieval of all user memories via API."""
        with allure.step("Prepare test data"):
            user_id = "test_user_123"
        
        with allure.step("Mock memory client"):
            mock_memory_client.get_all_memories.return_value = sample_memory_entries
            
            with patch('app.api.v1.endpoints.memory_endpoints.get_mem0_client') as mock_get_client:
                mock_get_client.return_value = mock_memory_client
                
                response = client.get(f"/api/v1/memory/all/{user_id}")
        
        with allure.step("Verify response"):
            assert response.status_code == 200
            response_data = response.json()
            
            assert response_data["success"] is True
            assert "Retrieved 2 memories" in response_data["message"]
            assert response_data["total_count"] == 2
            assert len(response_data["memories"]) == 2
            
            mock_memory_client.get_all_memories.assert_called_once_with(user_id)
    
    @allure.story("Memory Context")
    @allure.severity(allure.severity_level.NORMAL)
    def test_get_memory_context_success(
        self, 
        client: TestClient, 
        mock_memory_client: AsyncMock,
        sample_memory_entries: List[MemoryEntry]
    ):
        """Test successful memory context retrieval via API."""
        with allure.step("Prepare test data"):
            user_id = "test_user_123"
            limit = 10
            expected_context = "Memory: User prefers morning workouts\nMemory: User is a software developer"
        
        with allure.step("Mock memory client and convenience function"):
            mock_memory_client.get_memory_history.return_value = sample_memory_entries
            
            with patch('app.api.v1.endpoints.memory_endpoints.get_mem0_client') as mock_get_client:
                with patch('app.api.v1.endpoints.memory_endpoints.get_user_memory_context') as mock_get_context:
                    mock_get_client.return_value = mock_memory_client
                    mock_get_context.return_value = expected_context
                    
                    response = client.get(f"/api/v1/memory/context/{user_id}?limit={limit}")
        
        with allure.step("Verify response"):
            assert response.status_code == 200
            response_data = response.json()
            
            assert response_data["success"] is True
            assert "Retrieved memory context" in response_data["message"]
            assert response_data["context"] == expected_context
            assert response_data["memory_count"] == 2
    
    @allure.story("Memory Update")
    @allure.severity(allure.severity_level.NORMAL)
    def test_update_memory_success(self, client: TestClient, mock_memory_client: AsyncMock):
        """Test successful memory update via API."""
        with allure.step("Prepare test data"):
            request_data = {
                "memory_id": "mem_123",
                "data": {"memory": "Updated memory content"},
                "user_id": "test_user_123"
            }
            
            expected_response = {"id": "mem_123", "status": "updated"}
        
        with allure.step("Mock memory client"):
            mock_memory_client.update_memory.return_value = expected_response
            
            with patch('app.api.v1.endpoints.memory_endpoints.get_mem0_client') as mock_get_client:
                mock_get_client.return_value = mock_memory_client
                
                response = client.put("/api/v1/memory/update", json=request_data)
        
        with allure.step("Verify response"):
            assert response.status_code == 200
            response_data = response.json()
            
            assert response_data["success"] is True
            assert "Memory mem_123 successfully updated" in response_data["message"]
            assert response_data["data"] == expected_response
            
            mock_memory_client.update_memory.assert_called_once_with(
                memory_id=request_data["memory_id"],
                data=request_data["data"],
                user_id=request_data["user_id"]
            )
    
    @allure.story("Memory Deletion")
    @allure.severity(allure.severity_level.NORMAL)
    def test_delete_memory_success(self, client: TestClient, mock_memory_client: AsyncMock):
        """Test successful memory deletion via API."""
        with allure.step("Prepare test data"):
            request_data = {
                "memory_id": "mem_123",
                "user_id": "test_user_123"
            }
            
            expected_response = {"id": "mem_123", "status": "deleted"}
        
        with allure.step("Mock memory client"):
            mock_memory_client.delete_memory.return_value = expected_response
            
            with patch('app.api.v1.endpoints.memory_endpoints.get_mem0_client') as mock_get_client:
                mock_get_client.return_value = mock_memory_client
                
                response = client.delete("/api/v1/memory/delete", json=request_data)
        
        with allure.step("Verify response"):
            assert response.status_code == 200
            response_data = response.json()
            
            assert response_data["success"] is True
            assert "Memory mem_123 successfully deleted" in response_data["message"]
            assert response_data["data"] == expected_response
            
            mock_memory_client.delete_memory.assert_called_once_with(
                memory_id=request_data["memory_id"],
                user_id=request_data["user_id"]
            )
    
    @allure.story("Memory Deletion")
    @allure.severity(allure.severity_level.NORMAL)
    def test_delete_all_user_memories_success(self, client: TestClient, mock_memory_client: AsyncMock):
        """Test successful deletion of all user memories via API."""
        with allure.step("Prepare test data"):
            user_id = "test_user_123"
            expected_response = {"user_id": user_id, "status": "all_deleted"}
        
        with allure.step("Mock memory client"):
            mock_memory_client.delete_all_memories.return_value = expected_response
            
            with patch('app.api.v1.endpoints.memory_endpoints.get_mem0_client') as mock_get_client:
                mock_get_client.return_value = mock_memory_client
                
                response = client.delete(f"/api/v1/memory/delete-all/{user_id}")
        
        with allure.step("Verify response"):
            assert response.status_code == 200
            response_data = response.json()
            
            assert response_data["success"] is True
            assert "All memories successfully deleted" in response_data["message"]
            assert response_data["data"] == expected_response
            
            mock_memory_client.delete_all_memories.assert_called_once_with(user_id)
    
    @allure.story("Health Check")
    @allure.severity(allure.severity_level.NORMAL)
    def test_memory_health_check_success(self, client: TestClient, mock_memory_client: AsyncMock):
        """Test successful memory service health check via API."""
        with allure.step("Prepare test data"):
            health_status = {
                "status": "healthy",
                "timestamp": "2024-01-01T10:00:00",
                "service": "Mem0 AsyncMemoryClient",
                "message": "Memory service is operational"
            }
        
        with allure.step("Mock memory client"):
            mock_memory_client.health_check.return_value = health_status
            
            with patch('app.api.v1.endpoints.memory_endpoints.get_mem0_client') as mock_get_client:
                mock_get_client.return_value = mock_memory_client
                
                response = client.get("/api/v1/memory/health")
        
        with allure.step("Verify response"):
            assert response.status_code == 200
            response_data = response.json()
            
            assert response_data["status"] == "healthy"
            assert response_data["service"] == "Mem0 AsyncMemoryClient"
            assert response_data["message"] == "Memory service is operational"
            
            mock_memory_client.health_check.assert_called_once()
    
    @allure.story("Health Check")
    @allure.severity(allure.severity_level.NORMAL)
    def test_memory_health_check_failure(self, client: TestClient, mock_memory_client: AsyncMock):
        """Test memory service health check failure via API."""
        with allure.step("Mock health check failure"):
            mock_memory_client.health_check.side_effect = Exception("Service unavailable")
            
            with patch('app.api.v1.endpoints.memory_endpoints.get_mem0_client') as mock_get_client:
                mock_get_client.return_value = mock_memory_client
                
                response = client.get("/api/v1/memory/health")
        
        with allure.step("Verify error response"):
            assert response.status_code == 503
            response_data = response.json()
            assert "Memory service health check failed" in response_data["detail"]
    
    @allure.story("Service Unavailable")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_memory_service_unavailable(self, client: TestClient):
        """Test behavior when memory service is unavailable."""
        with allure.step("Mock service unavailable"):
            with patch('app.api.v1.endpoints.memory_endpoints.get_mem0_client') as mock_get_client:
                mock_get_client.side_effect = Exception("Service unavailable")
                
                response = client.get("/api/v1/memory/all/test_user")
        
        with allure.step("Verify error response"):
            assert response.status_code == 503
            response_data = response.json()
            assert "Memory service is currently unavailable" in response_data["detail"]
    
    @allure.story("Input Validation")
    @allure.severity(allure.severity_level.NORMAL)
    def test_search_memories_validation_error(self, client: TestClient, mock_memory_client: AsyncMock):
        """Test search memories with validation errors."""
        with allure.step("Prepare invalid test data"):
            request_data = {
                "query": "",  # Empty query should cause validation error
                "user_id": "test_user_123",
                "limit": 10
            }
        
        with allure.step("Mock memory client to raise validation error"):
            mock_memory_client.search_memories.side_effect = ValueError("Query cannot be empty")
            
            with patch('app.api.v1.endpoints.memory_endpoints.get_mem0_client') as mock_get_client:
                mock_get_client.return_value = mock_memory_client
                
                response = client.post("/api/v1/memory/search", json=request_data)
        
        with allure.step("Verify error response"):
            assert response.status_code == 400
            response_data = response.json()
            assert "Query cannot be empty" in response_data["detail"]
    
    @allure.story("Server Errors")
    @allure.severity(allure.severity_level.NORMAL)
    def test_internal_server_error_handling(self, client: TestClient, mock_memory_client: AsyncMock):
        """Test handling of internal server errors."""
        with allure.step("Prepare test data"):
            request_data = {
                "messages": [{"role": "user", "content": "test"}],
                "user_id": "test_user_123"
            }
        
        with allure.step("Mock internal server error"):
            mock_memory_client.add_memory.side_effect = Exception("Internal server error")
            
            with patch('app.api.v1.endpoints.memory_endpoints.get_mem0_client') as mock_get_client:
                mock_get_client.return_value = mock_memory_client
                
                response = client.post("/api/v1/memory/add", json=request_data)
        
        with allure.step("Verify error response"):
            assert response.status_code == 500
            response_data = response.json()
            assert "Failed to add memory" in response_data["detail"]


@allure.epic("API Endpoints")
@allure.feature("Memory Endpoints Integration")
class TestMemoryEndpointsIntegration:
    """Integration tests for memory endpoints with actual workflow scenarios."""
    
    @pytest.fixture
    def client(self) -> TestClient:
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    @allure.story("Complete Memory Workflow")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_complete_memory_workflow(self, client: TestClient):
        """Test a complete memory management workflow."""
        with allure.step("Mock memory client for complete workflow"):
            mock_client = AsyncMock()
            
            # Mock responses for the workflow
            add_response = {"id": "mem_123", "status": "success"}
            search_result = MemorySearchResult(
                memories=[
                    MemoryEntry(
                        id="mem_123",
                        memory="User prefers morning workouts",
                        user_id="test_user_123",
                        created_at=datetime.now(),
                        metadata={}
                    )
                ],
                total_count=1,
                query="workout"
            )
            
            mock_client.add_memory.return_value = add_response
            mock_client.search_memories.return_value = search_result
            mock_client.get_all_memories.return_value = search_result.memories
            
            with patch('app.api.v1.endpoints.memory_endpoints.get_mem0_client') as mock_get_client:
                mock_get_client.return_value = mock_client
                
                # Step 1: Add a memory
                with allure.step("Add memory"):
                    add_request = {
                        "messages": [
                            {"role": "user", "content": "I prefer morning workouts"},
                            {"role": "assistant", "content": "I'll remember that"}
                        ],
                        "user_id": "test_user_123"
                    }
                    
                    add_response_api = client.post("/api/v1/memory/add", json=add_request)
                    assert add_response_api.status_code == 200
                    assert add_response_api.json()["success"] is True
                
                # Step 2: Search for the memory
                with allure.step("Search memories"):
                    search_request = {
                        "query": "workout",
                        "user_id": "test_user_123",
                        "limit": 10
                    }
                    
                    search_response = client.post("/api/v1/memory/search", json=search_request)
                    assert search_response.status_code == 200
                    search_data = search_response.json()
                    assert search_data["success"] is True
                    assert search_data["total_count"] == 1
                
                # Step 3: Get all memories
                with allure.step("Get all memories"):
                    all_response = client.get("/api/v1/memory/all/test_user_123")
                    assert all_response.status_code == 200
                    all_data = all_response.json()
                    assert all_data["success"] is True
                    assert all_data["total_count"] == 1
                
                # Step 4: Check health
                with allure.step("Health check"):
                    mock_client.health_check.return_value = {
                        "status": "healthy",
                        "service": "Mem0 AsyncMemoryClient"
                    }
                    
                    health_response = client.get("/api/v1/memory/health")
                    assert health_response.status_code == 200
                    health_data = health_response.json()
                    assert health_data["status"] == "healthy" 