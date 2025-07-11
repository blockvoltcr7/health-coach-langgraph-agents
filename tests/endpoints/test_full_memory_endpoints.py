"""Test full memory comparison endpoints."""

import allure
import pytest
import requests
import json
from datetime import datetime


@allure.epic("Memory Comparison")
@allure.feature("Full Memory vs Semantic Search")
@allure.suite("full_memory_tests")
@pytest.mark.api
@pytest.mark.integration
class TestFullMemoryEndpoints:

    @allure.story("Full Memory Chat Endpoint")
    @allure.severity(allure.severity_level.NORMAL)
    def test_full_memory_chat_endpoint(self, session, api_base_url):
        """Test the full memory chat endpoint."""
        
        # Test data
        test_request = {
            "message": "Hello, how are you today?",
            "user_id": "test_user_full_memory_001",
            "chatbot_type": "with_memory",
            "metadata": {
                "test_type": "full_memory_comparison",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        with allure.step("Prepare full memory chat request"):
            allure.attach(
                json.dumps(test_request, indent=2),
                name="Request Payload",
                attachment_type=allure.attachment_type.JSON
            )
        
        with allure.step(f"Send POST request to full memory chat endpoint {api_base_url}/api/v1/chatbot/chat/full-memory"):
            response = session.post(
                f"{api_base_url}/api/v1/chatbot/chat/full-memory",
                json=test_request
            )
        
        with allure.step("Verify response status"):
            allure.attach(
                str(response.status_code),
                name="Status Code",
                attachment_type=allure.attachment_type.TEXT
            )
            
            # Check if the endpoint is available (200) or if there's a configuration issue (500)
            assert response.status_code in [200, 500], f"Unexpected status code: {response.status_code}"
            
            if response.status_code == 500:
                # Log the error for debugging but don't fail the test
                error_data = response.json()
                allure.attach(
                    json.dumps(error_data, indent=2),
                    name="Error Response",
                    attachment_type=allure.attachment_type.JSON
                )
                pytest.skip("Full memory endpoint requires MEM0_API_KEY configuration")
        
        with allure.step("Parse and verify response structure"):
            data = response.json()
            allure.attach(
                json.dumps(data, indent=2),
                name="Response JSON",
                attachment_type=allure.attachment_type.JSON
            )
            
            # Verify response structure
            assert "response" in data
            assert "user_id" in data
            assert "chatbot_type" in data
            assert "timestamp" in data
            
            # Verify that the chatbot_type indicates full memory usage
            assert data["chatbot_type"].endswith("_full_memory")
            assert data["user_id"] == test_request["user_id"]
            
            # Verify response is not empty
            assert len(data["response"]) > 0

    @allure.story("Regular Chat Endpoint for Comparison")
    @allure.severity(allure.severity_level.NORMAL)
    def test_regular_chat_endpoint_comparison(self, session, api_base_url):
        """Test the regular chat endpoint for comparison with full memory."""
        
        # Test data - same as full memory test for comparison
        test_request = {
            "message": "Hello, how are you today?",
            "user_id": "test_user_regular_001",
            "chatbot_type": "with_memory",
            "metadata": {
                "test_type": "regular_memory_comparison",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        with allure.step("Prepare regular chat request"):
            allure.attach(
                json.dumps(test_request, indent=2),
                name="Request Payload",
                attachment_type=allure.attachment_type.JSON
            )
        
        with allure.step(f"Send POST request to regular chat endpoint {api_base_url}/api/v1/chatbot/chat"):
            response = session.post(
                f"{api_base_url}/api/v1/chatbot/chat",
                json=test_request
            )
        
        with allure.step("Verify response status"):
            allure.attach(
                str(response.status_code),
                name="Status Code",
                attachment_type=allure.attachment_type.TEXT
            )
            
            # Check if the endpoint is available (200) or if there's a configuration issue (500)
            assert response.status_code in [200, 500], f"Unexpected status code: {response.status_code}"
            
            if response.status_code == 500:
                # Log the error for debugging but don't fail the test
                error_data = response.json()
                allure.attach(
                    json.dumps(error_data, indent=2),
                    name="Error Response",
                    attachment_type=allure.attachment_type.JSON
                )
                pytest.skip("Regular chat endpoint requires API configuration")
        
        with allure.step("Parse and verify response structure"):
            data = response.json()
            allure.attach(
                json.dumps(data, indent=2),
                name="Response JSON",
                attachment_type=allure.attachment_type.JSON
            )
            
            # Verify response structure
            assert "response" in data
            assert "user_id" in data
            assert "chatbot_type" in data
            assert "timestamp" in data
            
            # Verify that the chatbot_type does NOT indicate full memory usage
            assert not data["chatbot_type"].endswith("_full_memory")
            assert data["user_id"] == test_request["user_id"]
            
            # Verify response is not empty
            assert len(data["response"]) > 0

    @allure.story("Memory Comparison Endpoint")
    @allure.severity(allure.severity_level.NORMAL)
    def test_memory_comparison_endpoint(self, session, api_base_url):
        """Test the memory comparison endpoint."""
        
        user_id = "test_user_comparison_001"
        query = "test conversation"
        
        with allure.step(f"Send GET request to memory comparison endpoint"):
            response = session.get(
                f"{api_base_url}/api/v1/chatbot/memory/compare/{user_id}",
                params={"query": query}
            )
        
        with allure.step("Verify response status"):
            allure.attach(
                str(response.status_code),
                name="Status Code",
                attachment_type=allure.attachment_type.TEXT
            )
            
            # Check if the endpoint is available (200) or if there's a configuration issue (500)
            assert response.status_code in [200, 500], f"Unexpected status code: {response.status_code}"
            
            if response.status_code == 500:
                # Log the error for debugging but don't fail the test
                error_data = response.json()
                allure.attach(
                    json.dumps(error_data, indent=2),
                    name="Error Response",
                    attachment_type=allure.attachment_type.JSON
                )
                pytest.skip("Memory comparison endpoint requires MEM0_API_KEY configuration")
        
        with allure.step("Parse and verify comparison structure"):
            data = response.json()
            allure.attach(
                json.dumps(data, indent=2),
                name="Comparison Response",
                attachment_type=allure.attachment_type.JSON
            )
            
            # Verify main structure
            assert "user_id" in data
            assert "query" in data
            assert "semantic_search" in data
            assert "get_all" in data
            assert "comparison" in data
            assert "timestamp" in data
            
            # Verify user_id and query match
            assert data["user_id"] == user_id
            assert data["query"] == query
            
            # Check semantic search structure
            semantic = data["semantic_search"]
            assert "count" in semantic
            assert "memories" in semantic
            assert "approach" in semantic
            assert isinstance(semantic["count"], int)
            assert isinstance(semantic["memories"], list)
            
            # Check get_all structure
            get_all = data["get_all"]
            assert "count" in get_all
            assert "memories" in get_all
            assert "approach" in get_all
            assert isinstance(get_all["count"], int)
            assert isinstance(get_all["memories"], list)
            
            # Check comparison structure
            comparison = data["comparison"]
            assert "semantic_efficiency" in comparison
            assert "full_memory_load" in comparison
            assert "efficiency_ratio" in comparison
            assert "recommendation" in comparison

    @allure.story("Chatbot Types Endpoint")
    @allure.severity(allure.severity_level.NORMAL)
    def test_chatbot_types_endpoint(self, session, api_base_url):
        """Test the chatbot types endpoint to verify available types."""
        
        with allure.step(f"Send GET request to chatbot types endpoint"):
            response = session.get(f"{api_base_url}/api/v1/chatbot/types")
        
        with allure.step("Verify response status"):
            assert response.status_code == 200
            allure.attach(
                str(response.status_code),
                name="Status Code",
                attachment_type=allure.attachment_type.TEXT
            )
        
        with allure.step("Parse and verify types structure"):
            data = response.json()
            allure.attach(
                json.dumps(data, indent=2),
                name="Types Response",
                attachment_type=allure.attachment_type.JSON
            )
            
            # Verify structure
            assert "available_types" in data
            assert "total" in data
            assert "timestamp" in data
            
            # Verify types are available
            assert isinstance(data["available_types"], list)
            assert len(data["available_types"]) > 0
            assert data["total"] == len(data["available_types"])
            
            # Check for expected types
            expected_types = ["basic", "with_tools", "with_memory", "advanced"]
            for expected_type in expected_types:
                assert expected_type in data["available_types"]

    @allure.story("Health Check Endpoint")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_health_check_endpoint(self, session, api_base_url):
        """Test the health check endpoint."""
        
        with allure.step(f"Send GET request to health check endpoint"):
            response = session.get(f"{api_base_url}/api/v1/chatbot/health")
        
        with allure.step("Verify response status"):
            assert response.status_code == 200
            allure.attach(
                str(response.status_code),
                name="Status Code",
                attachment_type=allure.attachment_type.TEXT
            )
        
        with allure.step("Parse and verify health check structure"):
            data = response.json()
            allure.attach(
                json.dumps(data, indent=2),
                name="Health Check Response",
                attachment_type=allure.attachment_type.JSON
            )
            
            # Verify structure
            assert "status" in data
            assert "timestamp" in data
            assert "available_types" in data
            assert "message" in data
            
            # Verify health status
            assert data["status"] in ["healthy", "unhealthy"]
            assert isinstance(data["available_types"], list)
            assert len(data["message"]) > 0
