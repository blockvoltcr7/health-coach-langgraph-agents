"""Test AsyncMemoryClient integration with chatbot system."""

import pytest
import asyncio
import os
from unittest.mock import AsyncMock, patch
from dotenv import load_dotenv
import allure

from app.core.chatbot_config import ChatbotConfig, ChatbotType
from app.core.chatbot_factory import create_chatbot
from app.services.chatbot_service import ChatbotService

load_dotenv()


@allure.epic("Chatbot System")
@allure.feature("Mem0 Integration")
class TestAsyncMem0Integration:
    """Test AsyncMemoryClient integration with chatbot system."""
    
    @pytest.fixture
    def mem0_config(self):
        """Create a memory-enabled chatbot config."""
        return ChatbotConfig.create_chatbot_with_memory("Test Memory Bot")
    
    @pytest.fixture
    def chatbot_service(self):
        """Create a chatbot service instance."""
        return ChatbotService(session_timeout_minutes=5)
    
    @allure.story("AsyncMemoryClient Configuration")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_async_memory_client_creation(self, mem0_config):
        """Test that AsyncMemoryClient is properly created with API key."""
        with allure.step("Create memory-enabled chatbot"):
            chatbot = create_chatbot(mem0_config)
            
        with allure.step("Verify memory client is created"):
            assert chatbot.memory is not None
            assert hasattr(chatbot.memory, 'add')
            assert hasattr(chatbot.memory, 'search')
            
        with allure.step("Verify API key is configured"):
            # Check that the config has the API key
            api_key = os.getenv("MEM0_API_KEY")
            if api_key:
                assert chatbot.config.memory.api_key == api_key
            else:
                # If no API key, memory should be None or warn
                allure.attach(
                    "No MEM0_API_KEY found in environment",
                    name="API Key Status",
                    attachment_type=allure.attachment_type.TEXT
                )
    
    @allure.story("Memory Operations")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.asyncio
    async def test_memory_operations_with_mock(self, mem0_config):
        """Test memory operations with mocked AsyncMemoryClient."""
        with patch('app.core.chatbot_base.AsyncMemoryClient') as mock_client_class:
            # Setup mock
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.add.return_value = {"status": "success"}
            mock_client.search.return_value = [{"memory": "test memory"}]
            
            with allure.step("Create memory-enabled chatbot"):
                chatbot = create_chatbot(mem0_config)
                
            with allure.step("Test add memory operation"):
                await chatbot.add_memory("Test content", "test_user")
                
                # Verify add was called with correct format
                mock_client.add.assert_called_once()
                call_args = mock_client.add.call_args
                assert call_args[1]['user_id'] == "test_user"
                assert call_args[1]['output_format'] == "v1.1"
                assert len(call_args[1]['messages']) == 1
                assert call_args[1]['messages'][0]['role'] == "assistant"
                assert call_args[1]['messages'][0]['content'] == "Test content"
                
            with allure.step("Test search memory operation"):
                results = await chatbot.search_memory("test query", "test_user")
                
                # Verify search was called with correct format
                mock_client.search.assert_called_once()
                call_args = mock_client.search.call_args
                assert call_args[1]['query'] == "test query"
                assert call_args[1]['user_id'] == "test_user"
                assert call_args[1]['output_format'] == "v1.1"
                
                # Verify results
                assert len(results) == 1
                assert results[0]["memory"] == "test memory"
    
    @allure.story("Service Integration")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.asyncio
    async def test_service_memory_integration(self, chatbot_service):
        """Test memory integration through service layer."""
        with patch('app.core.chatbot_base.AsyncMemoryClient') as mock_client_class:
            # Setup mock
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.add.return_value = {"status": "success"}
            
            with allure.step("Create memory-enabled session"):
                session_id = chatbot_service.create_session(
                    user_id="test_user",
                    chatbot_type="with_memory"
                )
                
            with allure.step("Send message to trigger memory storage"):
                with patch.object(chatbot_service.get_session(session_id).chatbot, 'chat') as mock_chat:
                    mock_chat.return_value = "Test response"
                    
                    response = await chatbot_service.chat(session_id, "Test message")
                    
                    assert response == "Test response"
                    
            with allure.step("Verify session was created"):
                session = chatbot_service.get_session(session_id)
                assert session is not None
                assert session.user_id == "test_user"
                assert session.chatbot_type == "with_memory"
    
    @allure.story("Error Handling")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.asyncio
    async def test_memory_error_handling(self, mem0_config):
        """Test error handling in memory operations."""
        with patch('app.core.chatbot_base.AsyncMemoryClient') as mock_client_class:
            # Setup mock to raise exception
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.add.side_effect = Exception("Memory error")
            mock_client.search.side_effect = Exception("Search error")
            
            with allure.step("Create memory-enabled chatbot"):
                chatbot = create_chatbot(mem0_config)
                
            with allure.step("Test add memory error handling"):
                # Should not raise exception, just log error
                await chatbot.add_memory("Test content", "test_user")
                
            with allure.step("Test search memory error handling"):
                # Should return empty list on error
                results = await chatbot.search_memory("test query", "test_user")
                assert results == []
    
    @allure.story("Configuration Validation")
    @allure.severity(allure.severity_level.NORMAL)
    def test_memory_config_validation(self):
        """Test memory configuration validation."""
        with allure.step("Test default memory configuration"):
            config = ChatbotConfig.create_chatbot_with_memory()
            
            assert config.memory.enabled is True
            assert config.memory.output_format == "v1.1"
            assert config.memory.graph_store_provider == "neo4j"
            assert config.memory.enable_graph is True
            
        with allure.step("Test custom memory configuration"):
            config = ChatbotConfig(
                name="Custom Memory Bot",
                type=ChatbotType.WITH_MEMORY,
                memory={
                    "enabled": True,
                    "output_format": "v1.0",
                    "graph_store_provider": "custom"
                }
            )
            
            assert config.memory.enabled is True
            assert config.memory.output_format == "v1.0"
            assert config.memory.graph_store_provider == "custom" 