"""Comprehensive tests for the LangGraph chatbot system.

This module contains unit tests and integration tests for all components
of the chatbot system, including configuration, factory, service, and API layers.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any
import allure

from app.core import (
    ChatbotConfig,
    ChatbotType,
    LLMConfig,
    Mem0Config,
    ToolConfig,
    ChatbotFactory,
    create_basic_chatbot,
    create_tool_chatbot,
    create_memory_chatbot,
    create_advanced_chatbot,
    get_chatbot_config
)
from app.services import ChatbotService, ChatSession
from app.api.v1.schemas.chatbot_schemas import (
    ChatRequest,
    CreateSessionRequest,
    ChatbotTypeEnum
)


@allure.epic("Chatbot System")
@allure.feature("Configuration Management")
class TestChatbotConfig:
    """Test chatbot configuration classes."""
    
    @allure.story("Basic Configuration Creation")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_basic_config_creation(self):
        """Test creating a basic chatbot configuration."""
        with allure.step("Create basic chatbot config"):
            config = ChatbotConfig.create_basic_chatbot("Test Bot")
            
        with allure.step("Verify configuration properties"):
            assert config.name == "Test Bot"
            assert config.type == ChatbotType.BASIC
            assert config.llm.model == "gpt-4o-mini"
            assert config.llm.temperature == 0.0
            assert not config.memory.enabled
            assert not config.tools.enabled
            
        allure.attach(
            str(config.dict()),
            name="Basic Config",
            attachment_type=allure.attachment_type.JSON
        )
    
    @allure.story("Advanced Configuration Creation")
    @allure.severity(allure.severity_level.NORMAL)
    def test_advanced_config_creation(self):
        """Test creating an advanced chatbot configuration."""
        with allure.step("Create advanced chatbot config"):
            config = ChatbotConfig.create_advanced_chatbot("Advanced Bot")
            
        with allure.step("Verify advanced features are enabled"):
            assert config.name == "Advanced Bot"
            assert config.type == ChatbotType.WITH_TOOLS_AND_MEMORY
            assert config.memory.enabled
            assert config.tools.enabled
            assert "web_search" in config.tools.available_tools
            
        allure.attach(
            str(config.dict()),
            name="Advanced Config",
            attachment_type=allure.attachment_type.JSON
        )
    
    @allure.story("Predefined Configuration Access")
    @allure.severity(allure.severity_level.NORMAL)
    def test_predefined_configs(self):
        """Test accessing predefined configurations."""
        with allure.step("Get basic config"):
            basic_config = get_chatbot_config("basic")
            assert basic_config.type == ChatbotType.BASIC
            
        with allure.step("Get advanced config"):
            advanced_config = get_chatbot_config("advanced")
            assert advanced_config.type == ChatbotType.WITH_TOOLS_AND_MEMORY
            
        with allure.step("Test invalid config name"):
            with pytest.raises(ValueError):
                get_chatbot_config("invalid_config")


@allure.epic("Chatbot System")
@allure.feature("Factory Pattern")
class TestChatbotFactory:
    """Test chatbot factory functionality."""
    
    @allure.story("Factory Creation Methods")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_factory_creation_methods(self):
        """Test different factory creation methods."""
        with allure.step("Create basic chatbot via factory"):
            basic_bot = ChatbotFactory.create_basic_chatbot("Test prompt")
            assert basic_bot.config.type == ChatbotType.BASIC
            
        with allure.step("Create tool chatbot via factory"):
            tool_bot = ChatbotFactory.create_tool_chatbot("Tool prompt")
            assert tool_bot.config.type == ChatbotType.WITH_TOOLS
            
        with allure.step("Create memory chatbot via factory"):
            memory_bot = ChatbotFactory.create_memory_chatbot("Memory prompt")
            assert memory_bot.config.type == ChatbotType.WITH_MEMORY
            
        with allure.step("Create advanced chatbot via factory"):
            advanced_bot = ChatbotFactory.create_advanced_chatbot("Advanced prompt")
            assert advanced_bot.config.type == ChatbotType.WITH_TOOLS_AND_MEMORY
    
    @allure.story("Factory Configuration-Based Creation")
    @allure.severity(allure.severity_level.NORMAL)
    def test_factory_config_creation(self):
        """Test creating chatbots from configuration objects."""
        with allure.step("Create custom configuration"):
            config = ChatbotConfig(
                name="Custom Bot",
                type=ChatbotType.BASIC,
                system_prompt="Custom prompt"
            )
            
        with allure.step("Create chatbot from config"):
            bot = ChatbotFactory.create_chatbot(config)
            assert bot.config.name == "Custom Bot"
            assert bot.config.system_prompt == "Custom prompt"
    
    @allure.story("Factory Name-Based Creation")
    @allure.severity(allure.severity_level.NORMAL)
    def test_factory_name_creation(self):
        """Test creating chatbots from predefined names."""
        with allure.step("Create chatbot from predefined name"):
            bot = ChatbotFactory.create_from_name("basic")
            assert bot.config.type == ChatbotType.BASIC
            
        with allure.step("Test invalid name"):
            with pytest.raises(ValueError):
                ChatbotFactory.create_from_name("invalid_name")


@allure.epic("Chatbot System")
@allure.feature("Service Layer")
class TestChatbotService:
    """Test chatbot service functionality."""
    
    @pytest.fixture
    def service(self):
        """Create a chatbot service for testing."""
        return ChatbotService(session_timeout_minutes=1)
    
    @allure.story("Session Management")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_session_creation(self, service):
        """Test creating and managing sessions."""
        with allure.step("Create a new session"):
            session_id = service.create_session(
                user_id="test_user",
                chatbot_type="basic",
                metadata={"test": "data"}
            )
            assert session_id is not None
            
        with allure.step("Verify session exists"):
            session = service.get_session(session_id)
            assert session is not None
            assert session.user_id == "test_user"
            assert session.chatbot_type == "basic"
            assert session.metadata["test"] == "data"
            
        with allure.step("Get session info"):
            info = service.get_session_info(session_id)
            assert info is not None
            assert info["user_id"] == "test_user"
            
        allure.attach(
            str(info),
            name="Session Info",
            attachment_type=allure.attachment_type.JSON
        )
    
    @allure.story("Session Listing and Filtering")
    @allure.severity(allure.severity_level.NORMAL)
    def test_session_listing(self, service):
        """Test listing and filtering sessions."""
        with allure.step("Create multiple sessions"):
            session1 = service.create_session(user_id="user1", chatbot_type="basic")
            session2 = service.create_session(user_id="user2", chatbot_type="advanced")
            session3 = service.create_session(user_id="user1", chatbot_type="with_tools")
            
        with allure.step("List all sessions"):
            all_sessions = service.list_sessions()
            assert len(all_sessions) >= 3
            
        with allure.step("Filter sessions by user"):
            user1_sessions = service.list_sessions(user_id="user1")
            assert len(user1_sessions) == 2
            
        with allure.step("Verify session data"):
            session_ids = [s["session_id"] for s in user1_sessions]
            assert session1 in session_ids
            assert session3 in session_ids
    
    @allure.story("Session Deletion")
    @allure.severity(allure.severity_level.NORMAL)
    def test_session_deletion(self, service):
        """Test deleting sessions."""
        with allure.step("Create a session"):
            session_id = service.create_session(user_id="test_user")
            
        with allure.step("Verify session exists"):
            assert service.get_session(session_id) is not None
            
        with allure.step("Delete the session"):
            result = service.delete_session(session_id)
            assert result is True
            
        with allure.step("Verify session is deleted"):
            assert service.get_session(session_id) is None
            
        with allure.step("Try to delete non-existent session"):
            result = service.delete_session("non_existent")
            assert result is False
    
    @allure.story("Chat Functionality")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_chat_functionality(self, service):
        """Test basic chat functionality through service."""
        with allure.step("Create a session"):
            session_id = service.create_session(user_id="test_user", chatbot_type="basic")
            
        with allure.step("Mock the chatbot response"):
            # Mock the chatbot's chat method
            session = service.get_session(session_id)
            session.chatbot.chat = AsyncMock(return_value="Hello! How can I help you?")
            
        with allure.step("Send a chat message"):
            response = await service.chat(session_id, "Hello!")
            assert response == "Hello! How can I help you?"
            
        with allure.step("Verify session activity was updated"):
            session_info = service.get_session_info(session_id)
            assert session_info["message_count"] == 1
            
        allure.attach(
            response,
            name="Chat Response",
            attachment_type=allure.attachment_type.TEXT
        )
    
    @allure.story("Error Handling")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.asyncio
    async def test_error_handling(self, service):
        """Test error handling in service layer."""
        with allure.step("Test chat with non-existent session"):
            with pytest.raises(ValueError, match="Session not found"):
                await service.chat("non_existent_session", "Hello!")
                
        with allure.step("Test getting non-existent session info"):
            info = service.get_session_info("non_existent_session")
            assert info is None


@allure.epic("Chatbot System")
@allure.feature("Direct Chatbot Usage")
class TestDirectChatbotUsage:
    """Test using chatbots directly without service layer."""
    
    @allure.story("Basic Chatbot Creation and Usage")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_basic_chatbot_creation(self):
        """Test creating and configuring a basic chatbot."""
        with allure.step("Create basic chatbot"):
            chatbot = create_basic_chatbot("You are a helpful assistant.")
            
        with allure.step("Verify chatbot configuration"):
            assert chatbot.config.type == ChatbotType.BASIC
            assert chatbot.config.system_prompt == "You are a helpful assistant."
            assert chatbot.llm is not None
            assert chatbot.graph is not None
            
        with allure.step("Verify no tools or memory"):
            assert len(chatbot.tools) == 0
            assert chatbot.memory is None
    
    @allure.story("Tool-Enabled Chatbot")
    @allure.severity(allure.severity_level.NORMAL)
    def test_tool_chatbot_creation(self):
        """Test creating a tool-enabled chatbot."""
        with allure.step("Create tool-enabled chatbot"):
            chatbot = create_tool_chatbot("You are an assistant with tools.")
            
        with allure.step("Verify chatbot configuration"):
            assert chatbot.config.type == ChatbotType.WITH_TOOLS
            assert chatbot.config.tools.enabled
            assert "web_search" in chatbot.config.tools.available_tools
    
    @allure.story("Memory-Enabled Chatbot")
    @allure.severity(allure.severity_level.NORMAL)
    def test_memory_chatbot_creation(self):
        """Test creating a memory-enabled chatbot."""
        with allure.step("Create memory-enabled chatbot"):
            chatbot = create_memory_chatbot("You are an assistant with memory.")
            
        with allure.step("Verify chatbot configuration"):
            assert chatbot.config.type == ChatbotType.WITH_MEMORY
            assert chatbot.config.memory.enabled
    
    @allure.story("Advanced Chatbot")
    @allure.severity(allure.severity_level.NORMAL)
    def test_advanced_chatbot_creation(self):
        """Test creating an advanced chatbot."""
        with allure.step("Create advanced chatbot"):
            chatbot = create_advanced_chatbot("You are an advanced assistant.")
            
        with allure.step("Verify chatbot configuration"):
            assert chatbot.config.type == ChatbotType.WITH_TOOLS_AND_MEMORY
            assert chatbot.config.memory.enabled
            assert chatbot.config.tools.enabled


@allure.epic("Chatbot System")
@allure.feature("API Schema Validation")
class TestAPISchemas:
    """Test API schema validation."""
    
    @allure.story("Chat Request Validation")
    @allure.severity(allure.severity_level.NORMAL)
    def test_chat_request_validation(self):
        """Test chat request schema validation."""
        with allure.step("Create valid chat request"):
            request = ChatRequest(
                message="Hello, world!",
                session_id="test_session",
                user_id="test_user",
                metadata={"source": "test"}
            )
            assert request.message == "Hello, world!"
            assert request.session_id == "test_session"
            
        with allure.step("Test message validation"):
            with pytest.raises(ValueError):
                ChatRequest(message="")  # Empty message should fail
                
        with allure.step("Test message trimming"):
            request = ChatRequest(message="  Hello  ")
            assert request.message == "Hello"
    
    @allure.story("Session Request Validation")
    @allure.severity(allure.severity_level.NORMAL)
    def test_session_request_validation(self):
        """Test session creation request validation."""
        with allure.step("Create valid session request"):
            request = CreateSessionRequest(
                user_id="test_user",
                chatbot_type=ChatbotTypeEnum.BASIC,
                metadata={"source": "test"}
            )
            assert request.chatbot_type == ChatbotTypeEnum.BASIC
            assert request.user_id == "test_user"
            
        with allure.step("Test default values"):
            request = CreateSessionRequest()
            assert request.chatbot_type == ChatbotTypeEnum.BASIC
            assert request.user_id is None


@allure.epic("Chatbot System")
@allure.feature("Integration Tests")
class TestIntegration:
    """Integration tests for the complete chatbot system."""
    
    @allure.story("End-to-End Basic Flow")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_end_to_end_basic_flow(self):
        """Test complete flow from service creation to chat response."""
        with allure.step("Initialize service"):
            service = ChatbotService()
            
        with allure.step("Create session"):
            session_id = service.create_session(
                user_id="integration_user",
                chatbot_type="basic",
                metadata={"test": "integration"}
            )
            
        with allure.step("Mock chatbot response"):
            session = service.get_session(session_id)
            session.chatbot.chat = AsyncMock(return_value="Integration test response")
            
        with allure.step("Send chat message"):
            response = await service.chat(session_id, "Integration test message")
            assert response == "Integration test response"
            
        with allure.step("Verify session state"):
            session_info = service.get_session_info(session_id)
            assert session_info["message_count"] == 1
            assert session_info["user_id"] == "integration_user"
            
        with allure.step("Cleanup"):
            await service.shutdown()
            
        allure.attach(
            json.dumps(session_info, indent=2),
            name="Final Session State",
            attachment_type=allure.attachment_type.JSON
        )
    
    @allure.story("Multiple Sessions Management")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.asyncio
    async def test_multiple_sessions_management(self):
        """Test managing multiple concurrent sessions."""
        with allure.step("Initialize service"):
            service = ChatbotService()
            
        with allure.step("Create multiple sessions"):
            sessions = []
            for i in range(3):
                session_id = service.create_session(
                    user_id=f"user_{i}",
                    chatbot_type="basic",
                    metadata={"session_number": i}
                )
                sessions.append(session_id)
                
        with allure.step("Mock responses for all sessions"):
            for i, session_id in enumerate(sessions):
                session = service.get_session(session_id)
                session.chatbot.chat = AsyncMock(return_value=f"Response from session {i}")
                
        with allure.step("Send messages to all sessions"):
            responses = []
            for i, session_id in enumerate(sessions):
                response = await service.chat(session_id, f"Message to session {i}")
                responses.append(response)
                
        with allure.step("Verify all responses"):
            for i, response in enumerate(responses):
                assert response == f"Response from session {i}"
                
        with allure.step("Verify session isolation"):
            for i, session_id in enumerate(sessions):
                session_info = service.get_session_info(session_id)
                assert session_info["user_id"] == f"user_{i}"
                assert session_info["message_count"] == 1
                
        with allure.step("Cleanup"):
            await service.shutdown()
            
        allure.attach(
            json.dumps({"sessions_created": len(sessions), "responses": responses}, indent=2),
            name="Multiple Sessions Test Results",
            attachment_type=allure.attachment_type.JSON
        )


if __name__ == "__main__":
    # Run tests with allure reporting
    pytest.main([
        __file__,
        "-v",
        "--alluredir=allure-results",
        "--tb=short"
    ]) 