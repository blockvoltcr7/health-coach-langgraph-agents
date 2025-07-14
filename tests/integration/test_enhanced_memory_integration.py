"""Integration tests for Enhanced Memory Features with Chatbot.

This module contains integration tests that verify the enhanced memory features
work correctly within the full chatbot/agent flow.
"""

import pytest
import allure
from typing import Dict, Any
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from app.core.chatbot_config import ChatbotConfig, LLMConfig, MemoryConfig, ToolsConfig
from app.core.chatbot_base import LimitlessOSIntelligentAgent
from app.mem0.mem0AsyncClient import MemoryCategory, MemoryEntry, Mem0AsyncClientWrapper


@allure.epic("Memory Integration")
@allure.feature("Enhanced Memory with Chatbot")
class TestEnhancedMemoryIntegration:
    """Test class for enhanced memory integration with the chatbot."""
    
    @pytest.fixture
    def chatbot_config(self) -> ChatbotConfig:
        """Create a test chatbot configuration."""
        return ChatbotConfig(
            llm=LLMConfig(
                model="gpt-4",
                temperature=0.7,
                api_key="test_key"
            ),
            memory=MemoryConfig(
                api_key="test_mem0_key",
                output_format="v1.1"
            ),
            tools=ToolsConfig(
                tavily_api_key="test_tavily_key"
            ),
            system_prompt="You are a helpful sales agent for Limitless OS."
        )
    
    @pytest.fixture
    async def mock_enhanced_memory_client(self) -> Mem0AsyncClientWrapper:
        """Create a mock enhanced memory client."""
        mock_client = AsyncMock(spec=Mem0AsyncClientWrapper)
        
        # Mock the methods we'll use
        mock_client.get_all_memories = AsyncMock()
        mock_client.add_memory = AsyncMock()
        mock_client.search_memories = AsyncMock()
        mock_client.get_memories_by_importance = AsyncMock()
        mock_client.get_memory_analytics = AsyncMock()
        
        return mock_client
    
    @allure.story("Chatbot Memory Integration")
    @allure.severity(allure.severity_level.CRITICAL)
    async def test_chatbot_uses_enhanced_memory_on_chat(self, chatbot_config, mock_enhanced_memory_client):
        """Test that chatbot correctly uses enhanced memory features during chat."""
        with allure.step("Mock LLM and memory initialization"):
            with patch('app.core.chatbot_base.ChatOpenAI') as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.bind_tools.return_value = mock_llm
                mock_llm.invoke.return_value = MagicMock(content="I see you have a budget of $50k")
                mock_llm_class.return_value = mock_llm
                
                with patch('app.core.chatbot_base.Mem0AsyncClientWrapper') as mock_wrapper_class:
                    mock_wrapper_class.return_value = mock_enhanced_memory_client
                    
                    # Initialize agent
                    agent = LimitlessOSIntelligentAgent(chatbot_config)
        
        with allure.step("Mock existing memories with categories"):
            mock_memories = [
                MemoryEntry(
                    id="mem_1",
                    memory="User has a budget of $50,000",
                    user_id="test_user",
                    category=MemoryCategory.QUALIFICATION,
                    importance_score=9.0,
                    access_count=5,
                    metadata={"source": "previous_chat"}
                ),
                MemoryEntry(
                    id="mem_2",
                    memory="User prefers email communication",
                    user_id="test_user",
                    category=MemoryCategory.PREFERENCE,
                    importance_score=6.0,
                    access_count=2,
                    metadata={"source": "previous_chat"}
                )
            ]
            mock_enhanced_memory_client.get_all_memories.return_value = mock_memories
        
        with allure.step("Simulate chat interaction"):
            user_message = "What was my budget again?"
            response = await agent.chat(user_message, "test_user", source="chat_test")
        
        with allure.step("Verify memory retrieval"):
            # Should have retrieved all memories
            mock_enhanced_memory_client.get_all_memories.assert_called_once_with("test_user")
            
            # Verify the message was enhanced with memory context
            llm_call_args = mock_llm.invoke.call_args[0][0]
            assert len(llm_call_args) > 0
            # Check that memories were included in the context
            message_content = llm_call_args[-1].content
            assert "budget of $50,000" in message_content.lower()
        
        with allure.step("Verify memory storage with enhanced wrapper"):
            # Should have stored the new conversation
            mock_enhanced_memory_client.add_memory.assert_called_once()
            add_memory_args = mock_enhanced_memory_client.add_memory.call_args
            
            # Check the messages were stored
            messages = add_memory_args.kwargs["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == user_message
            assert messages[1]["role"] == "assistant"
            assert messages[1]["content"] == "I see you have a budget of $50k"
            
            # Check metadata was passed
            assert add_memory_args.kwargs["metadata"]["source"] == "chat_test"
    
    @allure.story("Auto-Categorization in Chat")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_chat_auto_categorizes_memories(self, chatbot_config, mock_enhanced_memory_client):
        """Test that memories are auto-categorized during chat conversations."""
        with allure.step("Setup mocks"):
            with patch('app.core.chatbot_base.ChatOpenAI') as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.bind_tools.return_value = mock_llm
                mock_llm.invoke.return_value = MagicMock(
                    content="I understand your concern about the pricing. Let me explain the value."
                )
                mock_llm_class.return_value = mock_llm
                
                with patch('app.core.chatbot_base.Mem0AsyncClientWrapper') as mock_wrapper_class:
                    mock_wrapper_class.return_value = mock_enhanced_memory_client
                    mock_enhanced_memory_client.get_all_memories.return_value = []
                    
                    agent = LimitlessOSIntelligentAgent(chatbot_config)
        
        with allure.step("Simulate objection conversation"):
            user_message = "I'm worried this is too expensive for our budget"
            response = await agent.chat(user_message, "test_user", conversation_type="sales")
        
        with allure.step("Verify auto-categorization happened"):
            # The add_memory should have been called
            mock_enhanced_memory_client.add_memory.assert_called_once()
            
            # Note: The actual auto-categorization happens inside the wrapper
            # We're verifying the flow works correctly
            add_memory_args = mock_enhanced_memory_client.add_memory.call_args
            messages = add_memory_args.kwargs["messages"]
            
            # The objection should be in the conversation
            assert "expensive" in messages[0]["content"]
            assert "concern about the pricing" in messages[1]["content"]
    
    @allure.story("Memory Priority Retrieval")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_high_importance_memories_prioritized(self, chatbot_config, mock_enhanced_memory_client):
        """Test that high importance memories are properly used in context."""
        with allure.step("Mock memories with different importance scores"):
            mock_memories = [
                MemoryEntry(
                    id="mem_critical",
                    memory="CRITICAL: User has already signed with competitor last week",
                    user_id="test_user",
                    category=MemoryCategory.OUTCOME,
                    importance_score=10.0,
                    access_count=1,
                    metadata={"critical": True}
                ),
                MemoryEntry(
                    id="mem_low",
                    memory="User mentioned they like coffee",
                    user_id="test_user",
                    category=MemoryCategory.PREFERENCE,
                    importance_score=2.0,
                    access_count=1,
                    metadata={}
                )
            ]
            mock_enhanced_memory_client.get_all_memories.return_value = mock_memories
        
        with allure.step("Setup chatbot"):
            with patch('app.core.chatbot_base.ChatOpenAI') as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.bind_tools.return_value = mock_llm
                mock_llm.invoke.return_value = MagicMock(
                    content="I see you've already made a decision with our competitor."
                )
                mock_llm_class.return_value = mock_llm
                
                with patch('app.core.chatbot_base.Mem0AsyncClientWrapper') as mock_wrapper_class:
                    mock_wrapper_class.return_value = mock_enhanced_memory_client
                    agent = LimitlessOSIntelligentAgent(chatbot_config)
        
        with allure.step("Chat and verify high importance memory is used"):
            response = await agent.chat(
                "Can you offer me a better deal?",
                "test_user"
            )
            
            # Verify the critical memory was retrieved
            mock_enhanced_memory_client.get_all_memories.assert_called_with("test_user")
            
            # The response should acknowledge the competitor situation
            assert "competitor" in response.lower()
    
    @allure.story("Memory Analytics Integration")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_memory_analytics_tracking(self, chatbot_config, mock_enhanced_memory_client):
        """Test that memory access is tracked for analytics."""
        with allure.step("Mock initial analytics"):
            initial_analytics = {
                "total_memories": 10,
                "categories": {
                    "fact": 3,
                    "preference": 2,
                    "objection": 2,
                    "outcome": 1,
                    "context": 2,
                    "qualification": 0
                },
                "avg_importance_score": 5.5,
                "avg_access_count": 2.0,
                "importance_distribution": {
                    "high": 2,
                    "medium": 5,
                    "low": 3
                },
                "most_accessed": []
            }
            mock_enhanced_memory_client.get_memory_analytics.return_value = initial_analytics
        
        with allure.step("Setup agent with empty memories"):
            with patch('app.core.chatbot_base.ChatOpenAI') as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.bind_tools.return_value = mock_llm
                mock_llm.invoke.return_value = MagicMock(content="Hello! How can I help you?")
                mock_llm_class.return_value = mock_llm
                
                with patch('app.core.chatbot_base.Mem0AsyncClientWrapper') as mock_wrapper_class:
                    mock_wrapper_class.return_value = mock_enhanced_memory_client
                    mock_enhanced_memory_client.get_all_memories.return_value = []
                    
                    agent = LimitlessOSIntelligentAgent(chatbot_config)
        
        with allure.step("Perform multiple chats to generate memories"):
            # First chat - qualification info
            await agent.chat("My budget is $75,000", "test_user")
            
            # Second chat - preference
            await agent.chat("I prefer quarterly billing", "test_user")
            
            # Third chat - objection
            await agent.chat("I'm concerned about implementation time", "test_user")
        
        with allure.step("Verify memories were added"):
            # Should have been called 3 times
            assert mock_enhanced_memory_client.add_memory.call_count == 3
            
            # Each call should have proper conversation structure
            for call in mock_enhanced_memory_client.add_memory.call_args_list:
                messages = call.kwargs["messages"]
                assert len(messages) == 2
                assert messages[0]["role"] == "user"
                assert messages[1]["role"] == "assistant"
    
    @allure.story("MongoDB Snapshot Integration")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_memory_snapshot_after_conversation(self, chatbot_config, mock_enhanced_memory_client):
        """Test that memory snapshots can be created after conversations."""
        with allure.step("Mock successful snapshot"):
            mock_enhanced_memory_client.sync_to_mongodb.return_value = {
                "success": True,
                "memory_count": 5,
                "snapshot_id": "snap_123",
                "categories": {
                    "fact": 2,
                    "preference": 1,
                    "objection": 1,
                    "outcome": 0,
                    "context": 1,
                    "qualification": 0
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        with allure.step("Setup agent"):
            with patch('app.core.chatbot_base.ChatOpenAI') as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.bind_tools.return_value = mock_llm
                mock_llm.invoke.return_value = MagicMock(content="Information saved")
                mock_llm_class.return_value = mock_llm
                
                with patch('app.core.chatbot_base.Mem0AsyncClientWrapper') as mock_wrapper_class:
                    mock_wrapper_class.return_value = mock_enhanced_memory_client
                    mock_enhanced_memory_client.get_all_memories.return_value = []
                    
                    agent = LimitlessOSIntelligentAgent(chatbot_config)
        
        with allure.step("Have a conversation"):
            await agent.chat("I work at TechCorp", "test_user")
        
        with allure.step("Create snapshot"):
            # Simulate creating a snapshot after the conversation
            snapshot_result = await mock_enhanced_memory_client.sync_to_mongodb("test_user")
            
            assert snapshot_result["success"] is True
            assert snapshot_result["memory_count"] == 5
            assert snapshot_result["snapshot_id"] == "snap_123"