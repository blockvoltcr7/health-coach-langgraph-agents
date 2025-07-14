"""Tests for Mem0 Async Client Wrapper.

This module contains comprehensive tests for the reusable Mem0 async client wrapper,
including all memory operations, error handling, and integration scenarios.
"""

import pytest
import allure
from typing import Dict, List, Any
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from app.mem0.mem0AsyncClient import (
    Mem0AsyncClientWrapper,
    MemoryConfig,
    MemoryEntry,
    MemorySearchResult,
    MemoryCategory,
    get_mem0_client,
    add_conversation_memory,
    search_user_memories,
    get_user_memory_context,
    shutdown_mem0_client
)


@allure.epic("Memory Management")
@allure.feature("Mem0 Async Client Wrapper")
class TestMem0AsyncClientWrapper:
    """Test class for Mem0 async client wrapper functionality."""
    
    @pytest.fixture
    def memory_config(self) -> MemoryConfig:
        """Create a test memory configuration."""
        return MemoryConfig(
            api_key="test_api_key_12345",
            output_format="v1.1",
            max_retries=2,
            timeout=10
        )
    
    @pytest.fixture
    def mock_mem0_client(self):
        """Create a mock Mem0 AsyncMemoryClient."""
        mock_client = AsyncMock()
        return mock_client
    
    @pytest.fixture
    async def client_wrapper(self, memory_config: MemoryConfig) -> Mem0AsyncClientWrapper:
        """Create a client wrapper instance for testing."""
        with patch('app.mem0.mem0AsyncClient.AsyncMemoryClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            wrapper = Mem0AsyncClientWrapper(memory_config)
            await wrapper._ensure_initialized()
            return wrapper
    
    @allure.story("Client Initialization")
    @allure.severity(allure.severity_level.CRITICAL)
    async def test_client_initialization_success(self, memory_config: MemoryConfig):
        """Test successful client initialization."""
        with allure.step("Initialize client wrapper with valid config"):
            with patch('app.mem0.mem0AsyncClient.AsyncMemoryClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                
                wrapper = Mem0AsyncClientWrapper(memory_config)
                
                assert wrapper.config.api_key == "test_api_key_12345"
                assert wrapper.config.output_format == "v1.1"
                assert not wrapper._initialized
        
        with allure.step("Ensure initialization"):
            await wrapper._ensure_initialized()
            
            assert wrapper._initialized
            assert wrapper._client is not None
            mock_client_class.assert_called_once_with(api_key="test_api_key_12345")
    
    @allure.story("Client Initialization")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_client_initialization_missing_api_key(self):
        """Test client initialization with missing API key."""
        with allure.step("Attempt to initialize without API key"):
            config = MemoryConfig(api_key=None)
            
            with pytest.raises(ValueError, match="MEM0_API_KEY is required"):
                Mem0AsyncClientWrapper(config)
    
    @allure.story("Client Initialization")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_client_initialization_mem0_unavailable(self):
        """Test client initialization when mem0 package is unavailable."""
        with allure.step("Mock mem0 as unavailable"):
            with patch('app.mem0.mem0AsyncClient.MEM0_AVAILABLE', False):
                config = MemoryConfig(api_key="test_key")
                
                with pytest.raises(ImportError, match="Mem0 package not found"):
                    Mem0AsyncClientWrapper(config)
    
    @allure.story("Memory Addition")
    @allure.severity(allure.severity_level.CRITICAL)
    async def test_add_memory_success(self, client_wrapper: Mem0AsyncClientWrapper):
        """Test successful memory addition."""
        with allure.step("Prepare test data"):
            messages = [
                {"role": "user", "content": "I like morning workouts"},
                {"role": "assistant", "content": "I'll remember you prefer morning workouts"}
            ]
            user_id = "test_user_123"
            metadata = {"source": "test", "timestamp": "2024-01-01"}
            
            expected_result = {"id": "mem_123", "status": "success"}
        
        with allure.step("Mock successful API response"):
            client_wrapper._client.add.return_value = expected_result
        
        with allure.step("Add memory"):
            result = await client_wrapper.add_memory(messages, user_id, metadata)
            
            assert result == expected_result
            client_wrapper._client.add.assert_called_once_with(
                messages=messages,
                user_id=user_id,
                output_format="v1.1",
                metadata=metadata
            )
    
    @allure.story("Memory Addition")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_add_memory_validation_errors(self, client_wrapper: Mem0AsyncClientWrapper):
        """Test memory addition with validation errors."""
        with allure.step("Test empty messages"):
            with pytest.raises(ValueError, match="Messages cannot be empty"):
                await client_wrapper.add_memory([], "user_123")
        
        with allure.step("Test missing user ID"):
            messages = [{"role": "user", "content": "test"}]
            with pytest.raises(ValueError, match="User ID is required"):
                await client_wrapper.add_memory(messages, "")
        
        with allure.step("Test invalid message format"):
            invalid_messages = [{"invalid": "format"}]
            with pytest.raises(ValueError, match="Each message must be a dict"):
                await client_wrapper.add_memory(invalid_messages, "user_123")
    
    @allure.story("Memory Search")
    @allure.severity(allure.severity_level.CRITICAL)
    async def test_search_memories_success(self, client_wrapper: Mem0AsyncClientWrapper):
        """Test successful memory search."""
        with allure.step("Prepare test data"):
            query = "workout preferences"
            user_id = "test_user_123"
            limit = 5
            
            mock_response = {
                "memories": [
                    {
                        "id": "mem_1",
                        "memory": "User prefers morning workouts",
                        "created_at": "2024-01-01T10:00:00Z",
                        "updated_at": "2024-01-01T10:00:00Z",
                        "metadata": {"source": "chat"}
                    },
                    {
                        "id": "mem_2",
                        "memory": "User likes cardio exercises",
                        "created_at": "2024-01-02T10:00:00Z",
                        "updated_at": "2024-01-02T10:00:00Z",
                        "metadata": {"source": "chat"}
                    }
                ]
            }
        
        with allure.step("Mock successful API response"):
            client_wrapper._client.search.return_value = mock_response
        
        with allure.step("Search memories"):
            result = await client_wrapper.search_memories(query, user_id, limit)
            
            assert isinstance(result, MemorySearchResult)
            assert result.total_count == 2
            assert result.query == query
            assert len(result.memories) == 2
            assert result.memories[0].memory == "User prefers morning workouts"
            
            client_wrapper._client.search.assert_called_once_with(
                query=query,
                user_id=user_id,
                limit=limit,
                output_format="v1.1"
            )
    
    @allure.story("Memory Retrieval")
    @allure.severity(allure.severity_level.CRITICAL)
    async def test_get_all_memories_success(self, client_wrapper: Mem0AsyncClientWrapper):
        """Test successful retrieval of all memories."""
        with allure.step("Prepare test data"):
            user_id = "test_user_123"
            
            mock_response = {
                "memories": [
                    {
                        "id": "mem_1",
                        "memory": "User is a software developer",
                        "created_at": "2024-01-01T10:00:00Z",
                        "metadata": {}
                    },
                    {
                        "id": "mem_2",
                        "memory": "User prefers Python programming",
                        "created_at": "2024-01-02T10:00:00Z",
                        "metadata": {}
                    }
                ]
            }
        
        with allure.step("Mock successful API response"):
            client_wrapper._client.get_all.return_value = mock_response
        
        with allure.step("Get all memories"):
            result = await client_wrapper.get_all_memories(user_id)
            
            assert len(result) == 2
            assert all(isinstance(mem, MemoryEntry) for mem in result)
            assert result[0].memory == "User is a software developer"
            assert result[1].memory == "User prefers Python programming"
            
            client_wrapper._client.get_all.assert_called_once_with(
                user_id=user_id,
                output_format="v1.1"
            )
    
    @allure.story("Memory Update")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_update_memory_success(self, client_wrapper: Mem0AsyncClientWrapper):
        """Test successful memory update."""
        with allure.step("Prepare test data"):
            memory_id = "mem_123"
            data = {"memory": "Updated memory content"}
            user_id = "test_user_123"
            expected_result = {"id": memory_id, "status": "updated"}
        
        with allure.step("Mock successful API response"):
            client_wrapper._client.update.return_value = expected_result
        
        with allure.step("Update memory"):
            result = await client_wrapper.update_memory(memory_id, data, user_id)
            
            assert result == expected_result
            client_wrapper._client.update.assert_called_once_with(
                memory_id=memory_id,
                data=data,
                user_id=user_id,
                output_format="v1.1"
            )
    
    @allure.story("Memory Deletion")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_delete_memory_success(self, client_wrapper: Mem0AsyncClientWrapper):
        """Test successful memory deletion."""
        with allure.step("Prepare test data"):
            memory_id = "mem_123"
            user_id = "test_user_123"
            expected_result = {"id": memory_id, "status": "deleted"}
        
        with allure.step("Mock successful API response"):
            client_wrapper._client.delete.return_value = expected_result
        
        with allure.step("Delete memory"):
            result = await client_wrapper.delete_memory(memory_id, user_id)
            
            assert result == expected_result
            client_wrapper._client.delete.assert_called_once_with(
                memory_id=memory_id,
                user_id=user_id
            )
    
    @allure.story("Memory Deletion")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_delete_all_memories_success(self, client_wrapper: Mem0AsyncClientWrapper):
        """Test successful deletion of all memories."""
        with allure.step("Prepare test data"):
            user_id = "test_user_123"
            expected_result = {"user_id": user_id, "status": "all_deleted"}
        
        with allure.step("Mock successful API response"):
            client_wrapper._client.delete_all.return_value = expected_result
        
        with allure.step("Delete all memories"):
            result = await client_wrapper.delete_all_memories(user_id)
            
            assert result == expected_result
            client_wrapper._client.delete_all.assert_called_once_with(user_id=user_id)
    
    @allure.story("Error Handling")
    @allure.severity(allure.severity_level.CRITICAL)
    async def test_retry_mechanism(self, client_wrapper: Mem0AsyncClientWrapper):
        """Test retry mechanism on failures."""
        with allure.step("Prepare test data"):
            messages = [{"role": "user", "content": "test"}]
            user_id = "test_user_123"
        
        with allure.step("Mock API to fail twice then succeed"):
            client_wrapper._client.add.side_effect = [
                Exception("Network error"),
                Exception("Timeout error"),
                {"id": "mem_123", "status": "success"}
            ]
        
        with allure.step("Add memory with retries"):
            result = await client_wrapper.add_memory(messages, user_id)
            
            assert result == {"id": "mem_123", "status": "success"}
            assert client_wrapper._client.add.call_count == 3
    
    @allure.story("Health Check")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_health_check_success(self, client_wrapper: Mem0AsyncClientWrapper):
        """Test successful health check."""
        with allure.step("Mock successful health check operations"):
            client_wrapper._client.add.return_value = {"id": "health_mem", "status": "success"}
            client_wrapper._client.delete_all.return_value = {"status": "deleted"}
        
        with allure.step("Perform health check"):
            result = await client_wrapper.health_check()
            
            assert result["status"] == "healthy"
            assert result["service"] == "Mem0 AsyncMemoryClient"
            assert "timestamp" in result
            assert result["message"] == "Memory service is operational"
    
    @allure.story("Health Check")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_health_check_failure(self, client_wrapper: Mem0AsyncClientWrapper):
        """Test health check failure."""
        with allure.step("Mock health check failure"):
            client_wrapper._client.add.side_effect = Exception("Service unavailable")
        
        with allure.step("Perform health check"):
            result = await client_wrapper.health_check()
            
            assert result["status"] == "unhealthy"
            assert result["service"] == "Mem0 AsyncMemoryClient"
            assert "error" in result
            assert result["message"] == "Memory service is experiencing issues"


@allure.epic("Memory Management")
@allure.feature("Global Client Management")
class TestGlobalClientManagement:
    """Test class for global client management functions."""
    
    @allure.story("Global Client Access")
    @allure.severity(allure.severity_level.CRITICAL)
    async def test_get_global_client(self):
        """Test getting the global client instance."""
        with allure.step("Clear any existing global client"):
            await shutdown_mem0_client()
        
        with allure.step("Mock AsyncMemoryClient"):
            with patch('app.mem0.mem0AsyncClient.AsyncMemoryClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                
                with patch.dict('os.environ', {'MEM0_API_KEY': 'test_key'}):
                    client1 = await get_mem0_client()
                    client2 = await get_mem0_client()
                    
                    # Should return the same instance
                    assert client1 is client2
                    assert isinstance(client1, Mem0AsyncClientWrapper)
        
        with allure.step("Cleanup"):
            await shutdown_mem0_client()
    
    @allure.story("Convenience Functions")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_add_conversation_memory_convenience(self):
        """Test convenience function for adding conversation memory."""
        with allure.step("Mock global client"):
            with patch('app.mem0.mem0AsyncClient.get_mem0_client') as mock_get_client:
                mock_client = AsyncMock()
                mock_client.add_memory.return_value = {"id": "conv_123", "status": "success"}
                mock_get_client.return_value = mock_client
                
                result = await add_conversation_memory(
                    user_message="Hello",
                    assistant_message="Hi there!",
                    user_id="user_123",
                    metadata={"source": "test"}
                )
                
                assert result == {"id": "conv_123", "status": "success"}
                mock_client.add_memory.assert_called_once_with(
                    [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!"}
                    ],
                    "user_123",
                    {"source": "test"}
                )
    
    @allure.story("Convenience Functions")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_search_user_memories_convenience(self):
        """Test convenience function for searching user memories."""
        with allure.step("Mock global client"):
            with patch('app.mem0.mem0AsyncClient.get_mem0_client') as mock_get_client:
                mock_client = AsyncMock()
                mock_search_result = MemorySearchResult(
                    memories=[],
                    total_count=0,
                    query="test query"
                )
                mock_client.search_memories.return_value = mock_search_result
                mock_get_client.return_value = mock_client
                
                result = await search_user_memories("test query", "user_123", 5)
                
                assert result == mock_search_result
                mock_client.search_memories.assert_called_once_with("test query", "user_123", 5)
    
    @allure.story("Convenience Functions")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_get_user_memory_context_convenience(self):
        """Test convenience function for getting user memory context."""
        with allure.step("Mock global client"):
            with patch('app.mem0.mem0AsyncClient.get_mem0_client') as mock_get_client:
                mock_client = AsyncMock()
                mock_memories = [
                    MemoryEntry(
                        id="mem_1",
                        memory="User likes Python",
                        user_id="user_123"
                    ),
                    MemoryEntry(
                        id="mem_2",
                        memory="User is a developer",
                        user_id="user_123"
                    )
                ]
                mock_client.get_memory_history.return_value = mock_memories
                mock_get_client.return_value = mock_client
                
                result = await get_user_memory_context("user_123", 10)
                
                expected_context = "Memory: User likes Python\nMemory: User is a developer"
                assert result == expected_context
                mock_client.get_memory_history.assert_called_once_with("user_123", 10)
    
    @allure.story("Convenience Functions")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_get_user_memory_context_no_memories(self):
        """Test memory context when no memories exist."""
        with allure.step("Mock global client with no memories"):
            with patch('app.mem0.mem0AsyncClient.get_mem0_client') as mock_get_client:
                mock_client = AsyncMock()
                mock_client.get_memory_history.return_value = []
                mock_get_client.return_value = mock_client
                
                result = await get_user_memory_context("user_123", 10)
                
                assert result == "No previous conversation history found."


@allure.epic("Memory Management")
@allure.feature("Data Models")
class TestDataModels:
    """Test class for Pydantic data models."""
    
    @allure.story("Memory Entry Model")
    @allure.severity(allure.severity_level.NORMAL)
    def test_memory_entry_model(self):
        """Test MemoryEntry model validation."""
        with allure.step("Create valid memory entry"):
            entry = MemoryEntry(
                id="mem_123",
                memory="Test memory content",
                user_id="user_123",
                created_at=datetime.now(),
                metadata={"source": "test"}
            )
            
            assert entry.id == "mem_123"
            assert entry.memory == "Test memory content"
            assert entry.user_id == "user_123"
            assert entry.metadata == {"source": "test"}
    
    @allure.story("Memory Search Result Model")
    @allure.severity(allure.severity_level.NORMAL)
    def test_memory_search_result_model(self):
        """Test MemorySearchResult model validation."""
        with allure.step("Create valid search result"):
            memories = [
                MemoryEntry(id="mem_1", memory="Memory 1", user_id="user_123"),
                MemoryEntry(id="mem_2", memory="Memory 2", user_id="user_123")
            ]
            
            result = MemorySearchResult(
                memories=memories,
                total_count=2,
                query="test query"
            )
            
            assert len(result.memories) == 2
            assert result.total_count == 2
            assert result.query == "test query"
    
    @allure.story("Memory Config Model")
    @allure.severity(allure.severity_level.NORMAL)
    def test_memory_config_model(self):
        """Test MemoryConfig model validation."""
        with allure.step("Create valid memory config"):
            config = MemoryConfig(
                api_key="test_key",
                output_format="v1.1",
                max_retries=5,
                timeout=60
            )
            
            assert config.api_key == "test_key"
            assert config.output_format == "v1.1"
            assert config.max_retries == 5
            assert config.timeout == 60
    
    @allure.story("Memory Config Model")
    @allure.severity(allure.severity_level.NORMAL)
    def test_memory_config_auto_api_key_loading(self):
        """Test automatic API key loading from environment."""
        with allure.step("Test auto-loading API key from environment"):
            with patch.dict('os.environ', {'MEM0_API_KEY': 'env_test_key'}):
                config = MemoryConfig()
                assert config.api_key == 'env_test_key'


@allure.epic("Memory Management")
@allure.feature("Enhanced Memory Features")
class TestEnhancedMemoryFeatures:
    """Test class for enhanced memory features including categorization and scoring."""
    
    @pytest.fixture
    async def client_wrapper(self) -> Mem0AsyncClientWrapper:
        """Create a client wrapper instance for testing."""
        with patch('app.mem0.mem0AsyncClient.AsyncMemoryClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            config = MemoryConfig(api_key="test_key")
            wrapper = Mem0AsyncClientWrapper(config)
            await wrapper._ensure_initialized()
            return wrapper
    
    @allure.story("Memory Categorization")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_memory_category_enum(self):
        """Test MemoryCategory enum values."""
        with allure.step("Verify all category values"):
            assert MemoryCategory.FACT.value == "fact"
            assert MemoryCategory.PREFERENCE.value == "preference"
            assert MemoryCategory.OBJECTION.value == "objection"
            assert MemoryCategory.OUTCOME.value == "outcome"
            assert MemoryCategory.CONTEXT.value == "context"
            assert MemoryCategory.QUALIFICATION.value == "qualification"
    
    @allure.story("Memory Entry Enhancement")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_enhanced_memory_entry_model(self):
        """Test enhanced MemoryEntry with category and scoring."""
        with allure.step("Create enhanced memory entry"):
            from datetime import datetime, timezone, timedelta
            now = datetime.now(timezone.utc)
            
            entry = MemoryEntry(
                id="mem_123",
                memory="User has a budget of $50,000",
                user_id="user_123",
                category=MemoryCategory.QUALIFICATION,
                importance_score=8.5,
                access_count=5,
                last_accessed=now - timedelta(days=2),
                created_at=now - timedelta(days=10),
                metadata={"source": "sales_call"}
            )
            
            assert entry.category == MemoryCategory.QUALIFICATION
            assert entry.importance_score == 8.5
            assert entry.access_count == 5
            assert entry.last_accessed < now
    
    @allure.story("Decay Factor Calculation")
    @allure.severity(allure.severity_level.NORMAL)
    def test_memory_decay_factor_calculation(self):
        """Test decay factor calculation based on last access time."""
        from datetime import datetime, timezone, timedelta
        
        with allure.step("Test fresh memory (0 days old)"):
            entry = MemoryEntry(
                memory="Fresh memory",
                user_id="user_123",
                importance_score=5.0,
                last_accessed=datetime.now(timezone.utc)
            )
            assert entry.decay_factor == 1.0
            assert entry.effective_score == 5.0
        
        with allure.step("Test 5-day old memory"):
            old_entry = MemoryEntry(
                memory="Old memory",
                user_id="user_123",
                importance_score=5.0,
                last_accessed=datetime.now(timezone.utc) - timedelta(days=5)
            )
            expected_decay = 0.95 ** 5
            assert abs(old_entry.decay_factor - expected_decay) < 0.001
            assert abs(old_entry.effective_score - (5.0 * expected_decay)) < 0.01
        
        with allure.step("Test memory without last_accessed"):
            no_access_entry = MemoryEntry(
                memory="Never accessed",
                user_id="user_123",
                importance_score=5.0
            )
            assert no_access_entry.decay_factor == 1.0
    
    @allure.story("Access Count Update")
    @allure.severity(allure.severity_level.NORMAL)
    def test_memory_access_update(self):
        """Test memory access count and timestamp update."""
        from datetime import datetime, timezone
        
        with allure.step("Create memory and update access"):
            entry = MemoryEntry(
                memory="Test memory",
                user_id="user_123",
                importance_score=5.0,
                access_count=0
            )
            
            original_count = entry.access_count
            original_score = entry.importance_score
            
            entry.update_access()
            
            assert entry.access_count == original_count + 1
            assert entry.importance_score == min(10.0, original_score * 1.05)
            assert entry.last_accessed is not None
            assert isinstance(entry.last_accessed, datetime)
    
    @allure.story("Auto-Categorization")
    @allure.severity(allure.severity_level.CRITICAL)
    async def test_auto_categorize_memory(self, client_wrapper: Mem0AsyncClientWrapper):
        """Test automatic memory categorization based on content."""
        test_cases = [
            # (messages, expected_category)
            ([
                {"role": "user", "content": "I have a budget of $50,000 for this project"},
                {"role": "assistant", "content": "Great, $50,000 budget noted"}
            ], MemoryCategory.QUALIFICATION),
            
            ([
                {"role": "user", "content": "I prefer morning meetings"},
                {"role": "assistant", "content": "I'll remember you prefer morning meetings"}
            ], MemoryCategory.PREFERENCE),
            
            ([
                {"role": "user", "content": "I'm concerned about the price, it seems expensive"},
                {"role": "assistant", "content": "I understand your concern about pricing"}
            ], MemoryCategory.OBJECTION),
            
            ([
                {"role": "user", "content": "We've decided to move forward with the contract"},
                {"role": "assistant", "content": "Excellent decision!"}
            ], MemoryCategory.OUTCOME),
            
            ([
                {"role": "user", "content": "I work as a software engineer at Google"},
                {"role": "assistant", "content": "Software engineer at Google, got it"}
            ], MemoryCategory.FACT),
            
            ([
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"}
            ], MemoryCategory.CONTEXT)
        ]
        
        for messages, expected_category in test_cases:
            with allure.step(f"Test categorization for {expected_category.value}"):
                category = client_wrapper._auto_categorize_memory(messages)
                assert category == expected_category, f"Expected {expected_category}, got {category}"
    
    @allure.story("Enhanced Memory Addition")
    @allure.severity(allure.severity_level.CRITICAL)
    async def test_add_memory_with_category_and_score(self, client_wrapper: Mem0AsyncClientWrapper):
        """Test adding memory with category and importance score."""
        with allure.step("Prepare test data"):
            messages = [
                {"role": "user", "content": "My budget is $100,000"},
                {"role": "assistant", "content": "$100,000 budget noted"}
            ]
            user_id = "test_user_123"
            
            expected_result = {"id": "mem_enhanced_123", "status": "success"}
            client_wrapper._client.add.return_value = expected_result
        
        with allure.step("Add memory with explicit category and score"):
            result = await client_wrapper.add_memory(
                messages=messages,
                user_id=user_id,
                category=MemoryCategory.QUALIFICATION,
                importance_score=9.0
            )
            
            assert result == expected_result
            
            # Verify the metadata was enhanced correctly
            call_args = client_wrapper._client.add.call_args
            metadata = call_args.kwargs["metadata"]
            assert metadata["category"] == "qualification"
            assert metadata["importance_score"] == 9.0
            assert metadata["access_count"] == 1
            assert "last_accessed" in metadata
            assert "auto_categorized" not in metadata
    
    @allure.story("Enhanced Memory Search")
    @allure.severity(allure.severity_level.CRITICAL)
    async def test_search_memories_with_filters(self, client_wrapper: Mem0AsyncClientWrapper):
        """Test searching memories with category and importance filters."""
        with allure.step("Prepare mock response"):
            mock_response = {
                "memories": [
                    {
                        "id": "mem_1",
                        "memory": "High importance qualification",
                        "metadata": {
                            "category": "qualification",
                            "importance_score": 8.5,
                            "access_count": 3
                        }
                    },
                    {
                        "id": "mem_2",
                        "memory": "Low importance context",
                        "metadata": {
                            "category": "context",
                            "importance_score": 2.0,
                            "access_count": 1
                        }
                    },
                    {
                        "id": "mem_3",
                        "memory": "Medium importance preference",
                        "metadata": {
                            "category": "preference",
                            "importance_score": 5.0,
                            "access_count": 2
                        }
                    }
                ]
            }
            client_wrapper._client.search.return_value = mock_response
        
        with allure.step("Search with category filter"):
            result = await client_wrapper.search_memories(
                query="test",
                user_id="user_123",
                category=MemoryCategory.QUALIFICATION
            )
            
            assert len(result.memories) == 1
            assert result.memories[0].category == MemoryCategory.QUALIFICATION
        
        with allure.step("Search with importance score filter"):
            client_wrapper._client.search.return_value = mock_response
            result = await client_wrapper.search_memories(
                query="test",
                user_id="user_123",
                min_importance_score=5.0
            )
            
            # Should return memories with score >= 5.0
            assert len(result.memories) == 2
            assert all(m.importance_score >= 5.0 for m in result.memories)
    
    @allure.story("Get Memories by Importance")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_get_memories_by_importance(self, client_wrapper: Mem0AsyncClientWrapper):
        """Test retrieving memories sorted by importance."""
        from datetime import datetime, timezone, timedelta
        
        with allure.step("Mock get_all_memories response"):
            mock_memories = [
                MemoryEntry(
                    id="mem_1",
                    memory="Very important",
                    user_id="user_123",
                    category=MemoryCategory.QUALIFICATION,
                    importance_score=9.0,
                    last_accessed=datetime.now(timezone.utc)
                ),
                MemoryEntry(
                    id="mem_2",
                    memory="Less important",
                    user_id="user_123",
                    category=MemoryCategory.CONTEXT,
                    importance_score=3.0,
                    last_accessed=datetime.now(timezone.utc) - timedelta(days=10)
                ),
                MemoryEntry(
                    id="mem_3",
                    memory="Medium importance",
                    user_id="user_123",
                    category=MemoryCategory.PREFERENCE,
                    importance_score=6.0,
                    last_accessed=datetime.now(timezone.utc) - timedelta(days=2)
                )
            ]
            
            # Mock the get_all_memories method
            client_wrapper.get_all_memories = AsyncMock(return_value=mock_memories)
        
        with allure.step("Get memories by importance with decay"):
            result = await client_wrapper.get_memories_by_importance(
                user_id="user_123",
                limit=2,
                min_score=2.0,
                include_decay=True
            )
            
            assert len(result) == 2
            # First should be the 9.0 score (fresh)
            assert result[0].importance_score == 9.0
            # Second should be the 6.0 score (2 days old, so some decay)
            assert result[1].importance_score == 6.0
        
        with allure.step("Get memories by importance without decay"):
            result = await client_wrapper.get_memories_by_importance(
                user_id="user_123",
                limit=3,
                include_decay=False
            )
            
            assert len(result) == 3
            # Should be sorted by raw importance score
            assert result[0].importance_score == 9.0
            assert result[1].importance_score == 6.0
            assert result[2].importance_score == 3.0
        
        with allure.step("Filter by category"):
            result = await client_wrapper.get_memories_by_importance(
                user_id="user_123",
                category=MemoryCategory.QUALIFICATION,
                limit=10
            )
            
            assert len(result) == 1
            assert result[0].category == MemoryCategory.QUALIFICATION
    
    @allure.story("Memory Analytics")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_get_memory_analytics(self, client_wrapper: Mem0AsyncClientWrapper):
        """Test memory analytics calculation."""
        with allure.step("Mock memories for analytics"):
            mock_memories = [
                MemoryEntry(
                    id=f"mem_{i}",
                    memory=f"Memory {i}",
                    user_id="user_123",
                    category=MemoryCategory.FACT if i < 3 else MemoryCategory.PREFERENCE,
                    importance_score=float(i + 1),
                    access_count=i * 2
                )
                for i in range(5)
            ]
            
            client_wrapper.get_all_memories = AsyncMock(return_value=mock_memories)
        
        with allure.step("Get analytics"):
            analytics = await client_wrapper.get_memory_analytics("user_123")
            
            assert analytics["total_memories"] == 5
            assert analytics["categories"]["fact"] == 3
            assert analytics["categories"]["preference"] == 2
            assert analytics["avg_importance_score"] == 3.0  # (1+2+3+4+5)/5
            assert analytics["avg_access_count"] == 4.0  # (0+2+4+6+8)/5
            assert analytics["importance_distribution"]["high"] == 0  # None >= 7
            assert analytics["importance_distribution"]["medium"] == 3  # 3,4,5 are [3,7)
            assert analytics["importance_distribution"]["low"] == 2  # 1,2 are < 3
            assert len(analytics["most_accessed"]) == 5
        
        with allure.step("Test empty analytics"):
            client_wrapper.get_all_memories = AsyncMock(return_value=[])
            empty_analytics = await client_wrapper.get_memory_analytics("user_123")
            
            assert empty_analytics["total_memories"] == 0
            assert empty_analytics["categories"] == {}
            assert empty_analytics["avg_importance_score"] == 0
            assert empty_analytics["avg_access_count"] == 0
    
    @allure.story("MongoDB Sync")
    @allure.severity(allure.severity_level.CRITICAL)
    async def test_sync_to_mongodb(self, client_wrapper: Mem0AsyncClientWrapper):
        """Test syncing memories to MongoDB."""
        from datetime import datetime, timezone
        
        with allure.step("Mock memories to sync"):
            mock_memories = [
                MemoryEntry(
                    id="mem_1",
                    memory="Budget is $50k",
                    user_id="user_123",
                    category=MemoryCategory.QUALIFICATION,
                    importance_score=8.0,
                    access_count=2,
                    last_accessed=datetime.now(timezone.utc),
                    created_at=datetime.now(timezone.utc),
                    metadata={"source": "chat"}
                )
            ]
            
            client_wrapper.get_all_memories = AsyncMock(return_value=mock_memories)
        
        with allure.step("Sync without repository (test mode)"):
            result = await client_wrapper.sync_to_mongodb("user_123", None)
            
            assert result["success"] is True
            assert result["memory_count"] == 1
            assert result["categories"]["qualification"] == 1
            assert "timestamp" in result
        
        with allure.step("Test sync with no memories"):
            client_wrapper.get_all_memories = AsyncMock(return_value=[])
            result = await client_wrapper.sync_to_mongodb("user_123", None)
            
            assert result["success"] is False
            assert result["message"] == "No memories found for user"
            assert result["memory_count"] == 0
    
    @allure.story("MongoDB Restore")
    @allure.severity(allure.severity_level.CRITICAL)
    async def test_restore_from_mongodb(self, client_wrapper: Mem0AsyncClientWrapper):
        """Test restoring memories from MongoDB snapshot."""
        from datetime import datetime, timezone
        
        with allure.step("Mock snapshot repository"):
            mock_repo = MagicMock()
            mock_snapshot = {
                "_id": "snapshot_123",
                "user_id": "user_123",
                "memories": [
                    {
                        "memory_id": "mem_1",
                        "content": "Restored memory content",
                        "category": "fact",
                        "importance_score": 7.0,
                        "metadata": {"restored": True}
                    }
                ],
                "snapshot_timestamp": datetime.now(timezone.utc)
            }
            mock_repo.find_by_id.return_value = mock_snapshot
        
        with allure.step("Mock successful restoration"):
            client_wrapper._client.add.return_value = {"id": "restored_mem_1"}
            client_wrapper._client.delete_all.return_value = {"status": "deleted"}
            
            result = await client_wrapper.restore_from_mongodb(
                user_id="user_123",
                snapshot_id="snapshot_123",
                snapshot_repository=mock_repo,
                clear_existing=True
            )
            
            assert result["success"] is True
            assert result["restored_count"] == 1
            assert result["total_in_snapshot"] == 1
            assert len(result["errors"]) == 0
        
        with allure.step("Test restore with wrong user ID"):
            mock_repo.find_by_id.return_value = {**mock_snapshot, "user_id": "wrong_user"}
            
            result = await client_wrapper.restore_from_mongodb(
                user_id="user_123",
                snapshot_id="snapshot_123",
                snapshot_repository=mock_repo
            )
            
            assert result["success"] is False
            assert result["error"] == "User ID mismatch"
    
    @allure.story("Enhanced Convenience Functions")
    @allure.severity(allure.severity_level.NORMAL)
    async def test_enhanced_convenience_functions(self):
        """Test enhanced convenience functions with categorization."""
        with allure.step("Test add_conversation_memory with category"):
            with patch('app.mem0.mem0AsyncClient.get_mem0_client') as mock_get_client:
                mock_client = AsyncMock()
                mock_client.add_memory.return_value = {"id": "mem_123"}
                mock_get_client.return_value = mock_client
                
                result = await add_conversation_memory(
                    user_message="I need this by end of month",
                    assistant_message="End of month timeline noted",
                    user_id="user_123",
                    category=MemoryCategory.QUALIFICATION,
                    importance_score=7.5
                )
                
                assert result == {"id": "mem_123"}
                mock_client.add_memory.assert_called_once()
                call_args = mock_client.add_memory.call_args
                assert call_args.kwargs["category"] == MemoryCategory.QUALIFICATION
                assert call_args.kwargs["importance_score"] == 7.5
        
        with allure.step("Test search_user_memories with filters"):
            with patch('app.mem0.mem0AsyncClient.get_mem0_client') as mock_get_client:
                mock_client = AsyncMock()
                mock_result = MemorySearchResult(memories=[], total_count=0, query="test")
                mock_client.search_memories.return_value = mock_result
                mock_get_client.return_value = mock_client
                
                result = await search_user_memories(
                    query="budget",
                    user_id="user_123",
                    limit=5,
                    category=MemoryCategory.QUALIFICATION,
                    min_importance_score=5.0
                )
                
                assert result == mock_result
                mock_client.search_memories.assert_called_once_with(
                    "budget", "user_123", 5, None,
                    MemoryCategory.QUALIFICATION, 5.0
                ) 