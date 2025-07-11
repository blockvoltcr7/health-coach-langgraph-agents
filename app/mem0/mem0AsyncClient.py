"""Mem0 Async Client Wrapper for Reusable Memory Management.

This module provides a comprehensive wrapper around the Mem0 AsyncMemoryClient
with enhanced functionality, error handling, and logging for use across
multiple APIs and services.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

try:
    from mem0 import AsyncMemoryClient
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    AsyncMemoryClient = None

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class MemoryEntry(BaseModel):
    """Pydantic model for memory entries."""
    id: Optional[str] = None
    memory: str = Field(..., description="The memory content")
    user_id: str = Field(..., description="User identifier")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class MemorySearchResult(BaseModel):
    """Pydantic model for memory search results."""
    memories: List[MemoryEntry] = Field(default_factory=list)
    total_count: int = 0
    query: str = ""


class MemoryConfig(BaseModel):
    """Configuration for Mem0 client."""
    api_key: Optional[str] = Field(default=None, description="Mem0 API key")
    output_format: str = Field(default="v1.1", description="Output format for mem0")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    
    def __init__(self, **data):
        """Initialize with automatic API key loading."""
        super().__init__(**data)
        if self.api_key is None:
            self.api_key = os.getenv("MEM0_API_KEY")


class Mem0AsyncClientWrapper:
    """Enhanced wrapper around Mem0 AsyncMemoryClient with additional functionality.
    
    This wrapper provides:
    - Enhanced error handling and logging
    - Retry mechanisms
    - Data validation with Pydantic models
    - Connection management
    - Standardized response formats
    - Comprehensive memory operations
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize the Mem0 client wrapper.
        
        Args:
            config: Optional MemoryConfig. If None, uses default configuration.
        """
        self.config = config or MemoryConfig()
        self._client: Optional[AsyncMemoryClient] = None
        self._initialized = False
        
        if not MEM0_AVAILABLE:
            logger.error("Mem0 is not available. Install with: uv add mem0ai")
            raise ImportError("Mem0 package not found. Install with: uv add mem0ai")
        
        if not self.config.api_key:
            logger.error("MEM0_API_KEY not found in environment variables")
            raise ValueError("MEM0_API_KEY is required. Set it in environment variables.")
    
    async def _ensure_initialized(self) -> None:
        """Ensure the client is properly initialized."""
        if not self._initialized:
            try:
                self._client = AsyncMemoryClient(api_key=self.config.api_key)
                self._initialized = True
                logger.info("Mem0 AsyncMemoryClient initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Mem0 client: {e}")
                raise
    
    @asynccontextmanager
    async def _with_retries(self, operation_name: str):
        """Context manager for retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                yield
                break
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Operation {operation_name} failed after {self.config.max_retries} attempts: {e}")
                    raise
                else:
                    logger.warning(f"Operation {operation_name} attempt {attempt + 1} failed: {e}. Retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def add_memory(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add a new memory entry.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            user_id: User identifier for the memory
            metadata: Optional additional metadata
            
        Returns:
            Dict[str, Any]: Response from mem0 API
            
        Raises:
            ValueError: If required parameters are missing
            Exception: If memory addition fails
        """
        await self._ensure_initialized()
        
        if not messages:
            raise ValueError("Messages cannot be empty")
        if not user_id:
            raise ValueError("User ID is required")
        
        # Validate message format
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must be a dict with 'role' and 'content' keys")
        
        async with self._with_retries("add_memory"):
            try:
                result = await self._client.add(
                    messages=messages,
                    user_id=user_id,
                    output_format=self.config.output_format,
                    metadata=metadata or {}
                )
                
                logger.info(f"Successfully added memory for user {user_id}")
                return result
                
            except Exception as e:
                logger.error(f"Failed to add memory for user {user_id}: {e}")
                raise
    
    async def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> MemorySearchResult:
        """Search memories for a specific user.
        
        Args:
            query: Search query string
            user_id: User identifier
            limit: Maximum number of results to return
            metadata_filter: Optional metadata filter
            
        Returns:
            MemorySearchResult: Structured search results
        """
        await self._ensure_initialized()
        
        if not query:
            raise ValueError("Query cannot be empty")
        if not user_id:
            raise ValueError("User ID is required")
        
        async with self._with_retries("search_memories"):
            try:
                result = await self._client.search(
                    query=query,
                    user_id=user_id,
                    limit=limit,
                    output_format=self.config.output_format
                )
                
                # Parse and structure the response
                memories = []
                if isinstance(result, dict):
                    memory_list = result.get("memories", [])
                elif isinstance(result, list):
                    memory_list = result
                else:
                    memory_list = []
                
                for mem in memory_list:
                    if isinstance(mem, dict):
                        memory_entry = MemoryEntry(
                            id=mem.get("id"),
                            memory=mem.get("memory", ""),
                            user_id=user_id,
                            created_at=mem.get("created_at"),
                            updated_at=mem.get("updated_at"),
                            metadata=mem.get("metadata", {})
                        )
                        memories.append(memory_entry)
                
                search_result = MemorySearchResult(
                    memories=memories,
                    total_count=len(memories),
                    query=query
                )
                
                logger.info(f"Found {len(memories)} memories for user {user_id} with query: {query}")
                return search_result
                
            except Exception as e:
                logger.error(f"Failed to search memories for user {user_id}: {e}")
                raise
    
    async def get_all_memories(self, user_id: str) -> List[MemoryEntry]:
        """Retrieve ALL memories for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List[MemoryEntry]: List of all memory entries for the user
        """
        await self._ensure_initialized()
        
        if not user_id:
            raise ValueError("User ID is required")
        
        async with self._with_retries("get_all_memories"):
            try:
                result = await self._client.get_all(
                    user_id=user_id,
                    output_format=self.config.output_format
                )
                
                # Parse and structure the response
                memories = []
                if isinstance(result, dict):
                    memory_list = result.get("memories", [])
                elif isinstance(result, list):
                    memory_list = result
                else:
                    memory_list = []
                
                for mem in memory_list:
                    if isinstance(mem, dict):
                        memory_entry = MemoryEntry(
                            id=mem.get("id"),
                            memory=mem.get("memory", ""),
                            user_id=user_id,
                            created_at=mem.get("created_at"),
                            updated_at=mem.get("updated_at"),
                            metadata=mem.get("metadata", {})
                        )
                        memories.append(memory_entry)
                
                logger.info(f"Retrieved {len(memories)} total memories for user {user_id}")
                return memories
                
            except Exception as e:
                logger.error(f"Failed to get all memories for user {user_id}: {e}")
                raise
    
    async def update_memory(
        self,
        memory_id: str,
        data: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Update an existing memory entry.
        
        Args:
            memory_id: ID of the memory to update
            data: Updated memory data
            user_id: User identifier
            
        Returns:
            Dict[str, Any]: Response from mem0 API
        """
        await self._ensure_initialized()
        
        if not memory_id:
            raise ValueError("Memory ID is required")
        if not user_id:
            raise ValueError("User ID is required")
        
        async with self._with_retries("update_memory"):
            try:
                result = await self._client.update(
                    memory_id=memory_id,
                    data=data,
                    user_id=user_id,
                    output_format=self.config.output_format
                )
                
                logger.info(f"Successfully updated memory {memory_id} for user {user_id}")
                return result
                
            except Exception as e:
                logger.error(f"Failed to update memory {memory_id} for user {user_id}: {e}")
                raise
    
    async def delete_memory(self, memory_id: str, user_id: str) -> Dict[str, Any]:
        """Delete a specific memory entry.
        
        Args:
            memory_id: ID of the memory to delete
            user_id: User identifier
            
        Returns:
            Dict[str, Any]: Response from mem0 API
        """
        await self._ensure_initialized()
        
        if not memory_id:
            raise ValueError("Memory ID is required")
        if not user_id:
            raise ValueError("User ID is required")
        
        async with self._with_retries("delete_memory"):
            try:
                result = await self._client.delete(
                    memory_id=memory_id,
                    user_id=user_id
                )
                
                logger.info(f"Successfully deleted memory {memory_id} for user {user_id}")
                return result
                
            except Exception as e:
                logger.error(f"Failed to delete memory {memory_id} for user {user_id}: {e}")
                raise
    
    async def delete_all_memories(self, user_id: str) -> Dict[str, Any]:
        """Delete ALL memories for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict[str, Any]: Response from mem0 API
        """
        await self._ensure_initialized()
        
        if not user_id:
            raise ValueError("User ID is required")
        
        async with self._with_retries("delete_all_memories"):
            try:
                result = await self._client.delete_all(user_id=user_id)
                
                logger.info(f"Successfully deleted all memories for user {user_id}")
                return result
                
            except Exception as e:
                logger.error(f"Failed to delete all memories for user {user_id}: {e}")
                raise
    
    async def get_memory_history(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[MemoryEntry]:
        """Get memory history for a user (chronologically ordered).
        
        Args:
            user_id: User identifier
            limit: Maximum number of memories to return
            
        Returns:
            List[MemoryEntry]: List of memory entries ordered by creation time
        """
        memories = await self.get_all_memories(user_id)
        
        # Sort by created_at if available
        sorted_memories = sorted(
            memories,
            key=lambda x: x.created_at or datetime.min,
            reverse=True  # Most recent first
        )
        
        return sorted_memories[:limit]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the Mem0 service.
        
        Returns:
            Dict[str, Any]: Health check status
        """
        try:
            await self._ensure_initialized()
            
            # Try a simple operation to test connectivity
            test_user_id = "health_check_user"
            test_messages = [
                {"role": "user", "content": "health check"},
                {"role": "assistant", "content": "system operational"}
            ]
            
            # Add and immediately delete a test memory
            await self.add_memory(test_messages, test_user_id)
            await self.delete_all_memories(test_user_id)
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "service": "Mem0 AsyncMemoryClient",
                "message": "Memory service is operational"
            }
            
        except Exception as e:
            logger.error(f"Mem0 health check failed: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "service": "Mem0 AsyncMemoryClient",
                "error": str(e),
                "message": "Memory service is experiencing issues"
            }
    
    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            # Mem0 AsyncMemoryClient doesn't have explicit close method
            # but we can clean up our references
            self._client = None
            self._initialized = False
            logger.info("Mem0 client connection closed")


# Global client instance for reuse across the application
_global_mem0_client: Optional[Mem0AsyncClientWrapper] = None


async def get_mem0_client(config: Optional[MemoryConfig] = None) -> Mem0AsyncClientWrapper:
    """Get the global Mem0 client instance.
    
    Args:
        config: Optional MemoryConfig. If None, uses default configuration.
        
    Returns:
        Mem0AsyncClientWrapper: The client instance
    """
    global _global_mem0_client
    
    if _global_mem0_client is None:
        _global_mem0_client = Mem0AsyncClientWrapper(config)
        await _global_mem0_client._ensure_initialized()
    
    return _global_mem0_client


async def shutdown_mem0_client() -> None:
    """Shutdown the global Mem0 client."""
    global _global_mem0_client
    
    if _global_mem0_client:
        await _global_mem0_client.close()
        _global_mem0_client = None
        logger.info("Global Mem0 client shutdown completed")


# Convenience functions for common operations
async def add_conversation_memory(
    user_message: str,
    assistant_message: str,
    user_id: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function to add a conversation to memory.
    
    Args:
        user_message: The user's message
        assistant_message: The assistant's response
        user_id: User identifier
        metadata: Optional metadata
        
    Returns:
        Dict[str, Any]: Response from mem0 API
    """
    client = await get_mem0_client()
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]
    return await client.add_memory(messages, user_id, metadata)


async def search_user_memories(
    query: str,
    user_id: str,
    limit: int = 10
) -> MemorySearchResult:
    """Convenience function to search user memories.
    
    Args:
        query: Search query
        user_id: User identifier
        limit: Maximum results
        
    Returns:
        MemorySearchResult: Search results
    """
    client = await get_mem0_client()
    return await client.search_memories(query, user_id, limit)


async def get_user_memory_context(user_id: str, limit: int = 20) -> str:
    """Get formatted memory context for a user.
    
    Args:
        user_id: User identifier
        limit: Maximum number of memories to include
        
    Returns:
        str: Formatted memory context string
    """
    client = await get_mem0_client()
    memories = await client.get_memory_history(user_id, limit)
    
    if not memories:
        return "No previous conversation history found."
    
    context_lines = []
    for memory in memories:
        context_lines.append(f"Memory: {memory.memory}")
    
    return "\n".join(context_lines)
