"""Mem0 Async Client Wrapper for Reusable Memory Management.

This module provides a comprehensive wrapper around the Mem0 AsyncMemoryClient
with enhanced functionality, error handling, and logging for use across
multiple APIs and services.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
import asyncio
from contextlib import asynccontextmanager
from enum import Enum

try:
    from mem0 import AsyncMemoryClient
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    AsyncMemoryClient = None

from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class MemoryCategory(str, Enum):
    """Categories for memory classification."""
    FACT = "fact"  # Factual information about the user
    PREFERENCE = "preference"  # User preferences and likes/dislikes
    OBJECTION = "objection"  # Sales objections and concerns
    OUTCOME = "outcome"  # Conversation outcomes and decisions
    CONTEXT = "context"  # General contextual information
    QUALIFICATION = "qualification"  # BANT qualification data


class MemoryEntry(BaseModel):
    """Enhanced Pydantic model for memory entries with categorization and scoring."""
    id: Optional[str] = None
    memory: str = Field(..., description="The memory content")
    user_id: str = Field(..., description="User identifier")
    category: Optional[MemoryCategory] = Field(None, description="Memory category")
    importance_score: float = Field(1.0, ge=0.0, le=10.0, description="Importance score (0-10)")
    access_count: int = Field(0, ge=0, description="Number of times accessed")
    last_accessed: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @property
    def decay_factor(self) -> float:
        """Calculate decay factor based on time since last access."""
        if not self.last_accessed:
            return 1.0
        
        days_since_access = (datetime.now(timezone.utc) - self.last_accessed).days
        # Decay formula: score * (0.95 ^ days_since_access)
        return 0.95 ** days_since_access
    
    @property
    def effective_score(self) -> float:
        """Calculate effective importance score with decay."""
        return self.importance_score * self.decay_factor
    
    def update_access(self) -> None:
        """Update access count and timestamp."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)
        # Boost importance slightly on access
        self.importance_score = min(10.0, self.importance_score * 1.05)


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
    
    def _auto_categorize_memory(self, messages: List[Dict[str, str]]) -> MemoryCategory:
        """Auto-categorize memory based on content analysis.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            MemoryCategory: Best matching category
        """
        # Combine all message content for analysis
        combined_content = " ".join(msg["content"].lower() for msg in messages)
        
        # Keywords for each category
        objection_keywords = ["expensive", "cost", "price", "budget", "concern", "worry", 
                            "not sure", "doubt", "problem", "issue", "difficult"]
        preference_keywords = ["like", "prefer", "want", "need", "love", "hate", 
                             "enjoy", "favorite", "dislike", "wish"]
        qualification_keywords = ["budget", "timeline", "authority", "decision", 
                                "approve", "purchase", "buy", "invest"]
        outcome_keywords = ["decided", "agreed", "accepted", "rejected", "closed",
                           "deal", "contract", "signed", "committed"]
        fact_keywords = ["work", "company", "years", "experience", "currently",
                        "responsible", "manage", "department", "team"]
        
        # Count keyword matches
        scores = {
            MemoryCategory.OBJECTION: sum(1 for kw in objection_keywords if kw in combined_content),
            MemoryCategory.PREFERENCE: sum(1 for kw in preference_keywords if kw in combined_content),
            MemoryCategory.QUALIFICATION: sum(1 for kw in qualification_keywords if kw in combined_content),
            MemoryCategory.OUTCOME: sum(1 for kw in outcome_keywords if kw in combined_content),
            MemoryCategory.FACT: sum(1 for kw in fact_keywords if kw in combined_content),
        }
        
        # Return category with highest score, default to CONTEXT
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)
        return MemoryCategory.CONTEXT
    
    async def add_memory(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        category: Optional[MemoryCategory] = None,
        importance_score: float = 1.0
    ) -> Dict[str, Any]:
        """Add a new memory entry with categorization and importance scoring.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            user_id: User identifier for the memory
            metadata: Optional additional metadata
            category: Memory category (fact, preference, objection, etc.)
            importance_score: Initial importance score (0-10)
            
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
        
        # Enhance metadata with category and scoring
        enhanced_metadata = metadata or {}
        if category:
            enhanced_metadata["category"] = category.value
        enhanced_metadata["importance_score"] = max(0.0, min(10.0, importance_score))
        enhanced_metadata["access_count"] = 1
        enhanced_metadata["last_accessed"] = datetime.now(timezone.utc).isoformat()
        
        # Auto-categorize based on content if no category provided
        if not category:
            category = self._auto_categorize_memory(messages)
            enhanced_metadata["category"] = category.value
            enhanced_metadata["auto_categorized"] = True
        
        async with self._with_retries("add_memory"):
            try:
                result = await self._client.add(
                    messages=messages,
                    user_id=user_id,
                    output_format=self.config.output_format,
                    metadata=enhanced_metadata
                )
                
                logger.info(f"Successfully added {category.value} memory for user {user_id}")
                return result
                
            except Exception as e:
                logger.error(f"Failed to add memory for user {user_id}: {e}")
                raise
    
    async def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        category: Optional[MemoryCategory] = None,
        min_importance_score: Optional[float] = None
    ) -> MemorySearchResult:
        """Search memories for a specific user with category and importance filtering.
        
        Args:
            query: Search query string
            user_id: User identifier
            limit: Maximum number of results to return
            metadata_filter: Optional metadata filter
            category: Optional category filter
            min_importance_score: Optional minimum importance score filter
            
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
                        mem_metadata = mem.get("metadata", {})
                        
                        # Extract enhanced fields from metadata
                        mem_category = mem_metadata.get("category", "context")
                        try:
                            mem_category_enum = MemoryCategory(mem_category)
                        except ValueError:
                            mem_category_enum = MemoryCategory.CONTEXT
                        
                        importance_score = float(mem_metadata.get("importance_score", 1.0))
                        
                        # Apply filters
                        if category and mem_category_enum != category:
                            continue
                        if min_importance_score and importance_score < min_importance_score:
                            continue
                        
                        memory_entry = MemoryEntry(
                            id=mem.get("id"),
                            memory=mem.get("memory", ""),
                            user_id=user_id,
                            category=mem_category_enum,
                            importance_score=importance_score,
                            access_count=mem_metadata.get("access_count", 0),
                            last_accessed=mem_metadata.get("last_accessed"),
                            created_at=mem.get("created_at"),
                            updated_at=mem.get("updated_at"),
                            metadata=mem_metadata
                        )
                        
                        # Update access on retrieval
                        memory_entry.update_access()
                        memories.append(memory_entry)
                
                # Sort by effective score (importance with decay)
                memories.sort(key=lambda m: m.effective_score, reverse=True)
                
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
                        mem_metadata = mem.get("metadata", {})
                        
                        # Extract enhanced fields from metadata
                        mem_category = mem_metadata.get("category", "context")
                        try:
                            mem_category_enum = MemoryCategory(mem_category)
                        except ValueError:
                            mem_category_enum = MemoryCategory.CONTEXT
                        
                        memory_entry = MemoryEntry(
                            id=mem.get("id"),
                            memory=mem.get("memory", ""),
                            user_id=user_id,
                            category=mem_category_enum,
                            importance_score=float(mem_metadata.get("importance_score", 1.0)),
                            access_count=mem_metadata.get("access_count", 0),
                            last_accessed=mem_metadata.get("last_accessed"),
                            created_at=mem.get("created_at"),
                            updated_at=mem.get("updated_at"),
                            metadata=mem_metadata
                        )
                        memories.append(memory_entry)
                
                logger.info(f"Retrieved {len(memories)} total memories for user {user_id}")
                return memories
                
            except Exception as e:
                logger.error(f"Failed to get all memories for user {user_id}: {e}")
                raise
    
    async def get_memories_by_importance(
        self,
        user_id: str,
        limit: int = 20,
        min_score: float = 0.0,
        category: Optional[MemoryCategory] = None,
        include_decay: bool = True
    ) -> List[MemoryEntry]:
        """Get memories sorted by importance score with optional filtering.
        
        Args:
            user_id: User identifier
            limit: Maximum number of memories to return
            min_score: Minimum importance score (0-10)
            category: Optional category filter
            include_decay: Whether to use effective score (with decay) or raw score
            
        Returns:
            List[MemoryEntry]: Memories sorted by importance
        """
        memories = await self.get_all_memories(user_id)
        
        # Apply filters
        filtered_memories = []
        for mem in memories:
            if category and mem.category != category:
                continue
            
            score = mem.effective_score if include_decay else mem.importance_score
            if score >= min_score:
                filtered_memories.append(mem)
        
        # Sort by score
        if include_decay:
            filtered_memories.sort(key=lambda m: m.effective_score, reverse=True)
        else:
            filtered_memories.sort(key=lambda m: m.importance_score, reverse=True)
        
        # Return limited results
        return filtered_memories[:limit]
    
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
    
    async def sync_to_mongodb(
        self,
        user_id: str,
        snapshot_repository: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Sync user memories to MongoDB as a snapshot.
        
        Args:
            user_id: User identifier
            snapshot_repository: Optional MemorySnapshotRepository instance
            
        Returns:
            Dict with sync results
        """
        try:
            # Get all memories for user
            memories = await self.get_all_memories(user_id)
            
            if not memories:
                return {
                    "success": False,
                    "message": "No memories found for user",
                    "memory_count": 0
                }
            
            # Convert memories to MongoDB format
            mongo_memories = []
            for mem in memories:
                mongo_memory = {
                    "memory_id": mem.id,
                    "content": mem.memory,
                    "category": mem.category.value if mem.category else "context",
                    "importance_score": mem.importance_score,
                    "access_count": mem.access_count,
                    "last_accessed": mem.last_accessed,
                    "created_at": mem.created_at,
                    "updated_at": mem.updated_at,
                    "metadata": mem.metadata
                }
                mongo_memories.append(mongo_memory)
            
            # Create snapshot in MongoDB if repository provided
            snapshot_id = None
            if snapshot_repository:
                from app.db.mongodb.schemas.memory_snapshot_schema import SnapshotStatus
                snapshot_id = snapshot_repository.create_snapshot(
                    user_id=user_id,
                    memories=mongo_memories,
                    status=SnapshotStatus.COMPLETED
                )
            
            logger.info(f"Successfully synced {len(memories)} memories for user {user_id}")
            
            return {
                "success": True,
                "memory_count": len(memories),
                "snapshot_id": snapshot_id,
                "categories": {
                    cat.value: sum(1 for m in memories if m.category == cat)
                    for cat in MemoryCategory
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to sync memories to MongoDB: {e}")
            return {
                "success": False,
                "error": str(e),
                "memory_count": 0
            }
    
    async def restore_from_mongodb(
        self,
        user_id: str,
        snapshot_id: str,
        snapshot_repository: Any,
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """Restore memories from a MongoDB snapshot.
        
        Args:
            user_id: User identifier
            snapshot_id: Snapshot ID to restore from
            snapshot_repository: MemorySnapshotRepository instance
            clear_existing: Whether to clear existing memories first
            
        Returns:
            Dict with restore results
        """
        try:
            # Get snapshot from MongoDB
            snapshot = snapshot_repository.find_by_id(snapshot_id)
            if not snapshot:
                return {
                    "success": False,
                    "error": "Snapshot not found"
                }
            
            # Verify user ID matches
            if snapshot["user_id"] != user_id:
                return {
                    "success": False,
                    "error": "User ID mismatch"
                }
            
            # Clear existing memories if requested
            if clear_existing:
                await self.delete_all_memories(user_id)
                logger.info(f"Cleared existing memories for user {user_id}")
            
            # Restore memories
            restored_count = 0
            errors = []
            
            for mongo_mem in snapshot.get("memories", []):
                try:
                    # Reconstruct conversation format
                    messages = [
                        {"role": "user", "content": "Restored memory"},
                        {"role": "assistant", "content": mongo_mem["content"]}
                    ]
                    
                    # Restore with original metadata
                    await self.add_memory(
                        messages=messages,
                        user_id=user_id,
                        metadata=mongo_mem.get("metadata", {}),
                        category=MemoryCategory(mongo_mem.get("category", "context")),
                        importance_score=mongo_mem.get("importance_score", 1.0)
                    )
                    restored_count += 1
                    
                except Exception as e:
                    errors.append(f"Memory {mongo_mem.get('memory_id', 'unknown')}: {str(e)}")
            
            logger.info(f"Restored {restored_count} memories for user {user_id}")
            
            return {
                "success": True,
                "restored_count": restored_count,
                "total_in_snapshot": len(snapshot.get("memories", [])),
                "errors": errors,
                "snapshot_date": snapshot["snapshot_timestamp"].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to restore memories from MongoDB: {e}")
            return {
                "success": False,
                "error": str(e),
                "restored_count": 0
            }
    
    async def get_memory_analytics(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Get analytics about user's memories.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict containing memory analytics
        """
        try:
            memories = await self.get_all_memories(user_id)
            
            if not memories:
                return {
                    "total_memories": 0,
                    "categories": {},
                    "avg_importance_score": 0,
                    "avg_access_count": 0
                }
            
            # Calculate analytics
            categories = {}
            total_importance = 0
            total_access = 0
            
            for mem in memories:
                # Category breakdown
                cat = mem.category.value if mem.category else "context"
                categories[cat] = categories.get(cat, 0) + 1
                
                # Averages
                total_importance += mem.importance_score
                total_access += mem.access_count
            
            # Calculate importance distribution
            importance_dist = {
                "high": sum(1 for m in memories if m.importance_score >= 7),
                "medium": sum(1 for m in memories if 3 <= m.importance_score < 7),
                "low": sum(1 for m in memories if m.importance_score < 3)
            }
            
            # Most accessed memories
            most_accessed = sorted(memories, key=lambda m: m.access_count, reverse=True)[:5]
            
            return {
                "total_memories": len(memories),
                "categories": categories,
                "avg_importance_score": round(total_importance / len(memories), 2),
                "avg_access_count": round(total_access / len(memories), 2),
                "importance_distribution": importance_dist,
                "most_accessed": [
                    {
                        "memory": m.memory[:100] + "..." if len(m.memory) > 100 else m.memory,
                        "access_count": m.access_count,
                        "importance_score": m.importance_score
                    }
                    for m in most_accessed
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory analytics: {e}")
            return {
                "error": str(e),
                "total_memories": 0
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
    metadata: Optional[Dict[str, Any]] = None,
    category: Optional[MemoryCategory] = None,
    importance_score: float = 1.0
) -> Dict[str, Any]:
    """Convenience function to add a conversation to memory with categorization.
    
    Args:
        user_message: The user's message
        assistant_message: The assistant's response
        user_id: User identifier
        metadata: Optional metadata
        category: Memory category (auto-detected if not provided)
        importance_score: Initial importance score (0-10)
        
    Returns:
        Dict[str, Any]: Response from mem0 API
    """
    client = await get_mem0_client()
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]
    return await client.add_memory(messages, user_id, metadata, category, importance_score)


async def search_user_memories(
    query: str,
    user_id: str,
    limit: int = 10,
    category: Optional[MemoryCategory] = None,
    min_importance_score: Optional[float] = None
) -> MemorySearchResult:
    """Convenience function to search user memories with filtering.
    
    Args:
        query: Search query
        user_id: User identifier
        limit: Maximum results
        category: Optional category filter
        min_importance_score: Optional minimum importance score
        
    Returns:
        MemorySearchResult: Search results
    """
    client = await get_mem0_client()
    return await client.search_memories(query, user_id, limit, None, category, min_importance_score)


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
