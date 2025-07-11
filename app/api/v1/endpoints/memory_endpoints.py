"""Memory Management API Endpoints.

This module provides RESTful endpoints for managing user memories using
the reusable Mem0 async client wrapper. These endpoints can be used
across multiple APIs and services.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from app.mem0.mem0AsyncClient import (
    get_mem0_client,
    Mem0AsyncClientWrapper,
    MemoryEntry,
    MemorySearchResult,
    MemoryConfig,
    add_conversation_memory,
    search_user_memories,
    get_user_memory_context
)

logger = logging.getLogger(__name__)

# Create the router
router = APIRouter(prefix="/memory", tags=["Memory Management"])


# Request/Response Models
class AddMemoryRequest(BaseModel):
    """Request model for adding a memory."""
    messages: List[Dict[str, str]] = Field(
        ..., 
        description="List of message dictionaries with 'role' and 'content' keys",
        json_schema_extra={
            "example": [
                {"role": "user", "content": "I prefer morning workouts"},
                {"role": "assistant", "content": "I'll remember you prefer morning workouts"}
            ]
        }
    )
    user_id: str = Field(..., description="User identifier")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional additional metadata"
    )


class AddConversationRequest(BaseModel):
    """Request model for adding a conversation to memory."""
    user_message: str = Field(..., description="The user's message")
    assistant_message: str = Field(..., description="The assistant's response")
    user_id: str = Field(..., description="User identifier")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional additional metadata"
    )


class SearchMemoryRequest(BaseModel):
    """Request model for searching memories."""
    query: str = Field(..., description="Search query string")
    user_id: str = Field(..., description="User identifier")
    limit: int = Field(default=10, description="Maximum number of results")


class UpdateMemoryRequest(BaseModel):
    """Request model for updating a memory."""
    memory_id: str = Field(..., description="ID of the memory to update")
    data: Dict[str, Any] = Field(..., description="Updated memory data")
    user_id: str = Field(..., description="User identifier")


class DeleteMemoryRequest(BaseModel):
    """Request model for deleting a memory."""
    memory_id: str = Field(..., description="ID of the memory to delete")
    user_id: str = Field(..., description="User identifier")


class MemoryResponse(BaseModel):
    """Response model for memory operations."""
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")


class MemoryListResponse(BaseModel):
    """Response model for memory list operations."""
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    memories: List[Dict[str, Any]] = Field(default_factory=list, description="List of memories")
    total_count: int = Field(default=0, description="Total number of memories")


class MemoryContextResponse(BaseModel):
    """Response model for memory context."""
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    context: str = Field(..., description="Formatted memory context")
    memory_count: int = Field(default=0, description="Number of memories included")


# Dependency to get mem0 client
async def get_memory_client() -> Mem0AsyncClientWrapper:
    """Dependency to get the mem0 client instance."""
    try:
        return await get_mem0_client()
    except Exception as e:
        logger.error(f"Failed to get mem0 client: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory service is currently unavailable"
        )


# API Endpoints
@router.post("/add", response_model=MemoryResponse)
async def add_memory(
    request: AddMemoryRequest,
    client: Mem0AsyncClientWrapper = Depends(get_memory_client)
) -> MemoryResponse:
    """Add a new memory entry.
    
    Args:
        request: Memory addition request
        client: Mem0 client instance
        
    Returns:
        MemoryResponse: Response with operation status
    """
    try:
        result = await client.add_memory(
            messages=request.messages,
            user_id=request.user_id,
            metadata=request.metadata
        )
        
        return MemoryResponse(
            success=True,
            message=f"Memory successfully added for user {request.user_id}",
            data=result
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add memory"
        )


@router.post("/add-conversation", response_model=MemoryResponse)
async def add_conversation(
    request: AddConversationRequest,
    client: Mem0AsyncClientWrapper = Depends(get_memory_client)
) -> MemoryResponse:
    """Add a conversation to memory (convenience endpoint).
    
    Args:
        request: Conversation addition request
        client: Mem0 client instance
        
    Returns:
        MemoryResponse: Response with operation status
    """
    try:
        result = await add_conversation_memory(
            user_message=request.user_message,
            assistant_message=request.assistant_message,
            user_id=request.user_id,
            metadata=request.metadata
        )
        
        return MemoryResponse(
            success=True,
            message=f"Conversation successfully added to memory for user {request.user_id}",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Failed to add conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add conversation to memory"
        )


@router.post("/search", response_model=MemoryListResponse)
async def search_memories(
    request: SearchMemoryRequest,
    client: Mem0AsyncClientWrapper = Depends(get_memory_client)
) -> MemoryListResponse:
    """Search memories for a specific user.
    
    Args:
        request: Memory search request
        client: Mem0 client instance
        
    Returns:
        MemoryListResponse: Search results
    """
    try:
        search_result = await client.search_memories(
            query=request.query,
            user_id=request.user_id,
            limit=request.limit
        )
        
        # Convert MemoryEntry objects to dictionaries
        memories_dict = [
            {
                "id": mem.id,
                "memory": mem.memory,
                "user_id": mem.user_id,
                "created_at": mem.created_at.isoformat() if mem.created_at else None,
                "updated_at": mem.updated_at.isoformat() if mem.updated_at else None,
                "metadata": mem.metadata
            }
            for mem in search_result.memories
        ]
        
        return MemoryListResponse(
            success=True,
            message=f"Found {search_result.total_count} memories for query: {request.query}",
            memories=memories_dict,
            total_count=search_result.total_count
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to search memories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search memories"
        )


@router.get("/all/{user_id}", response_model=MemoryListResponse)
async def get_all_user_memories(
    user_id: str,
    client: Mem0AsyncClientWrapper = Depends(get_memory_client)
) -> MemoryListResponse:
    """Get all memories for a specific user.
    
    Args:
        user_id: User identifier
        client: Mem0 client instance
        
    Returns:
        MemoryListResponse: All user memories
    """
    try:
        memories = await client.get_all_memories(user_id)
        
        # Convert MemoryEntry objects to dictionaries
        memories_dict = [
            {
                "id": mem.id,
                "memory": mem.memory,
                "user_id": mem.user_id,
                "created_at": mem.created_at.isoformat() if mem.created_at else None,
                "updated_at": mem.updated_at.isoformat() if mem.updated_at else None,
                "metadata": mem.metadata
            }
            for mem in memories
        ]
        
        return MemoryListResponse(
            success=True,
            message=f"Retrieved {len(memories)} memories for user {user_id}",
            memories=memories_dict,
            total_count=len(memories)
        )
        
    except Exception as e:
        logger.error(f"Failed to get all memories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve memories"
        )


@router.get("/context/{user_id}", response_model=MemoryContextResponse)
async def get_memory_context(
    user_id: str,
    limit: int = 20,
    client: Mem0AsyncClientWrapper = Depends(get_memory_client)
) -> MemoryContextResponse:
    """Get formatted memory context for a user.
    
    Args:
        user_id: User identifier
        limit: Maximum number of memories to include
        client: Mem0 client instance
        
    Returns:
        MemoryContextResponse: Formatted memory context
    """
    try:
        context = await get_user_memory_context(user_id, limit)
        memories = await client.get_memory_history(user_id, limit)
        
        return MemoryContextResponse(
            success=True,
            message=f"Retrieved memory context for user {user_id}",
            context=context,
            memory_count=len(memories)
        )
        
    except Exception as e:
        logger.error(f"Failed to get memory context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve memory context"
        )


@router.put("/update", response_model=MemoryResponse)
async def update_memory(
    request: UpdateMemoryRequest,
    client: Mem0AsyncClientWrapper = Depends(get_memory_client)
) -> MemoryResponse:
    """Update an existing memory entry.
    
    Args:
        request: Memory update request
        client: Mem0 client instance
        
    Returns:
        MemoryResponse: Response with operation status
    """
    try:
        result = await client.update_memory(
            memory_id=request.memory_id,
            data=request.data,
            user_id=request.user_id
        )
        
        return MemoryResponse(
            success=True,
            message=f"Memory {request.memory_id} successfully updated",
            data=result
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to update memory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update memory"
        )


@router.delete("/delete", response_model=MemoryResponse)
async def delete_memory(
    request: DeleteMemoryRequest,
    client: Mem0AsyncClientWrapper = Depends(get_memory_client)
) -> MemoryResponse:
    """Delete a specific memory entry.
    
    Args:
        request: Memory deletion request
        client: Mem0 client instance
        
    Returns:
        MemoryResponse: Response with operation status
    """
    try:
        result = await client.delete_memory(
            memory_id=request.memory_id,
            user_id=request.user_id
        )
        
        return MemoryResponse(
            success=True,
            message=f"Memory {request.memory_id} successfully deleted",
            data=result
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to delete memory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete memory"
        )


@router.delete("/delete-all/{user_id}", response_model=MemoryResponse)
async def delete_all_user_memories(
    user_id: str,
    client: Mem0AsyncClientWrapper = Depends(get_memory_client)
) -> MemoryResponse:
    """Delete all memories for a specific user.
    
    Args:
        user_id: User identifier
        client: Mem0 client instance
        
    Returns:
        MemoryResponse: Response with operation status
    """
    try:
        result = await client.delete_all_memories(user_id)
        
        return MemoryResponse(
            success=True,
            message=f"All memories successfully deleted for user {user_id}",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Failed to delete all memories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete all memories"
        )


@router.get("/health", response_model=Dict[str, Any])
async def memory_health_check(
    client: Mem0AsyncClientWrapper = Depends(get_memory_client)
) -> Dict[str, Any]:
    """Perform a health check of the memory service.
    
    Args:
        client: Mem0 client instance
        
    Returns:
        Dict[str, Any]: Health check status
    """
    try:
        health_status = await client.health_check()
        return health_status
        
    except Exception as e:
        logger.error(f"Memory health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory service health check failed"
        ) 