"""API schemas for the Limitless OS Sales Agent.

This module defines the request and response schemas for the sales agent API.
All schemas are focused on the specialized sales agent functionality.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class SalesAgentRequest(BaseModel):
    """Request schema for the sales agent endpoint.
    
    Attributes:
        message: User message to send to the sales agent
        user_id: User identifier for mem0 context (required for memory functionality)
        channel: Conversation channel (web, mobile, api)
        metadata: Additional context data
    """
    message: str = Field(..., description="User message", min_length=1)
    user_id: str = Field(..., description="User identifier for mem0 context")
    channel: Optional[str] = Field(
        default="web",
        description="Conversation channel"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context data"
    )


class SalesAgentResponse(BaseModel):
    """Response schema for the sales agent endpoint.
    
    Attributes:
        response: Sales agent response message
        user_id: User identifier that was used
        conversation_id: MongoDB conversation ID
        event: Conversation event (created/resumed)
        sales_stage: Current sales stage
        is_qualified: Whether lead is qualified
        agent_name: Name of the sales agent
        timestamp: Response timestamp
    """
    response: str = Field(..., description="Sales agent response message")
    user_id: str = Field(..., description="User identifier")
    conversation_id: str = Field(..., description="MongoDB conversation ID")
    event: str = Field(..., description="Conversation event (created/resumed)")
    sales_stage: Optional[str] = Field(None, description="Current sales stage")
    is_qualified: bool = Field(False, description="Whether lead is qualified")
    agent_name: str = Field(..., description="Sales agent name")
    timestamp: str = Field(..., description="Response timestamp")


class ErrorResponse(BaseModel):
    """Error response schema.
    
    Attributes:
        error: Error type or category
        message: Detailed error message
        timestamp: Error timestamp
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="Error timestamp")


class ConversationMessage(BaseModel):
    """Schema for a conversation message."""
    role: str = Field(..., description="Message role (user/assistant/agent)")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="Message timestamp")


class ConversationSummary(BaseModel):
    """Schema for conversation summary."""
    conversation_id: str = Field(..., description="Conversation ID")
    user_id: str = Field(..., description="User ID")
    status: str = Field(..., description="Conversation status")
    sales_stage: str = Field(..., description="Current sales stage")
    message_count: int = Field(..., description="Total message count")
    created_at: str = Field(..., description="Conversation creation time")
    updated_at: str = Field(..., description="Last update time")
    is_qualified: bool = Field(False, description="Whether lead is qualified")
    current_agent: Optional[str] = Field(None, description="Current agent")
    handoff_count: int = Field(0, description="Number of handoffs")
    qualification_score: float = Field(0.0, description="Qualification score")


class ConversationHistoryResponse(BaseModel):
    """Response schema for conversation history."""
    conversation_id: str = Field(..., description="Conversation ID")
    messages: List[ConversationMessage] = Field(..., description="Conversation messages")
    summary: ConversationSummary = Field(..., description="Conversation summary")


class UserConversationsResponse(BaseModel):
    """Response schema for user conversations list."""
    user_id: str = Field(..., description="User ID")
    conversations: List[Dict[str, Any]] = Field(..., description="List of conversations") 