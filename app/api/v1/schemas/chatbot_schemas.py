"""API schemas for the Limitless OS Sales Agent.

This module defines the request and response schemas for the sales agent API.
All schemas are focused on the specialized sales agent functionality.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class SalesAgentRequest(BaseModel):
    """Request schema for the sales agent endpoint.
    
    Attributes:
        message: User message to send to the sales agent
        user_id: User identifier for mem0 context (required for memory functionality)
        metadata: Additional context data
    """
    message: str = Field(..., description="User message", min_length=1)
    user_id: str = Field(..., description="User identifier for mem0 context")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context data"
    )


class SalesAgentResponse(BaseModel):
    """Response schema for the sales agent endpoint.
    
    Attributes:
        response: Sales agent response message
        user_id: User identifier that was used
        agent_name: Name of the sales agent
        timestamp: Response timestamp
    """
    response: str = Field(..., description="Sales agent response message")
    user_id: str = Field(..., description="User identifier")
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