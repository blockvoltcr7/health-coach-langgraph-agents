"""Pydantic models for multi-agent workflow integration.

These models provide structured data for workflow state management
and integration between LangGraph, MongoDB, and Mem0.
"""

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class TaskStatus(str, Enum):
    """Individual task status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QualificationField(BaseModel):
    """Single qualification field (BANT)."""
    value: Optional[Any] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    source: Optional[str] = None
    timestamp: Optional[datetime] = None


class QualificationData(BaseModel):
    """BANT qualification data structure."""
    budget: QualificationField = Field(default_factory=QualificationField)
    authority: QualificationField = Field(default_factory=QualificationField)
    need: QualificationField = Field(default_factory=QualificationField)
    timeline: QualificationField = Field(default_factory=QualificationField)
    
    @property
    def overall_confidence(self) -> float:
        """Calculate overall qualification confidence."""
        fields = [self.budget, self.authority, self.need, self.timeline]
        total_confidence = sum(f.confidence for f in fields)
        return total_confidence / 4
    
    @property
    def is_qualified(self) -> bool:
        """Check if lead is qualified (all fields have some data)."""
        return all(
            field.value is not None 
            for field in [self.budget, self.authority, self.need, self.timeline]
        )


class CurrentTask(BaseModel):
    """Current task being processed."""
    type: str = Field(..., description="Task type (qualification, objection_handling, closing)")
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    context: Dict[str, Any] = Field(default_factory=dict)
    iterations: int = Field(0, ge=0)
    max_iterations: int = Field(5, ge=1)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class AgentResponse(BaseModel):
    """Response from an agent."""
    response: str
    tools_used: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class RoutingDecision(BaseModel):
    """Routing decision made by supervisor."""
    from_agent: str
    to_agent: str
    reason: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage_before: str
    stage_after: str
    context: Optional[Dict[str, Any]] = None


class WorkflowState(BaseModel):
    """Complete workflow state for multi-agent system."""
    id: str = Field(..., description="Unique workflow identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: WorkflowStatus = WorkflowStatus.IN_PROGRESS
    current_stage: str = Field(..., description="Current sales stage")
    current_task: Optional[CurrentTask] = None
    agents_visited: List[str] = Field(default_factory=list)
    agent_responses: Dict[str, AgentResponse] = Field(default_factory=dict)
    routing_history: List[RoutingDecision] = Field(default_factory=list)
    qualification_data: QualificationData = Field(default_factory=QualificationData)
    objections: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('updated_at', mode='before')
    @classmethod
    def update_timestamp(cls, v):
        """Always update the timestamp when model is modified."""
        return datetime.now(timezone.utc)
    
    def add_routing_decision(self, decision: RoutingDecision) -> None:
        """Add a routing decision to history."""
        self.routing_history.append(decision)
        self.updated_at = datetime.now(timezone.utc)
    
    def record_agent_response(self, agent_name: str, response: AgentResponse) -> None:
        """Record an agent's response."""
        self.agent_responses[agent_name] = response
        if agent_name not in self.agents_visited:
            self.agents_visited.append(agent_name)
        self.updated_at = datetime.now(timezone.utc)
    
    def update_qualification(self, field: str, value: Any, confidence: float = 0.8) -> None:
        """Update a qualification field."""
        if hasattr(self.qualification_data, field):
            qual_field = QualificationField(
                value=value,
                confidence=confidence,
                source=self.current_task.assigned_agent if self.current_task else "unknown",
                timestamp=datetime.now(timezone.utc)
            )
            setattr(self.qualification_data, field, qual_field)
            self.updated_at = datetime.now(timezone.utc)
    
    def to_mongodb_doc(self) -> Dict[str, Any]:
        """Convert to MongoDB-compatible document."""
        doc = self.model_dump()
        # Convert datetime objects to ISO strings
        doc['created_at'] = self.created_at.isoformat()
        doc['updated_at'] = self.updated_at.isoformat()
        
        # Convert nested datetime objects
        for decision in doc.get('routing_history', []):
            if isinstance(decision.get('timestamp'), datetime):
                decision['timestamp'] = decision['timestamp'].isoformat()
        
        for agent, response in doc.get('agent_responses', {}).items():
            if isinstance(response.get('timestamp'), datetime):
                response['timestamp'] = response['timestamp'].isoformat()
        
        return doc
    
    @classmethod
    def from_mongodb_doc(cls, doc: Dict[str, Any]) -> 'WorkflowState':
        """Create from MongoDB document."""
        # Convert ISO strings back to datetime
        if isinstance(doc.get('created_at'), str):
            doc['created_at'] = datetime.fromisoformat(doc['created_at'].replace('Z', '+00:00'))
        if isinstance(doc.get('updated_at'), str):
            doc['updated_at'] = datetime.fromisoformat(doc['updated_at'].replace('Z', '+00:00'))
        
        # Handle nested conversions if needed
        return cls(**doc)


class AgentInteraction(BaseModel):
    """Track individual agent interaction."""
    conversation_id: str
    agent_name: str
    user_message: str
    agent_response: str
    tools_used: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sales_stage: str
    qualification_updates: Optional[Dict[str, QualificationField]] = None
    objections_raised: List[str] = Field(default_factory=list)
    objections_resolved: List[str] = Field(default_factory=list)
    next_agent: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MultiAgentConversationSummary(BaseModel):
    """Summary of multi-agent conversation."""
    conversation_id: str
    workflow_id: str
    status: WorkflowStatus
    current_stage: str
    agents_involved: List[str]
    total_interactions: int
    qualification_status: Dict[str, bool]
    objection_summary: Dict[str, int]
    routing_path: List[str]
    duration_seconds: Optional[float] = None
    outcome: Optional[str] = None