"""MongoDB schema for conversation collection.

This module defines the enhanced conversation schema for the Sales AI Closer system,
including comprehensive embedded documents for qualification, messages, objections,
handoffs, follow-ups, and agent metrics.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import logging

from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.results import UpdateResult
from bson import ObjectId

from app.db.mongodb.base_repository import BaseRepository
from app.db.mongodb.schemas.validation_schemas import (
    get_enhanced_conversation_schema,
    get_regex_patterns
)
from app.db.mongodb.validators import (
    ConversationValidator,
    ValidationError,
    create_stage_transition,
    SalesStageTransitionValidator
)

logger = logging.getLogger(__name__)


class ChannelType(str, Enum):
    """Supported conversation channels."""
    INSTAGRAM_DM = "instagram_dm"
    FACEBOOK_MESSENGER = "facebook_messenger"
    WHATSAPP = "whatsapp"
    SMS = "sms"
    WEB = "web"
    EMAIL = "email"
    # Legacy support
    INSTAGRAM = "instagram"


class ConversationStatus(str, Enum):
    """Conversation status states."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    SCHEDULED = "scheduled"
    # Legacy support
    INACTIVE = "inactive"
    CLOSED = "closed"


class SalesStage(str, Enum):
    """Sales pipeline stages."""
    LEAD = "lead"
    QUALIFICATION = "qualification"
    QUALIFIED = "qualified"
    OBJECTION_HANDLING = "objection_handling"
    CLOSING = "closing"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"
    FOLLOW_UP = "follow_up"
    # Legacy support
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED = "closed"


class MessageRole(str, Enum):
    """Message sender roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SUPERVISOR = "supervisor"
    QUALIFIER = "qualifier"
    OBJECTION_HANDLER = "objection_handler"
    CLOSER = "closer"


class AgentName(str, Enum):
    """Agent names for handoffs."""
    SUPERVISOR = "supervisor"
    QUALIFIER = "qualifier"
    OBJECTION_HANDLER = "objection_handler"
    CLOSER = "closer"
    ONBOARDING_SPECIALIST = "onboarding_specialist"


class ObjectionType(str, Enum):
    """Types of objections."""
    PRICE = "price"
    TIMING = "timing"
    TRUST = "trust"
    FEATURE = "feature"
    COMPETITOR = "competitor"
    AUTHORITY = "authority"


class FollowUpType(str, Enum):
    """Types of follow-ups."""
    RE_ENGAGEMENT = "re_engagement"
    OBJECTION_FOLLOW_UP = "objection_follow_up"
    CLOSE_ATTEMPT = "close_attempt"
    NURTURE = "nurture"
    ONBOARDING_CHECK = "onboarding_check"


# Enhanced Schema validation for MongoDB with full validation
CONVERSATION_SCHEMA = get_enhanced_conversation_schema(
    channel_types=[c.value for c in ChannelType],
    conversation_statuses=[s.value for s in ConversationStatus],
    sales_stages=[s.value for s in SalesStage],
    message_roles=[r.value for r in MessageRole],
    agent_names=[a.value for a in AgentName],
    objection_types=[o.value for o in ObjectionType],
    follow_up_types=[f.value for f in FollowUpType]
)

# Legacy CONVERSATION_SCHEMA for backward compatibility
LEGACY_CONVERSATION_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["user_id", "channel", "status", "sales_stage", "created_at", "updated_at"],
        "properties": {
            "_id": {"bsonType": "objectId"},
            "user_id": {
                "bsonType": ["string", "objectId"],
                "description": "User identifier - can be string or ObjectId"
            },
            "channel": {
                "bsonType": "string",
                "enum": [c.value for c in ChannelType]
            },
            "status": {
                "bsonType": "string",
                "enum": [s.value for s in ConversationStatus]
            },
            "sales_stage": {
                "bsonType": "string",
                "enum": [s.value for s in SalesStage]
            },
            "current_agent": {
                "bsonType": ["string", "null"],
                "enum": [a.value for a in AgentName] + [None]
            },
            "is_qualified": {
                "bsonType": ["bool", "null"]
            },
            "stage_history": {
                "bsonType": "array",
                "items": {
                    "bsonType": "object",
                    "required": ["stage", "timestamp"],
                    "properties": {
                        "stage": {
                            "bsonType": "string",
                            "enum": [s.value for s in SalesStage]
                        },
                        "timestamp": {"bsonType": ["date", "string"]},
                        "notes": {"bsonType": "string"}
                    }
                }
            },
            "qualification": {
                "bsonType": "object",
                "properties": {
                    "budget": {
                        "bsonType": ["object", "bool"],
                        "properties": {
                            "meets_criteria": {"bsonType": ["bool", "null"]},
                            "value": {"bsonType": ["string", "null"]},
                            "minimum_required": {"bsonType": "string"},
                            "confidence": {"bsonType": "double"},
                            "captured_at": {"bsonType": ["date", "string", "null"]}
                        }
                    },
                    "authority": {
                        "bsonType": ["object", "bool"],
                        "properties": {
                            "meets_criteria": {"bsonType": ["bool", "null"]},
                            "role": {"bsonType": ["string", "null"]},
                            "needs_approval": {"bsonType": ["bool", "null"]},
                            "confidence": {"bsonType": "double"}
                        }
                    },
                    "need": {
                        "bsonType": ["object", "bool"],
                        "properties": {
                            "meets_criteria": {"bsonType": ["bool", "null"]},
                            "pain_points": {"bsonType": "array"},
                            "use_case": {"bsonType": ["string", "null"]},
                            "confidence": {"bsonType": "double"}
                        }
                    },
                    "timeline": {
                        "bsonType": ["object", "bool"],
                        "properties": {
                            "meets_criteria": {"bsonType": ["bool", "null"]},
                            "timeframe": {"bsonType": ["string", "null"]},
                            "deadline": {"bsonType": ["date", "string", "null"]},
                            "urgency": {"bsonType": ["string", "null"]},
                            "confidence": {"bsonType": "double"}
                        }
                    },
                    "overall_score": {"bsonType": ["double", "int"]},
                    "qualified_at": {"bsonType": ["date", "string", "null"]},
                    "qualified_by": {"bsonType": ["string", "null"]},
                    "qualification_method": {"bsonType": ["string", "null"]},
                    "score": {"bsonType": ["double", "int"]},
                    "notes": {"bsonType": "string"}
                }
            },
            "messages": {
                "bsonType": "array",
                "items": {
                    "bsonType": "object",
                    "required": ["role", "content", "timestamp"],
                    "properties": {
                        "role": {
                            "bsonType": "string",
                            "enum": [r.value for r in MessageRole]
                        },
                        "content": {"bsonType": "string"},
                        "timestamp": {"bsonType": ["date", "string"]}
                    }
                }
            },
            "objections": {
                "bsonType": "array",
                "items": {
                    "bsonType": "object",
                    "properties": {
                        "objection_id": {"bsonType": "string"},
                        "type": {
                            "bsonType": "string",
                            "enum": [o.value for o in ObjectionType]
                        },
                        "content": {"bsonType": "string"},
                        "raised_at": {"bsonType": ["date", "string"]},
                        "severity": {
                            "bsonType": "string",
                            "enum": ["high", "medium", "low"]
                        },
                        "status": {
                            "bsonType": "string",
                            "enum": ["active", "resolved", "deferred"]
                        },
                        "handled_by": {"bsonType": "string"},
                        "handling_attempts": {"bsonType": "int"},
                        "resolution": {
                            "bsonType": "object",
                            "properties": {
                                "resolved": {"bsonType": "bool"},
                                "method": {"bsonType": ["string", "null"]},
                                "resolved_at": {"bsonType": ["date", "string", "null"]},
                                "resolution_notes": {"bsonType": ["string", "null"]},
                                "confidence": {"bsonType": "double"}
                            }
                        },
                        # Legacy support
                        "objection": {"bsonType": "string"},
                        "response": {"bsonType": "string"},
                        "resolved": {"bsonType": "bool"},
                        "timestamp": {"bsonType": ["date", "string"]}
                    }
                }
            },
            "handoffs": {
                "bsonType": "array",
                "items": {
                    "bsonType": "object",
                    "required": ["from_agent", "to_agent", "timestamp"],
                    "properties": {
                        "handoff_id": {"bsonType": "string"},
                        "from_agent": {
                            "bsonType": "string",
                            "enum": [a.value for a in AgentName]
                        },
                        "to_agent": {
                            "bsonType": "string",
                            "enum": [a.value for a in AgentName]
                        },
                        "timestamp": {"bsonType": ["date", "string"]},
                        "reason": {"bsonType": "string"},
                        "trigger": {
                            "bsonType": "object",
                            "properties": {
                                "type": {"bsonType": "string"},
                                "details": {"bsonType": "string"}
                            }
                        },
                        "context": {
                            "bsonType": "object",
                            "properties": {
                                "previous_stage": {"bsonType": "string"},
                                "new_stage": {"bsonType": "string"},
                                "confidence_score": {"bsonType": "double"},
                                "handoff_notes": {"bsonType": "string"}
                            }
                        },
                        "metrics": {
                            "bsonType": "object",
                            "properties": {
                                "messages_before_handoff": {"bsonType": "int"},
                                "time_in_previous_agent": {"bsonType": "int"}
                            }
                        }
                    }
                }
            },
            "follow_up": {
                "bsonType": "object",
                "properties": {
                    "required": {"bsonType": "bool"},
                    "scheduled_date": {"bsonType": ["date", "string", "null"]},
                    "type": {
                        "bsonType": ["string", "null"],
                        "enum": [f.value for f in FollowUpType] + [None]
                    },
                    "priority": {
                        "bsonType": ["string", "null"],
                        "enum": ["high", "medium", "low", None]
                    },
                    "assigned_agent": {"bsonType": ["string", "null"]},
                    "attempts": {"bsonType": "array"},
                    "max_attempts": {"bsonType": "int"},
                    "final_attempt_date": {"bsonType": ["date", "string", "null"]},
                    "context": {"bsonType": "object"}
                }
            },
            "agent_context": {
                "bsonType": "object",
                "properties": {
                    "assigned_at": {"bsonType": ["date", "string"]},
                    "previous_agent": {"bsonType": ["string", "null"]},
                    "handoff_reason": {"bsonType": ["string", "null"]},
                    "interaction_count": {"bsonType": "int"}
                }
            },
            "agent_metrics": {
                "bsonType": "object",
                "properties": {
                    "supervisor": {"bsonType": "object"},
                    "qualifier": {"bsonType": "object"},
                    "objection_handler": {"bsonType": "object"},
                    "closer": {"bsonType": "object"},
                    "total_interaction_time": {"bsonType": "int"},
                    "total_messages": {"bsonType": "int"},
                    "engagement_score": {"bsonType": "double"}
                }
            },
            "deal_details": {
                "bsonType": "object",
                "properties": {
                    "monthly_value": {"bsonType": "int"},
                    "setup_fee": {"bsonType": "int"},
                    "total_first_payment": {"bsonType": "int"},
                    "expected_ltv": {"bsonType": "int"},
                    "close_date": {"bsonType": ["date", "string"]},
                    "start_date": {"bsonType": ["date", "string"]},
                    "contract_length": {"bsonType": "string"},
                    "payment_method": {"bsonType": ["string", "null"]},
                    "discount_applied": {"bsonType": "int"},
                    "close_reason": {"bsonType": "string"}
                }
            },
            "metadata": {"bsonType": "object"},
            "created_at": {"bsonType": ["date", "string"]},
            "updated_at": {"bsonType": ["date", "string"]}
        }
    }
}


class ConversationSchema:
    """Helper class for creating and managing the conversation collection schema."""
    
    @staticmethod
    def create_collection(database: Database, drop_existing: bool = False) -> Collection:
        """Create the conversations collection with schema validation.
        
        Args:
            database: MongoDB database instance
            drop_existing: Whether to drop existing collection (use with caution!)
            
        Returns:
            Collection: The created collection
        """
        collection_name = "conversations"
        
        # Drop existing collection if requested
        if drop_existing and collection_name in database.list_collection_names():
            database.drop_collection(collection_name)
        
        # Create collection with schema validation
        collection = database.create_collection(
            collection_name,
            validator=CONVERSATION_SCHEMA
        )
        
        # Create indexes for optimal performance
        ConversationSchema.create_indexes(collection)
        
        return collection
    
    @staticmethod
    def create_indexes(collection: Collection) -> None:
        """Create performance indexes on the conversations collection.
        
        Args:
            collection: MongoDB collection instance
        """
        # Get existing indexes
        existing_indexes = {idx['name'] for idx in collection.list_indexes()}
        
        # Define indexes with explicit names
        indexes_to_create = [
            ([("user_id", 1), ("updated_at", -1)], "user_lookup_idx"),
            ([("sales_stage", 1), ("status", 1)], "sales_pipeline_idx"),
            ([("channel", 1)], "channel_idx"),
            ([("created_at", -1)], "recent_conversations_idx"),
            ([("follow_up.required", 1), ("follow_up.scheduled_date", 1)], "follow_up_scheduling_idx"),
            ([("current_agent", 1), ("status", 1), ("updated_at", -1)], "agent_workload_idx")
        ]
        
        # Create only missing indexes
        for index_spec, index_name in indexes_to_create:
            if index_name not in existing_indexes:
                try:
                    collection.create_index(index_spec, name=index_name)
                    logger.info(f"Created index: {index_name}")
                except Exception as e:
                    logger.warning(f"Could not create index {index_name}: {e}")
        
        # Additional qualification index
        if "qualification_pipeline_idx" not in existing_indexes:
            try:
                collection.create_index(
                    [("is_qualified", 1), ("qualification.overall_score", -1)],
                    name="qualification_pipeline_idx"
                )
                logger.info("Created index: qualification_pipeline_idx")
            except Exception as e:
                logger.warning(f"Could not create qualification index: {e}")
        
        # Objection analysis index (sparse for performance)
        if "objection_analysis_idx" not in existing_indexes:
            try:
                collection.create_index(
                    [("objections.status", 1), ("objections.type", 1)],
                    name="objection_analysis_idx",
                    sparse=True
                )
                logger.info("Created index: objection_analysis_idx")
            except Exception as e:
                logger.warning(f"Could not create objection index: {e}")
        
        # Text search index for message content
        if "message_search_idx" not in existing_indexes:
            try:
                collection.create_index(
                    [("messages.content", "text")],
                    name="message_search_idx",
                    default_language="english",
                    weights={"messages.content": 1}
                )
                logger.info("Created index: message_search_idx for full-text search")
            except Exception as e:
                logger.warning(f"Could not create text search index: {e}")
    
    @staticmethod
    def create_conversation_document(
        user_id: str,
        channel: str,
        initial_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        campaign: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new conversation document with enhanced structure.
        
        Args:
            user_id: User ID as string
            channel: Conversation channel
            initial_message: Optional initial user message
            metadata: Optional metadata dictionary
            source: Traffic source (e.g., instagram_story, facebook_ad)
            campaign: Marketing campaign identifier
            
        Returns:
            Dict[str, Any]: Conversation document ready for insertion
        """
        now = datetime.utcnow()
        
        # Enhanced metadata
        enhanced_metadata = metadata or {}
        if source:
            enhanced_metadata["source"] = source
        if campaign:
            enhanced_metadata["campaign"] = campaign
        enhanced_metadata["mem0_user_id"] = f"mem0_{user_id}"
        
        doc = {
            "user_id": user_id,  # Keep as string for flexibility
            "channel": channel,
            "status": ConversationStatus.ACTIVE.value,
            "sales_stage": SalesStage.LEAD.value,
            "current_agent": AgentName.SUPERVISOR.value,
            "is_qualified": None,
            "stage_history": [{
                "stage": SalesStage.LEAD.value,
                "timestamp": now.isoformat() + "Z",
                "notes": "Conversation initiated"
            }],
            "qualification": {
                "budget": {
                    "meets_criteria": None,
                    "value": None,
                    "minimum_required": "$297",
                    "confidence": 0.0,
                    "captured_at": None
                },
                "authority": {
                    "meets_criteria": None,
                    "role": None,
                    "needs_approval": None,
                    "confidence": 0.0
                },
                "need": {
                    "meets_criteria": None,
                    "pain_points": [],
                    "use_case": None,
                    "confidence": 0.0
                },
                "timeline": {
                    "meets_criteria": None,
                    "timeframe": None,
                    "deadline": None,
                    "urgency": None,
                    "confidence": 0.0
                },
                "overall_score": 0,
                "qualified_at": None,
                "qualified_by": None,
                "qualification_method": None
            },
            "messages": [],
            "objections": [],
            "handoffs": [],
            "follow_up": {
                "required": False,
                "scheduled_date": None,
                "type": None,
                "priority": None,
                "assigned_agent": None,
                "attempts": [],
                "max_attempts": 3,
                "final_attempt_date": None,
                "context": {}
            },
            "agent_context": {
                "assigned_at": now.isoformat() + "Z",
                "previous_agent": None,
                "handoff_reason": None,
                "interaction_count": 0
            },
            "agent_metrics": {
                "supervisor": {
                    "interaction_time": 0,
                    "messages_sent": 0,
                    "handoff_efficiency": 0.0
                },
                "qualifier": {
                    "interaction_time": 0,
                    "messages_sent": 0,
                    "qualification_completion": 0.0,
                    "data_capture_rate": 0.0
                },
                "objection_handler": {
                    "interaction_time": 0,
                    "messages_sent": 0,
                    "resolution_rate": 0.0,
                    "escalation_rate": 0.0
                },
                "closer": {
                    "interaction_time": 0,
                    "messages_sent": 0,
                    "close_attempt": False,
                    "close_success": False
                },
                "total_interaction_time": 0,
                "total_messages": 0,
                "engagement_score": 0.0
            },
            "metadata": enhanced_metadata,
            "created_at": now.isoformat() + "Z",
            "updated_at": now.isoformat() + "Z"
        }
        
        # Add initial message if provided
        if initial_message:
            doc["messages"].append({
                "role": MessageRole.USER.value,
                "content": initial_message,
                "timestamp": now.isoformat() + "Z"
            })
            doc["agent_metrics"]["total_messages"] = 1
            doc["agent_context"]["interaction_count"] = 1
        
        return doc


class ConversationRepository(BaseRepository[Dict[str, Any]]):
    """Repository for conversation collection operations with enhanced methods."""
    
    def __init__(self, database: Optional[Database] = None):
        """Initialize repository with validation middleware.
        
        Args:
            database: Optional database instance
        """
        super().__init__(database)
        # Import here to avoid circular imports
        from app.db.mongodb.middleware.validation_middleware import create_validation_middleware
        self.validation_middleware = create_validation_middleware(raise_on_error=False)
    
    @property
    def collection_name(self) -> str:
        """Get the collection name."""
        return "conversations"
    
    def create_one(self, document: Dict[str, Any]) -> Any:
        """Create a conversation with validation.
        
        Args:
            document: Document to insert
            
        Returns:
            InsertOneResult: Result of the insert operation
        """
        # Validate document before insertion
        validated_doc = self.validation_middleware.validate_before_insert(document)
        
        # Log validation issues if any
        is_valid, errors = ConversationValidator.validate_conversation_document(validated_doc)
        if not is_valid:
            logger.warning(f"Document validation warnings: {errors}")
        
        return super().create_one(validated_doc)
    
    def update_by_id(self, id: str, update: Dict[str, Any]) -> UpdateResult:
        """Update a conversation by ID with validation.
        
        Args:
            id: Document ID
            update: Update dictionary
            
        Returns:
            UpdateResult: Result of the update operation
        """
        # Validate update operations
        validated_update = self.validation_middleware.validate_before_update(update)
        
        return super().update_by_id(id, validated_update)
    
    def find_active_by_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Find active conversation for a user.
        
        Args:
            user_id: User ID as string
            
        Returns:
            Optional[Dict[str, Any]]: Active conversation or None
        """
        return self.find_one({
            "user_id": user_id,
            "status": ConversationStatus.ACTIVE.value
        })
    
    def find_by_sales_stage(
        self,
        stage: str,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Find conversations by sales stage.
        
        Args:
            stage: Sales stage
            status: Optional status filter
            limit: Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: List of conversations
        """
        filter = {"sales_stage": stage}
        if status:
            filter["status"] = status
            
        return self.find_many(
            filter,
            limit=limit,
            sort=[("updated_at", -1)]
        )
    
    def find_by_current_agent(
        self,
        agent: str,
        status: str = ConversationStatus.ACTIVE.value,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Find conversations assigned to a specific agent.
        
        Args:
            agent: Agent name
            status: Conversation status (default: active)
            limit: Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: List of conversations
        """
        return self.find_many(
            {"current_agent": agent, "status": status},
            limit=limit,
            sort=[("updated_at", -1)]
        )
    
    def find_follow_ups_due(
        self,
        before_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Find conversations with follow-ups due.
        
        Args:
            before_date: Find follow-ups due before this date (default: now)
            
        Returns:
            List[Dict[str, Any]]: List of conversations needing follow-up
        """
        if before_date is None:
            before_date = datetime.utcnow()
            
        return self.find_many({
            "follow_up.required": True,
            "follow_up.scheduled_date": {"$lte": before_date.isoformat() + "Z"}
        })
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str
    ) -> UpdateResult:
        """Add a message to a conversation and update metrics.
        
        Args:
            conversation_id: Conversation ID
            role: Message role
            content: Message content
            
        Returns:
            UpdateResult: Result of the update
        """
        # Validate role
        try:
            role = ConversationValidator.validate_enum_value(
                role,
                [r.value for r in MessageRole],
                "message.role"
            )
        except ValidationError:
            # Try agent names as fallback
            role = ConversationValidator.validate_enum_value(
                role,
                [a.value for a in AgentName],
                "message.role"
            )
        
        now = datetime.utcnow()
        
        # Determine which agent metrics to update
        agent_key = None
        if role in [MessageRole.SUPERVISOR.value, AgentName.SUPERVISOR.value]:
            agent_key = "supervisor"
        elif role in [MessageRole.QUALIFIER.value, AgentName.QUALIFIER.value]:
            agent_key = "qualifier"
        elif role in [MessageRole.OBJECTION_HANDLER.value, AgentName.OBJECTION_HANDLER.value]:
            agent_key = "objection_handler"
        elif role in [MessageRole.CLOSER.value, AgentName.CLOSER.value]:
            agent_key = "closer"
        
        update_dict = {
            "$push": {
                "messages": {
                    "role": role,
                    "content": content,
                    "timestamp": now.isoformat() + "Z"
                }
            },
            "$set": {"updated_at": now.isoformat() + "Z"},
            "$inc": {
                "agent_metrics.total_messages": 1,
                "agent_context.interaction_count": 1
            }
        }
        
        # Update specific agent metrics if applicable
        if agent_key:
            update_dict["$inc"][f"agent_metrics.{agent_key}.messages_sent"] = 1
        
        return self.update_by_id(conversation_id, update_dict)
    
    def update_sales_stage(
        self,
        conversation_id: str,
        new_stage: str,
        notes: str = "",
        context: Optional[Dict[str, Any]] = None,
        validate: bool = True,
        triggered_by: str = "system"
    ) -> UpdateResult:
        """Update the sales stage of a conversation with validation.
        
        Args:
            conversation_id: Conversation ID
            new_stage: New sales stage
            notes: Optional notes about the transition
            context: Context for validation
            validate: Whether to validate the transition
            triggered_by: Who/what triggered the transition
            
        Returns:
            UpdateResult: Result of the update
            
        Raises:
            ValueError: If transition is invalid and validate=True
        """
        # Get current conversation
        conversation = self.find_by_id(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        current_stage = conversation.get("sales_stage")
        
        # Validate transition if requested
        if validate and current_stage:
            is_valid, error_msg = SalesStageTransitionValidator.validate_transition(
                current_stage,
                new_stage,
                context
            )
            if not is_valid:
                raise ValueError(f"Invalid stage transition: {error_msg}")
        
        now = datetime.utcnow()
        
        # Build update query
        update_query = {
            "$set": {
                "sales_stage": new_stage,
                "updated_at": now.isoformat() + "Z"
            },
            "$push": {
                "stage_history": {
                    "stage": new_stage,
                    "timestamp": now.isoformat() + "Z",
                    "notes": notes
                }
            }
        }
        
        # Add stage transition if current stage exists
        if current_stage:
            transition = create_stage_transition(
                current_stage,
                new_stage,
                notes or "Stage progression",
                triggered_by,
                context
            )
            update_query["$push"]["stage_transitions"] = transition
        
        # Update is_qualified flag for qualified stage
        if new_stage == SalesStage.QUALIFIED.value:
            update_query["$set"]["is_qualified"] = True
            update_query["$set"]["qualification.qualified_at"] = now.isoformat() + "Z"
            update_query["$set"]["qualification.qualified_by"] = triggered_by
        
        return self.update_by_id(conversation_id, update_query)
    
    def add_handoff(
        self,
        conversation_id: str,
        from_agent: str,
        to_agent: str,
        reason: str,
        trigger_type: str = "manual",
        trigger_details: str = "",
        confidence_score: float = 0.9
    ) -> UpdateResult:
        """Add an enhanced handoff record to a conversation.
        
        Args:
            conversation_id: Conversation ID
            from_agent: Agent handing off
            to_agent: Agent receiving
            reason: Reason for handoff
            trigger_type: Type of trigger (user_intent, stage_complete, etc.)
            trigger_details: Details about the trigger
            confidence_score: Confidence in the handoff decision
            
        Returns:
            UpdateResult: Result of the update
        """
        now = datetime.utcnow()
        handoff_id = f"hnd_{ObjectId()}"
        
        # Get current conversation to determine stages
        conv = self.find_by_id(conversation_id)
        current_stage = conv.get("sales_stage", "") if conv else ""
        
        return self.update_by_id(
            conversation_id,
            {
                "$push": {
                    "handoffs": {
                        "handoff_id": handoff_id,
                        "from_agent": from_agent,
                        "to_agent": to_agent,
                        "timestamp": now.isoformat() + "Z",
                        "reason": reason,
                        "trigger": {
                            "type": trigger_type,
                            "details": trigger_details
                        },
                        "context": {
                            "previous_stage": current_stage,
                            "new_stage": current_stage,  # Update separately if stage changes
                            "confidence_score": confidence_score,
                            "handoff_notes": reason
                        },
                        "metrics": {
                            "messages_before_handoff": conv.get("agent_metrics", {}).get("total_messages", 0) if conv else 0,
                            "time_in_previous_agent": 0  # Calculate based on timestamps
                        }
                    }
                },
                "$set": {
                    "current_agent": to_agent,
                    "agent_context.previous_agent": from_agent,
                    "agent_context.handoff_reason": reason,
                    "agent_context.assigned_at": now.isoformat() + "Z",
                    "updated_at": now.isoformat() + "Z"
                }
            }
        )
    
    def update_qualification(
        self,
        conversation_id: str,
        budget_info: Optional[Dict[str, Any]] = None,
        authority_info: Optional[Dict[str, Any]] = None,
        need_info: Optional[Dict[str, Any]] = None,
        timeline_info: Optional[Dict[str, Any]] = None
    ) -> UpdateResult:
        """Update qualification information.
        
        Args:
            conversation_id: Conversation ID
            budget_info: Budget qualification data
            authority_info: Authority qualification data
            need_info: Need qualification data
            timeline_info: Timeline qualification data
            
        Returns:
            UpdateResult: Result of the update
        """
        update_dict = {"$set": {"updated_at": datetime.utcnow().isoformat() + "Z"}}
        
        # Validate confidence scores in each info dict
        for info, name in [
            (budget_info, "budget"),
            (authority_info, "authority"),
            (need_info, "need"),
            (timeline_info, "timeline")
        ]:
            if info and "confidence" in info and info["confidence"] is not None:
                try:
                    info["confidence"] = ConversationValidator.validate_score(
                        info["confidence"],
                        f"qualification.{name}.confidence"
                    ) / 100.0  # Convert to 0-1 range for storage
                except ValidationError as e:
                    logger.warning(f"Validation warning: {e}")
                    info["confidence"] = 0.0
        
        if budget_info:
            update_dict["$set"]["qualification.budget"] = budget_info
        if authority_info:
            update_dict["$set"]["qualification.authority"] = authority_info
        if need_info:
            update_dict["$set"]["qualification.need"] = need_info
        if timeline_info:
            update_dict["$set"]["qualification.timeline"] = timeline_info
        
        # Calculate overall score if all criteria are provided
        if all([budget_info, authority_info, need_info, timeline_info]):
            scores = []
            for info in [budget_info, authority_info, need_info, timeline_info]:
                if info.get("meets_criteria"):
                    scores.append(info.get("confidence", 0.5) * 100)
                else:
                    scores.append(0)
            
            overall_score = sum(scores) / len(scores)
            update_dict["$set"]["qualification.overall_score"] = overall_score
            
            # Let AI agent determine qualification status, not score-based rules
            # Just track the score for analytics
        
        return self.update_by_id(conversation_id, update_dict)
    
    def schedule_follow_up(
        self,
        conversation_id: str,
        scheduled_date: datetime,
        follow_up_type: str,
        priority: str = "medium",
        assigned_agent: str = AgentName.SUPERVISOR.value,
        context: Optional[Dict[str, Any]] = None
    ) -> UpdateResult:
        """Schedule a follow-up for a conversation.
        
        Args:
            conversation_id: Conversation ID
            scheduled_date: When to follow up
            follow_up_type: Type of follow-up
            priority: Follow-up priority (high, medium, low)
            assigned_agent: Agent assigned to follow-up
            context: Additional context for the follow-up
            
        Returns:
            UpdateResult: Result of the update
        """
        final_date = scheduled_date + timedelta(days=7)  # 7 days after initial follow-up
        
        return self.update_by_id(
            conversation_id,
            {
                "$set": {
                    "follow_up.required": True,
                    "follow_up.scheduled_date": scheduled_date.isoformat() + "Z",
                    "follow_up.type": follow_up_type,
                    "follow_up.priority": priority,
                    "follow_up.assigned_agent": assigned_agent,
                    "follow_up.final_attempt_date": final_date.isoformat() + "Z",
                    "follow_up.context": context or {},
                    "updated_at": datetime.utcnow().isoformat() + "Z"
                }
            }
        )