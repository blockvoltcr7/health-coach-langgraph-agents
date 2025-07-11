"""MongoDB validation schemas for enhanced data validation.

This module provides JSON Schema validation definitions for MongoDB
collections with regex patterns, constraints, and custom validation rules.
"""

from typing import Dict, Any, List


def get_regex_patterns() -> Dict[str, str]:
    """Get regex patterns for field validation.
    
    Returns:
        Dict[str, str]: Dictionary of regex patterns
    """
    return {
        "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "phone": r"^\+?[1-9]\d{1,14}$",  # E.164 format
        "iso_date": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z$",
        "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    }


def get_score_validation() -> Dict[str, Any]:
    """Get validation schema for score fields.
    
    Returns:
        Dict[str, Any]: Score validation schema
    """
    return {
        "bsonType": ["double", "int", "null"],
        "minimum": 0,
        "maximum": 100,
        "description": "Score must be between 0 and 100"
    }


def get_email_validation() -> Dict[str, Any]:
    """Get validation schema for email fields.
    
    Returns:
        Dict[str, Any]: Email validation schema
    """
    patterns = get_regex_patterns()
    return {
        "bsonType": ["string", "null"],
        "pattern": patterns["email"],
        "description": "Valid email format required"
    }


def get_phone_validation() -> Dict[str, Any]:
    """Get validation schema for phone fields.
    
    Returns:
        Dict[str, Any]: Phone validation schema
    """
    patterns = get_regex_patterns()
    return {
        "bsonType": ["string", "null"],
        "pattern": patterns["phone"],
        "description": "Valid E.164 phone format required"
    }


def get_date_validation() -> Dict[str, Any]:
    """Get validation schema for date fields.
    
    Returns:
        Dict[str, Any]: Date validation schema
    """
    patterns = get_regex_patterns()
    return {
        "bsonType": ["string", "date", "null"],
        "pattern": patterns["iso_date"],
        "description": "ISO 8601 date format with timezone required"
    }


def get_enhanced_conversation_schema(
    channel_types: List[str],
    conversation_statuses: List[str],
    sales_stages: List[str],
    message_roles: List[str],
    agent_names: List[str],
    objection_types: List[str],
    follow_up_types: List[str]
) -> Dict[str, Any]:
    """Get enhanced conversation schema with full validation.
    
    Args:
        channel_types: List of valid channel types
        conversation_statuses: List of valid conversation statuses
        sales_stages: List of valid sales stages
        message_roles: List of valid message roles
        agent_names: List of valid agent names
        objection_types: List of valid objection types
        follow_up_types: List of valid follow-up types
        
    Returns:
        Dict[str, Any]: Enhanced MongoDB validation schema
    """
    return {
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
                    "enum": channel_types,
                    "description": "Communication channel must be one of the defined types"
                },
                "status": {
                    "bsonType": "string",
                    "enum": conversation_statuses,
                    "description": "Conversation status must be one of the defined values"
                },
                "sales_stage": {
                    "bsonType": "string",
                    "enum": sales_stages,
                    "description": "Sales stage must be one of the defined values"
                },
                "current_agent": {
                    "bsonType": ["string", "null"],
                    "enum": agent_names + [None],
                    "description": "Current agent must be one of the defined agent names"
                },
                "is_qualified": {
                    "bsonType": ["bool", "null"],
                    "description": "Qualification status determined by AI agent"
                },
                "stage_history": {
                    "bsonType": "array",
                    "maxItems": 50,
                    "items": {
                        "bsonType": "object",
                        "required": ["stage", "timestamp"],
                        "properties": {
                            "stage": {
                                "bsonType": "string",
                                "enum": sales_stages
                            },
                            "timestamp": get_date_validation(),
                            "notes": {"bsonType": "string", "maxLength": 1000}
                        }
                    }
                },
                "stage_transitions": {
                    "bsonType": "array",
                    "maxItems": 100,
                    "items": {
                        "bsonType": "object",
                        "required": ["from", "to", "timestamp", "reason"],
                        "properties": {
                            "from": {
                                "bsonType": "string",
                                "enum": sales_stages
                            },
                            "to": {
                                "bsonType": "string",
                                "enum": sales_stages
                            },
                            "timestamp": get_date_validation(),
                            "reason": {"bsonType": "string", "maxLength": 500},
                            "triggered_by": {"bsonType": "string", "maxLength": 100},
                            "context": {"bsonType": "object"}
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
                                "confidence": get_score_validation(),
                                "captured_at": get_date_validation()
                            }
                        },
                        "authority": {
                            "bsonType": ["object", "bool"],
                            "properties": {
                                "meets_criteria": {"bsonType": ["bool", "null"]},
                                "role": {"bsonType": ["string", "null"]},
                                "needs_approval": {"bsonType": ["bool", "null"]},
                                "confidence": get_score_validation()
                            }
                        },
                        "need": {
                            "bsonType": ["object", "bool"],
                            "properties": {
                                "meets_criteria": {"bsonType": ["bool", "null"]},
                                "pain_points": {
                                    "bsonType": "array",
                                    "maxItems": 20,
                                    "items": {"bsonType": "string", "maxLength": 500}
                                },
                                "use_case": {"bsonType": ["string", "null"], "maxLength": 1000},
                                "confidence": get_score_validation()
                            }
                        },
                        "timeline": {
                            "bsonType": ["object", "bool"],
                            "properties": {
                                "meets_criteria": {"bsonType": ["bool", "null"]},
                                "timeframe": {"bsonType": ["string", "null"]},
                                "deadline": get_date_validation(),
                                "urgency": {
                                    "bsonType": ["string", "null"],
                                    "enum": ["high", "medium", "low", None]
                                },
                                "confidence": get_score_validation()
                            }
                        },
                        "overall_score": get_score_validation(),
                        "qualified_at": get_date_validation(),
                        "qualified_by": {
                            "bsonType": ["string", "null"],
                            "enum": agent_names + [None]
                        },
                        "qualification_method": {"bsonType": ["string", "null"]},
                        "score": get_score_validation(),  # Legacy field
                        "notes": {"bsonType": "string", "maxLength": 2000}
                    }
                },
                "messages": {
                    "bsonType": "array",
                    "maxItems": 10000,
                    "items": {
                        "bsonType": "object",
                        "required": ["role", "content", "timestamp"],
                        "properties": {
                            "role": {
                                "bsonType": "string",
                                "enum": message_roles,
                                "description": "Message role must be one of the defined values"
                            },
                            "content": {
                                "bsonType": "string",
                                "maxLength": 10000,
                                "description": "Message content with reasonable length limit"
                            },
                            "timestamp": get_date_validation()
                        }
                    }
                },
                "objections": {
                    "bsonType": "array",
                    "maxItems": 100,
                    "items": {
                        "bsonType": "object",
                        "properties": {
                            "objection_id": {"bsonType": "string"},
                            "type": {
                                "bsonType": "string",
                                "enum": objection_types
                            },
                            "content": {"bsonType": "string", "maxLength": 1000},
                            "raised_at": get_date_validation(),
                            "severity": {
                                "bsonType": "string",
                                "enum": ["high", "medium", "low"]
                            },
                            "status": {
                                "bsonType": "string",
                                "enum": ["active", "resolved", "deferred"]
                            },
                            "handled_by": {
                                "bsonType": "string",
                                "enum": agent_names
                            },
                            "handling_attempts": {
                                "bsonType": "int",
                                "minimum": 0,
                                "maximum": 10
                            },
                            "resolution": {
                                "bsonType": "object",
                                "properties": {
                                    "resolved": {"bsonType": "bool"},
                                    "method": {"bsonType": ["string", "null"]},
                                    "resolved_at": get_date_validation(),
                                    "resolution_notes": {"bsonType": ["string", "null"], "maxLength": 1000},
                                    "confidence": get_score_validation()
                                }
                            },
                            # Legacy fields
                            "objection": {"bsonType": "string"},
                            "response": {"bsonType": "string"},
                            "resolved": {"bsonType": "bool"},
                            "timestamp": get_date_validation()
                        }
                    }
                },
                "handoffs": {
                    "bsonType": "array",
                    "maxItems": 100,
                    "items": {
                        "bsonType": "object",
                        "required": ["from_agent", "to_agent", "timestamp"],
                        "properties": {
                            "handoff_id": {"bsonType": "string"},
                            "from_agent": {
                                "bsonType": "string",
                                "enum": agent_names,
                                "description": "From agent must be valid agent name"
                            },
                            "to_agent": {
                                "bsonType": "string",
                                "enum": agent_names,
                                "description": "To agent must be valid agent name"
                            },
                            "timestamp": get_date_validation(),
                            "reason": {"bsonType": "string", "maxLength": 500},
                            "trigger": {
                                "bsonType": "object",
                                "properties": {
                                    "type": {"bsonType": "string"},
                                    "details": {"bsonType": "string", "maxLength": 1000}
                                }
                            },
                            "context": {
                                "bsonType": "object",
                                "properties": {
                                    "previous_stage": {"bsonType": "string"},
                                    "new_stage": {"bsonType": "string"},
                                    "confidence_score": get_score_validation(),
                                    "handoff_notes": {"bsonType": "string", "maxLength": 1000}
                                }
                            },
                            "metrics": {
                                "bsonType": "object",
                                "properties": {
                                    "messages_before_handoff": {
                                        "bsonType": "int",
                                        "minimum": 0
                                    },
                                    "time_in_previous_agent": {
                                        "bsonType": "int",
                                        "minimum": 0,
                                        "description": "Time in seconds"
                                    }
                                }
                            }
                        }
                    }
                },
                "follow_up": {
                    "bsonType": "object",
                    "properties": {
                        "required": {"bsonType": "bool"},
                        "scheduled_date": get_date_validation(),
                        "type": {
                            "bsonType": ["string", "null"],
                            "enum": follow_up_types + [None]
                        },
                        "priority": {
                            "bsonType": ["string", "null"],
                            "enum": ["high", "medium", "low", None]
                        },
                        "assigned_agent": {
                            "bsonType": ["string", "null"],
                            "enum": agent_names + [None]
                        },
                        "attempts": {
                            "bsonType": "array",
                            "maxItems": 20,
                            "items": {
                                "bsonType": "object",
                                "properties": {
                                    "timestamp": get_date_validation(),
                                    "agent": {"bsonType": "string"},
                                    "result": {"bsonType": "string"},
                                    "notes": {"bsonType": "string", "maxLength": 500}
                                }
                            }
                        },
                        "max_attempts": {
                            "bsonType": "int",
                            "minimum": 1,
                            "maximum": 10
                        },
                        "final_attempt_date": get_date_validation(),
                        "context": {"bsonType": "object"}
                    }
                },
                "agent_context": {
                    "bsonType": "object",
                    "properties": {
                        "assigned_at": get_date_validation(),
                        "previous_agent": {
                            "bsonType": ["string", "null"],
                            "enum": agent_names + [None]
                        },
                        "handoff_reason": {"bsonType": ["string", "null"], "maxLength": 500},
                        "interaction_count": {
                            "bsonType": "int",
                            "minimum": 0
                        }
                    }
                },
                "agent_metrics": {
                    "bsonType": "object",
                    "properties": {
                        "supervisor": {"bsonType": "object"},
                        "qualifier": {"bsonType": "object"},
                        "objection_handler": {"bsonType": "object"},
                        "closer": {"bsonType": "object"},
                        "total_interaction_time": {
                            "bsonType": "int",
                            "minimum": 0,
                            "description": "Total time in seconds"
                        },
                        "total_messages": {
                            "bsonType": "int",
                            "minimum": 0
                        },
                        "engagement_score": get_score_validation()
                    }
                },
                "deal_details": {
                    "bsonType": "object",
                    "properties": {
                        "monthly_value": {
                            "bsonType": "int",
                            "minimum": 0,
                            "description": "Monthly value in cents"
                        },
                        "setup_fee": {
                            "bsonType": "int",
                            "minimum": 0,
                            "description": "Setup fee in cents"
                        },
                        "total_first_payment": {
                            "bsonType": "int",
                            "minimum": 0,
                            "description": "Total first payment in cents"
                        },
                        "expected_ltv": {
                            "bsonType": "int",
                            "minimum": 0,
                            "description": "Expected lifetime value in cents"
                        },
                        "close_date": get_date_validation(),
                        "start_date": get_date_validation(),
                        "contract_length": {"bsonType": "string", "maxLength": 100},
                        "payment_method": {
                            "bsonType": ["string", "null"],
                            "enum": ["credit_card", "ach", "wire", "check", None]
                        },
                        "discount_applied": {
                            "bsonType": "int",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Discount percentage"
                        },
                        "close_reason": {"bsonType": "string", "maxLength": 500}
                    }
                },
                "metadata": {
                    "bsonType": "object",
                    "description": "Flexible metadata field for additional data"
                },
                "created_at": get_date_validation(),
                "updated_at": get_date_validation()
            }
        }
    }


def get_metadata_validation_rules() -> Dict[str, Any]:
    """Get validation rules for metadata fields.
    
    Returns:
        Dict[str, Any]: Metadata validation rules
    """
    return {
        "email_fields": ["email", "user_email", "contact_email", "customer_email"],
        "phone_fields": ["phone", "user_phone", "contact_phone", "customer_phone", "mobile"],
        "validation": {
            "email": get_email_validation(),
            "phone": get_phone_validation()
        }
    }