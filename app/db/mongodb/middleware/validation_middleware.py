"""Validation middleware for MongoDB operations.

This module provides middleware functions that validate data
before database operations, ensuring data integrity and consistency.
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from functools import wraps

from app.db.mongodb.validators import (
    ConversationValidator,
    ValidationError,
    create_stage_transition
)

logger = logging.getLogger(__name__)


class ValidationMiddleware:
    """Middleware class for data validation in MongoDB operations."""
    
    def __init__(self, raise_on_error: bool = True):
        """Initialize validation middleware.
        
        Args:
            raise_on_error: Whether to raise exceptions on validation errors
        """
        self.raise_on_error = raise_on_error
        self.validator = ConversationValidator()
    
    def validate_before_insert(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document before insertion.
        
        Args:
            document: Document to validate
            
        Returns:
            Dict[str, Any]: Validated document
            
        Raises:
            ValidationError: If validation fails and raise_on_error is True
        """
        try:
            # Validate metadata if present
            if "metadata" in document:
                document["metadata"] = self.validator.validate_metadata(
                    document["metadata"]
                )
            
            # Validate qualification scores if present
            if "qualification" in document:
                document["qualification"] = self.validator.validate_qualification_scores(
                    document["qualification"]
                )
            
            # Validate dates
            date_fields = ["created_at", "updated_at"]
            for field in date_fields:
                if field in document and document[field]:
                    document[field] = self.validator.validate_iso_date(
                        document[field], field
                    )
            
            # Validate arrays
            if "messages" in document:
                self.validator.validate_array_size(
                    document["messages"],
                    self.validator.MAX_MESSAGES,
                    "messages"
                )
            
            return document
            
        except ValidationError as e:
            logger.error(f"Validation error on insert: {e}")
            if self.raise_on_error:
                raise
            return document
    
    def validate_before_update(
        self,
        update_dict: Dict[str, Any],
        validate_arrays: bool = True
    ) -> Dict[str, Any]:
        """Validate update operations.
        
        Args:
            update_dict: MongoDB update dictionary
            validate_arrays: Whether to validate array operations
            
        Returns:
            Dict[str, Any]: Validated update dictionary
            
        Raises:
            ValidationError: If validation fails and raise_on_error is True
        """
        try:
            # Handle $set operations
            if "$set" in update_dict:
                set_ops = update_dict["$set"]
                
                # Validate metadata fields
                for key, value in list(set_ops.items()):
                    if key.startswith("metadata.") and any(
                        contact_field in key
                        for contact_field in ["email", "phone", "mobile"]
                    ):
                        field_type = "email" if "email" in key else "phone"
                        if field_type == "email":
                            set_ops[key] = self.validator.validate_email(value, key)
                        else:
                            set_ops[key] = self.validator.validate_phone(value, key)
                    
                    # Validate score fields
                    elif "score" in key or "confidence" in key:
                        if value is not None:
                            set_ops[key] = self.validator.validate_score(value, key)
                    
                    # Validate date fields
                    elif key.endswith("_at") or key.endswith("_date"):
                        if value is not None:
                            set_ops[key] = self.validator.validate_iso_date(value, key)
            
            # Handle $push operations
            if "$push" in update_dict and validate_arrays:
                push_ops = update_dict["$push"]
                
                # Validate message additions
                if "messages" in push_ops:
                    msg = push_ops["messages"]
                    if isinstance(msg, dict) and "timestamp" in msg:
                        msg["timestamp"] = self.validator.validate_iso_date(
                            msg["timestamp"], "message.timestamp"
                        )
                
                # Validate handoff additions
                if "handoffs" in push_ops:
                    handoff = push_ops["handoffs"]
                    if isinstance(handoff, dict):
                        if "timestamp" in handoff:
                            handoff["timestamp"] = self.validator.validate_iso_date(
                                handoff["timestamp"], "handoff.timestamp"
                            )
                        if "context" in handoff and "confidence_score" in handoff["context"]:
                            handoff["context"]["confidence_score"] = self.validator.validate_score(
                                handoff["context"]["confidence_score"],
                                "handoff.confidence_score"
                            )
            
            return update_dict
            
        except ValidationError as e:
            logger.error(f"Validation error on update: {e}")
            if self.raise_on_error:
                raise
            return update_dict
    
    def create_validated_stage_transition(
        self,
        from_stage: str,
        to_stage: str,
        reason: str,
        triggered_by: str = "system",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a validated stage transition record.
        
        Args:
            from_stage: Current stage
            to_stage: New stage
            reason: Reason for transition
            triggered_by: What triggered the transition
            context: Additional context
            
        Returns:
            Dict[str, Any]: Validated stage transition
        """
        return create_stage_transition(
            from_stage=from_stage,
            to_stage=to_stage,
            reason=reason,
            triggered_by=triggered_by,
            context=context
        )
    
    def log_validation_error(
        self,
        error: ValidationError,
        operation: str,
        document_id: Optional[str] = None
    ) -> None:
        """Log validation error with context.
        
        Args:
            error: Validation error
            operation: Operation being performed
            document_id: Optional document ID
        """
        error_context = {
            "operation": operation,
            "field": error.field,
            "value": str(error.value)[:100],  # Truncate long values
            "message": error.message
        }
        
        if document_id:
            error_context["document_id"] = document_id
        
        logger.error(f"Validation error: {error_context}")


def validation_decorator(
    middleware: ValidationMiddleware,
    operation: str = "operation"
) -> Callable:
    """Decorator for adding validation to repository methods.
    
    Args:
        middleware: ValidationMiddleware instance
        operation: Name of the operation for logging
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Pre-validation logic could go here
                result = func(*args, **kwargs)
                # Post-validation logic could go here
                return result
            except ValidationError as e:
                middleware.log_validation_error(e, operation)
                if middleware.raise_on_error:
                    raise
                return None
        return wrapper
    return decorator


def create_validation_middleware(
    raise_on_error: bool = True
) -> ValidationMiddleware:
    """Factory function to create validation middleware.
    
    Args:
        raise_on_error: Whether to raise exceptions on validation errors
        
    Returns:
        ValidationMiddleware: Configured middleware instance
    """
    return ValidationMiddleware(raise_on_error=raise_on_error)