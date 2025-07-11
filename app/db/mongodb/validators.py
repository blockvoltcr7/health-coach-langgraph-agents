"""MongoDB validation functions for conversation data.

This module provides validation functions for various data formats
used in the conversation collection, including email, phone numbers,
scores, and other business data.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors with detailed messages."""
    
    def __init__(self, field: str, value: Any, message: str):
        self.field = field
        self.value = value
        self.message = message
        super().__init__(f"{field}: {message}")


class ConversationValidator:
    """Validator class for conversation data."""
    
    # Regex patterns
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9]+([._+%-][a-zA-Z0-9]+)*@[a-zA-Z0-9]+([.-][a-zA-Z0-9]+)*\.[a-zA-Z]{2,}$')
    PHONE_PATTERN = re.compile(r'^\+?[1-9]\d{4,14}$')  # E.164 format, min 5 digits
    ISO_DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$')
    
    # Constraints
    SCORE_MIN = 0
    SCORE_MAX = 100
    MAX_MESSAGES = 10000
    MAX_OBJECTIONS = 100
    MAX_STAGE_HISTORY = 50
    MAX_HANDOFFS = 100
    
    @classmethod
    def validate_email(cls, email: str, field_name: str = "email") -> str:
        """Validate email format.
        
        Args:
            email: Email address to validate
            field_name: Name of the field for error messages
            
        Returns:
            str: Validated email
            
        Raises:
            ValidationError: If email format is invalid
        """
        if not email:
            raise ValidationError(field_name, email, "Email cannot be empty")
            
        email = email.strip().lower()
        if not email or not cls.EMAIL_PATTERN.match(email):
            raise ValidationError(
                field_name, 
                email, 
                "Invalid email format. Expected format: user@example.com"
            )
        
        return email
    
    @classmethod
    def validate_phone(cls, phone: str, field_name: str = "phone") -> str:
        """Validate phone number format (E.164).
        
        Args:
            phone: Phone number to validate
            field_name: Name of the field for error messages
            
        Returns:
            str: Validated phone number
            
        Raises:
            ValidationError: If phone format is invalid
        """
        if not phone:
            raise ValidationError(field_name, phone, "Phone number cannot be empty")
        
        # Remove common formatting characters
        phone = re.sub(r'[\s\-\(\)]', '', phone)
        
        if not cls.PHONE_PATTERN.match(phone):
            raise ValidationError(
                field_name,
                phone,
                "Invalid phone format. Expected E.164 format: +1234567890"
            )
        
        return phone
    
    @classmethod
    def validate_score(
        cls, 
        score: Union[int, float], 
        field_name: str = "score"
    ) -> float:
        """Validate score is within 0-100 range.
        
        Args:
            score: Score value to validate
            field_name: Name of the field for error messages
            
        Returns:
            float: Validated score
            
        Raises:
            ValidationError: If score is out of range
        """
        try:
            score = float(score)
        except (TypeError, ValueError):
            raise ValidationError(
                field_name,
                score,
                f"Score must be a number, got {type(score).__name__}"
            )
        
        if not cls.SCORE_MIN <= score <= cls.SCORE_MAX:
            raise ValidationError(
                field_name,
                score,
                f"Score must be between {cls.SCORE_MIN} and {cls.SCORE_MAX}"
            )
        
        return score
    
    @classmethod
    def validate_iso_date(cls, date_str: str, field_name: str = "date") -> str:
        """Validate ISO 8601 date format with timezone.
        
        Args:
            date_str: Date string to validate
            field_name: Name of the field for error messages
            
        Returns:
            str: Validated date string
            
        Raises:
            ValidationError: If date format is invalid
        """
        if not date_str:
            raise ValidationError(field_name, date_str, "Date cannot be empty")
        
        if not cls.ISO_DATE_PATTERN.match(date_str):
            raise ValidationError(
                field_name,
                date_str,
                "Invalid date format. Expected ISO 8601: 2024-01-15T10:30:00Z"
            )
        
        # Verify it's a valid date
        try:
            datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except ValueError as e:
            raise ValidationError(field_name, date_str, f"Invalid date: {str(e)}")
        
        return date_str
    
    @classmethod
    def validate_array_size(
        cls,
        array: List[Any],
        max_size: int,
        field_name: str = "array"
    ) -> List[Any]:
        """Validate array doesn't exceed maximum size.
        
        Args:
            array: Array to validate
            max_size: Maximum allowed size
            field_name: Name of the field for error messages
            
        Returns:
            List[Any]: The array if valid
            
        Raises:
            ValidationError: If array exceeds max size
        """
        if not isinstance(array, list):
            raise ValidationError(
                field_name,
                array,
                f"Expected list, got {type(array).__name__}"
            )
        
        if len(array) > max_size:
            raise ValidationError(
                field_name,
                len(array),
                f"Array exceeds maximum size of {max_size} items"
            )
        
        return array
    
    @classmethod
    def validate_enum_value(
        cls,
        value: str,
        allowed_values: List[str],
        field_name: str = "enum_field"
    ) -> str:
        """Validate value is in allowed enum values.
        
        Args:
            value: Value to validate
            allowed_values: List of allowed values
            field_name: Name of the field for error messages
            
        Returns:
            str: The value if valid
            
        Raises:
            ValidationError: If value not in allowed values
        """
        if value not in allowed_values:
            raise ValidationError(
                field_name,
                value,
                f"Invalid value. Allowed values: {', '.join(allowed_values)}"
            )
        
        return value
    
    @classmethod
    def validate_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metadata fields if they contain contact information.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            Dict[str, Any]: Validated metadata
            
        Raises:
            ValidationError: If any validation fails
        """
        validated = metadata.copy()
        
        # Validate email if present
        if "email" in metadata and metadata["email"]:
            validated["email"] = cls.validate_email(metadata["email"], "metadata.email")
        
        # Validate phone if present
        if "phone" in metadata and metadata["phone"]:
            validated["phone"] = cls.validate_phone(metadata["phone"], "metadata.phone")
        
        # Validate any other contact fields
        for key in ["user_email", "contact_email", "customer_email"]:
            if key in metadata and metadata[key]:
                validated[key] = cls.validate_email(metadata[key], f"metadata.{key}")
        
        for key in ["user_phone", "contact_phone", "customer_phone", "mobile"]:
            if key in metadata and metadata[key]:
                validated[key] = cls.validate_phone(metadata[key], f"metadata.{key}")
        
        return validated
    
    @classmethod
    def validate_qualification_scores(
        cls, 
        qualification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate all scores in qualification data.
        
        Args:
            qualification: Qualification dictionary to validate
            
        Returns:
            Dict[str, Any]: Validated qualification data
            
        Raises:
            ValidationError: If any validation fails
        """
        validated = qualification.copy()
        
        # Validate overall score if present
        if "overall_score" in qualification and qualification["overall_score"] is not None:
            validated["overall_score"] = cls.validate_score(
                qualification["overall_score"],
                "qualification.overall_score"
            )
        
        # Validate individual BANT scores
        for bant_key in ["budget", "authority", "need", "timeline"]:
            if bant_key in qualification and isinstance(qualification[bant_key], dict):
                if "confidence" in qualification[bant_key] and qualification[bant_key]["confidence"] is not None:
                    validated[bant_key]["confidence"] = cls.validate_score(
                        qualification[bant_key]["confidence"],
                        f"qualification.{bant_key}.confidence"
                    )
        
        # Legacy score field
        if "score" in qualification and qualification["score"] is not None:
            validated["score"] = cls.validate_score(
                qualification["score"],
                "qualification.score"
            )
        
        return validated
    
    @classmethod
    def validate_conversation_document(
        cls,
        document: Dict[str, Any],
        is_update: bool = False
    ) -> Tuple[bool, List[str]]:
        """Validate an entire conversation document.
        
        Args:
            document: Document to validate
            is_update: Whether this is an update (partial document)
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Validate metadata if present
            if "metadata" in document:
                cls.validate_metadata(document["metadata"])
            
            # Validate qualification if present
            if "qualification" in document:
                cls.validate_qualification_scores(document["qualification"])
            
            # Validate arrays if present
            if "messages" in document:
                cls.validate_array_size(
                    document["messages"],
                    cls.MAX_MESSAGES,
                    "messages"
                )
            
            if "objections" in document:
                cls.validate_array_size(
                    document["objections"],
                    cls.MAX_OBJECTIONS,
                    "objections"
                )
            
            if "stage_history" in document:
                cls.validate_array_size(
                    document["stage_history"],
                    cls.MAX_STAGE_HISTORY,
                    "stage_history"
                )
            
            if "handoffs" in document:
                cls.validate_array_size(
                    document["handoffs"],
                    cls.MAX_HANDOFFS,
                    "handoffs"
                )
            
            # Validate dates in various fields
            date_fields = [
                "created_at",
                "updated_at",
                ("qualification", "qualified_at"),
                ("follow_up", "scheduled_date"),
                ("follow_up", "final_attempt_date"),
                ("agent_context", "assigned_at")
            ]
            
            for field in date_fields:
                if isinstance(field, tuple):
                    parent, child = field
                    if parent in document and child in document[parent] and document[parent][child]:
                        cls.validate_iso_date(
                            document[parent][child],
                            f"{parent}.{child}"
                        )
                else:
                    if field in document and document[field]:
                        cls.validate_iso_date(document[field], field)
            
        except ValidationError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"Unexpected validation error: {str(e)}")
        
        return len(errors) == 0, errors


def create_stage_transition(
    from_stage: str,
    to_stage: str,
    reason: str,
    triggered_by: str = "system",
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a stage transition record with proper validation.
    
    Args:
        from_stage: Current stage
        to_stage: New stage
        reason: Reason for transition
        triggered_by: What triggered the transition
        context: Additional context
        
    Returns:
        Dict[str, Any]: Stage transition record
    """
    now = datetime.now(timezone.utc)
    
    transition = {
        "from": from_stage,
        "to": to_stage,
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "reason": reason,
        "triggered_by": triggered_by,
        "context": context or {}
    }
    
    # Validate the timestamp
    ConversationValidator.validate_iso_date(transition["timestamp"], "transition.timestamp")
    
    return transition