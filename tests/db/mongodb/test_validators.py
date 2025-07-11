"""Tests for MongoDB validators.

This module contains comprehensive tests for the validation
functions and classes used in MongoDB operations.
"""

import pytest
from datetime import datetime

from app.db.mongodb.validators import (
    ConversationValidator,
    ValidationError,
    create_stage_transition
)


class TestConversationValidator:
    """Test cases for ConversationValidator class."""
    
    def test_validate_email_valid(self):
        """Test validation of valid email addresses."""
        valid_emails = [
            "user@example.com",
            "test.user@domain.co.uk",
            "user+tag@example.com",
            "123@numbers.com",
            "user_name@example-domain.com"
        ]
        
        for email in valid_emails:
            result = ConversationValidator.validate_email(email)
            assert result == email.lower()
    
    def test_validate_email_invalid(self):
        """Test validation of invalid email addresses."""
        invalid_emails = [
            "notanemail",
            "@example.com",
            "user@",
            "user @example.com",
            "user@example",
            "",
            "user..double@example.com",
            "user@.example.com"
        ]
        
        for email in invalid_emails:
            with pytest.raises(ValidationError) as exc_info:
                ConversationValidator.validate_email(email)
            assert "email" in str(exc_info.value).lower()
    
    def test_validate_phone_valid(self):
        """Test validation of valid phone numbers."""
        valid_phones = [
            "+14155552671",
            "+442071838750",
            "+33142685300",
            "14155552671",  # Without + is also valid
            "+1234567890"
        ]
        
        for phone in valid_phones:
            result = ConversationValidator.validate_phone(phone)
            assert result == phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    
    def test_validate_phone_invalid(self):
        """Test validation of invalid phone numbers."""
        invalid_phones = [
            "123",  # Too short
            "+1234567890123456",  # Too long
            "abc123",  # Contains letters
            "",
            "++123456789",  # Double plus
            "+0123456789"  # Starts with 0 after country code
        ]
        
        for phone in invalid_phones:
            with pytest.raises(ValidationError) as exc_info:
                ConversationValidator.validate_phone(phone)
            assert "phone" in str(exc_info.value).lower()
    
    def test_validate_score_valid(self):
        """Test validation of valid scores."""
        valid_scores = [0, 0.5, 50, 99.9, 100]
        
        for score in valid_scores:
            result = ConversationValidator.validate_score(score)
            assert result == float(score)
    
    def test_validate_score_invalid(self):
        """Test validation of invalid scores."""
        invalid_scores = [-1, -0.1, 100.1, 101, 1000, "not a number", None]
        
        for score in invalid_scores:
            with pytest.raises(ValidationError) as exc_info:
                ConversationValidator.validate_score(score)
            assert "score" in str(exc_info.value).lower()
    
    def test_validate_iso_date_valid(self):
        """Test validation of valid ISO dates."""
        valid_dates = [
            "2024-01-15T10:30:00Z",
            "2024-12-31T23:59:59Z",
            "2024-01-01T00:00:00.123Z",
            "2024-01-01T00:00:00.123456Z"  # 6 decimal places
        ]
        
        for date in valid_dates:
            result = ConversationValidator.validate_iso_date(date)
            assert result == date
    
    def test_validate_iso_date_invalid(self):
        """Test validation of invalid ISO dates."""
        invalid_dates = [
            "2024-01-15",  # Missing time
            "2024-01-15T10:30:00",  # Missing Z
            "2024/01/15T10:30:00Z",  # Wrong separator
            "2024-13-01T10:30:00Z",  # Invalid month
            "2024-01-32T10:30:00Z",  # Invalid day
            "2024-01-15T25:30:00Z",  # Invalid hour
            "",
            "not a date"
        ]
        
        for date in invalid_dates:
            with pytest.raises(ValidationError) as exc_info:
                ConversationValidator.validate_iso_date(date)
            assert "date" in str(exc_info.value).lower()
    
    def test_validate_array_size_valid(self):
        """Test validation of arrays within size limits."""
        arrays = [
            ([], 10),
            ([1, 2, 3], 5),
            (["a", "b", "c"], 3),
            (list(range(100)), 100)
        ]
        
        for array, max_size in arrays:
            result = ConversationValidator.validate_array_size(array, max_size)
            assert result == array
    
    def test_validate_array_size_invalid(self):
        """Test validation of arrays exceeding size limits."""
        invalid_cases = [
            ([1, 2, 3], 2),
            (list(range(101)), 100),
            ("not an array", 10)
        ]
        
        for array, max_size in invalid_cases:
            with pytest.raises(ValidationError) as exc_info:
                ConversationValidator.validate_array_size(array, max_size)
            assert "array" in str(exc_info.value).lower() or "list" in str(exc_info.value).lower()
    
    def test_validate_enum_value_valid(self):
        """Test validation of valid enum values."""
        allowed_values = ["active", "inactive", "paused"]
        
        for value in allowed_values:
            result = ConversationValidator.validate_enum_value(value, allowed_values)
            assert result == value
    
    def test_validate_enum_value_invalid(self):
        """Test validation of invalid enum values."""
        allowed_values = ["active", "inactive", "paused"]
        invalid_values = ["Active", "ACTIVE", "completed", ""]
        
        for value in invalid_values:
            with pytest.raises(ValidationError) as exc_info:
                ConversationValidator.validate_enum_value(value, allowed_values)
            assert "allowed values" in str(exc_info.value).lower()
    
    def test_validate_metadata_with_contacts(self):
        """Test validation of metadata containing contact information."""
        metadata = {
            "email": "USER@EXAMPLE.COM",
            "phone": "+1 (415) 555-2671",
            "user_email": "test@domain.com",
            "customer_phone": "14155552672",
            "other_field": "some value"
        }
        
        result = ConversationValidator.validate_metadata(metadata)
        
        # Check emails are lowercased
        assert result["email"] == "user@example.com"
        assert result["user_email"] == "test@domain.com"
        
        # Check phones are cleaned
        assert result["phone"] == "+14155552671"
        assert result["customer_phone"] == "14155552672"
        
        # Check other fields are unchanged
        assert result["other_field"] == "some value"
    
    def test_validate_metadata_with_invalid_contacts(self):
        """Test validation of metadata with invalid contact information."""
        metadata = {
            "email": "invalid.email",
            "other_field": "value"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ConversationValidator.validate_metadata(metadata)
        assert "email" in str(exc_info.value).lower()
    
    def test_validate_qualification_scores(self):
        """Test validation of qualification scores."""
        qualification = {
            "overall_score": 85.5,
            "score": 90,
            "budget": {
                "meets_criteria": True,
                "confidence": 0.8
            },
            "authority": {
                "meets_criteria": False,
                "confidence": 0.3
            },
            "need": {
                "confidence": None  # Should be skipped
            },
            "timeline": {
                "confidence": 0.95
            }
        }
        
        result = ConversationValidator.validate_qualification_scores(qualification)
        
        assert result["overall_score"] == 85.5
        assert result["score"] == 90
        assert result["budget"]["confidence"] == 0.8
        assert result["authority"]["confidence"] == 0.3
        assert result["timeline"]["confidence"] == 0.95
    
    def test_validate_qualification_scores_out_of_range(self):
        """Test validation of qualification scores out of range."""
        qualification = {
            "overall_score": 150  # Out of range
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ConversationValidator.validate_qualification_scores(qualification)
        assert "score" in str(exc_info.value).lower()
        assert "0 and 100" in str(exc_info.value)
    
    def test_validate_conversation_document_complete(self):
        """Test validation of a complete conversation document."""
        document = {
            "user_id": "user123",
            "channel": "web",
            "status": "active",
            "sales_stage": "lead",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            ],
            "metadata": {
                "email": "user@example.com",
                "phone": "+14155552671"
            },
            "qualification": {
                "overall_score": 75
            },
            "created_at": "2024-01-15T10:00:00Z",
            "updated_at": "2024-01-15T10:30:00Z"
        }
        
        is_valid, errors = ConversationValidator.validate_conversation_document(document)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_conversation_document_with_errors(self):
        """Test validation of document with multiple errors."""
        document = {
            "messages": [{"msg": "test"}] * 11000,  # Too many messages
            "metadata": {
                "email": "not-an-email"
            },
            "qualification": {
                "overall_score": 200  # Out of range
            },
            "created_at": "invalid-date"
        }
        
        is_valid, errors = ConversationValidator.validate_conversation_document(document)
        
        assert is_valid is False
        assert len(errors) > 0
        # Check that various errors are captured
        error_str = " ".join(errors).lower()
        assert "array exceeds" in error_str or "email" in error_str or "score" in error_str


class TestStageTransition:
    """Test cases for stage transition creation."""
    
    def test_create_stage_transition(self):
        """Test creation of stage transition record."""
        transition = create_stage_transition(
            from_stage="lead",
            to_stage="qualification",
            reason="User showed interest",
            triggered_by="user_message",
            context={"confidence": 0.9}
        )
        
        assert transition["from"] == "lead"
        assert transition["to"] == "qualification"
        assert transition["reason"] == "User showed interest"
        assert transition["triggered_by"] == "user_message"
        assert transition["context"]["confidence"] == 0.9
        
        # Check timestamp is valid ISO format
        ConversationValidator.validate_iso_date(transition["timestamp"])
    
    def test_create_stage_transition_minimal(self):
        """Test creation of stage transition with minimal data."""
        transition = create_stage_transition(
            from_stage="qualification",
            to_stage="qualified",
            reason="BANT criteria met"
        )
        
        assert transition["from"] == "qualification"
        assert transition["to"] == "qualified"
        assert transition["reason"] == "BANT criteria met"
        assert transition["triggered_by"] == "system"
        assert transition["context"] == {}


class TestValidationErrorHandling:
    """Test cases for ValidationError exception."""
    
    def test_validation_error_properties(self):
        """Test ValidationError has correct properties."""
        error = ValidationError("test_field", "bad_value", "Invalid format")
        
        assert error.field == "test_field"
        assert error.value == "bad_value"
        assert error.message == "Invalid format"
        assert str(error) == "test_field: Invalid format"
    
    def test_validation_error_in_try_except(self):
        """Test handling ValidationError in try-except blocks."""
        try:
            ConversationValidator.validate_email("invalid")
        except ValidationError as e:
            assert e.field == "email"
            assert "invalid" in str(e.value)
            assert "Invalid email format" in e.message


class TestArraySizeLimits:
    """Test cases for array size limit constants."""
    
    def test_max_limits_reasonable(self):
        """Test that max limits are reasonable values."""
        assert ConversationValidator.MAX_MESSAGES == 10000
        assert ConversationValidator.MAX_OBJECTIONS == 100
        assert ConversationValidator.MAX_STAGE_HISTORY == 50
        assert ConversationValidator.MAX_HANDOFFS == 100
    
    def test_score_range_limits(self):
        """Test score range limits."""
        assert ConversationValidator.SCORE_MIN == 0
        assert ConversationValidator.SCORE_MAX == 100