"""Test utilities for date formatting."""

from datetime import datetime


def format_iso_date() -> str:
    """Format current UTC datetime to ISO format with milliseconds.
    
    Returns ISO format: YYYY-MM-DDTHH:MM:SS.sssZ
    """
    now = datetime.utcnow()
    # Format with exactly 3 millisecond digits
    return now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def test_iso_date_format():
    """Test that our date format matches the validation pattern."""
    import re
    
    # The pattern from validation_schemas.py
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z$"
    
    # Test our formatted date
    formatted = format_iso_date()
    assert re.match(pattern, formatted), f"Date {formatted} doesn't match pattern"
    
    # Test format consistency
    assert formatted.endswith('Z')
    assert len(formatted) == 24  # YYYY-MM-DDTHH:MM:SS.sssZ


if __name__ == "__main__":
    test_iso_date_format()
    print(f"Sample formatted date: {format_iso_date()}")