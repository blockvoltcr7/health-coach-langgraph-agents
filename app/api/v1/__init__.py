"""
Main entrypoint for the v1 API.

This module provides the API router that includes all v1 endpoints.
"""

# The main API router is now defined in api.py to avoid conflicts
# Import it from there if needed
from .api import api_router

__all__ = ["api_router"]
