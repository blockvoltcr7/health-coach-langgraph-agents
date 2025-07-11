"""MongoDB configuration for Sales AI Closer system.

This module manages MongoDB connection settings, database configuration,
and collection names using Pydantic for validation and environment variables.
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, computed_field
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()


class MongoDBConfig(BaseModel):
    """Configuration for MongoDB connection and database settings.
    
    This class manages all MongoDB-related configuration including
    connection strings, database names, collection names, and connection
    pool settings.
    """
    
    # Connection credentials
    username: str = Field(description="MongoDB username")
    password: str = Field(description="MongoDB password")
    
    # Atlas cluster information
    cluster_url: str = Field(
        default="limitless-os.5ar2eh.mongodb.net",
        description="MongoDB Atlas cluster URL"
    )
    
    # Database configuration
    database_name: str = Field(
        default="limitless_os_sales",
        description="Database name for the sales system"
    )
    
    # Collection names
    conversations_collection: str = Field(
        default="conversations",
        description="Collection for storing conversation data"
    )
    
    # Connection pool settings
    max_pool_size: int = Field(default=50, description="Maximum connection pool size")
    min_pool_size: int = Field(default=10, description="Minimum connection pool size")
    
    # Timeout settings (in milliseconds)
    connect_timeout_ms: int = Field(default=10000, description="Connection timeout")
    server_selection_timeout_ms: int = Field(default=5000, description="Server selection timeout")
    
    # Additional connection options
    retry_writes: bool = Field(default=True, description="Enable retry writes")
    w: str = Field(default="majority", description="Write concern")
    app_name: str = Field(default="limitless-os", description="Application name")
    
    def __init__(self, **data):
        """Initialize MongoDB configuration with environment variables."""
        # Load credentials from environment if not provided
        if 'username' not in data:
            data['username'] = os.getenv('MONGODB_USERNAME', '')
        if 'password' not in data:
            data['password'] = os.getenv('MONGODB_PASSWORD', '')
            
        super().__init__(**data)
        
        # Validate credentials are present
        if not self.username or not self.password:
            raise ValueError(
                "MongoDB credentials not found. Please set MONGODB_USERNAME "
                "and MONGODB_PASSWORD environment variables."
            )
    
    @computed_field
    @property
    def connection_uri(self) -> str:
        """Generate the MongoDB connection URI.
        
        Returns:
            str: MongoDB Atlas connection string with all parameters
        """
        # URL encode username and password to handle special characters
        encoded_username = quote_plus(self.username)
        encoded_password = quote_plus(self.password)
        
        # Build the connection string
        uri = (
            f"mongodb+srv://{encoded_username}:{encoded_password}"
            f"@{self.cluster_url}/?"
            f"retryWrites={str(self.retry_writes).lower()}"
            f"&w={self.w}"
            f"&appName={self.app_name}"
        )
        
        return uri
    
    @property
    def connection_options(self) -> Dict[str, Any]:
        """Get connection options for MongoClient.
        
        Returns:
            Dict[str, Any]: Connection options dictionary
        """
        return {
            'maxPoolSize': self.max_pool_size,
            'minPoolSize': self.min_pool_size,
            'connectTimeoutMS': self.connect_timeout_ms,
            'serverSelectionTimeoutMS': self.server_selection_timeout_ms,
        }
    
    @property
    def collection_names(self) -> Dict[str, str]:
        """Get all collection names.
        
        Returns:
            Dict[str, str]: Mapping of collection types to names
        """
        return {
            'conversations': self.conversations_collection,
        }
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = 'forbid'


def get_mongodb_config() -> MongoDBConfig:
    """Get the default MongoDB configuration instance.
    
    Returns:
        MongoDBConfig: MongoDB configuration instance
    
    Raises:
        ValueError: If required environment variables are not set
    """
    return MongoDBConfig()