"""MongoDB utility functions.

This module provides utility functions for database initialization,
index creation, and other MongoDB-related operations.
"""

import logging
from typing import Dict, Any, List, Optional

from pymongo.database import Database
from pymongo.errors import CollectionInvalid, OperationFailure

from app.db.mongodb.client import get_database
from app.db.mongodb.schemas.conversation_schema import (
    ConversationSchema,
    CONVERSATION_SCHEMA
)

logger = logging.getLogger(__name__)


def initialize_database(database: Optional[Database] = None) -> Dict[str, Any]:
    """Initialize the MongoDB database with required collections and indexes.
    
    Args:
        database: MongoDB database instance. Uses default if not provided.
        
    Returns:
        Dict[str, Any]: Initialization results
    """
    db = database or get_database()
    results = {
        'database': db.name,
        'collections_created': [],
        'indexes_created': [],
        'errors': []
    }
    
    try:
        # Initialize conversations collection
        logger.info("Initializing conversations collection...")
        
        if "conversations" not in db.list_collection_names():
            collection = ConversationSchema.create_collection(db)
            results['collections_created'].append('conversations')
            logger.info("Conversations collection created successfully")
        else:
            # Collection exists, ensure indexes are created
            collection = db['conversations']
            try:
                ConversationSchema.create_indexes(collection)
                logger.info("Conversations collection already exists, indexes updated")
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
                # Don't fail the initialization, just log the warning
        
        # Get index information
        indexes = list(collection.list_indexes())
        results['indexes_created'] = [idx['name'] for idx in indexes if idx['name'] != '_id_']
        
    except Exception as e:
        error_msg = f"Failed to initialize database: {e}"
        logger.error(error_msg)
        results['errors'].append(error_msg)
    
    return results


def validate_collection_schema(
    collection_name: str,
    database: Optional[Database] = None
) -> Dict[str, Any]:
    """Validate that a collection has the correct schema.
    
    Args:
        collection_name: Name of the collection to validate
        database: MongoDB database instance. Uses default if not provided.
        
    Returns:
        Dict[str, Any]: Validation results
    """
    db = database or get_database()
    results = {
        'collection': collection_name,
        'exists': False,
        'has_validator': False,
        'validator_valid': False,
        'indexes': [],
        'document_count': 0
    }
    
    try:
        # Check if collection exists
        if collection_name not in db.list_collection_names():
            return results
        
        results['exists'] = True
        collection = db[collection_name]
        
        # Get collection info
        collection_info = db.command('collMod', collection_name, getParameter='validator')
        
        if 'validator' in collection_info:
            results['has_validator'] = True
            # For conversations, check if validator matches expected schema
            if collection_name == 'conversations':
                results['validator_valid'] = (
                    collection_info['validator'] == CONVERSATION_SCHEMA
                )
        
        # Get indexes
        results['indexes'] = [idx['name'] for idx in collection.list_indexes()]
        
        # Get document count
        results['document_count'] = collection.count_documents({})
        
    except Exception as e:
        logger.error(f"Error validating collection {collection_name}: {e}")
        
    return results


def create_sample_conversation(
    user_id: str,
    database: Optional[Database] = None
) -> Optional[str]:
    """Create a sample conversation for testing.
    
    Args:
        user_id: User ID for the conversation
        database: MongoDB database instance. Uses default if not provided.
        
    Returns:
        Optional[str]: Created conversation ID or None if failed
    """
    from app.db.mongodb.schemas.conversation_schema import (
        ConversationRepository,
        ConversationSchema,
        MessageRole
    )
    
    try:
        repo = ConversationRepository(database)
        
        # Create conversation document
        doc = ConversationSchema.create_conversation_document(
            user_id=user_id,
            channel="web",
            initial_message="Hello, I'm interested in learning about Limitless OS.",
            metadata={"mem0_user_id": f"mem0_{user_id}", "source": "website"}
        )
        
        # Insert the document
        result = repo.create_one(doc)
        conversation_id = str(result.inserted_id)
        
        # Add a response message
        repo.add_message(
            conversation_id,
            MessageRole.SUPERVISOR.value,
            "Hello! I'm excited to tell you about Limitless OS. It's a revolutionary "
            "AI-powered business transformation platform. What specific aspect would "
            "you like to know more about?"
        )
        
        logger.info(f"Created sample conversation: {conversation_id}")
        return conversation_id
        
    except Exception as e:
        logger.error(f"Failed to create sample conversation: {e}")
        return None


def drop_all_collections(
    database: Optional[Database] = None,
    confirm: bool = False
) -> Dict[str, Any]:
    """Drop all collections in the database.
    
    WARNING: This will delete all data!
    
    Args:
        database: MongoDB database instance. Uses default if not provided.
        confirm: Must be True to actually drop collections
        
    Returns:
        Dict[str, Any]: Results of the operation
    """
    if not confirm:
        return {
            'error': 'Must set confirm=True to drop collections',
            'collections': []
        }
    
    db = database or get_database()
    results = {
        'dropped': [],
        'errors': []
    }
    
    try:
        collections = db.list_collection_names()
        for collection_name in collections:
            try:
                db.drop_collection(collection_name)
                results['dropped'].append(collection_name)
                logger.info(f"Dropped collection: {collection_name}")
            except Exception as e:
                error_msg = f"Failed to drop {collection_name}: {e}"
                results['errors'].append(error_msg)
                logger.error(error_msg)
                
    except Exception as e:
        results['errors'].append(f"Failed to list collections: {e}")
        
    return results


def get_database_stats(database: Optional[Database] = None) -> Dict[str, Any]:
    """Get statistics about the database.
    
    Args:
        database: MongoDB database instance. Uses default if not provided.
        
    Returns:
        Dict[str, Any]: Database statistics
    """
    db = database or get_database()
    stats = {
        'database': db.name,
        'collections': {},
        'total_documents': 0,
        'total_size': 0
    }
    
    try:
        # Get database stats
        db_stats = db.command('dbStats')
        stats['total_size'] = db_stats.get('dataSize', 0)
        
        # Get stats for each collection
        for collection_name in db.list_collection_names():
            collection = db[collection_name]
            col_stats = db.command('collStats', collection_name)
            
            stats['collections'][collection_name] = {
                'count': collection.count_documents({}),
                'size': col_stats.get('size', 0),
                'indexes': len(list(collection.list_indexes()))
            }
            
            stats['total_documents'] += stats['collections'][collection_name]['count']
            
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        
    return stats


def check_connection_health(database: Optional[Database] = None) -> Dict[str, Any]:
    """Check the health of the MongoDB connection.
    
    Args:
        database: MongoDB database instance. Uses default if not provided.
        
    Returns:
        Dict[str, Any]: Health check results
    """
    from app.db.mongodb.client import get_mongodb_client
    
    try:
        client = get_mongodb_client()
        health = client.health_check()
        
        # Add database-specific checks
        db = database or get_database()
        health['database'] = db.name
        health['collections'] = db.list_collection_names()
        
        return health
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'connected': False
        }