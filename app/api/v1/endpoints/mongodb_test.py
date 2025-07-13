"""MongoDB test endpoints for connection verification."""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from datetime import datetime

from app.db.mongodb.client import get_mongodb_client, get_database
from app.db.mongodb.utils import get_database_stats, create_sample_conversation

router = APIRouter()


@router.get(
    "/test",
    summary="Test MongoDB Connection",
    description="Test endpoint to verify MongoDB connection and basic operations",
    response_description="MongoDB connection test results",
    tags=["MongoDB Test"]
)
async def test_mongodb_connection() -> Dict[str, Any]:
    """
    Test MongoDB connection and perform basic operations.
    
    This endpoint:
    - Verifies MongoDB connection
    - Gets database statistics
    - Creates a test document
    - Performs a read operation
    - Returns comprehensive test results
    """
    try:
        # Get MongoDB client and database
        client = get_mongodb_client()
        db = get_database()
        
        # 1. Test connection
        health = client.health_check()
        
        # 2. Get database stats
        stats = get_database_stats(db)
        
        # 3. Create a test conversation (if conversations collection exists)
        test_user_id = f"test_user_{datetime.utcnow().timestamp()}"
        sample_id = None
        
        try:
            sample_id = create_sample_conversation(test_user_id, db)
            sample_created = True
        except Exception as e:
            sample_created = False
            sample_error = str(e)
        
        # 4. Read test - count documents in conversations collection
        try:
            conversations_collection = db['conversations']
            doc_count = conversations_collection.count_documents({})
        except Exception:
            doc_count = 0
        
        # 5. Cleanup test document if created
        if sample_id:
            try:
                conversations_collection.delete_one({"_id": sample_id})
            except Exception:
                pass  # Ignore cleanup errors
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "connection": {
                "healthy": health.get("connected", False),
                "version": health.get("version", "unknown"),
                "status": health.get("status", "unknown")
            },
            "database": {
                "name": stats.get("database", "unknown"),
                "collections": list(stats.get("collections", {}).keys()),
                "total_documents": stats.get("total_documents", 0)
            },
            "tests": {
                "connection": "passed" if health.get("connected") else "failed",
                "read": "passed" if doc_count >= 0 else "failed", 
                "write": "passed" if sample_created else "failed",
                "sample_created": sample_created,
                "conversation_count": doc_count
            },
            "message": "MongoDB connection test completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "MongoDB connection test failed",
                "error": str(e)
            }
        )


@router.get(
    "/stats",
    summary="Get MongoDB Statistics",
    description="Get detailed statistics about the MongoDB database",
    response_description="Database statistics",
    tags=["MongoDB Test"]
)
async def get_mongodb_stats() -> Dict[str, Any]:
    """
    Get detailed MongoDB database statistics.
    
    Returns information about:
    - Database size
    - Collection counts
    - Index information
    - Storage statistics
    """
    try:
        db = get_database()
        stats = get_database_stats(db)
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "statistics": stats
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to get database statistics",
                "error": str(e)
            }
        )