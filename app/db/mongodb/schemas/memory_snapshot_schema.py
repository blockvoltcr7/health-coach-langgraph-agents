"""MongoDB schema for memory snapshots.

This module defines the schema and repository for storing Mem0 memory snapshots
in MongoDB for backup, synchronization, and analysis purposes.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from enum import Enum
import logging

from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.results import InsertOneResult, UpdateResult
from bson import ObjectId

from app.db.mongodb.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class SnapshotStatus(str, Enum):
    """Status of memory snapshots."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


def get_memory_snapshot_schema() -> Dict[str, Any]:
    """Get the JSON schema for memory snapshot validation.
    
    Returns:
        Dict containing MongoDB JSON schema for memory snapshots
    """
    return {
        "bsonType": "object",
        "required": ["user_id", "snapshot_timestamp", "status", "created_at"],
        "properties": {
            "user_id": {
                "bsonType": "string",
                "description": "User identifier from Mem0"
            },
            "snapshot_timestamp": {
                "bsonType": "date",
                "description": "When the snapshot was taken"
            },
            "status": {
                "enum": ["pending", "in_progress", "completed", "failed", "partial"],
                "description": "Snapshot status"
            },
            "memories": {
                "bsonType": "array",
                "description": "Array of memory entries",
                "items": {
                    "bsonType": "object",
                    "required": ["memory_id", "content", "category"],
                    "properties": {
                        "memory_id": {"bsonType": "string"},
                        "content": {"bsonType": "string"},
                        "category": {
                            "enum": ["fact", "preference", "objection", "outcome", "context", "qualification"]
                        },
                        "importance_score": {
                            "bsonType": "double",
                            "minimum": 0.0,
                            "maximum": 10.0
                        },
                        "access_count": {
                            "bsonType": "int",
                            "minimum": 0
                        },
                        "last_accessed": {"bsonType": "date"},
                        "created_at": {"bsonType": "date"},
                        "updated_at": {"bsonType": "date"},
                        "metadata": {"bsonType": "object"}
                    }
                }
            },
            "memory_count": {
                "bsonType": "int",
                "minimum": 0,
                "description": "Total number of memories in snapshot"
            },
            "category_breakdown": {
                "bsonType": "object",
                "description": "Count of memories by category",
                "properties": {
                    "fact": {"bsonType": "int", "minimum": 0},
                    "preference": {"bsonType": "int", "minimum": 0},
                    "objection": {"bsonType": "int", "minimum": 0},
                    "outcome": {"bsonType": "int", "minimum": 0},
                    "context": {"bsonType": "int", "minimum": 0},
                    "qualification": {"bsonType": "int", "minimum": 0}
                }
            },
            "sync_metadata": {
                "bsonType": "object",
                "description": "Synchronization metadata",
                "properties": {
                    "last_sync": {"bsonType": "date"},
                    "sync_source": {"bsonType": "string"},
                    "sync_duration_ms": {"bsonType": "int"},
                    "errors": {
                        "bsonType": "array",
                        "items": {"bsonType": "string"}
                    }
                }
            },
            "created_at": {
                "bsonType": "date",
                "description": "When the snapshot record was created"
            },
            "updated_at": {
                "bsonType": "date",
                "description": "When the snapshot record was last updated"
            }
        }
    }


class MemorySnapshotRepository(BaseRepository):
    """Repository for memory snapshot operations."""
    
    def __init__(self, database: Optional[Database] = None):
        """Initialize memory snapshot repository.
        
        Args:
            database: Optional MongoDB database instance
        """
        super().__init__(collection_name="memory_snapshots", database=database)
        self._ensure_indexes()
    
    def _ensure_indexes(self) -> None:
        """Ensure required indexes exist."""
        try:
            # Index on user_id and snapshot_timestamp for queries
            self.collection.create_index(
                [("user_id", 1), ("snapshot_timestamp", -1)],
                name="user_snapshot_idx"
            )
            
            # Index on status for finding pending snapshots
            self.collection.create_index(
                [("status", 1)],
                name="status_idx"
            )
            
            # TTL index to auto-delete old snapshots after 30 days
            self.collection.create_index(
                [("created_at", 1)],
                name="ttl_idx",
                expireAfterSeconds=30 * 24 * 60 * 60  # 30 days
            )
            
            logger.info("Memory snapshot indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating memory snapshot indexes: {e}")
    
    def create_snapshot(
        self,
        user_id: str,
        memories: List[Dict[str, Any]],
        status: SnapshotStatus = SnapshotStatus.COMPLETED
    ) -> str:
        """Create a new memory snapshot.
        
        Args:
            user_id: User identifier
            memories: List of memory entries
            status: Snapshot status
            
        Returns:
            str: Snapshot ID
        """
        # Calculate category breakdown
        category_breakdown = {
            "fact": 0,
            "preference": 0,
            "objection": 0,
            "outcome": 0,
            "context": 0,
            "qualification": 0
        }
        
        for memory in memories:
            category = memory.get("category", "context")
            if category in category_breakdown:
                category_breakdown[category] += 1
        
        snapshot_doc = {
            "user_id": user_id,
            "snapshot_timestamp": datetime.now(timezone.utc),
            "status": status.value,
            "memories": memories,
            "memory_count": len(memories),
            "category_breakdown": category_breakdown,
            "sync_metadata": {
                "last_sync": datetime.now(timezone.utc),
                "sync_source": "mem0",
                "sync_duration_ms": 0,
                "errors": []
            },
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        
        result = self.create_one(snapshot_doc)
        return str(result.inserted_id)
    
    def get_latest_snapshot(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest snapshot for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Optional[Dict]: Latest snapshot or None
        """
        return self.find_one(
            filter={"user_id": user_id, "status": SnapshotStatus.COMPLETED.value},
            sort=[("snapshot_timestamp", -1)]
        )
    
    def get_snapshots_by_date_range(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get snapshots within a date range.
        
        Args:
            user_id: User identifier
            start_date: Start date
            end_date: End date
            
        Returns:
            List[Dict]: List of snapshots
        """
        return self.find_many(
            filter={
                "user_id": user_id,
                "snapshot_timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            },
            sort=[("snapshot_timestamp", -1)]
        )
    
    def update_snapshot_status(
        self,
        snapshot_id: str,
        status: SnapshotStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """Update snapshot status.
        
        Args:
            snapshot_id: Snapshot ID
            status: New status
            error_message: Optional error message
            
        Returns:
            bool: True if updated
        """
        update_doc = {
            "$set": {
                "status": status.value,
                "updated_at": datetime.now(timezone.utc)
            }
        }
        
        if error_message and status == SnapshotStatus.FAILED:
            update_doc["$push"] = {"sync_metadata.errors": error_message}
        
        result = self.update_by_id(snapshot_id, update_doc)
        return result.modified_count > 0
    
    def get_memory_changes(
        self,
        user_id: str,
        since_snapshot_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get memory changes since a specific snapshot.
        
        Args:
            user_id: User identifier
            since_snapshot_id: Reference snapshot ID
            
        Returns:
            Dict containing changes analysis
        """
        # Get reference snapshot
        if since_snapshot_id:
            reference = self.find_by_id(since_snapshot_id)
        else:
            reference = self.get_latest_snapshot(user_id)
        
        if not reference:
            return {"error": "No reference snapshot found"}
        
        # Get latest snapshot
        latest = self.get_latest_snapshot(user_id)
        if not latest or latest["_id"] == reference["_id"]:
            return {"changes": False, "message": "No new snapshot available"}
        
        # Analyze changes
        ref_memories = {m["memory_id"]: m for m in reference.get("memories", [])}
        latest_memories = {m["memory_id"]: m for m in latest.get("memories", [])}
        
        added = [m for mid, m in latest_memories.items() if mid not in ref_memories]
        removed = [m for mid, m in ref_memories.items() if mid not in latest_memories]
        modified = []
        
        for mid, latest_mem in latest_memories.items():
            if mid in ref_memories:
                ref_mem = ref_memories[mid]
                if (latest_mem.get("content") != ref_mem.get("content") or
                    latest_mem.get("importance_score") != ref_mem.get("importance_score")):
                    modified.append({
                        "memory_id": mid,
                        "old": ref_mem,
                        "new": latest_mem
                    })
        
        return {
            "changes": True,
            "reference_snapshot": str(reference["_id"]),
            "latest_snapshot": str(latest["_id"]),
            "added_count": len(added),
            "removed_count": len(removed),
            "modified_count": len(modified),
            "added": added[:10],  # Limit to first 10
            "removed": removed[:10],
            "modified": modified[:10]
        }


def create_memory_snapshot_collection(database: Database) -> Collection:
    """Create memory snapshot collection with validation.
    
    Args:
        database: MongoDB database instance
        
    Returns:
        Collection: Created collection
    """
    try:
        # Create collection with validation
        database.create_collection(
            "memory_snapshots",
            validator={"$jsonSchema": get_memory_snapshot_schema()}
        )
        logger.info("Memory snapshot collection created with validation")
    except Exception as e:
        if "already exists" in str(e):
            logger.info("Memory snapshot collection already exists")
        else:
            logger.error(f"Error creating memory snapshot collection: {e}")
    
    return database["memory_snapshots"]