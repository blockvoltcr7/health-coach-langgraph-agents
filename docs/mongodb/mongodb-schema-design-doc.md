# MongoDB Schema Design Document for Limitless OS Multi-Agent Sales System

## 1. Executive Summary

This document outlines the MongoDB schema design and indexing strategy for the Limitless OS Multi-Agent Sales System. The system manages sales conversations across multiple channels (primarily Instagram DMs) using an intelligent agent orchestration model with specialized agents for qualification, objection handling, and closing deals.

## 2. Database Architecture

### 2.1 Database Structure
- **Production Database**: `limitless_os_sales`
  - Primary collection: `conversations`
- **Development Database**: `limitless_os_sales_dev`
  - Test collection: `conversations_test`

### 2.2 Document Schema

```javascript
{
  "_id": ObjectId,
  
  // User Identification
  "user_id": String,           // Instagram handle or unique identifier
  "user_info": {
    "name": String,
    "email": String,
    "instagram_handle": String,
    "created_at": ISODate,
    "last_interaction": ISODate
  },
  
  // Sales Process State
  "status": String,            // "active", "inactive", "completed", "abandoned"
  "sales_stage": String,       // "qualification", "objection_handling", "closing", "not_qualified"
  "is_qualified": Boolean,     // null during assessment
  "current_agent": String,     // "supervisor", "qualifier", "objection_handler", "closer"
  
  // Conversation Management
  "channel": String,           // "instagram_dm", "facebook_messenger", etc.
  "conversation_history": [{
    "role": String,          // "user" or "agent"
    "content": String,
    "timestamp": ISODate,
    "agent_type": String     // Which agent responded
  }],
  
  // Memory and Context (Mem0 Integration)
  "memories": [{
    "key": String,           // "budget", "need", "objection"
    "value": Mixed,
    "timestamp": ISODate,
    "extracted_by": String   // Which agent extracted this
  }],
  
  // Sales Intelligence
  "qualification_details": {
    "budget": String,
    "need": String,
    "authority": String,
    "timeline": String,
    "score": Number          // Qualification score
  },
  
  "objections": [{
    "type": String,          // "price", "timing", "trust"
    "description": String,
    "raised_at": ISODate,
    "resolved": Boolean,
    "resolution": String
  }],
  
  // Follow-up Management
  "follow_up": {
    "required": Boolean,
    "scheduled_date": ISODate,
    "reason": String,
    "notes": String,
    "reminder_sent": Boolean
  },
  
  // Metadata and Analytics
  "metadata": {
    "source": String,
    "campaign": String,
    "tags": [String],
    "lead_score": Number,
    "conversion_probability": Number
  },
  
  // Timestamps
  "created_at": ISODate,
  "updated_at": ISODate,
  "completed_at": ISODate
}
```

## 3. Indexing Strategy

### 3.1 Index Overview

We've implemented 7 strategic indexes (including the default `_id`) to optimize query performance for the multi-agent sales workflow:

### 3.2 Index Definitions

#### 1. **Primary Key Index** (`_id_`)
```javascript
{ "_id": 1 }
```
- **Purpose**: Default MongoDB index for unique document identification
- **Use Case**: Direct document lookups by ID

#### 2. **User Lookup Index** (`user_lookup_idx`)
```javascript
{ "user_id": 1, "updated_at": -1 }
```
- **Purpose**: Fast retrieval of user conversations with recent activity first
- **Use Cases**:
  - Finding all conversations for a specific user
  - Loading user's most recent conversation
  - User history retrieval for context
- **Performance**: O(log n) lookup + sorted results

#### 3. **Sales Pipeline Index** (`sales_pipeline_idx`)
```javascript
{ "sales_stage": 1, "status": 1 }
```
- **Purpose**: Efficient filtering and analytics across sales stages
- **Use Cases**:
  - Count conversations in each stage
  - Find all active conversations in "qualification"
  - Pipeline analytics and reporting
  - Agent workload distribution
- **Performance**: Supports compound queries on both fields

#### 4. **Channel Index** (`channel_idx`)
```javascript
{ "channel": 1 }
```
- **Purpose**: Quick filtering by communication channel
- **Use Cases**:
  - Retrieve all Instagram DM conversations
  - Channel-specific analytics
  - Multi-channel support scaling
- **Performance**: Direct channel filtering

#### 5. **Recent Conversations Index** (`recent_conversations_idx`)
```javascript
{ "created_at": -1 }
```
- **Purpose**: Efficient sorting by creation date (newest first)
- **Use Cases**:
  - Dashboard showing recent conversations
  - Time-based analytics
  - New lead prioritization
- **Performance**: Pre-sorted results without collection scan

#### 6. **Follow-up Scheduling Index** (`follow_up_scheduling_idx`)
```javascript
{ "follow_up.required": 1, "follow_up.scheduled_date": 1 }
```
- **Purpose**: Optimize follow-up task management
- **Use Cases**:
  - Find all conversations requiring follow-up
  - Today's follow-up tasks
  - Overdue follow-ups
  - Automated reminder systems
- **Performance**: Compound index for filtered date ranges

#### 7. **Agent Workload Index** (`agent_workload_idx`)
```javascript
{ "current_agent": 1, "status": 1, "updated_at": -1 }
```
- **Purpose**: Efficient agent task distribution and monitoring
- **Use Cases**:
  - Count active conversations per agent
  - Find oldest unhandled conversations per agent
  - Load balancing decisions
  - Agent performance metrics
- **Performance**: Three-field compound for complex workload queries

## 4. Query Optimization Examples

### 4.1 Common Query Patterns

```javascript
// 1. Get user's most recent conversation
db.conversations.find({ "user_id": "user123" })
  .sort({ "updated_at": -1 })
  .limit(1)
// Uses: user_lookup_idx

// 2. Find all active qualification stage conversations
db.conversations.find({ 
  "sales_stage": "qualification",
  "status": "active" 
})
// Uses: sales_pipeline_idx

// 3. Get today's follow-ups
db.conversations.find({
  "follow_up.required": true,
  "follow_up.scheduled_date": {
    $gte: ISODate("2024-01-15T00:00:00Z"),
    $lt: ISODate("2024-01-16T00:00:00Z")
  }
})
// Uses: follow_up_scheduling_idx

// 4. Agent workload check
db.conversations.find({
  "current_agent": "qualifier",
  "status": "active"
}).sort({ "updated_at": -1 })
// Uses: agent_workload_idx
```

## 5. Performance Characteristics

### 5.1 Index Impact
- **Without Indexes**: Collection scans O(n) for 1M documents = ~1-2 seconds
- **With Indexes**: Index lookups O(log n) for 1M documents = ~1-5 milliseconds

### 5.2 Storage Overhead
- Each index adds ~10-20% storage overhead
- Total index size for 1M documents: ~100-200MB
- Acceptable trade-off for query performance gains

### 5.3 Write Performance
- Slight overhead on inserts/updates (maintaining 6 additional indexes)
- Negligible impact: ~1-2ms additional latency
- Benefits far outweigh costs for read-heavy workload

## 6. Scaling Considerations

### 6.1 Future Index Candidates
```javascript
// Text search for conversation content
{ "conversation_history.content": "text" }

// Compound index for time-based user queries
{ "user_id": 1, "created_at": -1 }

// TTL index for automatic cleanup
{ "completed_at": 1 }, { expireAfterSeconds: 7776000 } // 90 days
```

### 6.2 Sharding Strategy
When scaling beyond single server:
- Shard key: `{ "user_id": "hashed" }`
- Ensures even distribution
- Keeps user conversations co-located

## 7. Best Practices

### 7.1 Query Guidelines
1. Always filter by indexed fields first
2. Use compound indexes efficiently (left-to-right)
3. Avoid regex queries on non-indexed fields
4. Use projections to limit returned data

### 7.2 Maintenance
1. Monitor index usage with `db.conversations.aggregate([{$indexStats:{}}])`
2. Rebuild indexes periodically if fragmented
3. Review slow query log monthly
4. Add new indexes based on actual query patterns

## 8. Integration with Application

### 8.1 Supervisor Agent Queries
```python
# Find user's active conversation
{"user_id": user_id, "status": "active"}

# Route to appropriate agent
{"_id": conversation_id, "sales_stage": 1, "current_agent": 1}
```

### 8.2 Memory Synchronization
- Mem0 memories synced to `memories` array
- Indexed user lookups ensure fast memory retrieval
- Supports both real-time and batch operations

## 9. Conclusion

This schema and indexing strategy provides:
- ✅ Sub-millisecond query performance for common operations
- ✅ Scalability to millions of conversations
- ✅ Efficient multi-agent workflow support
- ✅ Comprehensive sales process tracking
- ✅ Seamless integration with Mem0 and LangGraph

The design balances performance, scalability, and maintainability while supporting the complex requirements of an AI-driven multi-agent sales system.