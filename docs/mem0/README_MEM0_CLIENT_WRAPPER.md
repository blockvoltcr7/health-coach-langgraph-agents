# Mem0 Async Client Wrapper Documentation

## Overview

The **Mem0 Async Client Wrapper** is a comprehensive, reusable wrapper around the Mem0 AsyncMemoryClient that provides enhanced functionality, error handling, and logging for use across multiple APIs and services in your application.

## Features

- ✅ **Enhanced Error Handling**: Comprehensive error handling with detailed logging
- ✅ **Retry Mechanisms**: Automatic retry with exponential backoff for failed operations
- ✅ **Data Validation**: Pydantic models for request/response validation
- ✅ **Connection Management**: Proper initialization and connection lifecycle management
- ✅ **Standardized Response Formats**: Consistent response structures across all operations
- ✅ **Global Client Instance**: Singleton pattern for efficient resource usage
- ✅ **Convenience Functions**: Helper functions for common operations
- ✅ **Health Monitoring**: Built-in health check functionality
- ✅ **Memory Operations**: Complete CRUD operations for memory management

## Installation

Ensure you have the required dependencies:

```bash
# Install mem0 package
uv add mem0ai

# Install other dependencies (already in requirements.txt)
uv add pydantic python-dotenv
```

## Quick Start

### 1. Environment Setup

Set your Mem0 API key:

```bash
export MEM0_API_KEY="your_mem0_api_key_here"
```

### 2. Basic Usage

```python
import asyncio
from app.mem0.mem0AsyncClient import Mem0AsyncClientWrapper, MemoryConfig

async def basic_example():
    # Create configuration
    config = MemoryConfig(
        api_key="your_api_key",  # Or use environment variable
        output_format="v1.1",
        max_retries=3,
        timeout=30
    )
    
    # Create client wrapper
    client = Mem0AsyncClientWrapper(config)
    
    # Add a memory
    result = await client.add_memory(
        messages=[
            {"role": "user", "content": "I prefer morning workouts"},
            {"role": "assistant", "content": "I'll remember you prefer morning workouts"}
        ],
        user_id="user_123",
        metadata={"source": "chat"}
    )
    
    # Search memories
    search_result = await client.search_memories(
        query="workout preferences",
        user_id="user_123",
        limit=10
    )
    
    # Get all memories
    all_memories = await client.get_all_memories("user_123")

# Run the example
asyncio.run(basic_example())
```

### 3. Using Global Client (Recommended)

```python
from app.mem0.mem0AsyncClient import get_mem0_client, shutdown_mem0_client

async def global_client_example():
    # Get the global client instance (reusable across your app)
    client = await get_mem0_client()
    
    # Use the client for operations
    await client.add_memory(messages, user_id)
    
    # The client is automatically managed and reused
    # Clean up when your app shuts down
    await shutdown_mem0_client()
```

### 4. Convenience Functions

```python
from app.mem0.mem0AsyncClient import (
    add_conversation_memory,
    search_user_memories,
    get_user_memory_context
)

# Add a conversation easily
await add_conversation_memory(
    user_message="Hello!",
    assistant_message="Hi there!",
    user_id="user_123"
)

# Search memories
results = await search_user_memories("hello", "user_123", 5)

# Get formatted context
context = await get_user_memory_context("user_123", limit=20)
```

## API Reference

### Core Classes

#### `MemoryConfig`

Configuration class for the Mem0 client.

```python
class MemoryConfig(BaseModel):
    api_key: Optional[str] = None  # Auto-loads from MEM0_API_KEY
    output_format: str = "v1.1"
    max_retries: int = 3
    timeout: int = 30
```

#### `MemoryEntry`

Pydantic model representing a memory entry.

```python
class MemoryEntry(BaseModel):
    id: Optional[str] = None
    memory: str
    user_id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
```

#### `MemorySearchResult`

Pydantic model for search results.

```python
class MemorySearchResult(BaseModel):
    memories: List[MemoryEntry] = Field(default_factory=list)
    total_count: int = 0
    query: str = ""
```

### Core Methods

#### `Mem0AsyncClientWrapper`

Main wrapper class with comprehensive memory operations.

##### `add_memory(messages, user_id, metadata=None)`

Add a new memory entry.

**Parameters:**
- `messages`: List of message dictionaries with 'role' and 'content' keys
- `user_id`: User identifier for the memory
- `metadata`: Optional additional metadata

**Returns:** Dict with API response

**Example:**
```python
result = await client.add_memory(
    messages=[
        {"role": "user", "content": "I like Python"},
        {"role": "assistant", "content": "Noted!"}
    ],
    user_id="user_123",
    metadata={"source": "chat"}
)
```

##### `search_memories(query, user_id, limit=10)`

Search memories for a specific user.

**Parameters:**
- `query`: Search query string
- `user_id`: User identifier
- `limit`: Maximum number of results

**Returns:** `MemorySearchResult` object

##### `get_all_memories(user_id)`

Retrieve all memories for a user.

**Parameters:**
- `user_id`: User identifier

**Returns:** List of `MemoryEntry` objects

##### `update_memory(memory_id, data, user_id)`

Update an existing memory entry.

##### `delete_memory(memory_id, user_id)`

Delete a specific memory entry.

##### `delete_all_memories(user_id)`

Delete all memories for a user.

##### `get_memory_history(user_id, limit=50)`

Get chronologically ordered memory history.

##### `health_check()`

Perform a health check of the Mem0 service.

### Global Functions

#### `get_mem0_client(config=None)`

Get the global Mem0 client instance.

#### `shutdown_mem0_client()`

Shutdown the global Mem0 client.

### Convenience Functions

#### `add_conversation_memory(user_message, assistant_message, user_id, metadata=None)`

Add a conversation to memory easily.

#### `search_user_memories(query, user_id, limit=10)`

Search user memories with convenience function.

#### `get_user_memory_context(user_id, limit=20)`

Get formatted memory context string.

## API Endpoints

The wrapper is integrated with FastAPI endpoints for REST API access:

### Memory Management Endpoints

- `POST /api/v1/memory/add` - Add a memory
- `POST /api/v1/memory/add-conversation` - Add a conversation
- `POST /api/v1/memory/search` - Search memories
- `GET /api/v1/memory/all/{user_id}` - Get all user memories
- `GET /api/v1/memory/context/{user_id}` - Get memory context
- `PUT /api/v1/memory/update` - Update a memory
- `DELETE /api/v1/memory/delete` - Delete a memory
- `DELETE /api/v1/memory/delete-all/{user_id}` - Delete all user memories
- `GET /api/v1/memory/health` - Health check

### Example API Usage

```bash
# Add a memory
curl -X POST "http://localhost:8000/api/v1/memory/add" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I prefer morning workouts"},
      {"role": "assistant", "content": "Noted!"}
    ],
    "user_id": "user_123",
    "metadata": {"source": "api"}
  }'

# Search memories
curl -X POST "http://localhost:8000/api/v1/memory/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "workout",
    "user_id": "user_123",
    "limit": 5
  }'

# Get all memories
curl "http://localhost:8000/api/v1/memory/all/user_123"

# Health check
curl "http://localhost:8000/api/v1/memory/health"
```

## Error Handling

The wrapper provides comprehensive error handling:

### Validation Errors

```python
try:
    await client.add_memory([], "user_123")  # Empty messages
except ValueError as e:
    print(f"Validation error: {e}")
```

### Retry Mechanism

The wrapper automatically retries failed operations with exponential backoff:

```python
# Configure retry behavior
config = MemoryConfig(
    api_key="your_key",
    max_retries=5,  # Will retry up to 5 times
    timeout=30
)
```

### Service Health Monitoring

```python
# Check service health
health_status = await client.health_check()
if health_status["status"] == "healthy":
    print("Service is operational")
else:
    print(f"Service issue: {health_status['error']}")
```

## Integration Examples

### Sales Agent Integration

```python
from app.mem0.mem0AsyncClient import get_mem0_client, add_conversation_memory

async def sales_agent_workflow(user_id: str, user_message: str, agent_response: str):
    # Add conversation to memory
    await add_conversation_memory(
        user_message=user_message,
        assistant_message=agent_response,
        user_id=user_id,
        metadata={
            "stage": "qualification",
            "intent": "health_coaching"
        }
    )
    
    # Search for customer objections
    client = await get_mem0_client()
    objections = await client.search_memories(
        query="concerned cost price",
        user_id=user_id,
        limit=5
    )
    
    # Get complete customer context
    context = await get_user_memory_context(user_id, limit=20)
    return context
```

### Multi-Agent System Integration

```python
async def multi_agent_memory_sharing(user_id: str):
    client = await get_mem0_client()
    
    # Agent 1: Qualification agent adds memory
    await client.add_memory(
        messages=[
            {"role": "user", "content": "I have a $5000 budget"},
            {"role": "qualification_agent", "content": "Budget qualified: $5000"}
        ],
        user_id=user_id,
        metadata={"agent": "qualification", "stage": "budget_check"}
    )
    
    # Agent 2: Closer agent retrieves context
    memories = await client.get_all_memories(user_id)
    budget_info = [m for m in memories if "budget" in m.memory.lower()]
    
    return budget_info
```

## Testing

The wrapper includes comprehensive tests:

```bash
# Run all memory client tests
pytest tests/mem0/test_mem0_async_client_wrapper.py -v

# Run API endpoint tests
pytest tests/endpoints/test_memory_endpoints.py -v

# Run with Allure reporting
pytest tests/mem0/ --alluredir=allure-results
allure serve allure-results
```

## Best Practices

### 1. Use Global Client for Production

```python
# ✅ Good: Use global client for efficiency
client = await get_mem0_client()

# ❌ Avoid: Creating new instances repeatedly
client = Mem0AsyncClientWrapper(config)  # Don't do this in production
```

### 2. Handle Errors Gracefully

```python
try:
    result = await client.add_memory(messages, user_id)
except ValueError as e:
    # Handle validation errors
    logger.warning(f"Invalid input: {e}")
except Exception as e:
    # Handle service errors
    logger.error(f"Service error: {e}")
```

### 3. Use Metadata for Context

```python
await client.add_memory(
    messages=messages,
    user_id=user_id,
    metadata={
        "agent_type": "sales_qualifier",
        "conversation_stage": "objection_handling",
        "timestamp": datetime.now().isoformat(),
        "source": "web_chat"
    }
)
```

### 4. Clean Up in Tests

```python
# Always clean up test data
await client.delete_all_memories("test_user_123")
```

## Configuration

### Environment Variables

```bash
# Required
MEM0_API_KEY=your_mem0_api_key

# Optional (with defaults)
MEM0_OUTPUT_FORMAT=v1.1
MEM0_MAX_RETRIES=3
MEM0_TIMEOUT=30
```

### Application Configuration

```python
# Custom configuration
config = MemoryConfig(
    api_key="your_key",
    output_format="v1.1",
    max_retries=5,
    timeout=60
)

client = Mem0AsyncClientWrapper(config)
```

## Performance Considerations

1. **Use Global Client**: Reuse the global client instance across your application
2. **Batch Operations**: When possible, batch multiple memory operations
3. **Limit Results**: Use appropriate limits for search and retrieval operations
4. **Monitor Health**: Regularly check service health in production
5. **Handle Retries**: Configure retry settings based on your use case

## Troubleshooting

### Common Issues

#### 1. Import Errors

```python
# If you get import errors, ensure mem0 is installed
# uv add mem0ai
```

#### 2. API Key Issues

```python
# Verify your API key is set
import os
print(os.getenv("MEM0_API_KEY"))
```

#### 3. Connection Issues

```python
# Check service health
health = await client.health_check()
print(health)
```

#### 4. Validation Errors

```python
# Ensure proper message format
messages = [
    {"role": "user", "content": "message"},      # ✅ Correct
    {"role": "assistant", "content": "response"} # ✅ Correct
]

# Not this:
# {"message": "content"}  # ❌ Wrong format
```

## Contributing

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Use proper error handling and logging
5. Follow Pydantic V2 best practices

## License

This wrapper is part of the health-coach-langgraph-agents project and follows the same license terms. 