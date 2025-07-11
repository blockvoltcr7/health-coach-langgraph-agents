# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Project**: Health Coach LangGraph Agents (Currently implementing Sales AI Closer)
**Type**: FastAPI-based multi-agent AI platform using LangGraph
**Status**: Production-ready sales agent with active multi-agent routing development

## Essential Commands

### Development & Testing

```bash
# Start development server
uv sync                                    # Install/sync dependencies
uv run uvicorn app.main:app --reload      # Run dev server (default port 8000)
uv run uvicorn app.main:app --reload --port 8080  # Run on specific port

# Run tests
./test_runner.sh all                      # Run all tests
./test_runner.sh file tests/endpoints/test_fastapi_endpoints.py  # Run specific file
./test_runner.sh group "API Endpoints"    # Run test group
./test_runner.sh list-files              # List available test files
./test_runner.sh list-groups             # List test groups

# Test options
-e <env>    # Environment (dev/uat/prod) [default: dev]
-s, --skip  # Skip opening Allure report
-q, --quiet # Run with minimal output
-k <expr>   # Only run tests matching expression
```

### Task Management (Task Master)

```bash
# Daily workflow
task-master next                          # Get next available task
task-master show <id>                     # View task details
task-master set-status --id=<id> --status=done  # Mark task complete

# Task updates
task-master update-subtask --id=<id> --prompt="implementation notes"  # Log progress
task-master expand --id=<id> --research   # Break task into subtasks

# Research & context
task-master research --query="latest React Query v5 practices" --id=<task-ids>  # Get fresh info
```

### Database Operations (MongoDB)

```python
# MongoDB is available at app/db/mongodb/
from app.db.mongodb.client import get_mongodb_client, get_database
from app.db.mongodb.schemas.conversation_schema import ConversationRepository

# Initialize database
from app.db.mongodb.utils import initialize_database
initialize_database()  # Creates collections with schema validation
```

## High-Level Architecture

### Core Components

1. **FastAPI Application** (`app/main.py`)
   - Modular API structure with versioning
   - Health checks and monitoring endpoints
   - CORS and middleware configuration

2. **LangGraph Agent** (`app/core/chatbot_base.py`)
   - Supervisor pattern with tool delegation
   - State machine for conversation flow
   - Persistent memory via mem0
   - Web search and datetime tools

3. **MongoDB Integration** (`app/db/mongodb/`)
   - Singleton connection pattern
   - Repository pattern for data access
   - Conversation schema with embedded documents
   - Performance-optimized indexes

4. **Memory System** (`app/mem0/`)
   - Async wrapper for mem0 operations
   - Complete conversation history retrieval
   - Cross-session memory persistence

### Agent Architecture

**Current Implementation**: Limitless OS Sales Agent

Key features:
- **Memory**: Full conversation history with `get_all_memories()`
- **Tools**: Web search (Tavily), datetime awareness
- **State Management**: LangGraph state machine with message history
- **Sales Process**: BANT qualification, objection handling, stage tracking

### MongoDB Schema (Conversations Collection)

```javascript
{
  user_id: ObjectId,
  channel: "web" | "mobile" | "api",
  status: "active" | "inactive" | "closed",
  sales_stage: "lead" | "qualified" | "proposal" | "negotiation" | "closed",
  stage_history: [{stage, timestamp, notes}],
  qualification: {budget, authority, need, timeline, score, notes},
  messages: [{role, content, timestamp}],
  objections: [{objection, response, resolved, timestamp}],
  handoffs: [{from_agent, to_agent, reason, timestamp}],
  metadata: {},  // Stores mem0_user_id and other flexible data
  created_at: Date,
  updated_at: Date
}
```

## API Endpoints

### Active Endpoints

- `GET /` - API documentation
- `GET /health` - Health check
- `POST /api/v1/chatbot/chat/full-memory` - Main sales agent chat
- Memory CRUD endpoints at `/api/v1/memory/`

### Request/Response Format

```python
# Chat request
{
  "user_id": "string",
  "session_id": "string", 
  "input": "message text"
}

# Chat response
{
  "output": "agent response",
  "session_id": "string"
}
```

## Environment Configuration

### Required API Keys (.env)

```bash
# LLM Providers (at least one required)
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key  # For Claude models

# Memory & Tools
MEM0_API_KEY=your_key
TAVILY_API_KEY=your_key  # Web search

# MongoDB
MONGODB_USERNAME=your_username
MONGODB_PASSWORD=your_password

# Monitoring (optional)
LANGSMITH_API_KEY=your_key
LANGSMITH_TRACING=true
```

## Development Patterns

### Adding New Features

1. **New Endpoints**: Add to `app/api/v1/endpoints/`
2. **Schemas**: Define in `app/api/v1/schemas/`
3. **Business Logic**: Implement in `app/services/`
4. **Database Models**: Use repository pattern in `app/db/mongodb/`

### Testing Strategy

- **Unit Tests**: Test individual functions
- **Integration Tests**: Test API endpoints
- **Test Categories**: `@pytest.mark.smoke`, `@pytest.mark.integration`
- **Environment-based**: Use `-e uat` for UAT environment

### Working with MongoDB

```python
# Get repository
repo = ConversationRepository()

# Create conversation
conversation_id = repo.create_one(doc).inserted_id

# Add message
repo.add_message(conversation_id, "supervisor", "message")

# Update sales stage
repo.update_sales_stage(conversation_id, "qualified", "BANT met")
```

## Deployment

### Docker
```bash
docker build -t health-coach-agents .
docker run -p 8000:8000 health-coach-agents
```

### Platform Configs
- **Railway**: `railway.json`
- **Render**: `render.yaml`
- **Local**: Use `uv run`

## Current Development Focus

1. **Multi-Agent Routing**: Implementing supervisor → qualifier agent routing
2. **MongoDB Integration**: Conversation persistence and analytics
3. **Performance**: Query optimization and indexing
4. **Testing**: Expanding integration test coverage

## Important Notes

- **Task Master Integration**: Use `task-master` commands for project management
- **Memory Persistence**: All conversations stored in both mem0 and MongoDB
- **Agent Names**: supervisor, qualifier, objection_handler, closer
- **Message Roles**: user, assistant, or specific agent names
- **Metadata Field**: Use for mem0_user_id and extensibility

## Quick Debugging

```bash
# Check MongoDB connection
python -c "from app.db.mongodb.client import get_mongodb_client; print(get_mongodb_client().health_check())"

# View current task
task-master next

# Check API health
curl http://localhost:8000/health

# View logs with context
uv run uvicorn app.main:app --log-level debug
```

---

**Remember**: The project is transitioning from health coach to sales agent focus. Current implementation is the Limitless OS Sales Agent with full conversation persistence and multi-agent capabilities.