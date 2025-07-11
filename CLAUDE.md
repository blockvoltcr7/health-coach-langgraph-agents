# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Health Coach AI Platform built with FastAPI, LangGraph, and Mem0. It provides multi-agent AI workflows for personalized health coaching, featuring persistent memory management, vector search capabilities, and integration with multiple AI providers (OpenAI, Google Gemini, ElevenLabs). The platform uses MongoDB Atlas for document storage and vector search to enable RAG applications.

## Essential Commands

### Development Server
```bash
# Start development server (recommended)
uv run uvicorn app.main:app --reload

# Alternative with activated venv
source .venv/bin/activate && uvicorn app.main:app --reload
```

### Testing
```bash
# Run all tests with Allure reporting
uv run pytest --alluredir=allure-results -v

# Run specific test file
uv run pytest tests/test_hello.py -v

# Run LangGraph agent tests
uv run pytest tests/langgraph/test_langgraph_smoke.py -v

# Run Mem0 memory tests
uv run pytest tests/mem0/test_mem0_memory_add_search.py -v

# Run MongoDB vector search tests
uv run pytest tests/db-tests/test_mongo_vector_search.py -v
uv run pytest tests/db-tests/test_mongo_medspa_data.py -v

# Run tests by marker
uv run pytest -m api -v
uv run pytest -m integration -v
uv run pytest -m smoke -v

# Generate and serve Allure report
allure serve allure-results
```

### MongoDB Utilities
```bash
# View MedSpa data in MongoDB
uv run python tests/db-tests/view_medspa_data.py

# Search MedSpa data interactively
uv run python tests/db-tests/search_medspa_demo.py

# Launch MongoDB Vector Search Gradio App
./gradio/launch.sh
# Or directly:
uv run python gradio/mongodb_vector_search_app.py
```

### LangGraph Demos
```bash
# Run interactive LangGraph agents demo
uv run python tests/langgraph/demo_langgraph_agents.py

# Run LangGraph tests standalone
cd tests/langgraph && uv run python test_langgraph_smoke.py
```

### Mem0 Memory Operations
```bash
# Run Mem0 memory add and search demo
uv run python tests/mem0/test_mem0_memory_add_search.py

# Ensure environment variable is set
export MEM0_API_KEY="your-mem0-api-key"
```

### Dependency Management
Always use `uv` for package management, never `pip`:
```bash
# Install dependencies
uv sync

# Add new dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Update dependencies
uv sync --upgrade
```

## Architecture

### Core Structure
- **app/api/v1/**: Versioned API endpoints
- **app/core/**: Configuration and security
- **app/models/**: Pydantic data models
- **app/services/**: Business logic and AI integrations
- **app/agents/**: CrewAI agent implementations
- **app/tools/**: AI tools and utilities
- **tests/**: Comprehensive test suite with Allure reporting
  - **tests/langgraph/**: LangGraph multi-agent workflow tests and demos
  - **tests/mem0/**: Mem0 memory management tests
  - **tests/db-tests/**: MongoDB integration tests and utilities
  - **tests/test-data/**: Test data files (e.g., med-spa-test-data.md)
  - **tests/ai-tests/**: AI service integration tests
- **gradio/**: Gradio web applications
  - **mongodb_vector_search_app.py**: Interactive MongoDB vector search interface
- **ai-specs/**: AI implementation specifications for LangGraph and Mem0
- **docs/langgraph/**: Detailed LangGraph implementation documentation

### Key Integrations
- **LangGraph**: State-based multi-agent workflow orchestration (v0.2.70)
- **Mem0**: Async memory management for personalized AI interactions
- **OpenAI**: GPT models and image generation (text-embedding-ada-002)
- **Google Gemini**: Text-to-speech and language models
- **CrewAI**: Multi-agent AI orchestration
- **ElevenLabs**: Voice synthesis
- **ChromaDB/LanceDB**: Vector storage
- **MongoDB Atlas**: Document storage with vector search capabilities
- **Voyage AI**: Advanced embedding models (voyage-3-large)

## Development Guidelines

### Dependency Management
- Always use `uv` commands, never `pip`
- Update `pyproject.toml` when adding dependencies
- Commit both `pyproject.toml` and `uv.lock`

### Testing Requirements
- All new features must have corresponding tests
- Use pytest with Allure annotations:
  ```python
  @allure.epic("Core Functionality")
  @allure.feature("Feature Name")
  class TestFeatureName:
      @allure.story("Test Scenario")
      @allure.severity(allure.severity_level.CRITICAL)
      def test_something(self):
          with allure.step("Step description"):
              # test code
  ```
- Use appropriate markers: `@pytest.mark.api`, `@pytest.mark.integration`, `@pytest.mark.slow`

### Pydantic V2 Best Practices
- Use `Optional[bool]` instead of `bool` for API fields that might return null
- Use `model_validate_json()` for JSON validation
- Add logging with `allure.attach()` for debugging API responses
- Use `Field(default_factory=list)` for default collections
- Implement custom validation with `@field_validator` decorator

### AI Service Integration
- Store API keys in environment variables
- Add proper error handling for external API calls
- Include response logging for debugging
- Test both success and error scenarios

## API Endpoints

Main endpoints:
- `GET /health` - Health check
- `GET /api/v1/hello` - Basic endpoint
- `POST /api/v1/crewai` - CrewAI agent execution
- `POST /api/v1/gemini/podcast` - Multi-speaker TTS generation
- `POST /api/v1/auth` - Authentication
- `GET /api/v1/users` - User management

## Environment Setup

Required for AI features:
```bash
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
VOYAGE_AI_API_KEY=your_key_here  # Optional, falls back to OpenAI
MONGO_DB_PASSWORD=your_password_here  # For MongoDB tests
MEM0_API_KEY=your_key_here  # For Mem0 memory operations
```

## MongoDB Vector Search

### Overview
The project includes MongoDB Atlas vector search implementation for building RAG applications. It supports both Voyage AI and OpenAI embeddings with automatic fallback. For detailed architecture, see `RAG_CHAT_ARCHITECTURE.md`.

### Key Components
- **EmbeddingProvider**: Flexible class supporting multiple embedding providers
- **Vector Search Tests**: Comprehensive test suite in `tests/db-tests/`
- **MedSpa Demo**: Example implementation with real-world data
- **Gradio Interface**: Full-featured web UI with RAG chat in `gradio/`

### Usage
```python
# Example embedding generation
embedding_provider = EmbeddingProvider()
embeddings = embedding_provider.embed_documents(["text to embed"])
query_embedding = embedding_provider.embed_query("search query")

# MongoDB vector search pipeline
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 100,
            "limit": 10
        }
    }
]
```

### Atlas Configuration
Create a vector search index in MongoDB Atlas:
```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "type": "knnVector",
        "dimensions": 1536,  # 1024 for Voyage AI
        "similarity": "cosine"
      }
    }
  }
}
```

## LangGraph Multi-Agent Workflows

### Overview
LangGraph enables state-based orchestration of multiple AI agents that work together sequentially, passing information through a shared state. The implementation follows a graph-based execution model where agents are nodes and state transitions are edges.

### Key Components
- **AgentState**: TypedDict defining the shared state structure
- **StateGraph**: Manages workflow structure and execution
- **Agent Nodes**: Individual AI agents that process and update state
- **Sequential Flow**: `START → Agent 1 → Agent 2 → END`

### Usage Pattern
```python
# Define state structure
class AgentState(TypedDict):
    messages: list[str]
    current_message: str
    agent_1_response: str
    agent_2_response: str
    step_count: int

# Build workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent_1", agent_1_function)
workflow.add_node("agent_2", agent_2_function)
workflow.add_edge(START, "agent_1")
workflow.add_edge("agent_1", "agent_2")
workflow.add_edge("agent_2", END)
```

### Health Coaching Implementation
The project includes specifications for a weekly health analysis system that:
1. Retrieves user memories from Mem0
2. Processes through specialized agents:
   - Risk Analysis Agent
   - Health Progress Agent
   - Workout Summary Agent
   - Final Consolidation Agent

## Mem0 Memory Management

### Overview
Mem0 provides persistent memory storage for AI conversations, enabling personalized interactions across sessions. The integration uses the async client for efficient memory operations.

### Key Features
- **User-Scoped Memory**: Memories tied to specific user IDs
- **Semantic Search**: Query memories by content similarity
- **Conversation History**: Store user/assistant interactions
- **v1.1 Output Format**: Structured response format

### Usage Pattern
```python
from mem0 import MemoryClient

# Initialize async client
memory = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))

# Add memory
await memory.add(
    messages=[
        {"role": "user", "content": "I prefer morning workouts"},
        {"role": "assistant", "content": "Noted! I'll suggest morning workout routines"}
    ],
    user_id="user123",
    output_format="v1.1"
)

# Search memories
results = await memory.search(
    query="workout preferences",
    user_id="user123",
    limit=10
)
```

## Deployment

The project supports multiple deployment platforms:
- **Render**: Uses `render.yaml` configuration
- **Railway**: Uses `railway.json` configuration
- **Docker**: Multi-stage builds with UV support

Always run tests before deployment:
```bash
uv run pytest && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Important Reminders

- Always use `uv` for package management, never `pip`
- Run tests before committing changes
- Use Allure annotations for comprehensive test reporting
- Set all required environment variables before running AI features
- Follow existing code patterns and conventions
- Check `docs/langgraph/README_LANGGRAPH_AGENTS.md` for detailed LangGraph examples
- Review `ai-specs/` for implementation specifications
