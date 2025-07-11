# LangGraph Chatbot System

A comprehensive, reusable chatbot system built with LangGraph, OpenAI LLM, and optional mem0 graph memory integration. This system provides a flexible, scalable architecture for creating different types of chatbots with FastAPI endpoints.

## 🚀 Features

- **Multiple Chatbot Types**: Basic, Tool-enabled, Memory-enabled, and Advanced chatbots
- **OpenAI LLM Integration**: Configurable OpenAI models with customizable parameters
- **Mem0 Graph Memory**: Optional persistent memory with graph-based relationships
- **Tool Integration**: Web search capabilities via Tavily
- **Streaming Support**: Real-time streaming responses
- **Session Management**: Persistent conversations with automatic cleanup
- **FastAPI Integration**: Complete REST API with comprehensive documentation
- **Factory Pattern**: Easy chatbot creation with predefined configurations
- **Comprehensive Testing**: Full test suite with Allure reporting

## 📁 Project Structure

```
app/
├── core/                          # Core chatbot components
│   ├── chatbot_config.py         # Configuration management
│   ├── chatbot_base.py           # Base chatbot classes
│   ├── chatbot_factory.py        # Factory pattern implementation
│   └── __init__.py               # Core exports
├── services/                      # Business logic layer
│   ├── chatbot_service.py        # Service layer implementation
│   └── __init__.py               # Service exports
├── api/v1/                       # API layer
│   ├── schemas/
│   │   └── chatbot_schemas.py    # Pydantic schemas
│   └── endpoints/
│       └── chatbot_endpoints.py  # FastAPI endpoints
├── examples/                     # Usage examples
│   └── chatbot_example.py        # Demonstration script
└── langgraph/                    # Existing LangGraph workflows
    └── closer_ai_workflow.py     # Multi-agent sales workflow
```

## 🛠️ Installation

### Prerequisites

```bash
# Install required dependencies
uv pip add langgraph langchain-openai fastapi uvicorn

# Optional: For memory support
uv pip add mem0ai

# Optional: For web search tools
uv pip add langchain-tavily

# Optional: For testing
uv pip add pytest allure-pytest
```

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional - for memory support
MEM0_API_KEY=your_mem0_api_key

# Optional - for web search
TAVILY_API_KEY=your_tavily_api_key
```

## 🔧 Quick Start

### 1. Basic Usage

```python
from app.core import create_basic_chatbot

# Create a basic chatbot
chatbot = create_basic_chatbot("You are a helpful assistant.")

# Chat with the bot
response = await chatbot.chat("Hello! How can you help me?")
print(response)
```

### 2. Using the Service Layer

```python
from app.services import ChatbotService

# Initialize service
service = ChatbotService()

# Create a session
session_id = service.create_session(
    user_id="user123",
    chatbot_type="basic"
)

# Chat through the service
response = await service.chat(session_id, "Hello!")
print(response)
```

### 3. FastAPI Integration

```python
# The API is automatically available at /api/v1/chatbot/
# Key endpoints:
# POST /api/v1/chatbot/chat - Send a message
# POST /api/v1/chatbot/sessions - Create a session
# GET /api/v1/chatbot/sessions - List sessions
# POST /api/v1/chatbot/chat/stream - Stream responses
```

## 📖 Detailed Usage

### Chatbot Types

#### 1. Basic Chatbot
```python
from app.core import create_basic_chatbot

chatbot = create_basic_chatbot(
    system_prompt="You are a helpful assistant."
)
```

#### 2. Tool-Enabled Chatbot
```python
from app.core import create_tool_chatbot

chatbot = create_tool_chatbot(
    system_prompt="You are an assistant with web search capabilities."
)
```

#### 3. Memory-Enabled Chatbot
```python
from app.core import create_memory_chatbot

chatbot = create_memory_chatbot(
    system_prompt="You are an assistant with memory capabilities."
)
```

#### 4. Advanced Chatbot (Tools + Memory)
```python
from app.core import create_advanced_chatbot

chatbot = create_advanced_chatbot(
    system_prompt="You are an advanced assistant with tools and memory."
)
```

### Custom Configuration

```python
from app.core import ChatbotConfig, ChatbotType, LLMConfig, ToolConfig
from app.core import ChatbotFactory

# Create custom configuration
config = ChatbotConfig(
    name="Custom Support Bot",
    type=ChatbotType.WITH_TOOLS,
    llm=LLMConfig(
        model="gpt-4",
        temperature=0.3,
        max_tokens=500
    ),
    tools=ToolConfig(
        enabled=True,
        available_tools=["web_search"],
        max_search_results=3
    ),
    system_prompt="You are a customer support assistant."
)

# Create chatbot from config
chatbot = ChatbotFactory.create_chatbot(config)
```

### Session Management

```python
from app.services import ChatbotService

service = ChatbotService()

# Create sessions
session1 = service.create_session(user_id="user1", chatbot_type="basic")
session2 = service.create_session(user_id="user2", chatbot_type="advanced")

# Chat in different sessions
response1 = await service.chat(session1, "Hello from session 1")
response2 = await service.chat(session2, "Hello from session 2")

# List user sessions
sessions = service.list_sessions(user_id="user1")

# Get session info
info = service.get_session_info(session1)
```

### Streaming Responses

```python
# Direct chatbot streaming
async for chunk in chatbot.chat_stream("Tell me a story"):
    print(chunk, end="", flush=True)

# Service layer streaming
async for chunk in service.chat_stream(session_id, "Tell me a story"):
    print(chunk, end="", flush=True)
```

## 🌐 API Endpoints

### Chat Endpoints

#### POST `/api/v1/chatbot/chat`
Send a message to the chatbot.

```json
{
  "message": "Hello, how can you help me?",
  "session_id": "optional-session-id",
  "user_id": "optional-user-id",
  "metadata": {"source": "web"}
}
```

#### POST `/api/v1/chatbot/chat/stream`
Stream a chatbot response.

```json
{
  "message": "Tell me about AI",
  "session_id": "optional-session-id",
  "user_id": "optional-user-id"
}
```

### Session Management

#### POST `/api/v1/chatbot/sessions`
Create a new chat session.

```json
{
  "user_id": "user123",
  "chatbot_type": "basic",
  "metadata": {"source": "web"}
}
```

#### GET `/api/v1/chatbot/sessions`
List chat sessions (optionally filtered by user).

#### GET `/api/v1/chatbot/sessions/{session_id}`
Get information about a specific session.

#### DELETE `/api/v1/chatbot/sessions/{session_id}`
Delete a chat session.

### Configuration

#### GET `/api/v1/chatbot/chatbot-types`
Get available chatbot types.

#### POST `/api/v1/chatbot/sessions/custom`
Create a session with custom configuration.

```json
{
  "config": {
    "name": "Custom Bot",
    "llm": {
      "model": "gpt-4",
      "temperature": 0.7
    },
    "tools": {
      "enabled": true,
      "available_tools": ["web_search"]
    },
    "system_prompt": "You are a specialized assistant."
  },
  "user_id": "user123"
}
```

## 🔧 Configuration Options

### LLM Configuration
```python
LLMConfig(
    provider="openai",           # LLM provider
    model="gpt-4o-mini",        # Model name
    temperature=0.0,            # Response randomness (0.0-2.0)
    max_tokens=1000,            # Maximum response length
    top_p=1.0,                  # Top-p sampling (0.0-1.0)
    frequency_penalty=0.0,      # Frequency penalty (-2.0-2.0)
    presence_penalty=0.0,       # Presence penalty (-2.0-2.0)
    api_key="your-api-key"      # API key (optional, uses env var)
)
```

### Memory Configuration
```python
Mem0Config(
    enabled=True,                      # Enable memory
    api_key="your-mem0-key",          # Mem0 API key
    graph_store_provider="neo4j",      # Graph store provider
    enable_graph=True,                 # Enable graph features
    output_format="v1.1",              # Output format
    custom_prompt="Custom extraction"  # Custom entity extraction prompt
)
```

### Tool Configuration
```python
ToolConfig(
    enabled=True,                    # Enable tools
    tavily_api_key="your-key",      # Tavily API key
    max_search_results=2,           # Maximum search results
    available_tools=["web_search"]  # Available tools
)
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest tests/chatbot/ -v

# Run with Allure reporting
pytest tests/chatbot/ --alluredir=allure-results

# Generate Allure report
allure serve allure-results
```

### Test Categories
- **Unit Tests**: Core component testing
- **Integration Tests**: Service layer testing
- **API Tests**: FastAPI endpoint testing
- **End-to-End Tests**: Complete workflow testing

## 📊 Monitoring and Logging

The system includes comprehensive logging:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Logs include:
# - Session creation/deletion
# - Chat interactions
# - Error handling
# - Performance metrics
```

## 🔐 Security Considerations

- **API Key Management**: Use environment variables
- **Session Isolation**: Each session is isolated
- **Input Validation**: Comprehensive Pydantic validation
- **Error Handling**: Graceful error responses
- **Rate Limiting**: Consider implementing rate limiting

## 🚀 Performance Optimization

### Memory Management
- Automatic session cleanup
- Configurable session timeouts
- Efficient memory usage

### Streaming
- Real-time response streaming
- Server-Sent Events (SSE)
- Chunked responses

### Caching
- Session-based conversation history
- Configurable memory persistence

## 🔄 Integration with Existing Code

The new chatbot system coexists with the existing `closer_ai_workflow.py`:

```python
# Use existing multi-agent workflow
from app.langgraph.closer_ai_workflow import handle_instagram_message

# Use new chatbot system
from app.core import create_basic_chatbot

# Both systems can be used together
```

## 📈 Scaling Considerations

### Horizontal Scaling
- Stateless service design
- Session data can be externalized
- Load balancer friendly

### Vertical Scaling
- Configurable resource limits
- Memory cleanup mechanisms
- Efficient async operations

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Not Set**
   ```bash
   export OPENAI_API_KEY=your_key
   ```

2. **Memory Features Not Working**
   ```bash
   uv pip add mem0ai
   export MEM0_API_KEY=your_key
   ```

3. **Web Search Not Available**
   ```bash
   uv pip add langchain-tavily
   export TAVILY_API_KEY=your_key
   ```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Use Allure for test reporting
5. Follow Pydantic V2 best practices

## 📚 Examples

See `app/examples/chatbot_example.py` for comprehensive usage examples.

## 🔗 Related Documentation

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Mem0 Documentation](https://docs.mem0.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic V2 Documentation](https://docs.pydantic.dev/latest/)

---

This chatbot system provides a solid foundation for building sophisticated conversational AI applications with LangGraph, OpenAI, and optional memory capabilities. The modular design makes it easy to extend and customize for specific use cases. 