# Health Coach LangGraph Agents

A fresh start repository for building intelligent health coaching agents using LangGraph and FastAPI. This project provides a clean foundation with a simple FastAPI backend that will be extended to support sophisticated AI health coaching workflows.

## 🎯 Project Goals

This repository is designed to build advanced health coaching agents using:
- **LangGraph**: For creating complex, multi-step AI agent workflows
- **FastAPI**: As the backend API framework
- **Health Coaching Logic**: Personalized health recommendations and coaching
- **Agent Orchestration**: Multi-agent systems for comprehensive health guidance

## 🚀 Current Status

**Phase 1: Foundation Setup** ✅
- Clean FastAPI application with basic endpoints
- Development environment configured
- Testing framework ready
- Deployment configurations prepared

**Phase 2: LangGraph Integration** 🔄 (Next)
- LangGraph agent framework setup
- Basic health coaching agent implementation
- Agent workflow orchestration

**Phase 3: Health Coaching Features** 📋 (Planned)
- Personalized health assessments
- Nutrition guidance agents
- Exercise recommendation systems
- Progress tracking and analytics

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/health-coach-langgraph-agents.git
   cd health-coach-langgraph-agents
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Start the development server:**
   ```bash
   uv run uvicorn app.main:app --reload
   ```

4. **Visit the API:**
   - Application: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Alternative Docs: http://localhost:8000/redoc

## 📁 Project Structure

```
health-coach-langgraph-agents/
├── app/                          # FastAPI application
│   ├── api/                      # API routes and endpoints
│   │   └── v1/                   # API version 1
│   │       ├── api.py           # Main API router
│   │       └── endpoints/        # Individual endpoint modules
│   │           └── hello_world_v1.py
│   ├── core/                     # Core application logic
│   ├── models/                   # Pydantic models
│   ├── services/                 # Business logic services
│   └── main.py                   # FastAPI app entry point
├── tests/                        # Test suite
│   ├── endpoints/               # API endpoint tests
│   ├── config/                  # Test configuration
│   └── utils/                   # Test utilities
├── docs/                        # Documentation
├── deployment/                  # Deployment configurations
│   ├── Dockerfile              # Docker configuration
│   ├── render.yaml             # Render deployment
│   └── railway.json            # Railway deployment
├── pyproject.toml              # Project dependencies
├── uv.lock                     # Dependency lock file
└── README.md                   # This file
```

## 🔌 API Endpoints

### Current Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint - API welcome message |
| GET | `/health` | Health check endpoint |
| GET | `/api/v1/hello` | Hello world endpoint |

### Planned LangGraph Agent Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| POST | `/api/v1/agents/health-assessment` | Initial health assessment agent | 📋 Planned |
| POST | `/api/v1/agents/nutrition-coach` | Nutrition guidance agent | 📋 Planned |
| POST | `/api/v1/agents/exercise-planner` | Exercise recommendation agent | 📋 Planned |
| GET | `/api/v1/agents/workflows` | List available agent workflows | 📋 Planned |

## 🛠 Development

### Running the Application

```bash
# Start development server with auto-reload
uv run uvicorn app.main:app --reload

# Start on specific port
uv run uvicorn app.main:app --reload --port 8080

# Start with specific host
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Adding New Dependencies

```bash
# Add runtime dependency
uv add langgraph langchain

# Add development dependency
uv add --dev pytest-asyncio

# Add with version constraints
uv add "langgraph>=0.1.0"
```

### Environment Variables

Create a `.env` file for local development:

```bash
# API Configuration
API_ENV=development
API_DEBUG=true

# Future: LangGraph Configuration
# LANGCHAIN_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here

# Future: Health Coaching APIs
# NUTRITION_API_KEY=your_key_here
# FITNESS_API_KEY=your_key_here
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=app

# Run specific test file
uv run pytest tests/endpoints/test_fastapi_endpoints.py

# Run with Allure reporting
uv run pytest --alluredir=allure-results
allure serve allure-results
```

### Test Structure

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test API endpoints and workflows
- **Agent Tests**: Test LangGraph agent behaviors (planned)

## 🚀 Deployment

### Quick Deploy Options

**Render.com:**
```bash
# Using render.yaml configuration
git push origin main
# Connect repository to Render dashboard
```

**Railway:**
```bash
# Using railway.json configuration
railway up
```

**Docker:**
```bash
# Build and run locally
docker build -t health-coach-agents .
docker run -p 8000:8000 health-coach-agents
```

### Environment-Specific Deployments

- **Development**: Local development with hot reload
- **Staging**: Testing environment with production-like setup
- **Production**: Optimized for performance and reliability

## 🤝 Contributing

We welcome contributions to build the future of AI health coaching!

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/langgraph-nutrition-agent
   ```
3. **Make your changes**
4. **Add tests for new functionality**
5. **Run the test suite:**
   ```bash
   uv run pytest
   ```
6. **Submit a pull request**

### Contribution Areas

- 🤖 **LangGraph Agent Development**: Build new health coaching agents
- 🔧 **API Development**: Extend FastAPI endpoints
- 📊 **Health Data Integration**: Connect health APIs and data sources
- 🧪 **Testing**: Improve test coverage and agent testing
- 📚 **Documentation**: Enhance project documentation

## 🎯 Roadmap

### Phase 1: Foundation ✅
- [x] Clean FastAPI setup
- [x] Basic endpoint structure
- [x] Testing framework
- [x] Deployment configurations

### Phase 2: LangGraph Integration 🔄
- [ ] LangGraph framework setup
- [ ] Basic agent workflow implementation
- [ ] Agent state management
- [ ] Workflow orchestration

### Phase 3: Health Coaching Agents 📋
- [ ] Health assessment agent
- [ ] Nutrition coaching agent
- [ ] Exercise planning agent
- [ ] Progress tracking system

### Phase 4: Advanced Features 🚀
- [ ] Multi-agent coordination
- [ ] Personalization engine
- [ ] Integration with health APIs
- [ ] Real-time coaching workflows

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join project discussions for questions and ideas
- **Documentation**: Check the `/docs` directory for detailed guides

---

**Ready to build the future of AI health coaching?** 🏥🤖

Start by exploring the current API endpoints, then dive into LangGraph agent development!
