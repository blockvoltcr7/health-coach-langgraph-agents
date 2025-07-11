"""
Limitless OS Sales Agent API

FastAPI application providing an intelligent sales agent powered by LangGraph and mem0.

The API provides a specialized sales agent with:
- Complete memory retrieval for full context
- Web search capabilities for research
- Date/time awareness for timely responses
- Advanced sales psychology and techniques
- Focus on qualifying leads and closing deals

To run this application:
1. Set up your .env file with required API keys
2. Activate your virtual environment: `source .venv/bin/activate`
3. Run: `uvicorn app.main:app --reload`
4. Visit http://localhost:8000/docs for interactive API documentation
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.api import api_router
import uvicorn

# Enhanced API metadata for Swagger documentation
tags_metadata = [
    {
        "name": "Health",
        "description": "Health check and system status endpoints",
    },
    {
        "name": "Hello World",
        "description": "Basic example endpoints for testing API connectivity",
    },
    {
        "name": "Limitless OS Sales Agent",
        "description": "Specialized sales agent for Limitless OS with memory, tools, and advanced sales techniques",
        "externalDocs": {
            "description": "LangGraph Documentation",
            "url": "https://langchain-ai.github.io/langgraph/",
        },
    },
]

app = FastAPI(
    title="Limitless OS Sales Agent API",
    version="1.0.0",
    description="""
    🤖 **Limitless OS Sales Agent** - Your AI-powered sales companion
    
    ## Features
    
    * **Advanced Sales Psychology**: Consultative selling techniques and objection handling
    * **Complete Memory Context**: Full conversation history for personalized interactions
    * **Web Search Integration**: Real-time prospect research and market intelligence
    * **Date/Time Awareness**: Timely follow-ups and deadline creation
    * **Lead Qualification**: Systematic assessment of prospects based on BANT criteria
    * **Deal Closing**: Strategic techniques to drive commitments and close deals
    
    ## Quick Start
    
    1. **Set up environment variables**:
       ```bash
       OPENAI_API_KEY=your_openai_key     # Required
       MEM0_API_KEY=your_mem0_key         # Required for memory
       TAVILY_API_KEY=your_tavily_key     # Required for web search
       ```
    
    2. **Start selling with the agent**:
       ```bash
       POST /api/v1/chatbot/chat/full-memory
       {
         "message": "I'm interested in learning about AI automation",
         "user_id": "prospect_123"
       }
       ```
    
    ## Available Endpoints
    
    ### Core Endpoints
    * **`GET /health`**: System health check
    * **`GET /api/v1/hello`**: Hello world test endpoint
    
    ### Sales Agent Endpoints
    * **`POST /api/v1/chatbot/chat/full-memory`**: Engage with the sales agent
    * **`GET /api/v1/chatbot/health`**: Sales agent health check
    
    ## Sales Process
    
    The agent follows a structured sales process:
    
    1. **Qualification**: Assess prospect fit (need, budget, authority, timeline)
    2. **Objection Handling**: Address concerns and provide solutions
    3. **Closing**: Drive toward commitment with customized proposals
    4. **Follow-up**: Maintain engagement for future opportunities
    
    ## Limitless OS Service
    
    The agent sells Limitless OS - an AI-powered business transformation platform:
    * Intelligent process automation
    * Advanced analytics and insights
    * 24/7 AI-powered operations
    * Scalable solutions for all business sizes
    * 40-60% cost reduction through automation
    * 3-5x productivity improvements
    
    ## Authentication
    
    Currently supports optional Bearer token authentication for prospect identification.
    """,
    terms_of_service="https://limitlessos.com/terms/",
    contact={
        "name": "Limitless OS Sales Team",
        "url": "https://limitlessos.com/contact/",
        "email": "sales@limitlessos.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get(
    "/",
    tags=["Health"],
    summary="API Root",
    description="Root endpoint providing API information and navigation links",
    response_description="API welcome message with navigation links"
)
async def root():
    """
    🏠 **API Root Endpoint**
    
    Welcome to the Limitless OS Sales Agent API! This endpoint provides
    basic information about the API and links to key resources.
    
    **Quick Links:**
    - 📚 Interactive API Docs: `/docs`
    - 📖 ReDoc Documentation: `/redoc`
    - 🔍 OpenAPI Schema: `/openapi.json`
    - ❤️ Health Check: `/health`
    
    **API Base URL:** `/api/v1`
    """
    return {
        "message": "🤖 Limitless OS Sales Agent API",
        "version": "1.0.0",
        "description": "AI-powered sales agent for Limitless OS",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_schema": "/openapi.json"
        },
        "api_base": "/api/v1",
        "endpoints": {
            "health": "/health",
            "hello": "/api/v1/hello",
            "sales_agent": "/api/v1/chatbot/chat/full-memory",
            "agent_health": "/api/v1/chatbot/health"
        },
        "features": [
            "Advanced sales psychology and techniques",
            "Complete memory context for personalization",
            "Web search for prospect research",
            "Date/time awareness for timely responses",
            "Lead qualification and scoring",
            "Deal closing strategies"
        ],
        "service": {
            "name": "Limitless OS",
            "description": "AI-powered business transformation platform",
            "benefits": [
                "40-60% cost reduction through automation",
                "3-5x productivity improvements",
                "24/7 AI-powered operations",
                "Scalable solutions for all business sizes"
            ]
        }
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Health Check",
    description="System health check endpoint for monitoring and load balancers",
    response_description="Health status and system information"
)
async def health_check():
    """
    ❤️ **Health Check Endpoint**
    
    This endpoint is used for:
    - Load balancer health checks
    - System monitoring
    - Service availability verification
    - Container orchestration health probes
    
    Returns system status and basic information.
    """
    return {
        "status": "healthy",
        "message": "Health Coach LangGraph Agents API is running successfully",
        "service": "health-coach-langgraph-agents",
        "version": "1.0.0",
        "timestamp": "2024-01-01T00:00:00Z"
    }




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
