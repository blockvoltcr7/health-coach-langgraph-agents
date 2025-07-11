from fastapi import APIRouter
from app.api.v1.endpoints.hello_world_v1 import router as hello_world_router
from app.api.v1.endpoints.chatbot_endpoints import router as sales_agent_router


api_router = APIRouter()
api_router.include_router(hello_world_router, tags=["Hello World"])
api_router.include_router(sales_agent_router, prefix="/chatbot", tags=["Limitless OS Sales Agent"])
