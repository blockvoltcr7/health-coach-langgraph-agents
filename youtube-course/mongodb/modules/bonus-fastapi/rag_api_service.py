"""
BONUS Module: FastAPI RAG Service
Time: 45 minutes
Goal: Build a complete production-ready RAG API service
"""

import os
import asyncio
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
import json
import hashlib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI
import voyageai
import redis.asyncio as redis
import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
request_count = Counter('rag_requests_total', 'Total requests', ['endpoint', 'status'])
request_duration = Histogram('rag_request_duration_seconds', 'Request duration', ['endpoint'])
active_requests = Gauge('rag_active_requests', 'Active requests')
token_usage = Counter('rag_token_usage_total', 'Token usage', ['provider', 'operation'])
cache_hits = Counter('rag_cache_hits_total', 'Cache hits', ['cache_type'])
cache_misses = Counter('rag_cache_misses_total', 'Cache misses', ['cache_type'])

# Security
security = HTTPBearer()

# Global clients
motor_client: Optional[AsyncIOMotorClient] = None
redis_client: Optional[redis.Redis] = None
openai_client: Optional[AsyncOpenAI] = None
voyage_client: Optional[voyageai.Client] = None

# Pydantic Models
class EmbeddingRequest(BaseModel):
    """Request model for embedding generation"""
    text: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(default="voyage-3-large", pattern="^(voyage-3-large|text-embedding-ada-002)$")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()

class SearchRequest(BaseModel):
    """Request model for vector search"""
    query: str = Field(..., min_length=1, max_length=500)
    collection: str = Field(default="documents")
    limit: int = Field(default=5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = None
    include_metadata: bool = Field(default=True)
    rerank: bool = Field(default=True)

class ChatRequest(BaseModel):
    """Request model for RAG chat"""
    message: str = Field(..., min_length=1, max_length=1000)
    conversation_id: Optional[str] = None
    collection: str = Field(default="documents")
    model: str = Field(default="gpt-3.5-turbo")
    stream: bool = Field(default=False)
    max_tokens: int = Field(default=500, ge=50, le=2000)
    temperature: float = Field(default=0.7, ge=0, le=1)

class DocumentIngestionRequest(BaseModel):
    """Request model for document ingestion"""
    documents: List[Dict[str, str]]
    collection: str = Field(default="documents")
    embedding_model: str = Field(default="voyage-3-large")
    batch_size: int = Field(default=50, ge=1, le=100)

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str]

class EmbeddingResponse(BaseModel):
    """Embedding generation response"""
    embedding: List[float]
    model: str
    dimensions: int
    cached: bool
    processing_time: float

class SearchResult(BaseModel):
    """Individual search result"""
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    """Search response"""
    results: List[SearchResult]
    total_results: int
    processing_time: float
    cached: bool
    reranked: bool

class ChatResponse(BaseModel):
    """Chat response"""
    response: str
    conversation_id: str
    sources: List[Dict[str, Any]]
    processing_time: float
    tokens_used: int

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime
    request_id: str

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting RAG API Service...")
    
    # Initialize clients
    global motor_client, redis_client, openai_client, voyage_client
    
    motor_client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
    
    try:
        redis_client = await redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True
        )
        await redis_client.ping()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None
    
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_AI_API_KEY"))
    
    logger.info("All services initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG API Service...")
    
    if motor_client:
        motor_client.close()
    
    if redis_client:
        await redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="RAG API Service",
    description="Production-ready Retrieval-Augmented Generation API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for metrics
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    """Track request metrics"""
    start_time = time.time()
    active_requests.inc()
    
    try:
        response = await call_next(request)
        request_count.labels(
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        return response
    finally:
        request_duration.labels(endpoint=request.url.path).observe(
            time.time() - start_time
        )
        active_requests.dec()

# Dependency for authentication
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key"""
    api_key = credentials.credentials
    
    # In production, validate against database or external service
    valid_keys = os.getenv("VALID_API_KEYS", "").split(",")
    
    if not api_key or api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return api_key

# Cache utilities
async def get_cache_key(prefix: str, content: str) -> str:
    """Generate cache key"""
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return f"{prefix}:{content_hash}"

async def get_cached(key: str) -> Optional[str]:
    """Get value from cache"""
    if not redis_client:
        return None
    
    try:
        value = await redis_client.get(key)
        if value:
            cache_hits.labels(cache_type=key.split(":")[0]).inc()
            return value
        else:
            cache_misses.labels(cache_type=key.split(":")[0]).inc()
    except Exception as e:
        logger.error(f"Cache get error: {e}")
    
    return None

async def set_cached(key: str, value: str, ttl: int = 3600):
    """Set value in cache"""
    if not redis_client:
        return
    
    try:
        await redis_client.setex(key, ttl, value)
    except Exception as e:
        logger.error(f"Cache set error: {e}")

# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = {}
    
    # Check MongoDB
    try:
        await motor_client.admin.command('ping')
        services["mongodb"] = "healthy"
    except Exception:
        services["mongodb"] = "unhealthy"
    
    # Check Redis
    if redis_client:
        try:
            await redis_client.ping()
            services["redis"] = "healthy"
        except Exception:
            services["redis"] = "unhealthy"
    else:
        services["redis"] = "not_configured"
    
    # Check OpenAI
    services["openai"] = "configured" if os.getenv("OPENAI_API_KEY") else "not_configured"
    
    # Check Voyage AI
    services["voyage"] = "configured" if os.getenv("VOYAGE_AI_API_KEY") else "not_configured"
    
    overall_status = "healthy" if all(
        s in ["healthy", "configured"] for s in services.values()
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version="1.0.0",
        services=services
    )

@app.post("/api/v1/embeddings", response_model=EmbeddingResponse)
async def generate_embedding(
    request: EmbeddingRequest,
    api_key: str = Depends(verify_api_key)
):
    """Generate embeddings for text"""
    start_time = time.time()
    
    # Check cache
    cache_key = await get_cache_key(f"embedding:{request.model}", request.text)
    cached_embedding = await get_cached(cache_key)
    
    if cached_embedding:
        embedding = json.loads(cached_embedding)
        return EmbeddingResponse(
            embedding=embedding,
            model=request.model,
            dimensions=len(embedding),
            cached=True,
            processing_time=time.time() - start_time
        )
    
    try:
        # Generate embedding
        if request.model == "voyage-3-large" and voyage_client:
            result = voyage_client.embed(
                texts=[request.text],
                model="voyage-3-large",
                input_type="document"
            )
            embedding = result.embeddings[0]
            token_usage.labels(provider="voyage", operation="embedding").inc(result.total_tokens)
        else:
            # Fallback to OpenAI
            response = await openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=request.text
            )
            embedding = response.data[0].embedding
            token_usage.labels(provider="openai", operation="embedding").inc(response.usage.total_tokens)
        
        # Cache the result
        await set_cached(cache_key, json.dumps(embedding), ttl=3600)
        
        return EmbeddingResponse(
            embedding=embedding,
            model=request.model,
            dimensions=len(embedding),
            cached=False,
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/search", response_model=SearchResponse)
async def vector_search(
    request: SearchRequest,
    api_key: str = Depends(verify_api_key)
):
    """Perform vector search"""
    start_time = time.time()
    
    # Check cache for search results
    cache_key = await get_cache_key(
        "search",
        f"{request.query}:{json.dumps(request.filters, sort_keys=True)}"
    )
    cached_results = await get_cached(cache_key)
    
    if cached_results:
        results = json.loads(cached_results)
        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            total_results=len(results),
            processing_time=time.time() - start_time,
            cached=True,
            reranked=False
        )
    
    try:
        # Generate query embedding
        embedding_response = await generate_embedding(
            EmbeddingRequest(text=request.query),
            api_key
        )
        query_embedding = embedding_response.embedding
        
        # Perform vector search
        db = motor_client[os.getenv("MONGODB_DATABASE", "rag_course")]
        collection = db[request.collection]
        
        # Build search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": request.limit * 10,
                    "limit": request.limit * 3 if request.rerank else request.limit
                }
            }
        ]
        
        # Add filters if provided
        if request.filters:
            pipeline[0]["$vectorSearch"]["filter"] = request.filters
        
        # Add projection
        pipeline.append({
            "$project": {
                "content": 1,
                "title": 1,
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        })
        
        # Execute search
        results = []
        async for doc in collection.aggregate(pipeline):
            results.append({
                "content": doc.get("content", ""),
                "score": doc.get("score", 0),
                "metadata": doc.get("metadata", {}) if request.include_metadata else None
            })
        
        # Rerank if requested
        if request.rerank and results and voyage_client:
            try:
                documents = [r["content"] for r in results]
                reranking = voyage_client.rerank(
                    query=request.query,
                    documents=documents,
                    model="rerank-2-lite",
                    top_k=request.limit
                )
                
                reranked_results = []
                for rank_result in reranking.results:
                    original = results[rank_result.index]
                    original["score"] = rank_result.relevance_score
                    reranked_results.append(original)
                
                results = reranked_results
                reranked = True
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")
                results = results[:request.limit]
                reranked = False
        else:
            results = results[:request.limit]
            reranked = False
        
        # Cache results
        await set_cached(cache_key, json.dumps(results), ttl=300)
        
        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            total_results=len(results),
            processing_time=time.time() - start_time,
            cached=False,
            reranked=reranked
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat", response_model=ChatResponse)
async def rag_chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """RAG-powered chat endpoint"""
    start_time = time.time()
    
    try:
        # Generate or retrieve conversation ID
        if not request.conversation_id:
            request.conversation_id = hashlib.md5(
                f"{api_key}:{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()
        
        # Retrieve relevant context
        search_response = await vector_search(
            SearchRequest(
                query=request.message,
                collection=request.collection,
                limit=5,
                rerank=True
            ),
            api_key
        )
        
        # Build context
        context = "\n\n".join([
            result.content for result in search_response.results
        ])
        
        # Generate response
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the provided context to answer questions accurately."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {request.message}"
            }
        ]
        
        if request.stream:
            # Streaming response
            async def generate():
                stream = await openai_client.chat.completions.create(
                    model=request.model,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stream=True
                )
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            
            return StreamingResponse(generate(), media_type="text/plain")
        else:
            # Non-streaming response
            response = await openai_client.chat.completions.create(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            generated_response = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Track token usage
            token_usage.labels(provider="openai", operation="completion").inc(tokens_used)
            
            # Store conversation in background
            background_tasks.add_task(
                store_conversation,
                request.conversation_id,
                request.message,
                generated_response,
                search_response.results
            )
            
            return ChatResponse(
                response=generated_response,
                conversation_id=request.conversation_id,
                sources=[
                    {
                        "content": r.content[:200] + "...",
                        "score": r.score
                    }
                    for r in search_response.results[:3]
                ],
                processing_time=time.time() - start_time,
                tokens_used=tokens_used
            )
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ingest")
async def ingest_documents(
    request: DocumentIngestionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Ingest documents into the knowledge base"""
    try:
        # Validate documents
        if not request.documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        # Queue ingestion task
        task_id = hashlib.md5(
            f"{datetime.utcnow().isoformat()}:{len(request.documents)}".encode()
        ).hexdigest()
        
        background_tasks.add_task(
            process_document_ingestion,
            task_id,
            request.documents,
            request.collection,
            request.embedding_model,
            request.batch_size
        )
        
        return {
            "task_id": task_id,
            "status": "processing",
            "document_count": len(request.documents),
            "message": "Documents queued for ingestion"
        }
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Retrieve conversation history"""
    try:
        db = motor_client[os.getenv("MONGODB_DATABASE", "rag_course")]
        collection = db["conversations"]
        
        conversation = await collection.find_one({"_id": conversation_id})
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return conversation
        
    except Exception as e:
        logger.error(f"Conversation retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

# Background tasks
async def store_conversation(
    conversation_id: str,
    user_message: str,
    assistant_response: str,
    sources: List[SearchResult]
):
    """Store conversation in database"""
    try:
        db = motor_client[os.getenv("MONGODB_DATABASE", "rag_course")]
        collection = db["conversations"]
        
        await collection.update_one(
            {"_id": conversation_id},
            {
                "$push": {
                    "messages": {
                        "timestamp": datetime.utcnow(),
                        "user": user_message,
                        "assistant": assistant_response,
                        "sources": [s.dict() for s in sources]
                    }
                },
                "$set": {
                    "updated_at": datetime.utcnow()
                },
                "$setOnInsert": {
                    "created_at": datetime.utcnow()
                }
            },
            upsert=True
        )
    except Exception as e:
        logger.error(f"Failed to store conversation: {e}")

async def process_document_ingestion(
    task_id: str,
    documents: List[Dict[str, str]],
    collection_name: str,
    embedding_model: str,
    batch_size: int
):
    """Process document ingestion in background"""
    try:
        db = motor_client[os.getenv("MONGODB_DATABASE", "rag_course")]
        collection = db[collection_name]
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Generate embeddings for batch
            texts = [doc.get("content", "") for doc in batch]
            
            if embedding_model == "voyage-3-large" and voyage_client:
                result = voyage_client.embed(
                    texts=texts,
                    model="voyage-3-large",
                    input_type="document"
                )
                embeddings = result.embeddings
            else:
                # Use OpenAI
                embeddings = []
                for text in texts:
                    response = await openai_client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=text
                    )
                    embeddings.append(response.data[0].embedding)
            
            # Prepare documents with embeddings
            docs_with_embeddings = []
            for doc, embedding in zip(batch, embeddings):
                doc_with_embedding = {
                    **doc,
                    "embedding": embedding,
                    "ingested_at": datetime.utcnow(),
                    "task_id": task_id
                }
                docs_with_embeddings.append(doc_with_embedding)
            
            # Insert batch
            await collection.insert_many(docs_with_embeddings)
            
            logger.info(f"Ingested batch {i//batch_size + 1} for task {task_id}")
        
        # Update task status
        await db["ingestion_tasks"].update_one(
            {"_id": task_id},
            {
                "$set": {
                    "status": "completed",
                    "completed_at": datetime.utcnow(),
                    "documents_ingested": len(documents)
                }
            },
            upsert=True
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed for task {task_id}: {e}")
        
        # Update task status to failed
        await db["ingestion_tasks"].update_one(
            {"_id": task_id},
            {
                "$set": {
                    "status": "failed",
                    "error": str(e),
                    "failed_at": datetime.utcnow()
                }
            },
            upsert=True
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.utcnow(),
            request_id=hashlib.md5(
                f"{request.url.path}:{datetime.utcnow()}".encode()
            ).hexdigest()[:12]
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if os.getenv("DEBUG") else None,
            timestamp=datetime.utcnow(),
            request_id=hashlib.md5(
                f"{request.url.path}:{datetime.utcnow()}".encode()
            ).hexdigest()[:12]
        ).dict()
    )

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "rag_api_service:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("ENV") == "development",
        log_level="info",
        access_log=True
    )