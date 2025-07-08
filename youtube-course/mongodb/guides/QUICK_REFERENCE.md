# 🚀 MongoDB RAG Quick Reference Guide

## 📋 Essential Commands

### MongoDB Atlas Setup
```bash
# Create vector index via Atlas UI
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "type": "knnVector",
        "dimensions": 1536,  # OpenAI
        # dimensions": 1024,  # Voyage AI
        "similarity": "cosine"
      }
    }
  }
}
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install pymongo openai voyageai redis fastapi uvicorn

# Environment variables
export MONGODB_URI="mongodb+srv://..."
export OPENAI_API_KEY="sk-..."
export VOYAGE_AI_API_KEY="pa-..."
```

## 🧮 Embedding Generation

### OpenAI Embeddings
```python
from openai import OpenAI

client = OpenAI()
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input="Your text here"
)
embedding = response.data[0].embedding  # 1536 dimensions
```

### Voyage AI Embeddings
```python
import voyageai

voyage_client = voyageai.Client()
result = voyage_client.embed(
    texts=["Your text here"],
    model="voyage-3-large",
    input_type="document"  # or "query"
)
embedding = result.embeddings[0]  # 1024 dimensions
```

### Smart Fallback Pattern
```python
try:
    # Try Voyage AI first (better quality, lower cost)
    embedding = generate_voyage_embedding(text)
except Exception:
    # Fallback to OpenAI
    embedding = generate_openai_embedding(text)
```

## 🔍 Vector Search Patterns

### Basic Vector Search
```python
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 100,
            "limit": 5
        }
    }
]
results = collection.aggregate(pipeline)
```

### Filtered Vector Search
```python
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding", 
            "queryVector": query_embedding,
            "numCandidates": 200,
            "limit": 10,
            "filter": {
                "category": "tutorial",
                "date": {"$gte": datetime(2024, 1, 1)}
            }
        }
    }
]
```

### With Reranking
```python
# Get 3x candidates for reranking
candidates = get_vector_search_results(limit=15)

# Rerank with Voyage AI
reranking = voyage_client.rerank(
    query=query,
    documents=[c.content for c in candidates],
    model="rerank-2-lite",
    top_k=5
)
```

## 📄 Document Processing

### Basic Chunking
```python
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks
```

### Token-Based Chunking
```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def chunk_by_tokens(text, max_tokens=400):
    tokens = encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks
```

## 💬 RAG Implementation

### Basic RAG Pipeline
```python
async def rag_pipeline(query: str):
    # 1. Generate query embedding
    query_embedding = await generate_embedding(query)
    
    # 2. Vector search
    context_docs = await vector_search(query_embedding)
    
    # 3. Build context
    context = "\n\n".join([doc.content for doc in context_docs])
    
    # 4. Generate response
    response = await generate_response(query, context)
    
    return response
```

### Streaming Response
```python
async def stream_rag_response(query: str, context: str):
    stream = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Answer using context."},
            {"role": "user", "content": f"Context: {context}\n\nQ: {query}"}
        ],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

## 🛡️ Error Handling

### Retry Pattern
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def resilient_api_call():
    return await risky_operation()
```

### Circuit Breaker
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure = None
        self.state = "closed"
    
    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure > self.timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "open"
            raise e
```

## 🚀 Performance Optimization

### Caching with Redis
```python
import redis
import json

cache = redis.Redis(decode_responses=True)

def get_cached_embedding(text: str):
    key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
    cached = cache.get(key)
    
    if cached:
        return json.loads(cached)
    
    embedding = generate_embedding(text)
    cache.setex(key, 3600, json.dumps(embedding))  # 1 hour TTL
    
    return embedding
```

### Batch Processing
```python
async def batch_generate_embeddings(texts: List[str], batch_size=100):
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = await generate_embeddings_batch(batch)
        embeddings.extend(batch_embeddings)
    
    return embeddings
```

## 📊 Cost Optimization

### Model Selection by Use Case
| Use Case | Embedding Model | Cost/1M tokens | Quality |
|----------|----------------|----------------|---------|
| General | text-embedding-ada-002 | $0.10 | Good |
| Specialized | voyage-3-large | $0.12 | Excellent |
| Budget | text-embedding-3-small | $0.02 | Fair |

### Token Usage Tracking
```python
def track_token_usage(operation: str, tokens: int, model: str):
    costs = {
        "text-embedding-ada-002": 0.0001,  # per 1K tokens
        "voyage-3-large": 0.00012,
        "gpt-3.5-turbo": 0.002,
        "gpt-4": 0.03
    }
    
    cost = (tokens / 1000) * costs.get(model, 0)
    
    # Log to database or monitoring system
    log_usage({
        "operation": operation,
        "tokens": tokens,
        "model": model,
        "cost": cost,
        "timestamp": datetime.utcnow()
    })
```

## 🐳 Docker Commands

### Build and Run
```bash
# Build image
docker build -t rag-api:latest .

# Run container
docker run -d \
  --name rag-api \
  -p 8000:8000 \
  -e MONGODB_URI=$MONGODB_URI \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  rag-api:latest

# View logs
docker logs -f rag-api

# Stop and remove
docker stop rag-api
docker rm rag-api
```

### Docker Compose
```bash
# Start all services
docker-compose up -d

# View status
docker-compose ps

# View logs
docker-compose logs -f rag-api

# Stop all services
docker-compose down
```

## ☸️ Kubernetes Commands

### Deployment
```bash
# Apply configurations
kubectl apply -f k8s/

# Check deployment
kubectl get deployments -n rag-system
kubectl get pods -n rag-system
kubectl get svc -n rag-system

# View logs
kubectl logs -f deployment/rag-api -n rag-system

# Scale deployment
kubectl scale deployment rag-api --replicas=5 -n rag-system

# Update image
kubectl set image deployment/rag-api rag-api=rag-api:v2 -n rag-system
```

### Troubleshooting
```bash
# Describe pod
kubectl describe pod <pod-name> -n rag-system

# Get events
kubectl get events -n rag-system

# Execute command in pod
kubectl exec -it <pod-name> -n rag-system -- /bin/bash

# Port forward for debugging
kubectl port-forward svc/rag-api-service 8000:80 -n rag-system
```

## 🔧 Debugging Tips

### MongoDB Connection Issues
```python
# Test connection
from pymongo import MongoClient

client = MongoClient(uri, serverSelectionTimeoutMS=5000)
try:
    client.admin.command('ping')
    print("✅ Connected to MongoDB")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

### API Rate Limits
```python
# Add delays between requests
import time

for request in requests:
    try:
        result = process_request(request)
    except RateLimitError:
        time.sleep(60)  # Wait 1 minute
        result = process_request(request)
```

### Memory Optimization
```python
# Process large datasets in chunks
def process_large_dataset(documents, chunk_size=1000):
    for i in range(0, len(documents), chunk_size):
        chunk = documents[i:i + chunk_size]
        process_chunk(chunk)
        
        # Force garbage collection if needed
        import gc
        gc.collect()
```

## 📚 Additional Resources

### Documentation
- [MongoDB Atlas Vector Search](https://www.mongodb.com/docs/atlas/atlas-search/vector-search/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Voyage AI Docs](https://docs.voyageai.com/)
- [FastAPI](https://fastapi.tiangolo.com/)

### Tools
- [MongoDB Compass](https://www.mongodb.com/products/compass) - GUI for MongoDB
- [Postman](https://www.postman.com/) - API testing
- [k9s](https://k9scli.io/) - Kubernetes CLI UI
- [Lens](https://k8slens.dev/) - Kubernetes IDE

### Community
- [MongoDB Community Forum](https://www.mongodb.com/community/forums/)
- [OpenAI Community](https://community.openai.com/)
- [FastAPI Discord](https://discord.gg/fastapi)

---

💡 **Pro Tip**: Keep this guide handy while building your RAG systems. Most common issues can be solved by referring to these patterns!