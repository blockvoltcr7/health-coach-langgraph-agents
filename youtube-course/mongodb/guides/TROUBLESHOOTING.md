# 🔧 RAG System Troubleshooting Guide

## 🚨 Common Issues & Solutions

### 1. MongoDB Connection Issues

#### Problem: "ServerSelectionTimeoutError"
```python
# Error: pymongo.errors.ServerSelectionTimeoutError: No servers found
```

**Solutions:**
```python
# 1. Check connection string format
# Correct format:
uri = "mongodb+srv://username:password@cluster.mongodb.net/database?retryWrites=true"

# 2. Verify network access
# Add your IP to Atlas whitelist or use 0.0.0.0/0 for development

# 3. Test with longer timeout
client = MongoClient(uri, serverSelectionTimeoutMS=10000)  # 10 seconds

# 4. Debug connection
try:
    client.admin.command('ping')
    print("✅ Connected successfully")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    # Check: Username/password, cluster status, network
```

#### Problem: "Authentication failed"
```python
# Solutions:
# 1. URL-encode special characters in password
from urllib.parse import quote_plus
password = quote_plus("p@ssw#rd!")
uri = f"mongodb+srv://user:{password}@cluster.mongodb.net"

# 2. Verify user has correct permissions
# In Atlas: Database Access > Edit user > Check roles

# 3. Use connection string from Atlas directly
# Atlas > Connect > Connect your application > Copy string
```

### 2. Vector Search Issues

#### Problem: "No results from vector search"
```python
# Debugging steps:
def debug_vector_search(collection, query_embedding):
    # 1. Check if documents have embeddings
    count = collection.count_documents({"embedding": {"$exists": True}})
    print(f"Documents with embeddings: {count}")
    
    # 2. Verify embedding dimensions
    sample = collection.find_one({"embedding": {"$exists": True}})
    if sample:
        print(f"Embedding dimensions: {len(sample['embedding'])}")
        print(f"Query dimensions: {len(query_embedding)}")
    
    # 3. Check index exists (in Atlas UI)
    # Atlas > Search > View Indexes
    
    # 4. Try simple search without filters
    simple_pipeline = [{
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 10,
            "limit": 1
        }
    }]
    
    results = list(collection.aggregate(simple_pipeline))
    print(f"Simple search results: {len(results)}")
    
    # 5. Verify index status is "READY"
    # Check in Atlas UI - index might still be building
```

#### Problem: "Dimension mismatch error"
```python
# Error: Dimensions don't match index configuration

# Solution 1: Use correct model
def get_embedding_model(dimensions):
    if dimensions == 1536:
        return "text-embedding-ada-002"  # OpenAI
    elif dimensions == 1024:
        return "voyage-3-large"  # Voyage AI
    else:
        raise ValueError(f"No model for {dimensions} dimensions")

# Solution 2: Multiple indexes for different models
indexes = {
    "openai": "embedding_openai_1536",
    "voyage": "embedding_voyage_1024"
}

# Solution 3: Standardize embeddings
def standardize_embedding(embedding, target_dim=1536):
    current_dim = len(embedding)
    if current_dim == target_dim:
        return embedding
    elif current_dim < target_dim:
        # Pad with zeros
        return embedding + [0] * (target_dim - current_dim)
    else:
        # Truncate
        return embedding[:target_dim]
```

### 3. Embedding Generation Issues

#### Problem: "Rate limit exceeded"
```python
# Error: openai.error.RateLimitError

# Solution 1: Implement exponential backoff
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
def generate_embedding_with_retry(text):
    return openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )

# Solution 2: Add delays between requests
def batch_generate_with_delay(texts, delay=0.1):
    embeddings = []
    for text in texts:
        embedding = generate_embedding(text)
        embeddings.append(embedding)
        time.sleep(delay)  # Avoid rate limits
    return embeddings

# Solution 3: Use different API keys
api_keys = ["key1", "key2", "key3"]
current_key_index = 0

def rotate_api_key():
    global current_key_index
    current_key_index = (current_key_index + 1) % len(api_keys)
    openai_client.api_key = api_keys[current_key_index]
```

#### Problem: "API key not valid"
```python
# Debugging steps:
import os

# 1. Check environment variable
print(f"API Key exists: {bool(os.getenv('OPENAI_API_KEY'))}")
print(f"Key prefix: {os.getenv('OPENAI_API_KEY')[:8]}...")  # Should be 'sk-...'

# 2. Test API key
try:
    response = openai_client.models.list()
    print("✅ API key is valid")
except Exception as e:
    print(f"❌ API key error: {e}")

# 3. Common issues:
# - Extra spaces in key
# - Using wrong key (e.g., secret key vs API key)
# - Key revoked or expired
# - Organization mismatch
```

### 4. Performance Issues

#### Problem: "Slow vector search"
```python
# Optimization strategies:

# 1. Reduce numCandidates
def optimize_search_performance(query_embedding, target_time=1.0):
    for num_candidates in [50, 100, 200, 500, 1000]:
        start = time.time()
        
        pipeline = [{
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": num_candidates,
                "limit": 10
            }
        }]
        
        results = list(collection.aggregate(pipeline))
        elapsed = time.time() - start
        
        print(f"Candidates: {num_candidates}, Time: {elapsed:.2f}s, Results: {len(results)}")
        
        if elapsed < target_time:
            return num_candidates

# 2. Add indexes for filter fields
db.collection.create_index([("category", 1)])
db.collection.create_index([("timestamp", -1)])
db.collection.create_index([("category", 1), ("timestamp", -1)])  # Compound

# 3. Use projection to reduce data transfer
pipeline = [
    {"$vectorSearch": {...}},
    {"$project": {
        "title": 1,
        "summary": {"$substr": ["$content", 0, 200]},  # First 200 chars
        "score": {"$meta": "vectorSearchScore"},
        "_id": 0
    }}
]
```

#### Problem: "High memory usage"
```python
# Memory optimization techniques:

# 1. Process in chunks
def process_large_collection(collection, chunk_size=1000):
    total_docs = collection.count_documents({})
    
    for skip in range(0, total_docs, chunk_size):
        chunk = list(collection.find().skip(skip).limit(chunk_size))
        process_chunk(chunk)
        
        # Force garbage collection
        import gc
        gc.collect()

# 2. Use generators instead of lists
def document_generator(collection, batch_size=100):
    cursor = collection.find().batch_size(batch_size)
    for doc in cursor:
        yield doc

# 3. Store embeddings as binary
import numpy as np

def compress_embedding(embedding):
    # Convert to float32 (50% size reduction)
    arr = np.array(embedding, dtype=np.float32)
    return arr.tobytes()

def decompress_embedding(binary_data):
    arr = np.frombuffer(binary_data, dtype=np.float32)
    return arr.tolist()
```

### 5. LLM Response Issues

#### Problem: "Empty or irrelevant responses"
```python
# Debugging and solutions:

def debug_rag_response(query, context, response):
    print(f"Query: {query}")
    print(f"Context length: {len(context)} chars")
    print(f"Response: {response}")
    
    # 1. Check if context is relevant
    if not context or len(context) < 100:
        print("⚠️ Context too short or empty")
        return "insufficient_context"
    
    # 2. Verify prompt structure
    messages = [
        {"role": "system", "content": "Answer based on the context."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]
    
    # 3. Test with different models
    models = ["gpt-3.5-turbo", "gpt-4"]
    for model in models:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7
        )
        print(f"\n{model}: {response.choices[0].message.content[:100]}...")
    
    # 4. Improve prompt
    improved_prompt = f"""Use the following context to answer the question. 
    If the answer is not in the context, say "I don't have that information."
    
    Context: {context}
    
    Question: {query}
    
    Answer:"""
```

#### Problem: "Token limit exceeded"
```python
# Error: maximum context length exceeded

# Solution 1: Truncate context
def truncate_context(context, max_tokens=3000):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(context)
    
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    
    return context

# Solution 2: Summarize context
def summarize_context(documents, max_tokens=2000):
    # First pass: extract key points
    summaries = []
    
    for doc in documents:
        summary_prompt = f"Summarize in 2 sentences: {doc['content'][:500]}"
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=100
        )
        summaries.append(response.choices[0].message.content)
    
    return "\n".join(summaries)

# Solution 3: Use larger context models
model_limits = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-32k": 32768
}
```

### 6. Deployment Issues

#### Problem: "Container crashes on startup"
```bash
# Debugging steps:

# 1. Check logs
docker logs container_name
kubectl logs pod_name

# 2. Common causes:
# - Missing environment variables
# - Port already in use
# - Insufficient memory

# 3. Debug interactively
docker run -it --rm \
  -e MONGODB_URI=$MONGODB_URI \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  your-image:latest \
  /bin/bash

# Then run app manually:
python app.py
```

#### Problem: "Out of memory in production"
```yaml
# Kubernetes solution:
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: rag-api
    resources:
      requests:
        memory: "1Gi"
      limits:
        memory: "2Gi"  # Increase limit
    env:
    - name: PYTHONUNBUFFERED
      value: "1"  # Reduce memory buffering
```

### 7. Caching Issues

#### Problem: "Redis connection refused"
```python
# Solutions:

# 1. Fallback to local cache
class HybridCache:
    def __init__(self):
        try:
            self.redis = redis.Redis(host='localhost', port=6379)
            self.redis.ping()
            self.use_redis = True
        except:
            print("⚠️ Redis unavailable, using local cache")
            self.use_redis = False
            self.local_cache = {}
    
    def get(self, key):
        if self.use_redis:
            return self.redis.get(key)
        return self.local_cache.get(key)
    
    def set(self, key, value, ttl=3600):
        if self.use_redis:
            self.redis.setex(key, ttl, value)
        else:
            self.local_cache[key] = value

# 2. Implement cache warming
def warm_cache_on_startup():
    common_queries = [
        "What is RAG?",
        "How does vector search work?",
        "MongoDB setup guide"
    ]
    
    for query in common_queries:
        try:
            # Generate and cache embeddings
            embedding = generate_embedding(query)
            cache.set(f"emb:{query}", embedding)
            
            # Pre-run searches
            results = vector_search(query)
            cache.set(f"search:{query}", results)
        except Exception as e:
            print(f"Cache warming failed for '{query}': {e}")
```

## 🛠️ Diagnostic Tools

### System Health Check Script
```python
def run_system_diagnostics():
    """Comprehensive system health check"""
    
    results = {
        "mongodb": False,
        "openai": False,
        "voyage": False,
        "redis": False,
        "vector_search": False
    }
    
    # 1. Test MongoDB
    try:
        client = MongoClient(os.getenv("MONGODB_URI"))
        client.admin.command('ping')
        results["mongodb"] = True
    except Exception as e:
        print(f"❌ MongoDB: {e}")
    
    # 2. Test OpenAI
    try:
        openai_client.models.list()
        results["openai"] = True
    except Exception as e:
        print(f"❌ OpenAI: {e}")
    
    # 3. Test Voyage AI
    try:
        voyage_client.embed(texts=["test"], model="voyage-3-large")
        results["voyage"] = True
    except Exception as e:
        print(f"❌ Voyage AI: {e}")
    
    # 4. Test Redis
    try:
        r = redis.Redis()
        r.ping()
        results["redis"] = True
    except Exception as e:
        print(f"❌ Redis: {e}")
    
    # 5. Test vector search
    try:
        collection = db.test_collection
        test_doc = {
            "content": "test",
            "embedding": [0.1] * 1536
        }
        collection.insert_one(test_doc)
        
        pipeline = [{
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": [0.1] * 1536,
                "numCandidates": 10,
                "limit": 1
            }
        }]
        
        list(collection.aggregate(pipeline))
        results["vector_search"] = True
        
        # Cleanup
        collection.delete_one({"_id": test_doc["_id"]})
    except Exception as e:
        print(f"❌ Vector Search: {e}")
    
    # Summary
    print("\n📊 System Health Summary:")
    for component, status in results.items():
        emoji = "✅" if status else "❌"
        print(f"{emoji} {component}: {'OK' if status else 'FAILED'}")
    
    return results

# Run diagnostics
if __name__ == "__main__":
    run_system_diagnostics()
```

### Performance Profiling
```python
import cProfile
import pstats
from functools import wraps

def profile_function(func):
    """Decorator to profile function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions
        
        return result
    return wrapper

# Usage
@profile_function
def slow_operation():
    # Your code here
    pass
```

## 📞 Getting Help

### Information to Provide
When seeking help, include:

1. **Error message** (full traceback)
2. **Environment details**:
   ```python
   import sys
   print(f"Python: {sys.version}")
   print(f"MongoDB: {pymongo.__version__}")
   print(f"OpenAI: {openai.__version__}")
   ```
3. **Minimal reproducible example**
4. **What you've already tried**

### Community Resources
- MongoDB Community Forum
- OpenAI Community Forum
- Stack Overflow (tags: mongodb-atlas, vector-search)
- GitHub Issues (for specific libraries)

---

🔧 **Remember**: Most issues have been encountered before. Search error messages, check logs carefully, and test each component independently!