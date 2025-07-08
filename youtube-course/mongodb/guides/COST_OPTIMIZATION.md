# 💰 RAG System Cost Optimization Guide

## 📊 Cost Breakdown

### Typical RAG System Costs (Monthly)
| Component | Light Usage | Medium Usage | Heavy Usage |
|-----------|------------|--------------|-------------|
| Embeddings | $10-50 | $100-500 | $1000+ |
| LLM Calls | $20-100 | $200-1000 | $2000+ |
| Database | $0-50 | $50-200 | $500+ |
| Compute | $10-50 | $100-300 | $500+ |
| **Total** | **$40-250** | **$450-2000** | **$4000+** |

## 🎯 Embedding Cost Optimization

### 1. Choose the Right Model
```python
# Cost per 1M tokens (approximate)
EMBEDDING_COSTS = {
    "text-embedding-3-small": 0.02,    # Cheapest, lower quality
    "text-embedding-3-large": 0.13,    # Good balance
    "text-embedding-ada-002": 0.10,    # Legacy, good quality
    "voyage-3-large": 0.12,            # Best quality for domain-specific
    "voyage-3-lite": 0.08,             # Cheaper Voyage option
}

# Choose based on use case
def select_embedding_model(use_case):
    if use_case == "proof_of_concept":
        return "text-embedding-3-small"
    elif use_case == "production_general":
        return "text-embedding-ada-002"
    elif use_case == "production_specialized":
        return "voyage-3-large"
```

### 2. Implement Caching
```python
import hashlib
import redis
import json

class EmbeddingCache:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 86400 * 30  # 30 days
        
    def get_or_generate(self, text, model="text-embedding-ada-002"):
        # Create cache key
        cache_key = f"emb:{model}:{hashlib.md5(text.encode()).hexdigest()}"
        
        # Check cache
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached), True  # (embedding, from_cache)
        
        # Generate new embedding
        embedding = generate_embedding(text, model)
        
        # Cache it
        self.redis.setex(cache_key, self.ttl, json.dumps(embedding))
        
        return embedding, False

# Usage tracking
cache = EmbeddingCache(redis_client)
embedding, from_cache = cache.get_or_generate("Your text")
if from_cache:
    print("💰 Saved embedding cost!")
```

### 3. Batch Processing
```python
def batch_generate_embeddings(texts, batch_size=100):
    """
    OpenAI charges the same for batch requests
    But you save on network overhead
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Single API call for entire batch
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=batch  # Multiple texts in one request
        )
        
        embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(embeddings)
    
    return all_embeddings

# Cost comparison
single_cost = len(texts) * 0.0001  # Multiple API calls
batch_cost = len(texts) * 0.0001   # Same token cost, but faster
```

### 4. Deduplicate Before Embedding
```python
def deduplicate_texts(texts):
    """Remove duplicates before generating embeddings"""
    seen = set()
    unique_texts = []
    duplicate_indices = {}
    
    for i, text in enumerate(texts):
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in seen:
            # Track which indices are duplicates
            if text_hash not in duplicate_indices:
                duplicate_indices[text_hash] = []
            duplicate_indices[text_hash].append(i)
        else:
            seen.add(text_hash)
            unique_texts.append((i, text))
    
    return unique_texts, duplicate_indices

# Generate embeddings only for unique texts
unique_texts, duplicates = deduplicate_texts(documents)
unique_embeddings = generate_embeddings([t[1] for t in unique_texts])

# Map back to original indices
all_embeddings = [None] * len(documents)
for (idx, _), embedding in zip(unique_texts, unique_embeddings):
    all_embeddings[idx] = embedding
    
# Fill in duplicates
for text_hash, indices in duplicates.items():
    # Find the embedding for this text
    source_embedding = all_embeddings[indices[0]]
    for idx in indices[1:]:
        all_embeddings[idx] = source_embedding
```

## 💬 LLM Cost Optimization

### 1. Model Selection Strategy
```python
class SmartModelSelector:
    """Choose the cheapest model that can handle the task"""
    
    # Cost per 1K tokens (input/output)
    MODEL_COSTS = {
        "gpt-3.5-turbo": (0.0005, 0.0015),
        "gpt-4o-mini": (0.00015, 0.0006),  
        "gpt-4o": (0.0025, 0.01),
        "gpt-4": (0.03, 0.06),
    }
    
    def select_model(self, query_complexity, max_tokens=500):
        """Select model based on query complexity"""
        if query_complexity == "simple":
            # Simple factual questions
            return "gpt-3.5-turbo"
        elif query_complexity == "moderate":
            # Reasoning required
            return "gpt-4o-mini"
        elif query_complexity == "complex":
            # Complex analysis
            return "gpt-4o"
        else:
            # Critical accuracy needed
            return "gpt-4"
    
    def estimate_cost(self, model, input_tokens, output_tokens):
        input_cost, output_cost = self.MODEL_COSTS[model]
        total_cost = (input_tokens / 1000 * input_cost) + (output_tokens / 1000 * output_cost)
        return total_cost
```

### 2. Optimize Context Length
```python
def optimize_context(retrieved_docs, max_tokens=2000):
    """
    Reduce context to save tokens while maintaining quality
    """
    # Sort by relevance score
    sorted_docs = sorted(retrieved_docs, key=lambda x: x['score'], reverse=True)
    
    # Take only highly relevant content
    optimized_context = []
    token_count = 0
    
    for doc in sorted_docs:
        doc_tokens = count_tokens(doc['content'])
        
        if token_count + doc_tokens <= max_tokens:
            optimized_context.append(doc['content'])
            token_count += doc_tokens
        else:
            # Truncate the last document if needed
            remaining_tokens = max_tokens - token_count
            if remaining_tokens > 100:  # Worth including partial
                truncated = truncate_to_tokens(doc['content'], remaining_tokens)
                optimized_context.append(truncated)
            break
    
    return "\n\n".join(optimized_context)
```

### 3. Response Caching
```python
class ResponseCache:
    """Cache LLM responses for common queries"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 3600  # 1 hour for responses
    
    def get_or_generate(self, query, context, model="gpt-3.5-turbo"):
        # Create cache key from query + context hash
        context_hash = hashlib.md5(context.encode()).hexdigest()[:8]
        cache_key = f"resp:{model}:{hashlib.md5(query.encode()).hexdigest()}:{context_hash}"
        
        # Check cache
        cached = self.redis.get(cache_key)
        if cached:
            return cached, True
        
        # Generate new response
        response = generate_llm_response(query, context, model)
        
        # Cache it
        self.redis.setex(cache_key, self.ttl, response)
        
        return response, False
```

### 4. Implement Quotas
```python
class UserQuotaManager:
    """Manage per-user quotas to control costs"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.quotas = {
            "free": {"daily_tokens": 10000, "daily_requests": 100},
            "basic": {"daily_tokens": 100000, "daily_requests": 1000},
            "pro": {"daily_tokens": 1000000, "daily_requests": 10000}
        }
    
    def check_quota(self, user_id, user_tier, tokens_needed):
        # Keys for tracking
        token_key = f"quota:tokens:{user_id}:{datetime.utcnow().date()}"
        request_key = f"quota:requests:{user_id}:{datetime.utcnow().date()}"
        
        # Get current usage
        used_tokens = int(self.redis.get(token_key) or 0)
        used_requests = int(self.redis.get(request_key) or 0)
        
        # Check limits
        limits = self.quotas[user_tier]
        if used_tokens + tokens_needed > limits["daily_tokens"]:
            raise QuotaExceededError("Daily token limit exceeded")
        if used_requests >= limits["daily_requests"]:
            raise QuotaExceededError("Daily request limit exceeded")
        
        # Update usage
        self.redis.incrby(token_key, tokens_needed)
        self.redis.incr(request_key)
        
        # Set expiry to reset at midnight
        self.redis.expire(token_key, 86400)
        self.redis.expire(request_key, 86400)
        
        return True
```

## 🗄️ Database Cost Optimization

### 1. MongoDB Atlas Tier Selection
```python
def recommend_atlas_tier(document_count, monthly_searches):
    """Recommend appropriate MongoDB Atlas tier"""
    
    if document_count < 5000 and monthly_searches < 10000:
        return {
            "tier": "M0 (Free)",
            "cost": "$0",
            "limits": "512MB storage, shared resources"
        }
    elif document_count < 50000 and monthly_searches < 100000:
        return {
            "tier": "M10",
            "cost": "$57/month",
            "limits": "10GB storage, 2GB RAM"
        }
    elif document_count < 500000 and monthly_searches < 1000000:
        return {
            "tier": "M30",
            "cost": "$195/month",
            "limits": "40GB storage, 8GB RAM"
        }
    else:
        return {
            "tier": "M40+",
            "cost": "$500+/month",
            "limits": "Custom sizing needed"
        }
```

### 2. Optimize Storage
```python
def optimize_document_storage(document):
    """Reduce storage costs by optimizing document structure"""
    
    # Store embedding as binary instead of array
    if 'embedding' in document:
        # Convert to bytes (50% storage saving)
        import numpy as np
        embedding_array = np.array(document['embedding'], dtype=np.float32)
        document['embedding_binary'] = embedding_array.tobytes()
        del document['embedding']
    
    # Compress large text fields
    if 'content' in document and len(document['content']) > 1000:
        import gzip
        compressed = gzip.compress(document['content'].encode())
        document['content_compressed'] = compressed
        document['content_size'] = len(document['content'])
        del document['content']
    
    return document
```

### 3. Archive Old Data
```python
def archive_old_documents(days_old=90):
    """Move old documents to cheaper storage"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days_old)
    
    # Find old documents
    old_docs = list(db.documents.find({
        "last_accessed": {"$lt": cutoff_date}
    }))
    
    if old_docs:
        # Move to archive collection (can be on cheaper tier)
        db.documents_archive.insert_many(old_docs)
        
        # Remove from main collection
        db.documents.delete_many({
            "_id": {"$in": [doc["_id"] for doc in old_docs]}
        })
        
        print(f"Archived {len(old_docs)} documents")
```

## 🖥️ Compute Cost Optimization

### 1. Serverless vs Always-On
```python
# Serverless configuration (AWS Lambda, Google Cloud Functions)
def calculate_serverless_cost(monthly_requests, avg_duration_ms=200, memory_mb=512):
    # AWS Lambda pricing
    request_cost = monthly_requests * 0.0000002  # $0.20 per 1M requests
    
    # Compute cost
    gb_seconds = (memory_mb / 1024) * (avg_duration_ms / 1000) * monthly_requests
    compute_cost = gb_seconds * 0.0000166667  # $0.0000166667 per GB-second
    
    return request_cost + compute_cost

# Always-on container (e.g., ECS, Cloud Run)
def calculate_container_cost(cpu=0.5, memory_gb=1, hours=730):  # 730 hours/month
    # Approximate cloud pricing
    cpu_cost = cpu * 0.04 * hours  # $0.04 per vCPU hour
    memory_cost = memory_gb * 0.004 * hours  # $0.004 per GB hour
    
    return cpu_cost + memory_cost

# Recommendation
def recommend_compute_strategy(monthly_requests):
    serverless_cost = calculate_serverless_cost(monthly_requests)
    container_cost = calculate_container_cost()
    
    if serverless_cost < container_cost:
        return f"Use serverless (${serverless_cost:.2f}/month)"
    else:
        return f"Use containers (${container_cost:.2f}/month)"
```

### 2. Auto-scaling Configuration
```yaml
# Kubernetes HPA for cost optimization
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 1  # Scale down to 1 during low traffic
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 80  # Higher threshold = fewer pods = lower cost
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 minutes before scaling down
      policies:
      - type: Percent
        value: 50  # Scale down gradually
        periodSeconds: 60
```

## 📊 Cost Monitoring

### 1. Real-time Cost Tracking
```python
class CostTracker:
    def __init__(self):
        self.costs = defaultdict(float)
    
    def track_embedding(self, model, tokens):
        costs = {
            "text-embedding-ada-002": 0.0001,  # per 1K tokens
            "voyage-3-large": 0.00012,
        }
        cost = (tokens / 1000) * costs.get(model, 0)
        self.costs['embeddings'] += cost
        return cost
    
    def track_llm(self, model, input_tokens, output_tokens):
        model_costs = {
            "gpt-3.5-turbo": (0.0005, 0.0015),
            "gpt-4": (0.03, 0.06)
        }
        input_cost, output_cost = model_costs.get(model, (0, 0))
        cost = (input_tokens / 1000 * input_cost) + (output_tokens / 1000 * output_cost)
        self.costs['llm'] += cost
        return cost
    
    def get_daily_cost(self):
        return sum(self.costs.values())
    
    def get_projected_monthly_cost(self):
        # Based on current daily rate
        return self.get_daily_cost() * 30
```

### 2. Cost Alerts
```python
def setup_cost_alerts(daily_limit=100):
    """Alert when costs exceed thresholds"""
    
    current_cost = cost_tracker.get_daily_cost()
    
    if current_cost > daily_limit:
        send_alert(f"⚠️ Daily cost ${current_cost:.2f} exceeds limit ${daily_limit}")
        
        # Implement cost-saving measures
        enable_aggressive_caching()
        switch_to_cheaper_models()
        implement_rate_limiting()
    
    elif current_cost > daily_limit * 0.8:
        send_warning(f"📊 Daily cost ${current_cost:.2f} approaching limit")
```

## 💡 Cost Optimization Checklist

### Before Production
- [ ] Implement embedding caching
- [ ] Set up response caching for common queries
- [ ] Configure appropriate MongoDB Atlas tier
- [ ] Choose optimal embedding model
- [ ] Implement request deduplication
- [ ] Set up user quotas
- [ ] Configure auto-scaling

### Weekly Review
- [ ] Analyze cache hit rates
- [ ] Review model usage distribution
- [ ] Check for unused indexes
- [ ] Identify most expensive queries
- [ ] Review storage growth
- [ ] Optimize slow queries

### Monthly Review
- [ ] Analyze cost trends
- [ ] Review tier sizing
- [ ] Evaluate model performance vs cost
- [ ] Archive old data
- [ ] Update quotas based on usage
- [ ] Negotiate enterprise discounts

## 🎯 Target Metrics

| Metric | Good | Excellent |
|--------|------|-----------|
| Embedding Cache Hit Rate | >30% | >60% |
| Response Cache Hit Rate | >20% | >40% |
| Cost per 1K Queries | <$1 | <$0.50 |
| Average Response Time | <2s | <1s |
| Database Storage Growth | <10%/month | <5%/month |

---

💰 **Remember**: The best optimization is the one that doesn't compromise user experience. Always balance cost savings with quality!