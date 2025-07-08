# 🗄️ MongoDB Vector Search Cheatsheet

## 🏗️ Index Creation

### Basic Vector Index
```javascript
// MongoDB Atlas UI - JSON Editor
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "type": "knnVector",
        "dimensions": 1536,
        "similarity": "cosine"
      }
    }
  }
}
```

### Advanced Vector Index with Filters
```javascript
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "embedding": {
        "type": "knnVector",
        "dimensions": 1536,
        "similarity": "cosine"
      },
      "category": {
        "type": "string",
        "filterable": true
      },
      "timestamp": {
        "type": "date",
        "filterable": true
      },
      "tags": {
        "type": "string",
        "filterable": true,
        "multi": true
      },
      "priority": {
        "type": "number",
        "filterable": true
      }
    }
  }
}
```

### Multiple Embedding Fields
```javascript
{
  "mappings": {
    "fields": {
      "title_embedding": {
        "type": "knnVector",
        "dimensions": 1536,
        "similarity": "cosine"
      },
      "content_embedding": {
        "type": "knnVector",
        "dimensions": 1536,
        "similarity": "cosine"
      },
      "voyage_embedding": {
        "type": "knnVector",
        "dimensions": 1024,
        "similarity": "cosine"
      }
    }
  }
}
```

## 🔍 Vector Search Queries

### Basic Search
```python
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

### Search with Metadata
```python
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 150,
            "limit": 10
        }
    },
    {
        "$project": {
            "title": 1,
            "content": 1,
            "category": 1,
            "score": {"$meta": "vectorSearchScore"},
            "_id": 0
        }
    }
]
```

### Filtered Search
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
                "category": {"$in": ["tutorial", "guide"]},
                "priority": {"$gte": 3},
                "timestamp": {"$gte": datetime(2024, 1, 1)}
            }
        }
    }
]
```

### Compound Filtering
```python
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 300,
            "limit": 20,
            "filter": {
                "$and": [
                    {"category": "technical"},
                    {
                        "$or": [
                            {"priority": {"$gte": 4}},
                            {"tags": {"$in": ["important", "urgent"]}}
                        ]
                    }
                ]
            }
        }
    }
]
```

### Multi-Stage Pipeline
```python
pipeline = [
    # Stage 1: Vector search
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 500,
            "limit": 50
        }
    },
    # Stage 2: Add metadata
    {
        "$addFields": {
            "search_score": {"$meta": "vectorSearchScore"},
            "search_timestamp": "$$NOW"
        }
    },
    # Stage 3: Lookup related documents
    {
        "$lookup": {
            "from": "related_docs",
            "localField": "_id",
            "foreignField": "parent_id",
            "as": "related"
        }
    },
    # Stage 4: Sort and limit
    {
        "$sort": {"search_score": -1}
    },
    {
        "$limit": 10
    }
]
```

## 📊 Performance Optimization

### Index Hints
```javascript
// Specify dimensions based on your embedding model
{
  "dimensions": 1536,    // OpenAI ada-002
  // "dimensions": 1024, // Voyage AI voyage-3-large
  // "dimensions": 768,  // Sentence transformers
  // "dimensions": 384,  // MiniLM
}
```

### numCandidates Guidelines
```python
# numCandidates affects accuracy vs performance

# For small datasets (<10K documents)
"numCandidates": limit * 10

# For medium datasets (10K-100K documents)
"numCandidates": limit * 20

# For large datasets (>100K documents)
"numCandidates": min(limit * 50, 10000)

# With filters (increase candidates)
"numCandidates": limit * 30
```

### Batch Operations
```python
# Batch insert with embeddings
def batch_insert_with_embeddings(documents, batch_size=100):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        # Generate embeddings for batch
        texts = [doc['content'] for doc in batch]
        embeddings = generate_embeddings_batch(texts)
        
        # Add embeddings to documents
        for doc, emb in zip(batch, embeddings):
            doc['embedding'] = emb
        
        # Insert batch
        collection.insert_many(batch)
```

## 🛠️ Maintenance Commands

### Index Management
```javascript
// List all search indexes
db.collection.getSearchIndexes()

// Drop search index
db.collection.dropSearchIndex("index_name")

// Update index (drop and recreate)
// Note: This will cause downtime
db.collection.dropSearchIndex("vector_index")
// Then create new index through Atlas UI
```

### Collection Statistics
```javascript
// Get collection stats
db.collection.stats()

// Count documents with embeddings
db.collection.countDocuments({"embedding": {"$exists": true}})

// Find documents missing embeddings
db.collection.find({"embedding": {"$exists": false}}).limit(10)

// Average embedding dimensions
db.collection.aggregate([
  {"$match": {"embedding": {"$exists": true}}},
  {"$limit": 1},
  {"$project": {"embedding_size": {"$size": "$embedding"}}}
])
```

### Data Validation
```python
# Validate embedding dimensions
def validate_embeddings(collection):
    pipeline = [
        {"$match": {"embedding": {"$exists": True}}},
        {"$project": {
            "dim": {"$size": "$embedding"},
            "title": 1
        }},
        {"$group": {
            "_id": "$dim",
            "count": {"$sum": 1},
            "examples": {"$push": "$title"}
        }}
    ]
    
    results = list(collection.aggregate(pipeline))
    for r in results:
        print(f"Dimension {r['_id']}: {r['count']} documents")
```

## 🔐 Security Best Practices

### Connection String Security
```python
# Use environment variables
uri = os.getenv("MONGODB_URI")

# Use connection string with minimal privileges
uri = "mongodb+srv://read_user:password@cluster.mongodb.net/db?retryWrites=true"

# Enable TLS/SSL
uri = "mongodb+srv://...?tls=true&tlsAllowInvalidCertificates=false"
```

### Role-Based Access
```javascript
// Create read-only user for search
db.createUser({
  user: "search_user",
  pwd: "secure_password",
  roles: [
    { role: "read", db: "rag_database" }
  ]
})

// Create write user for ingestion
db.createUser({
  user: "ingest_user",
  pwd: "secure_password",
  roles: [
    { role: "readWrite", db: "rag_database" }
  ]
})
```

## 📈 Monitoring Queries

### Search Performance
```javascript
// Monitor slow queries
db.currentOp({
  "active": true,
  "secs_running": {"$gt": 3},
  "ns": /vector_search/
})

// Get search index statistics
db.collection.aggregate([
  {"$searchMeta": {
    "index": "vector_index"
  }}
])
```

### Usage Analytics
```python
# Track search patterns
def log_search(query, results_count, search_time):
    db.search_analytics.insert_one({
        "query": query,
        "results_count": results_count,
        "search_time_ms": search_time * 1000,
        "timestamp": datetime.utcnow(),
        "hour": datetime.utcnow().hour,
        "day_of_week": datetime.utcnow().weekday()
    })

# Analyze search patterns
pipeline = [
    {"$group": {
        "_id": "$hour",
        "count": {"$sum": 1},
        "avg_time": {"$avg": "$search_time_ms"}
    }},
    {"$sort": {"_id": 1}}
]
```

## 🚨 Common Issues & Solutions

### Issue: Slow Vector Search
```python
# Solution 1: Increase numCandidates gradually
for candidates in [100, 200, 500, 1000]:
    start = time.time()
    results = search_with_candidates(candidates)
    elapsed = time.time() - start
    print(f"Candidates: {candidates}, Time: {elapsed:.2f}s")

# Solution 2: Add filtering to reduce search space
# Use compound indexes for filter fields
```

### Issue: Dimension Mismatch
```python
# Solution: Handle multiple embedding models
def get_index_for_embedding(embedding):
    dim = len(embedding)
    if dim == 1536:
        return "openai_index"
    elif dim == 1024:
        return "voyage_index"
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
```

### Issue: Out of Memory
```python
# Solution: Use cursor with batch size
cursor = collection.find({}).batch_size(100)
for doc in cursor:
    process_document(doc)
    # Process in batches to avoid memory issues
```

## 💡 Pro Tips

1. **Pre-filter Strategy**: Apply filters before vector search when possible
2. **Index Naming**: Use descriptive names like `embedding_openai_1536_cosine`
3. **Backup Indexes**: Keep backup indexes during migrations
4. **Monitor Usage**: Track embedding dimensions to catch issues early
5. **Test Thoroughly**: Always test index changes in staging first

---

📌 **Remember**: Vector indexes in MongoDB Atlas are managed through the UI, not through standard MongoDB commands!