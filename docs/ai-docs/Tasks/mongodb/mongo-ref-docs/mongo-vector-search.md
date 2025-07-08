## Using Voyage AI with MongoDB Atlas Vector Search

Voyage AI provides embedding and reranking models that integrate with MongoDB Atlas Vector Search to power AI-driven applications like semantic search, recommendation systems, and Retrieval-Augmented Generation (RAG). This guide covers setup, embedding generation, vector search, and best practices with Python code examples.

### 1. Set Up MongoDB Atlas
- **Create an Atlas Cluster**:
  - Sign up or log in at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).
  - Create a cluster (e.g., M0 free tier for testing or a dedicated cluster for production).
  - Ensure MongoDB version 6.0 or higher for Vector Search support.
- **Enable Vector Search**:
  - In the Atlas UI, go to your cluster, select the **Search** tab, and enable Vector Search.
  - Create a vector search index on your collection (see step 3).

### 2. Generate Embeddings with Voyage AI
Voyage AI’s models convert text (or other data) into vectors. You’ll need a Voyage AI API key from [voyageai.com](https://www.voyageai.com).

#### Prerequisites
Install required libraries:
```bash
pip install voyageai pymongo
```

#### Generate Document Embeddings
Batch-generate embeddings for documents and store them in MongoDB:

```python
import voyageai
from pymongo import MongoClient

# Initialize clients
vo = voyageai.Client(api_key="your-voyage-ai-api-key")
client = MongoClient("your-mongodb-atlas-connection-string")
db = client["your_database"]
collection = db["your_collection"]

# Sample documents
documents = [
    {"text": "Holiday cookie recipes without tree nuts", "metadata": {"category": "recipes"}},
    {"text": "Best practices for secure coding in Python", "metadata": {"category": "coding"}}
]

# Generate and store embeddings
for doc in documents:
    embedding = vo.embed([doc["text"]], model="voyage-3-large").embeddings[0]
    doc["embedding"] = embedding
    collection.insert_one(doc)

print("Documents inserted with embeddings.")
```

#### Generate Query Embeddings
Generate embeddings for search queries:
```python
query = "Find cookie recipes without nuts"
query_embedding = vo.embed([query], model="voyage-3-large").embeddings[0]
```

### 3. Create a Vector Search Index
In the Atlas UI, create a vector search index for your collection:
```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536, // Matches voyage-3-large
      "similarity": "cosine" // Options: cosine, euclidean, dotProduct
    }
  ]
}
```
- Ensure `numDimensions` matches your model (e.g., 1536 for voyage-3-large).
- Use **cosine** similarity for text-based semantic search.

### 4. Perform Vector Search
Use the `$vectorSearch` aggregation stage:

```python
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 100, // 10x limit for better recall
            "limit": 10
        }
    },
    {
        "$project": {
            "text": 1,
            "metadata": 1,
            "score": {"$meta": "vectorSearchScore"}
        }
    }
]

results = collection.aggregate(pipeline)
for result in results:
    print(f"Text: {result['text']}, Score: {result['score']}")
```

### 5. Asynchronous Implementation with Motor
For high-performance apps, use Motor (async MongoDB driver):

```python
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import voyageai

async def search_with_motor():
    vo = voyageai.Client(api_key="your-voyage-ai-api-key")
    client = AsyncIOMotorClient("your-mongodb-atlas-connection-string")
    db = client["your_database"]
    collection = db["your_collection"]

    query = "Find cookie recipes without nuts"
    query_embedding = vo.embed([query], model="voyage-3-large").embeddings[0]

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": 10
            }
        },
        {
            "$project": {
                "text": 1,
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    async for result in collection.aggregate(pipeline):
        print(f"Text: {result['text']}, Score: {result['score']}")

asyncio.run(search_with_motor())
```

### 6. Optional: Reranking
Refine results using Voyage AI’s reranking model:

```python
initial_results = [result["text"] for result in collection.aggregate(pipeline)]
reranked_results = vo.rerank(query, initial_results, model="rerank-1")
for result in reranked_results:
    print(f"Text: {result['text']}, Relevance Score: {result['relevance_score']}")
```

### 7. Best Practices
- **Embedding Consistency**: Use the same model (e.g., voyage-3-large) for indexing and querying.
- **Performance**:
  - Set `numCandidates` to 10x `limit` for optimal recall.
  - Use batch inserts for documents to improve throughput.
  - Configure connection pooling (e.g., `maxPoolSize=50`) for PyMongo/Motor.
- **Error Handling**:
  ```python
  try:
      embedding = vo.embed([doc["text"]], model="voyage-3-large").embeddings[0]
  except voyageai.VoyageAIError as e:
      print(f"Embedding error: {e}")
      # Add retry logic
  ```
- **Metadata Filtering**: Add filters to `$vectorSearch` (e.g., `filter: {"metadata.category": "recipes"}`).

### 8. Integration with LangChain
Simplify RAG workflows with LangChain:

```python
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import VoyageAIEmbeddings

embeddings = VoyageAIEmbeddings(voyage_api_key="your-voyage-ai-api-key", model="voyage-3-large")
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name="vector_index"
)

query = "Find cookie recipes without nuts"
results = vector_store.similarity_search(query, k=10)
for doc in results:
    print(doc.page_content)
```

### 9. Migration from Other Vector Databases
To migrate from Pinecone, Weaviate, or Supabase:
- Export embeddings and metadata.
- Store embeddings as float arrays in MongoDB with metadata for filtering.
- Recreate vector indexes in Atlas, ensuring `numDimensions` and `similarity` match.
- Test queries to verify consistency.

### 10. Performance Optimization
- **Quantization**: Use Voyage AI’s quantization (e.g., int8) to reduce storage and latency.
- **Monitoring**: Use Atlas metrics to track query performance and resource usage.
- **Benchmarking**: Test with varying `numCandidates` and evaluate accuracy with metrics like MRR or Precision@K.

---

## Notes
- **Future Integration**: MongoDB plans to introduce auto-embedding and native reranking in Atlas by late 2025, simplifying workflows.
- **Resources**:
  - [MongoDB Atlas Vector Search Docs](https://www.mongodb.com/docs/atlas/atlas-search/vector-search/)
  - [Voyage AI Docs](https://www.voyageai.com/docs)
  - [MongoDB GenAI Showcase](https://github.com/mongodb-developer/GenAI-Showcase)
- **Cost**: Monitor storage and compute costs for large-scale vector data.

For further assistance, consult the official MongoDB and Voyage AI documentation or reach out to their support teams. Let me know if you need help with specific use cases or additional code examples!