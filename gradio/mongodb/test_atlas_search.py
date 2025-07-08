#!/usr/bin/env python3
"""
Test Atlas Search directly with both index names
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
from mongodb_vector_search_app import EmbeddingProvider

load_dotenv()

# Connect to MongoDB
mongo_password = os.getenv("MONGO_DB_PASSWORD")
uri = f"mongodb+srv://health-coach-ai-sami:{mongo_password}@cluster0-health-coach-a.69bhzsd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0-health-coach-ai"
client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())

db = client["health_coach_ai"]
collection = db["medspa_services"]

# Get embedding provider
embedding_provider = EmbeddingProvider()

# Generate query embedding
query = "How much is the CEO Drip?"
query_embedding = embedding_provider.embed_query(query)

print(f"Testing query: '{query}'")
print(f"Embedding dimensions: {len(query_embedding)}")
print("-" * 50)

# Test with "default" index
print("\n1. Testing with index name 'default':")
try:
    pipeline = [
        {
            "$vectorSearch": {
                "index": "default",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 50,
                "limit": 5
            }
        },
        {
            "$project": {
                "content": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    results = list(collection.aggregate(pipeline))
    print(f"✅ Found {len(results)} results")
    
    if results:
        for i, result in enumerate(results[:3], 1):
            print(f"\nResult {i}:")
            print(f"Score: {result['score']:.4f}")
            print(f"Content: {result['content'][:100]}...")
            
            # Check if it contains CEO Drip
            if "CEO Drip" in result['content'] and "$299" in result['content']:
                print("✅ FOUND CEO DRIP WITH PRICE!")
    
except Exception as e:
    print(f"❌ Error: {e}")

# Test with text search as comparison
print("\n\n2. Text search comparison:")
text_results = list(collection.find(
    {"content": {"$regex": "CEO Drip.*\\$299", "$options": "i"}},
    {"content": 1}
).limit(1))

if text_results:
    print("✅ Text search found CEO Drip with price")
    print(f"Content: {text_results[0]['content'][:200]}...")

client.close()