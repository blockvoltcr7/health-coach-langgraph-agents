#!/usr/bin/env python3
"""
Debug script to test the search functionality directly
"""

import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from our app
from mongodb_vector_search_app import EmbeddingProvider, search_context_for_chat

# Load environment variables
load_dotenv()

def test_direct_search():
    """Test search functionality directly"""
    
    print("🔍 Testing MongoDB Vector Search Debug")
    print("=" * 50)
    
    # Test MongoDB connection
    mongo_password = os.getenv("MONGO_DB_PASSWORD")
    if not mongo_password:
        print("❌ MONGO_DB_PASSWORD not found")
        return
    
    uri = f"mongodb+srv://health-coach-ai-sami:{mongo_password}@cluster0-health-coach-a.69bhzsd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0-health-coach-ai"
    
    try:
        client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
        client.admin.command('ping')
        print("✅ MongoDB connected successfully")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return
    
    # Check collection
    db = client["health_coach_ai"]
    collection = db["medspa_services"]
    
    doc_count = collection.count_documents({})
    print(f"📊 Documents in collection: {doc_count}")
    
    # Test embedding provider
    try:
        embedding_provider = EmbeddingProvider()
        print(f"✅ Embedding provider initialized: {embedding_provider.provider}")
    except Exception as e:
        print(f"❌ Embedding provider failed: {e}")
        return
    
    # Test a simple text search first
    print("\n📝 Testing text search for 'CEO Drip'...")
    text_results = list(collection.find(
        {"content": {"$regex": "CEO Drip", "$options": "i"}},
        {"content": 1, "metadata": 1}
    ).limit(3))
    
    print(f"Found {len(text_results)} results with text search")
    if text_results:
        print(f"First result preview: {text_results[0]['content'][:100]}...")
    
    # Test vector search
    print("\n🎯 Testing vector search...")
    query = "How much is the CEO Drip?"
    
    try:
        # Generate query embedding
        query_embedding = embedding_provider.embed_query(query)
        print(f"✅ Query embedding generated (dimensions: {len(query_embedding)})")
        
        # Try vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 50,
                    "limit": 5
                }
            },
            {
                "$project": {
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        try:
            results = list(collection.aggregate(pipeline))
            print(f"✅ Vector search returned {len(results)} results")
            
            if results:
                print("\nTop results:")
                for i, result in enumerate(results[:3], 1):
                    print(f"\n{i}. Score: {result.get('score', 0):.4f}")
                    print(f"   Content: {result['content'][:150]}...")
                    
        except Exception as e:
            print(f"❌ Vector search failed: {e}")
            print("This might mean the vector index is not configured in Atlas")
            
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
    
    # Check if documents have embeddings
    print("\n🔍 Checking document embeddings...")
    sample_doc = collection.find_one()
    if sample_doc and 'embedding' in sample_doc:
        print(f"✅ Documents have embeddings (dimensions: {len(sample_doc['embedding'])})")
    else:
        print("❌ Documents don't have embeddings")
    
    # Check indexes
    print("\n📑 Collection indexes:")
    for index in collection.list_indexes():
        print(f"  - {index['name']}: {index.get('key', 'N/A')}")
    
    client.close()
    print("\n✅ Debug complete!")


if __name__ == "__main__":
    test_direct_search()