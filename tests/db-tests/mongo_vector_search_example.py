#!/usr/bin/env python3
"""
MongoDB Vector Search Example
This script demonstrates how to:
1. Connect to MongoDB Atlas
2. Insert documents with embeddings
3. Create a vector search index
4. Perform vector searches

Requirements:
- Set MONGO_DB_PASSWORD in your environment
- Set either VOYAGE_AI_API_KEY or OPENAI_API_KEY
"""

import os
import time
from typing import List
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
from dotenv import load_dotenv
import voyageai
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()


class MongoVectorSearchDemo:
    def __init__(self):
        # MongoDB connection
        mongo_password = os.getenv("MONGO_DB_PASSWORD")
        if not mongo_password:
            raise ValueError("MONGO_DB_PASSWORD not found in environment")
            
        uri = f"mongodb+srv://health-coach-ai-sami:{mongo_password}@cluster0-health-coach-a.69bhzsd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0-health-coach-ai"
        self.client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
        
        # Test connection
        self.client.admin.command('ping')
        print("✅ Connected to MongoDB!")
        
        # Set up database and collection
        self.db = self.client["health_coach_ai"]
        self.collection = self.db["vector_search_demo"]
        
        # Set up embedding provider
        self.setup_embeddings()
        
    def setup_embeddings(self):
        """Set up Voyage AI or OpenAI embeddings."""
        # Try Voyage AI first
        voyage_key = os.getenv("VOYAGE_AI_API_KEY")
        if voyage_key:
            try:
                self.voyage_client = voyageai.Client(api_key=voyage_key)
                # Test it
                self.voyage_client.embed(["test"], model="voyage-3-large", input_type="document")
                self.embedding_provider = "voyage"
                print("✅ Using Voyage AI embeddings")
                return
            except Exception as e:
                print(f"⚠️  Voyage AI not available: {e}")
        
        # Fall back to OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.openai_embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=openai_key
            )
            self.embedding_provider = "openai"
            print("✅ Using OpenAI embeddings")
        else:
            raise ValueError("No embedding provider available. Set VOYAGE_AI_API_KEY or OPENAI_API_KEY")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self.embedding_provider == "voyage":
            try:
                time.sleep(0.5)  # Rate limiting
                result = self.voyage_client.embed(texts, model="voyage-3-large", input_type="document")
                return result.embeddings
            except Exception as e:
                print(f"Error with Voyage AI: {e}")
                # Try fallback to OpenAI if available
                if hasattr(self, 'openai_embeddings'):
                    return self.openai_embeddings.embed_documents(texts)
                raise
        else:
            return self.openai_embeddings.embed_documents(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a search query."""
        if self.embedding_provider == "voyage":
            try:
                time.sleep(0.5)  # Rate limiting
                result = self.voyage_client.embed([query], model="voyage-3-large", input_type="query")
                return result.embeddings[0]
            except Exception as e:
                print(f"Error with Voyage AI: {e}")
                # Try fallback to OpenAI if available
                if hasattr(self, 'openai_embeddings'):
                    return self.openai_embeddings.embed_query(query)
                raise
        else:
            return self.openai_embeddings.embed_query(query)
    
    def insert_sample_data(self):
        """Insert sample documents with embeddings."""
        print("\n📝 Inserting sample documents...")
        
        # Sample documents
        documents = [
            {
                "title": "Python Best Practices",
                "content": "Python coding best practices include using virtual environments, following PEP 8 style guide, writing comprehensive tests, and using type hints for better code clarity.",
                "category": "programming",
                "tags": ["python", "best-practices", "coding"]
            },
            {
                "title": "Machine Learning Basics",
                "content": "Machine learning is a subset of AI that enables systems to learn from data. Key concepts include supervised learning, unsupervised learning, neural networks, and model evaluation.",
                "category": "ai",
                "tags": ["machine-learning", "ai", "data-science"]
            },
            {
                "title": "Web Development with FastAPI",
                "content": "FastAPI is a modern Python web framework for building APIs. It offers automatic documentation, type safety, high performance, and easy integration with async/await.",
                "category": "web",
                "tags": ["fastapi", "python", "api", "web"]
            },
            {
                "title": "Database Design Principles",
                "content": "Good database design involves normalization, proper indexing, defining relationships, and optimizing queries. Consider scalability and performance from the start.",
                "category": "database",
                "tags": ["database", "sql", "design"]
            },
            {
                "title": "MongoDB Vector Search",
                "content": "MongoDB Atlas Vector Search enables semantic search using embeddings. It supports similarity search, hybrid search, and filtering with high performance at scale.",
                "category": "database",
                "tags": ["mongodb", "vector-search", "embeddings"]
            }
        ]
        
        # Generate embeddings
        texts = [doc["content"] for doc in documents]
        print(f"🔄 Generating embeddings for {len(texts)} documents...")
        embeddings = self.embed_texts(texts)
        
        # Add embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc["embedding"] = embedding
        
        # Clear existing data
        self.collection.delete_many({})
        
        # Insert documents
        result = self.collection.insert_many(documents)
        print(f"✅ Inserted {len(result.inserted_ids)} documents")
        
    def create_search_index_instructions(self):
        """Print instructions for creating vector search index."""
        print("\n📋 Vector Search Index Configuration:")
        print("Please create a vector search index in MongoDB Atlas with this configuration:")
        print("""
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }
  ]
}
        """)
        print("Index name: 'vector_index'")
        print("Note: The number of dimensions should match your embedding model")
        
    def search_documents(self, query: str, limit: int = 3):
        """Perform vector search."""
        print(f"\n🔍 Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Try Atlas vector search first
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "title": 1,
                    "content": 1,
                    "category": 1,
                    "tags": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        try:
            results = list(self.collection.aggregate(pipeline))
            
            if results:
                print(f"\n✅ Found {len(results)} results using Atlas Vector Search:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result['title']} (Score: {result.get('score', 'N/A'):.4f})")
                    print(f"   Category: {result['category']}")
                    print(f"   Tags: {', '.join(result['tags'])}")
                    print(f"   Preview: {result['content'][:100]}...")
            else:
                # Fallback to regular search if vector search not available
                print("\n⚠️  Atlas Vector Search not available, using fallback method")
                results = list(self.collection.find({}).limit(limit))
                
                if results:
                    print(f"\nShowing {len(results)} documents:")
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}. {result['title']}")
                        print(f"   Category: {result['category']}")
                        print(f"   Preview: {result['content'][:100]}...")
                        
        except Exception as e:
            print(f"\n❌ Search error: {e}")
            print("Make sure you've created the vector search index in Atlas")
    
    def run_demo(self):
        """Run the complete demo."""
        print("\n" + "="*60)
        print("MongoDB Vector Search Demo")
        print("="*60)
        
        # Insert sample data
        self.insert_sample_data()
        
        # Show index creation instructions
        self.create_search_index_instructions()
        
        # Wait a moment for indexing
        print("\n⏳ Waiting for documents to be indexed...")
        time.sleep(2)
        
        # Perform searches
        test_queries = [
            "How to write better Python code?",
            "Tell me about artificial intelligence and machine learning",
            "Building modern web APIs",
            "Database optimization techniques",
            "Semantic search with embeddings"
        ]
        
        for query in test_queries:
            self.search_documents(query)
            print("\n" + "-"*60)
        
        print("\n✅ Demo complete!")
        
    def cleanup(self):
        """Clean up resources."""
        self.client.close()


if __name__ == "__main__":
    try:
        demo = MongoVectorSearchDemo()
        demo.run_demo()
        demo.cleanup()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()