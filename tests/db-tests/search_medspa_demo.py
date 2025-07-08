#!/usr/bin/env python3
"""
Demo script to search MedSpa data in MongoDB.
Works with both vector search (if configured) and text search (fallback).
"""

import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import OperationFailure
import certifi
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import voyageai
import time

# Load environment variables
load_dotenv()


class SearchDemo:
    def __init__(self):
        # MongoDB connection
        mongo_password = os.getenv("MONGO_DB_PASSWORD")
        if not mongo_password:
            raise ValueError("MONGO_DB_PASSWORD not found")
            
        uri = f"mongodb+srv://health-coach-ai-sami:{mongo_password}@cluster0-health-coach-a.69bhzsd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0-health-coach-ai"
        self.client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
        
        # Test connection
        self.client.admin.command('ping')
        print("✅ Connected to MongoDB!")
        
        # Set up collection
        self.db = self.client["health_coach_ai"]
        self.collection = self.db["medspa_services"]
        
        # Set up embeddings (try Voyage first, fall back to OpenAI)
        self.setup_embeddings()
        
    def setup_embeddings(self):
        """Set up embedding provider."""
        # Try Voyage AI first
        voyage_key = os.getenv("VOYAGE_AI_API_KEY")
        if voyage_key:
            try:
                self.voyage_client = voyageai.Client(api_key=voyage_key)
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
            print("⚠️  No embedding provider available - using text search only")
            self.embedding_provider = None
    
    def embed_query(self, query: str):
        """Generate embedding for a query."""
        if self.embedding_provider == "voyage":
            try:
                time.sleep(0.5)  # Rate limiting
                result = self.voyage_client.embed([query], model="voyage-3-large", input_type="query")
                return result.embeddings[0]
            except Exception:
                # Fall back to OpenAI if available
                if hasattr(self, 'openai_embeddings'):
                    return self.openai_embeddings.embed_query(query)
        elif self.embedding_provider == "openai":
            return self.openai_embeddings.embed_query(query)
        return None
    
    def vector_search(self, query: str, limit: int = 5):
        """Perform vector search."""
        query_embedding = self.embed_query(query)
        if not query_embedding:
            return None
            
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
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        try:
            results = list(self.collection.aggregate(pipeline))
            return results
        except OperationFailure:
            return None
    
    def text_search(self, query: str, limit: int = 5):
        """Perform text-based search as fallback."""
        # Extract key terms from query
        keywords = query.lower().split()
        
        # Build regex pattern
        pattern = "|".join(keywords)
        
        # Search
        results = list(self.collection.find(
            {"content": {"$regex": pattern, "$options": "i"}},
            {"content": 1, "metadata": 1}
        ).limit(limit))
        
        return results
    
    def search(self, query: str, limit: int = 5):
        """Search with vector search or fall back to text search."""
        print(f"\n🔍 Searching for: '{query}'")
        print("-" * 60)
        
        # Try vector search first
        results = self.vector_search(query, limit)
        
        if results is not None:
            print("✅ Using Atlas Vector Search")
        else:
            print("⚠️  Vector search not available, using text search")
            results = self.text_search(query, limit)
        
        if not results:
            print("❌ No results found")
            return
        
        print(f"\n📊 Found {len(results)} results:")
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Result {i}")
            if 'score' in result:
                print(f"   Score: {result['score']:.4f}")
            
            # Extract key info from content
            content = result['content']
            lines = content.split('\n')
            
            # Find title/header
            title = "N/A"
            for line in lines:
                if line.strip() and (line.startswith('#') or line.startswith('**')):
                    title = line.strip('#* ')
                    break
            
            print(f"   Title: {title}")
            print(f"   Type: {result['metadata'].get('doc_type', 'N/A')}")
            print(f"   Preview: {content[:150].strip()}...")
            
            # Extract price if present
            if '$' in content:
                import re
                prices = re.findall(r'\$\d+(?:,\d+)?', content)
                if prices:
                    print(f"   Price: {prices[0]}")
    
    def run_demo(self):
        """Run search demonstrations."""
        print("\n" + "="*60)
        print("MedSpa Search Demo")
        print("="*60)
        
        # Check document count
        count = self.collection.count_documents({})
        print(f"\n📊 Total documents: {count}")
        
        if count == 0:
            print("⚠️  No documents found. Run the test to insert data first.")
            return
        
        # Demo searches
        demo_queries = [
            "I need energy and mental clarity for work",
            "Best treatment for glowing skin",
            "Athletic recovery after marathon", 
            "How much does NAD therapy cost?",
            "VIP membership options",
            "Hangover relief treatment",
            "Weight loss IV therapy"
        ]
        
        for query in demo_queries:
            self.search(query)
            input("\nPress Enter for next search...")
    
    def interactive_search(self):
        """Interactive search mode."""
        print("\n" + "="*60)
        print("Interactive Search Mode")
        print("="*60)
        print("Type your search query (or 'quit' to exit)")
        
        while True:
            query = input("\n🔍 Search: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if query:
                self.search(query)
    
    def close(self):
        """Close database connection."""
        self.client.close()


if __name__ == "__main__":
    try:
        demo = SearchDemo()
        
        # Run demo searches
        demo.run_demo()
        
        # Offer interactive mode
        choice = input("\n\nWould you like to try interactive search? (y/n): ")
        if choice.lower() == 'y':
            demo.interactive_search()
        
        demo.close()
        print("\n✅ Demo complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()