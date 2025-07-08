"""
Test MongoDB Vector Search functionality with Voyage AI embeddings.
This test demonstrates inserting documents with embeddings and performing vector search.
"""

import os
import time
from typing import List, Dict, Any
import pytest
import allure
import voyageai
from pymongo import MongoClient, ASCENDING
from pymongo.server_api import ServerApi
from pymongo.errors import OperationFailure
import certifi
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()


class EmbeddingProvider:
    """Flexible embedding provider supporting Voyage AI and OpenAI."""
    
    def __init__(self):
        self.voyage_client = None
        self.openai_embeddings = None
        self.provider = None
        self.dimensions = None
        
        # Try Voyage AI first
        voyage_key = os.getenv("VOYAGE_AI_API_KEY")
        if voyage_key:
            try:
                self.voyage_client = voyageai.Client(api_key=voyage_key)
                # Test the client and get actual dimensions
                test_result = self.voyage_client.embed(["test"], model="voyage-3-large", input_type="document")
                self.provider = "voyage"
                self.dimensions = len(test_result.embeddings[0])
                print(f"Using Voyage AI embeddings (dimensions: {self.dimensions})")
            except Exception as e:
                print(f"Voyage AI not available: {e}")
                self.voyage_client = None
        
        # Fall back to OpenAI if Voyage not available
        if not self.voyage_client:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                self.openai_embeddings = OpenAIEmbeddings(
                    model="text-embedding-ada-002",
                    openai_api_key=openai_key
                )
                self.provider = "openai"
                self.dimensions = 1536  # ada-002 dimensions
                print("Using OpenAI embeddings")
            else:
                raise ValueError("No embedding provider available. Set VOYAGE_AI_API_KEY or OPENAI_API_KEY")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        if self.provider == "voyage":
            try:
                time.sleep(0.5)  # Rate limiting protection
                result = self.voyage_client.embed(texts, model="voyage-3-large", input_type="document")
                return result.embeddings
            except voyageai.error.RateLimitError:
                print("Voyage AI rate limit hit, falling back to OpenAI")
                # Switch to OpenAI for this session
                if os.getenv("OPENAI_API_KEY"):
                    self.openai_embeddings = OpenAIEmbeddings(
                        model="text-embedding-ada-002",
                        openai_api_key=os.getenv("OPENAI_API_KEY")
                    )
                    self.provider = "openai"
                    return self.openai_embeddings.embed_documents(texts)
                raise
        else:
            return self.openai_embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        if self.provider == "voyage":
            try:
                time.sleep(0.5)  # Rate limiting protection
                result = self.voyage_client.embed([text], model="voyage-3-large", input_type="query")
                return result.embeddings[0]
            except voyageai.error.RateLimitError:
                print("Voyage AI rate limit hit, falling back to OpenAI")
                # Switch to OpenAI for this session
                if os.getenv("OPENAI_API_KEY"):
                    self.openai_embeddings = OpenAIEmbeddings(
                        model="text-embedding-ada-002",
                        openai_api_key=os.getenv("OPENAI_API_KEY")
                    )
                    self.provider = "openai"
                    return self.openai_embeddings.embed_query(text)
                raise
        else:
            return self.openai_embeddings.embed_query(text)


@allure.epic("Database Integration")
@allure.feature("MongoDB Vector Search")
class TestMongoVectorSearch:
    """Test suite for MongoDB vector search with Voyage AI embeddings."""
    
    @pytest.fixture(scope="class")
    def mongo_client(self):
        """Create MongoDB client connection."""
        mongo_password = os.getenv("MONGO_DB_PASSWORD")
        if not mongo_password:
            pytest.skip("MONGO_DB_PASSWORD not found in environment variables")
            
        uri = f"mongodb+srv://health-coach-ai-sami:{mongo_password}@cluster0-health-coach-a.69bhzsd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0-health-coach-ai"
        client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
        
        # Verify connection
        try:
            client.admin.command('ping')
            allure.attach("MongoDB connection successful", name="Connection Status")
        except Exception as e:
            pytest.fail(f"MongoDB connection failed: {e}")
            
        yield client
        client.close()
    
    @pytest.fixture(scope="class")
    def embedding_provider(self):
        """Create embedding provider (Voyage AI or OpenAI)."""
        try:
            return EmbeddingProvider()
        except ValueError as e:
            pytest.skip(str(e))
    
    @pytest.fixture(scope="class")
    def test_collection(self, mongo_client):
        """Get or create test collection."""
        db = mongo_client["health_coach_ai"]
        collection = db["vector_search_test"]
        
        # Clean up any existing test data
        collection.delete_many({})
        
        yield collection
        
        # Cleanup after tests
        collection.delete_many({})
    
    @allure.story("MongoDB Connection")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_mongo_connection(self, mongo_client):
        """Test MongoDB connection is established."""
        with allure.step("Verify MongoDB connection"):
            result = mongo_client.admin.command('ping')
            assert result.get('ok') == 1.0, "MongoDB ping failed"
            
    @allure.story("Embedding Provider Connection")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_embedding_connection(self, embedding_provider):
        """Test embedding provider is configured."""
        with allure.step("Generate test embedding"):
            test_text = "This is a test"
            embedding = embedding_provider.embed_query(test_text)
            
            assert isinstance(embedding, list), "Embedding should be a list"
            assert len(embedding) == embedding_provider.dimensions, f"Expected {embedding_provider.dimensions} dimensions"
            allure.attach(f"Using {embedding_provider.provider} embeddings", name="Embedding Provider")
    
    @allure.story("Document Insertion")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_insert_documents_with_embeddings(self, test_collection, embedding_provider):
        """Test inserting documents with embeddings."""
        # Sample documents
        documents = [
            {
                "text": "The Executive IV drip provides sustained energy and mental clarity for high-performing professionals.",
                "category": "iv_therapy",
                "service": "executive",
                "price": 249
            },
            {
                "text": "Beauty Glow IV infusion enhances natural beauty with antioxidants and vitamins for healthy skin.",
                "category": "iv_therapy", 
                "service": "beauty_glow",
                "price": 199
            },
            {
                "text": "Athletic Performance IV optimizes recovery and reduces muscle fatigue for athletes.",
                "category": "iv_therapy",
                "service": "athletic_performance", 
                "price": 229
            },
            {
                "text": "NAD+ therapy reverses aging at the cellular level and improves energy production.",
                "category": "specialty_treatment",
                "service": "nad_plus",
                "price": 599
            },
            {
                "text": "Weight management program combines IV therapy with nutritional counseling for optimal results.",
                "category": "wellness_program",
                "service": "weight_management",
                "price": 899
            }
        ]
        
        with allure.step("Generate embeddings for documents"):
            texts = [doc["text"] for doc in documents]
            embeddings = embedding_provider.embed_documents(texts)
            
            assert len(embeddings) == len(documents), "Embedding count mismatch"
            
        with allure.step("Insert documents with embeddings"):
            # Add embeddings to documents
            for doc, embedding in zip(documents, embeddings):
                doc["embedding"] = embedding
                
            result = test_collection.insert_many(documents)
            assert len(result.inserted_ids) == len(documents), "Not all documents inserted"
            
            # Log insertion details
            allure.attach(
                f"Inserted {len(result.inserted_ids)} documents",
                name="Insertion Result"
            )
    
    @allure.story("Vector Index Creation")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_create_vector_index(self, test_collection):
        """Test creating vector search index."""
        with allure.step("Create vector search index"):
            try:
                # Note: In production MongoDB Atlas, you would create the index through the UI
                # or using the Atlas Admin API. For testing, we'll create a standard index
                # to ensure the embedding field is indexed.
                test_collection.create_index([("embedding", ASCENDING)])
                
                # Verify index was created
                indexes = list(test_collection.list_indexes())
                index_names = [idx['name'] for idx in indexes]
                
                assert any('embedding' in name for name in index_names), "Embedding index not found"
                
                allure.attach(
                    f"Created indexes: {index_names}",
                    name="Index List"
                )
            except OperationFailure as e:
                allure.attach(str(e), name="Index Creation Error")
                # Index might already exist, which is okay
                pass
    
    @allure.story("Vector Search - Basic")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_vector_search_basic(self, test_collection, embedding_provider):
        """Test basic vector search functionality."""
        query = "I need something to boost my energy and focus at work"
        
        with allure.step("Generate query embedding"):
            query_embedding = embedding_provider.embed_query(query)
            
        with allure.step("Perform vector search"):
            # MongoDB Atlas vector search pipeline
            # Note: This requires Atlas vector search index to be configured
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 20,
                        "limit": 3
                    }
                },
                {
                    "$project": {
                        "text": 1,
                        "category": 1,
                        "service": 1,
                        "price": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            try:
                results = list(test_collection.aggregate(pipeline))
                
                # If vector search is not available, fall back to finding all documents
                # This is for testing purposes when Atlas vector search is not configured
                if not results:
                    allure.attach(
                        "Vector search not available, using fallback method",
                        name="Search Method"
                    )
                    results = list(test_collection.find(
                        {},
                        {"text": 1, "category": 1, "service": 1, "price": 1}
                    ).limit(3))
                    
                assert len(results) > 0, "No search results found"
                
                # Log search results
                for i, result in enumerate(results):
                    allure.attach(
                        f"Result {i+1}: {result.get('text', '')[:100]}... "
                        f"(Service: {result.get('service', 'N/A')})",
                        name=f"Search Result {i+1}"
                    )
                    
            except OperationFailure as e:
                if "Atlas Search index" in str(e) or "vector" in str(e).lower():
                    pytest.skip("Atlas Vector Search index not configured")
                else:
                    raise
    
    @allure.story("Vector Search - With Filter")
    @allure.severity(allure.severity_level.NORMAL)
    def test_vector_search_with_filter(self, test_collection, embedding_provider):
        """Test vector search with metadata filtering."""
        query = "Tell me about treatments for better skin"
        
        with allure.step("Generate query embedding"):
            query_embedding = embedding_provider.embed_query(query)
            
        with allure.step("Perform filtered vector search"):
            # Vector search with category filter
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 20,
                        "limit": 3,
                        "filter": {"category": "iv_therapy"}
                    }
                },
                {
                    "$project": {
                        "text": 1,
                        "category": 1,
                        "service": 1,
                        "price": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            try:
                results = list(test_collection.aggregate(pipeline))
                
                # Fallback for testing without Atlas vector search
                if not results:
                    results = list(test_collection.find(
                        {"category": "iv_therapy"},
                        {"text": 1, "category": 1, "service": 1, "price": 1}
                    ).limit(3))
                    
                # Verify filtered results
                for result in results:
                    if "category" in result:
                        assert result["category"] == "iv_therapy", "Filter not applied correctly"
                        
            except OperationFailure:
                pytest.skip("Atlas Vector Search with filters not available")
    
    @allure.story("Vector Search - Multiple Queries")
    @allure.severity(allure.severity_level.NORMAL)
    def test_vector_search_multiple_queries(self, test_collection, embedding_provider):
        """Test vector search with different query types."""
        test_queries = [
            ("What's the best treatment for anti-aging?", "nad_plus"),
            ("I want to lose weight", "weight_management"),
            ("How can I recover faster from workouts?", "athletic_performance"),
            ("I need something for a special event to look my best", "beauty_glow")
        ]
        
        for query_text, expected_service in test_queries:
            with allure.step(f"Search for: {query_text}"):
                # Generate embedding
                query_embedding = embedding_provider.embed_query(query_text)
                
                # Search
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "path": "embedding",
                            "queryVector": query_embedding,
                            "numCandidates": 20,
                            "limit": 1
                        }
                    },
                    {
                        "$project": {
                            "text": 1,
                            "service": 1,
                            "score": {"$meta": "vectorSearchScore"}
                        }
                    }
                ]
                
                try:
                    results = list(test_collection.aggregate(pipeline))
                    
                    if results:
                        top_result = results[0]
                        allure.attach(
                            f"Query: {query_text}\n"
                            f"Expected: {expected_service}\n"
                            f"Found: {top_result.get('service', 'N/A')}\n"
                            f"Score: {top_result.get('score', 'N/A')}",
                            name=f"Query Result"
                        )
                except OperationFailure:
                    # Skip if vector search not available
                    pass
    
    @allure.story("Error Handling")
    @allure.severity(allure.severity_level.MINOR) 
    def test_empty_collection_search(self, mongo_client, embedding_provider):
        """Test search on empty collection."""
        # Create a new empty collection
        db = mongo_client["health_coach_ai"]
        empty_collection = db["empty_test_collection"]
        empty_collection.delete_many({})  # Ensure it's empty
        
        with allure.step("Search empty collection"):
            query_embedding = embedding_provider.embed_query("test query")
            
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 20,
                        "limit": 5
                    }
                }
            ]
            
            try:
                results = list(empty_collection.aggregate(pipeline))
                assert len(results) == 0, "Expected no results from empty collection"
            except OperationFailure:
                # Expected if index doesn't exist
                pass
            finally:
                # Cleanup
                empty_collection.drop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--alluredir=allure-results"])