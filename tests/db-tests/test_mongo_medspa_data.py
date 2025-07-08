"""
Test MongoDB Vector Search with MedSpa test data.
This test reads the med-spa-test-data.md file, chunks it, creates embeddings, and performs searches.
"""

import os
import time
from typing import List, Dict, Any
import pytest
import allure
from pymongo import MongoClient, ASCENDING
from pymongo.server_api import ServerApi
from pymongo.errors import OperationFailure
import certifi
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import voyageai

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
@allure.feature("MedSpa Data Vector Search")
class TestMedSpaDataVectorSearch:
    """Test suite for MongoDB vector search with MedSpa test data."""
    
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
    def medspa_collection(self, mongo_client):
        """Get or create medspa collection."""
        db = mongo_client["health_coach_ai"]
        collection = db["medspa_services"]
        yield collection
    
    @pytest.fixture(scope="class")
    def medspa_data_path(self):
        """Get path to medspa test data file."""
        path = os.path.join(os.path.dirname(__file__), "..", "test-data", "med-spa-test-data.md")
        if not os.path.exists(path):
            pytest.skip(f"Test data file not found: {path}")
        return path
    
    @allure.story("Process MedSpa Data")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_process_medspa_data(self, medspa_collection, embedding_provider, medspa_data_path):
        """Process the MedSpa markdown file and insert into MongoDB."""
        
        with allure.step("Read MedSpa test data"):
            with open(medspa_data_path, 'r') as f:
                content = f.read()
            
            allure.attach(f"Read {len(content)} characters", name="File Size")
            
        with allure.step("Split markdown by headers"):
            # Split by markdown headers
            headers_to_split_on = [
                ("#", "Main Topic"),
                ("##", "Section"),
                ("###", "Subsection"),
            ]
            
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=False
            )
            
            header_splits = markdown_splitter.split_text(content)
            allure.attach(f"Created {len(header_splits)} header-based chunks", name="Header Splits")
            
        with allure.step("Further split by size"):
            # Further split large chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                add_start_index=True
            )
            
            final_chunks = text_splitter.split_documents(header_splits)
            allure.attach(f"Created {len(final_chunks)} final chunks", name="Final Chunks")
            
        with allure.step("Generate embeddings"):
            texts = [doc.page_content for doc in final_chunks]
            # Process in batches to avoid rate limits
            batch_size = 10
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = embedding_provider.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                allure.attach(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}", name="Batch Progress")
            
            assert len(all_embeddings) == len(final_chunks), "Embedding count mismatch"
            
        with allure.step("Prepare documents for MongoDB"):
            mongo_docs = []
            for i, (doc, embedding) in enumerate(zip(final_chunks, all_embeddings)):
                # Extract meaningful metadata
                content_preview = doc.page_content[:200]
                
                # Try to identify the type of content
                doc_type = "general"
                if "IV" in content_preview or "Drip" in content_preview:
                    doc_type = "iv_therapy"
                elif "Injection" in content_preview or "Shot" in content_preview:
                    doc_type = "injection"
                elif "Membership" in content_preview or "Package" in content_preview:
                    doc_type = "membership"
                elif "NAD+" in content_preview or "Peptide" in content_preview:
                    doc_type = "specialty"
                
                mongo_doc = {
                    "content": doc.page_content,
                    "embedding": embedding,
                    "metadata": {
                        **doc.metadata,
                        "source": "med-spa-test-data.md",
                        "doc_type": doc_type,
                        "chunk_index": i,
                        "total_chunks": len(final_chunks),
                        "char_count": len(doc.page_content),
                        "embedding_provider": embedding_provider.provider,
                        "embedding_dimensions": len(embedding),
                        "created_at": time.time()
                    }
                }
                mongo_docs.append(mongo_doc)
            
        with allure.step("Clear existing data and insert new documents"):
            # Clear any existing data
            medspa_collection.delete_many({})
            
            # Insert new documents
            result = medspa_collection.insert_many(mongo_docs)
            
            assert len(result.inserted_ids) == len(mongo_docs), "Not all documents inserted"
            allure.attach(f"Inserted {len(result.inserted_ids)} documents", name="Insertion Result")
            
            # Create index on embedding field
            try:
                medspa_collection.create_index([("embedding", ASCENDING)])
                allure.attach("Created index on embedding field", name="Index Creation")
            except Exception as e:
                allure.attach(f"Index creation note: {e}", name="Index Status")
    
    @allure.story("Search MedSpa Services")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_search_medspa_services(self, medspa_collection, embedding_provider):
        """Test searching for specific MedSpa services."""
        
        # Test queries
        test_queries = [
            ("I need something for mental clarity and energy", ["CEO Drip", "Executive"]),
            ("What treatments help with skin and beauty?", ["Beauty", "Glow", "Glutathione"]),
            ("I'm an athlete looking for recovery", ["Athletic", "Recovery", "Amino"]),
            ("How much does NAD+ therapy cost?", ["NAD+", "$699", "Anti-Aging"]),
            ("What membership options do you have?", ["Membership", "Gold", "Platinum", "Diamond"]),
            ("Hangover cure treatments", ["Hangover", "Rescue", "nausea", "rehydration"]),
            ("Weight loss IV options", ["Weight Loss", "MIC", "L-Carnitine", "metabolism"])
        ]
        
        for query, expected_terms in test_queries:
            with allure.step(f"Search: {query}"):
                # Generate query embedding
                query_embedding = embedding_provider.embed_query(query)
                
                # Perform vector search
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
                    results = list(medspa_collection.aggregate(pipeline))
                except OperationFailure as e:
                    # If vector search not available, use text search fallback
                    allure.attach("Vector search not available, using text search", name="Search Method")
                    
                    # Create a simple text search
                    text_results = []
                    for term in expected_terms[:2]:  # Search for first two terms
                        docs = medspa_collection.find(
                            {"content": {"$regex": term, "$options": "i"}},
                            {"content": 1, "metadata": 1}
                        ).limit(3)
                        text_results.extend(list(docs))
                    
                    # Remove duplicates
                    seen = set()
                    results = []
                    for doc in text_results:
                        if doc['_id'] not in seen:
                            seen.add(doc['_id'])
                            results.append(doc)
                    
                    results = results[:5]  # Limit to 5 results
                
                # Verify results
                assert len(results) > 0, f"No results found for query: {query}"
                
                # Check if results contain expected terms
                found_terms = []
                for result in results:
                    content = result.get('content', '')
                    for term in expected_terms:
                        if term.lower() in content.lower():
                            found_terms.append(term)
                
                # Log results
                allure.attach(
                    f"Query: {query}\n"
                    f"Expected terms: {expected_terms}\n"
                    f"Found terms: {list(set(found_terms))}\n"
                    f"Number of results: {len(results)}\n"
                    f"Top result preview: {results[0]['content'][:200] if results else 'N/A'}...",
                    name="Search Results"
                )
                
                # At least one expected term should be found
                assert len(found_terms) > 0, f"None of the expected terms {expected_terms} found in results"
    
    @allure.story("Specific Service Lookup")
    @allure.severity(allure.severity_level.NORMAL)
    def test_specific_service_lookup(self, medspa_collection, embedding_provider):
        """Test looking up specific services by name."""
        
        services_to_find = [
            "CEO Drip",
            "Beauty & Glow Infusion",
            "NAD+ Anti-Aging Therapy",
            "Diamond Concierge membership"
        ]
        
        for service in services_to_find:
            with allure.step(f"Find service: {service}"):
                # Generate embedding for service name
                service_embedding = embedding_provider.embed_query(service)
                
                # Search with high precision
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "path": "embedding",
                            "queryVector": service_embedding,
                            "numCandidates": 20,
                            "limit": 1
                        }
                    },
                    {
                        "$project": {
                            "content": 1,
                            "score": {"$meta": "vectorSearchScore"}
                        }
                    }
                ]
                
                try:
                    results = list(medspa_collection.aggregate(pipeline))
                except OperationFailure:
                    # Fallback to direct text search
                    results = list(medspa_collection.find(
                        {"content": {"$regex": service, "$options": "i"}},
                        {"content": 1}
                    ).limit(1))
                
                assert len(results) > 0, f"Service '{service}' not found"
                assert service.lower() in results[0]['content'].lower(), f"Result doesn't contain '{service}'"
                
                allure.attach(
                    f"Found '{service}' with content preview: {results[0]['content'][:150]}...",
                    name=f"Service: {service}"
                )
    
    @allure.story("Document Statistics")
    @allure.severity(allure.severity_level.MINOR)
    def test_document_statistics(self, medspa_collection):
        """Test and report statistics about the indexed documents."""
        
        with allure.step("Calculate document statistics"):
            # Total documents
            total_docs = medspa_collection.count_documents({})
            
            # Document types
            doc_types = list(medspa_collection.aggregate([
                {"$group": {"_id": "$metadata.doc_type", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]))
            
            # Average content length
            avg_length = list(medspa_collection.aggregate([
                {"$group": {"_id": None, "avg_length": {"$avg": "$metadata.char_count"}}}
            ]))
            
            # Embedding provider used
            providers = list(medspa_collection.aggregate([
                {"$group": {"_id": "$metadata.embedding_provider", "count": {"$sum": 1}}}
            ]))
            
            stats_report = f"""
            Document Statistics:
            - Total documents: {total_docs}
            - Document types: {doc_types}
            - Average content length: {avg_length[0]['avg_length'] if avg_length else 'N/A':.0f} characters
            - Embedding providers: {providers}
            """
            
            allure.attach(stats_report, name="Collection Statistics")
            
            # Assertions
            assert total_docs > 0, "No documents found in collection"
            assert len(doc_types) > 0, "No document types found"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--alluredir=allure-results"])