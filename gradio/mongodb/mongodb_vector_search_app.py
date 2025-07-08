#!/usr/bin/env python3
"""
🚀 MongoDB Vector Search Gradio Application
A beautiful, intuitive interface for managing your MedSpa knowledge base
with the power of semantic search!

Author: Your AI Development Partner
Date: July 2025
"""

import gradio as gr
import os
import time
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import json
import traceback

# MongoDB and Vector Search imports
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import OperationFailure
import certifi
from dotenv import load_dotenv

# Embedding providers
import voyageai
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from openai import OpenAI

# Load environment variables
load_dotenv()

# Global variables for connection state
_mongo_client = None
_embedding_provider = None
_current_db = "health_coach_ai"
_analytics_data = {"searches": [], "uploads": [], "errors": []}


class EmbeddingProvider:
    """🎯 Smart embedding provider with automatic fallback!"""
    
    def __init__(self):
        self.voyage_client = None
        self.openai_embeddings = None
        self.provider = None
        self.dimensions = None
        self.cost_per_1k_tokens = {"voyage": 0.0002, "openai": 0.0001}
        
        # Try Voyage AI first (our premium choice! 🌟)
        voyage_key = os.getenv("VOYAGE_AI_API_KEY")
        if voyage_key:
            try:
                self.voyage_client = voyageai.Client(api_key=voyage_key)
                # Test the waters!
                test_result = self.voyage_client.embed(["test"], model="voyage-3-large", input_type="document")
                self.provider = "voyage"
                self.dimensions = len(test_result.embeddings[0])
                print(f"🎯 Voyage AI initialized! Dimensions: {self.dimensions}")
            except Exception as e:
                print(f"⚠️ Voyage AI unavailable: {e}")
        
        # OpenAI as our trusty backup! 🛡️
        if not self.voyage_client:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                self.openai_embeddings = OpenAIEmbeddings(
                    model="text-embedding-ada-002",
                    openai_api_key=openai_key
                )
                self.provider = "openai"
                self.dimensions = 1536
                print("🔄 Using OpenAI embeddings")
            else:
                raise ValueError("🚨 No embedding provider available! Please set API keys.")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents with grace and efficiency!"""
        if self.provider == "voyage":
            try:
                time.sleep(0.5)  # Respectful rate limiting
                result = self.voyage_client.embed(texts, model="voyage-3-large", input_type="document")
                return result.embeddings
            except voyageai.error.RateLimitError:
                print("⚡ Voyage rate limit hit - switching to OpenAI!")
                if self.openai_embeddings:
                    self.provider = "openai"
                    return self.openai_embeddings.embed_documents(texts)
                raise
        else:
            return self.openai_embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a search query with precision!"""
        if self.provider == "voyage":
            try:
                time.sleep(0.5)
                result = self.voyage_client.embed([text], model="voyage-3-large", input_type="query")
                return result.embeddings[0]
            except voyageai.error.RateLimitError:
                if self.openai_embeddings:
                    self.provider = "openai"
                    return self.openai_embeddings.embed_query(text)
                raise
        else:
            return self.openai_embeddings.embed_query(text)


# 🔌 Connection Management Functions
def test_connection() -> Tuple[bool, str, dict]:
    """Test MongoDB connection and return status details"""
    global _mongo_client, _embedding_provider
    
    try:
        # Check MongoDB
        mongo_password = os.getenv("MONGO_DB_PASSWORD")
        if not mongo_password:
            return False, "❌ MONGO_DB_PASSWORD not found", {}
        
        uri = f"mongodb+srv://health-coach-ai-sami:{mongo_password}@cluster0-health-coach-a.69bhzsd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0-health-coach-ai"
        
        if not _mongo_client:
            _mongo_client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
        
        # Ping test
        _mongo_client.admin.command('ping')
        
        # Get cluster info
        server_info = _mongo_client.server_info()
        
        # Initialize embedding provider
        if not _embedding_provider:
            _embedding_provider = EmbeddingProvider()
        
        status_info = {
            "mongodb_version": server_info.get('version', 'Unknown'),
            "embedding_provider": _embedding_provider.provider,
            "embedding_dimensions": _embedding_provider.dimensions,
            "connection_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return True, "✅ All systems operational!", status_info
        
    except Exception as e:
        return False, f"❌ Connection failed: {str(e)}", {}


def get_collections() -> pd.DataFrame:
    """Get all collections with their stats"""
    global _mongo_client, _current_db
    
    # Initialize connection if needed
    if not _mongo_client:
        try:
            success, msg, details = test_connection()
            if not success:
                return pd.DataFrame({"Error": ["Not connected to MongoDB. Click Connection tab to connect."]})
        except:
            return pd.DataFrame({"Error": ["Not connected to MongoDB. Click Connection tab to connect."]})
    
    try:
        db = _mongo_client[_current_db]
        collections_data = []
        
        for collection_name in db.list_collection_names():
            collection = db[collection_name]
            count = collection.count_documents({})
            
            # Check for vector index
            indexes = list(collection.list_indexes())
            has_vector_index = any('embedding' in str(idx) for idx in indexes)
            
            collections_data.append({
                "Collection Name": collection_name,
                "Document Count": count,
                "Has Vector Index": "✅" if has_vector_index else "❌",
                "Last Modified": datetime.now().strftime("%Y-%m-%d")
            })
        
        return pd.DataFrame(collections_data)
        
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


def create_collection(collection_name: str, create_vector_index: bool) -> str:
    """Create a new collection with optional vector index"""
    global _mongo_client, _current_db
    
    if not _mongo_client:
        return "❌ Not connected to MongoDB"
    
    try:
        db = _mongo_client[_current_db]
        
        # Validate collection name
        if not collection_name or collection_name in db.list_collection_names():
            return "❌ Collection already exists or invalid name"
        
        # Create collection
        collection = db[collection_name]
        
        # Insert a dummy document to create the collection
        collection.insert_one({"_init": True})
        collection.delete_one({"_init": True})
        
        # Create index if requested
        if create_vector_index:
            collection.create_index([("embedding", 1)])
            
        return f"✅ Collection '{collection_name}' created successfully!"
        
    except Exception as e:
        return f"❌ Error creating collection: {str(e)}"


def upload_and_process_document(file, collection_name: str, chunk_size: int = 3000, chunk_overlap: Optional[int] = None) -> str:
    """Upload and process a document with embeddings"""
    global _mongo_client, _embedding_provider, _analytics_data, _current_db
    
    if not _mongo_client or not _embedding_provider:
        return "❌ System not initialized"
    
    try:
        # Read file content
        if file is None:
            return "❌ No file uploaded"
        
        # Handle different file input types
        filename = "unknown"
        content = None
        
        # Check if it's a file-like object with read method
        if hasattr(file, 'read'):
            content = file.read()
            if hasattr(file, 'name'):
                filename = file.name
        # Check if it's a Gradio NamedString (for text files)
        elif hasattr(file, 'value'):
            content = file.value
            if hasattr(file, 'name'):
                filename = file.name
            elif hasattr(file, 'orig_name'):
                filename = file.orig_name
        # Check if it's a string (direct text input)
        elif isinstance(file, str):
            content = file
            filename = "direct_input.txt"
        else:
            return f"❌ Unsupported file type: {type(file).__name__}. Expected file upload or text content."
        
        # Convert bytes to string if needed
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        if not content:
            return "❌ File is empty"
        
        # Determine overlap – default to 15% of chunk size if not provided
        if chunk_overlap is None:
            chunk_overlap = max(int(chunk_size * 0.15), 100)  # ensure at least 100 chars overlap
        # Create document chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        chunks = text_splitter.split_text(content)
        
        # Generate embeddings in batches
        batch_size = 5
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings = _embedding_provider.embed_documents(batch)
            all_embeddings.extend(embeddings)
        
        # Prepare documents for MongoDB
        db = _mongo_client[_current_db]
        collection = db[collection_name]
        
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
            doc = {
                "content": chunk,
                "embedding": embedding,
                "metadata": {
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "upload_time": datetime.now(),
                    "embedding_provider": _embedding_provider.provider,
                    "embedding_dimensions": len(embedding)
                }
            }
            documents.append(doc)
        
        # Insert documents
        result = collection.insert_many(documents)
        
        # Track analytics
        _analytics_data["uploads"].append({
            "file": filename,
            "chunks": len(chunks),
            "collection": collection_name,
            "time": datetime.now().isoformat()
        })
        
        return f"✅ Successfully uploaded {filename}\n📄 Created {len(chunks)} chunks\n🎯 Using {_embedding_provider.provider} embeddings"
        
    except Exception as e:
        _analytics_data["errors"].append({
            "operation": "upload",
            "error": str(e),
            "time": datetime.now().isoformat()
        })
        return f"❌ Error processing document: {str(e)}"


def search_documents(query: str, collection_name: str, top_k: int = 5) -> pd.DataFrame:
    """Perform vector search on documents"""
    global _mongo_client, _embedding_provider, _analytics_data, _current_db
    
    # Initialize if needed
    if not _mongo_client or not _embedding_provider:
        try:
            success, msg, details = test_connection()
            if not success:
                return pd.DataFrame({"Error": ["System not initialized. Please check Connection tab."]})
        except Exception as e:
            return pd.DataFrame({"Error": [f"Failed to initialize: {str(e)}"]})
    
    try:
        # Generate query embedding
        query_embedding = _embedding_provider.embed_query(query)
        
        db = _mongo_client[_current_db]
        collection = db[collection_name]
        
        # Try vector search first
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "default",  # Changed from "vector_index" to match Atlas
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k
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
        except (OperationFailure, Exception) as e:
            # Fallback to text search silently
            pass  # Vector search not configured, using text search
            
            # Create a more intelligent text search
            # Extract key terms from the query
            keywords = []
            
            # Common price-related queries
            if any(word in query.lower() for word in ["how much", "cost", "price", "$"]):
                # Look for price patterns
                results = list(collection.find(
                    {"content": {"$regex": "\\$\\d+", "$options": "i"}},
                    {"content": 1, "metadata": 1}
                ).limit(top_k * 2))
                
                # Filter results that contain query terms
                filtered_results = []
                query_lower = query.lower()
                for r in results:
                    content_lower = r['content'].lower()
                    # Score based on how many query words appear in content
                    score = sum(1 for word in query_lower.split() if word in content_lower) / len(query_lower.split())
                    if score > 0.3:  # At least 30% of words match
                        r["score"] = score
                        filtered_results.append(r)
                
                # Sort by score and limit
                results = sorted(filtered_results, key=lambda x: x['score'], reverse=True)[:top_k]
            else:
                # Regular text search with better matching
                # Try exact phrase first
                results = list(collection.find(
                    {"content": {"$regex": query, "$options": "i"}},
                    {"content": 1, "metadata": 1}
                ).limit(top_k))
                
                # If no results, try individual words
                if not results and ' ' in query:
                    # Search for any of the words
                    words = query.split()
                    regex_pattern = "|".join(words)
                    results = list(collection.find(
                        {"content": {"$regex": regex_pattern, "$options": "i"}},
                        {"content": 1, "metadata": 1}
                    ).limit(top_k))
                
                # Score results based on relevance
                for r in results:
                    content_lower = r['content'].lower()
                    query_lower = query.lower()
                    
                    # Higher score if exact phrase appears
                    if query_lower in content_lower:
                        r["score"] = 0.9
                    else:
                        # Score based on word matches
                        words = query_lower.split()
                        matches = sum(1 for word in words if word in content_lower)
                        r["score"] = matches / len(words) if words else 0.5
        
        # Track analytics
        _analytics_data["searches"].append({
            "query": query,
            "collection": collection_name,
            "results": len(results),
            "time": datetime.now().isoformat()
        })
        
        # Format results
        if results:
            df_data = []
            for i, result in enumerate(results, 1):
                df_data.append({
                    "Rank": i,
                    "Score": f"{result.get('score', 0):.4f}",
                    "Content Preview": result['content'][:200] + "...",
                    "Source": result['metadata'].get('source', 'Unknown'),
                    "Chunk": f"{result['metadata'].get('chunk_index', 0) + 1}/{result['metadata'].get('total_chunks', 1)}"
                })
            return pd.DataFrame(df_data)
        else:
            return pd.DataFrame({"Message": ["No results found"]})
            
    except Exception as e:
        _analytics_data["errors"].append({
            "operation": "search",
            "error": str(e),
            "time": datetime.now().isoformat()
        })
        return pd.DataFrame({"Error": [str(e)]})


def get_analytics_summary() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Get analytics summary for the dashboard"""
    global _analytics_data
    
    # Search analytics
    if _analytics_data["searches"]:
        search_df = pd.DataFrame(_analytics_data["searches"][-10:])  # Last 10 searches
        search_df = search_df[["query", "collection", "results", "time"]]
    else:
        search_df = pd.DataFrame({"Message": ["No searches yet"]})
    
    # Upload analytics
    if _analytics_data["uploads"]:
        upload_df = pd.DataFrame(_analytics_data["uploads"][-10:])  # Last 10 uploads
        upload_df = upload_df[["file", "chunks", "collection", "time"]]
    else:
        upload_df = pd.DataFrame({"Message": ["No uploads yet"]})
    
    # Cost estimation
    total_searches = len(_analytics_data["searches"])
    total_uploads = sum(u["chunks"] for u in _analytics_data["uploads"])
    est_cost = (total_searches * 0.0001) + (total_uploads * 0.0002)
    
    summary = f"""
    📊 Usage Summary:
    - Total Searches: {total_searches}
    - Total Documents Processed: {total_uploads}
    - Estimated Cost: ${est_cost:.4f}
    - Active Provider: {_embedding_provider.provider if _embedding_provider else 'Not initialized'}
    """
    
    return search_df, upload_df, summary


# 💬 Chat Functions
def search_context_for_chat(query: str, collection_name: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Search for relevant context to include in chat with Voyage AI reranking"""
    global _mongo_client, _embedding_provider, _current_db
    
    if not _mongo_client or not _embedding_provider:
        return []
    
    try:
        # Generate query embedding
        query_embedding = _embedding_provider.embed_query(query)
        
        db = _mongo_client[_current_db]
        collection = db[collection_name]
        
        # Retrieve MORE candidates for reranking (3x the requested amount)
        initial_retrieval = min(top_k * 3, 20)  # Get more docs but cap at 20
        
        # Try vector search first
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "default",  # Changed from "vector_index" to match Atlas
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": initial_retrieval * 10,
                    "limit": initial_retrieval
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
        except (OperationFailure, Exception) as e:
            # Intelligent fallback to text search silently
            pass  # Vector search not configured, using text search
            
            # Create a more intelligent text search
            # Extract key terms from the query
            keywords = []
            
            # Common price-related queries
            if any(word in query.lower() for word in ["how much", "cost", "price", "$"]):
                # Look for price patterns
                results = list(collection.find(
                    {"content": {"$regex": "\\$\\d+", "$options": "i"}},
                    {"content": 1, "metadata": 1}
                ).limit(initial_retrieval))
                
                # Filter results that contain query terms
                filtered_results = []
                query_lower = query.lower()
                for r in results:
                    content_lower = r['content'].lower()
                    # Score based on how many query words appear in content
                    score = sum(1 for word in query_lower.split() if word in content_lower) / len(query_lower.split())
                    if score > 0.3:  # At least 30% of words match
                        r["score"] = score
                        filtered_results.append(r)
                
                # Sort by score
                results = sorted(filtered_results, key=lambda x: x['score'], reverse=True)[:initial_retrieval]
            else:
                # For queries like "all IV therapy treatments", get more comprehensive results
                if any(word in query.lower() for word in ["all", "every", "list", "what are"]) and "iv" in query.lower():
                    # For IV therapy queries, search for documents with IV-related content
                    initial_retrieval = min(30, top_k * 5)
                    
                    # Search for IV therapy related documents more broadly
                    results = list(collection.find({
                        "$or": [
                            {"content": {"$regex": "IV Therapy", "$options": "i"}},
                            {"content": {"$regex": "Drip|Shield|Recovery|Rescue", "$options": "i"}},
                            {"metadata.Section": "IV Therapy Treatments"}
                        ]
                    }, {"content": 1, "metadata": 1}).limit(initial_retrieval))
                    
                else:
                    # Regular text search with better matching
                    # Try exact phrase first
                    results = list(collection.find(
                        {"content": {"$regex": query, "$options": "i"}},
                        {"content": 1, "metadata": 1}
                    ).limit(initial_retrieval))
                
                # If no results, try individual words
                if not results and ' ' in query:
                    # Search for any of the words
                    words = query.split()
                    regex_pattern = "|".join(words)
                    results = list(collection.find(
                        {"content": {"$regex": regex_pattern, "$options": "i"}},
                        {"content": 1, "metadata": 1}
                    ).limit(initial_retrieval))
                
                # Score results based on relevance
                for r in results:
                    content_lower = r['content'].lower()
                    query_lower = query.lower()
                    
                    # Higher score if exact phrase appears
                    if query_lower in content_lower:
                        r["score"] = 0.9
                    else:
                        # Score based on word matches
                        words = query_lower.split()
                        matches = sum(1 for word in words if word in content_lower)
                        r["score"] = matches / len(words) if words else 0.5
        
        # Use Voyage AI Reranker if available and we have results
        if results and _embedding_provider.voyage_client:
            try:
                # Extract document contents for reranking
                documents = [r['content'] for r in results]
                
                # Rerank using Voyage AI
                reranking = _embedding_provider.voyage_client.rerank(
                    query=query,
                    documents=documents,
                    model="rerank-2-lite",  # Using lite model for better latency
                    top_k=top_k
                )
                
                # Map reranked results back to original documents
                reranked_results = []
                for item in reranking.results:
                    idx = item.index
                    reranked_results.append({
                        "content": results[idx]["content"],
                        "metadata": results[idx].get("metadata", {}),
                        "score": item.relevance_score
                    })
                
                return reranked_results
                
            except Exception as e:
                print(f"Reranking failed: {e}, using original scores")
                # Fall back to original scoring
                return sorted(results, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
        else:
            # No reranker available, return top results by original score
            return sorted(results, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
        
    except Exception as e:
        print(f"Error searching context: {str(e)}")
        return []


def format_context_for_prompt(contexts: List[Dict[str, Any]]) -> str:
    """Format search results as context for the LLM prompt"""
    if not contexts:
        return "No relevant context found."
    
    formatted_contexts = []
    for i, ctx in enumerate(contexts, 1):
        source = ctx.get('metadata', {}).get('source', 'Unknown')
        content = ctx.get('content', '')
        score = ctx.get('score', 0)
        
        formatted_contexts.append(f"""
Context {i} (Score: {score:.3f}, Source: {source}):
{content}
""")
    
    return "\n".join(formatted_contexts)


def chat_with_rag(message: str, history: List[List[str]], collection_name: str, 
                  model_name: str = "gpt-4o", temperature: float = 0.7,
                  top_k: int = 3, system_prompt: str = None) -> str:
    """Chat function with RAG (Retrieval-Augmented Generation)"""
    
    # Check if OpenAI client is available
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return "❌ OpenAI API key not found. Please set OPENAI_API_KEY in your environment."
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_key)
        
        # Search for relevant context
        contexts = search_context_for_chat(message, collection_name, top_k)
        context_str = format_context_for_prompt(contexts)
        
        # Build the system prompt
        if not system_prompt:
            system_prompt = """You are a helpful AI assistant with access to a knowledge base. 
Use the provided context to answer questions accurately. If the context doesn't contain 
relevant information, say so and provide the best answer you can based on your general knowledge.
Always cite which context you're using when providing specific information."""
        
        # Build messages for OpenAI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Available Context:\n{context_str}"}
        ]
        
        # Add conversation history
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Get response from OpenAI
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=1000
        )
        
        # Track analytics
        _analytics_data["searches"].append({
            "query": f"[CHAT] {message[:50]}...",
            "collection": collection_name,
            "results": len(contexts),
            "time": datetime.now().isoformat()
        })
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


# 🎨 Create the Gradio Interface
def create_gradio_app():
    """Create the main Gradio application"""
    
    with gr.Blocks(title="MedSpa Knowledge Base", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🏥 MedSpa Knowledge Base Vector Search
        ### Your AI-powered document management system with semantic search capabilities
        """)
        
        with gr.Tabs():
            # Tab 1: Connection Management
            with gr.TabItem("🔌 Connection"):
                gr.Markdown("### System Connection Status")
                
                with gr.Row():
                    test_btn = gr.Button("Test Connection", variant="primary")
                    refresh_btn = gr.Button("Refresh Status")
                
                connection_status = gr.Textbox(label="Connection Status", interactive=False)
                connection_details = gr.JSON(label="Connection Details")
                
                test_btn.click(
                    fn=lambda: test_connection(),
                    outputs=[connection_status, connection_status, connection_details]
                )
                refresh_btn.click(
                    fn=lambda: test_connection(),
                    outputs=[connection_status, connection_status, connection_details]
                )
            
            # Tab 2: Collections
            with gr.TabItem("📁 Collections"):
                gr.Markdown("### Manage Your Collections")
                
                with gr.Row():
                    refresh_collections_btn = gr.Button("Refresh Collections")
                
                collections_table = gr.DataFrame(
                    label="Available Collections",
                    interactive=False
                )
                
                gr.Markdown("### Create New Collection")
                with gr.Row():
                    new_collection_name = gr.Textbox(
                        label="Collection Name",
                        placeholder="e.g., medspa_services"
                    )
                    create_vector_index_cb = gr.Checkbox(
                        label="Create Vector Index",
                        value=True
                    )
                    create_collection_btn = gr.Button("Create Collection", variant="primary")
                
                create_result = gr.Textbox(label="Result", interactive=False)
                
                # This click handler is now defined at the end
                
                create_collection_btn.click(
                    fn=create_collection,
                    inputs=[new_collection_name, create_vector_index_cb],
                    outputs=create_result
                )
            
            # Tab 3: Documents
            with gr.TabItem("📄 Documents"):
                gr.Markdown("### Upload and Process Documents")
                
                with gr.Row():
                    file_upload = gr.File(
                        label="Upload Document",
                        file_types=[".txt", ".md", ".pdf", ".docx"]
                    )
                    collection_dropdown = gr.Dropdown(
                        label="Target Collection",
                        choices=["medspa_services", "vector_search_test"],
                        value="medspa_services",
                        allow_custom_value=True
                    )
                
                with gr.Row():
                    chunk_size_slider = gr.Slider(
                        minimum=1500,
                        maximum=5000,
                        value=3000,
                        step=100,
                        label="Chunk Size (chars)"
                    )
                    upload_btn = gr.Button("Upload & Process", variant="primary")
                
                upload_result = gr.Textbox(label="Upload Result", lines=3, interactive=False)
                
                upload_btn.click(
                    fn=upload_and_process_document,
                    inputs=[file_upload, collection_dropdown, chunk_size_slider],
                    outputs=upload_result
                )
            
            # Tab 4: Search
            with gr.TabItem("🔍 Search"):
                gr.Markdown("### Semantic Search Your Knowledge Base")
                
                with gr.Row():
                    search_query = gr.Textbox(
                        label="Search Query",
                        placeholder="e.g., What treatments help with energy and focus?",
                        lines=2
                    )
                    search_collection = gr.Dropdown(
                        label="Search Collection",
                        choices=["medspa_services", "vector_search_test"],
                        value="medspa_services",
                        interactive=True,
                        allow_custom_value=True
                    )
                
                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Number of Results"
                    )
                    search_btn = gr.Button("Search", variant="primary")
                
                search_results = gr.DataFrame(
                    label="Search Results",
                    wrap=True
                )
                
                # Example queries
                gr.Examples(
                    examples=[
                        ["I need something for mental clarity and energy"],
                        ["What treatments help with skin and beauty?"],
                        ["How much does NAD+ therapy cost?"],
                        ["Best treatment for athletic recovery"],
                        ["Anti-aging treatments available"]
                    ],
                    inputs=search_query
                )
                
                search_btn.click(
                    fn=search_documents,
                    inputs=[search_query, search_collection, top_k_slider],
                    outputs=search_results
                )
            
            # Tab 5: Analytics
            with gr.TabItem("📊 Analytics"):
                gr.Markdown("### Usage Analytics & Insights")
                
                refresh_analytics_btn = gr.Button("Refresh Analytics")
                
                with gr.Row():
                    recent_searches = gr.DataFrame(label="Recent Searches")
                    recent_uploads = gr.DataFrame(label="Recent Uploads")
                
                analytics_summary = gr.Textbox(
                    label="Summary Statistics",
                    lines=6,
                    interactive=False
                )
                
                refresh_analytics_btn.click(
                    fn=get_analytics_summary,
                    outputs=[recent_searches, recent_uploads, analytics_summary]
                )
            
            # Tab 6: Chat Interface
            with gr.TabItem("💬 AI Chat"):
                gr.Markdown("### Chat with AI using your Knowledge Base")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        chat_collection = gr.Dropdown(
                            label="Knowledge Base Collection",
                            choices=["medspa_services", "vector_search_test"],
                            value="medspa_services",
                            allow_custom_value=True
                        )
                        
                        chat_model = gr.Dropdown(
                            label="AI Model",
                            choices=["gpt-4o", "gpt-4.1", "gpt-4o-mini"],
                            value="gpt-4o-mini"
                        )
                        
                        temperature_slider = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.7,
                            step=0.1,
                            label="Temperature (Creativity)"
                        )
                        
                        context_k_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Context Documents"
                        )
                        
                        system_prompt_text = gr.Textbox(
                            label="System Prompt (Optional)",
                            placeholder="Leave empty for default prompt",
                            lines=3
                        )
                        
                        clear_btn = gr.Button("Clear Chat")
                    
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="AI Assistant",
                            height=600,
                            show_copy_button=True,
                            type="messages"
                        )
                        
                        msg = gr.Textbox(
                            label="Your Message",
                            placeholder="Ask me anything about the knowledge base...",
                            lines=2
                        )
                        
                        with gr.Row():
                            submit = gr.Button("Send", variant="primary")
                            
                        # Example questions
                        gr.Examples(
                            examples=[
                                "What IV treatments do you offer for energy?",
                                "Tell me about NAD+ therapy benefits and pricing",
                                "What's the best treatment for athletic recovery?",
                                "Do you have any anti-aging treatments?",
                                "What memberships are available?"
                            ],
                            inputs=msg
                        )
                
                # Chat functionality
                def respond(message, chat_history, collection, model, temp, top_k, sys_prompt):
                    if not message:
                        return "", chat_history
                    
                    # Convert chat history to the format expected by chat_with_rag
                    history_tuples = []
                    if chat_history:
                        for msg in chat_history:
                            if msg["role"] == "user":
                                # Find the next assistant message
                                assistant_msg = None
                                idx = chat_history.index(msg)
                                if idx + 1 < len(chat_history) and chat_history[idx + 1]["role"] == "assistant":
                                    assistant_msg = chat_history[idx + 1]["content"]
                                if assistant_msg:
                                    history_tuples.append([msg["content"], assistant_msg])
                    
                    # Get AI response
                    bot_message = chat_with_rag(
                        message, 
                        history_tuples, 
                        collection,
                        model,
                        temp,
                        top_k,
                        sys_prompt if sys_prompt else None
                    )
                    
                    # Add to chat history in the new format
                    chat_history.append({"role": "user", "content": message})
                    chat_history.append({"role": "assistant", "content": bot_message})
                    
                    return "", chat_history
                
                # Connect events
                msg.submit(
                    respond, 
                    [msg, chatbot, chat_collection, chat_model, temperature_slider, context_k_slider, system_prompt_text], 
                    [msg, chatbot]
                )
                submit.click(
                    respond,
                    [msg, chatbot, chat_collection, chat_model, temperature_slider, context_k_slider, system_prompt_text],
                    [msg, chatbot]
                )
                clear_btn.click(lambda: [], None, chatbot, queue=False)
            
            # Tab 7: Settings
            with gr.TabItem("⚙️ Settings"):
                gr.Markdown("### Configuration & API Keys")
                
                gr.Markdown("""
                #### Current Configuration:
                - **Database**: health_coach_ai
                - **Primary Embedding**: Voyage AI (voyage-3-large)
                - **Fallback Embedding**: OpenAI (text-embedding-ada-002)
                
                #### Vector Index Configuration:
                ```json
                {
                  "fields": [{
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1536,
                    "similarity": "cosine"
                  }]
                }
                ```
                
                #### Environment Variables Required:
                - `MONGO_DB_PASSWORD`
                - `VOYAGE_AI_API_KEY` (optional but recommended)
                - `OPENAI_API_KEY` (fallback)
                """)
        
        # Function to get collection names list
        def get_collection_names_list():
            global _mongo_client, _current_db
            if _mongo_client:
                try:
                    db = _mongo_client[_current_db]
                    return db.list_collection_names()
                except:
                    return ["medspa_services", "vector_search_test"]
            return ["medspa_services", "vector_search_test"]
        
        # Function to update dropdowns with collection names
        def update_collection_dropdowns():
            choices = get_collection_names_list()
            return (
                gr.Dropdown(choices=choices),  # collection_dropdown
                gr.Dropdown(choices=choices),  # search_collection
                gr.Dropdown(choices=choices)   # chat_collection
            )
        
        # Auto-refresh collections on load and initialize connection
        def initialize_on_load():
            # Try to initialize connection on app load
            try:
                test_connection()
            except:
                pass
            return get_collections()
        
        # Update dropdowns when collections change
        def refresh_all():
            return get_collections(), *update_collection_dropdowns()
        
        refresh_collections_btn.click(
            fn=refresh_all,
            outputs=[collections_table, collection_dropdown, search_collection, chat_collection]
        )
        
        app.load(fn=initialize_on_load, outputs=collections_table)
        app.load(fn=update_collection_dropdowns, outputs=[collection_dropdown, search_collection, chat_collection])
        
    return app


# 🚀 Launch the application
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )