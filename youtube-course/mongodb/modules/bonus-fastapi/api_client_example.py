"""
FastAPI RAG Client Example
Demonstrates how to interact with the RAG API service
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class RAGAPIClient:
    """Client for interacting with the RAG API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("RAG_API_KEY", "test-api-key")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def health_check(self) -> Dict:
        """Check API health"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                return await response.json()
    
    async def generate_embedding(self, text: str, model: str = "voyage-3-large") -> Dict:
        """Generate embedding for text"""
        payload = {
            "text": text,
            "model": model
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/embeddings",
                headers=self.headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"API error: {error}")
                return await response.json()
    
    async def search(
        self,
        query: str,
        collection: str = "documents",
        limit: int = 5,
        filters: Optional[Dict] = None,
        rerank: bool = True
    ) -> Dict:
        """Perform vector search"""
        payload = {
            "query": query,
            "collection": collection,
            "limit": limit,
            "filters": filters,
            "rerank": rerank
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/search",
                headers=self.headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"API error: {error}")
                return await response.json()
    
    async def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        collection: str = "documents",
        model: str = "gpt-3.5-turbo",
        stream: bool = False
    ) -> Dict:
        """Send chat message with RAG"""
        payload = {
            "message": message,
            "conversation_id": conversation_id,
            "collection": collection,
            "model": model,
            "stream": stream
        }
        
        async with aiohttp.ClientSession() as session:
            if stream:
                # Handle streaming response
                async with session.post(
                    f"{self.base_url}/api/v1/chat",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        raise Exception(f"API error: {error}")
                    
                    full_response = ""
                    async for chunk in response.content:
                        chunk_text = chunk.decode('utf-8')
                        full_response += chunk_text
                        print(chunk_text, end='', flush=True)
                    
                    return {"response": full_response}
            else:
                # Handle regular response
                async with session.post(
                    f"{self.base_url}/api/v1/chat",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        raise Exception(f"API error: {error}")
                    return await response.json()
    
    async def ingest_documents(
        self,
        documents: List[Dict[str, str]],
        collection: str = "documents",
        embedding_model: str = "voyage-3-large",
        batch_size: int = 50
    ) -> Dict:
        """Ingest documents into knowledge base"""
        payload = {
            "documents": documents,
            "collection": collection,
            "embedding_model": embedding_model,
            "batch_size": batch_size
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/ingest",
                headers=self.headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"API error: {error}")
                return await response.json()
    
    async def get_conversation(self, conversation_id: str) -> Dict:
        """Retrieve conversation history"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/v1/conversations/{conversation_id}",
                headers=self.headers
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"API error: {error}")
                return await response.json()

async def demonstrate_api_usage():
    """Demonstrate various API operations"""
    print("🚀 RAG API CLIENT DEMONSTRATION\n")
    
    # Initialize client
    client = RAGAPIClient()
    
    # 1. Health check
    print("1️⃣ Health Check")
    print("="*60)
    try:
        health = await client.health_check()
        print(f"Status: {health['status']}")
        print(f"Services: {json.dumps(health['services'], indent=2)}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # 2. Generate embedding
    print("\n2️⃣ Generate Embedding")
    print("="*60)
    try:
        embedding_result = await client.generate_embedding(
            "MongoDB vector search enables semantic search capabilities"
        )
        print(f"Model: {embedding_result['model']}")
        print(f"Dimensions: {embedding_result['dimensions']}")
        print(f"Cached: {embedding_result['cached']}")
        print(f"Processing time: {embedding_result['processing_time']:.3f}s")
        print(f"First 5 values: {embedding_result['embedding'][:5]}")
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
    
    # 3. Vector search
    print("\n3️⃣ Vector Search")
    print("="*60)
    try:
        search_result = await client.search(
            query="How to implement vector search in MongoDB?",
            limit=3,
            rerank=True
        )
        print(f"Total results: {search_result['total_results']}")
        print(f"Cached: {search_result['cached']}")
        print(f"Reranked: {search_result['reranked']}")
        print(f"Processing time: {search_result['processing_time']:.3f}s")
        
        print("\nResults:")
        for i, result in enumerate(search_result['results'], 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   Content: {result['content'][:150]}...")
    except Exception as e:
        print(f"❌ Search failed: {e}")
    
    # 4. Chat interaction
    print("\n4️⃣ Chat Interaction")
    print("="*60)
    try:
        # Non-streaming chat
        print("Non-streaming response:")
        chat_result = await client.chat(
            message="What are the best practices for RAG systems?",
            model="gpt-3.5-turbo"
        )
        print(f"Response: {chat_result['response'][:200]}...")
        print(f"Conversation ID: {chat_result['conversation_id']}")
        print(f"Tokens used: {chat_result['tokens_used']}")
        print(f"Processing time: {chat_result['processing_time']:.3f}s")
        
        # Streaming chat
        print("\n\nStreaming response:")
        print("User: How do I optimize embeddings for cost?")
        print("Assistant: ", end='')
        await client.chat(
            message="How do I optimize embeddings for cost?",
            conversation_id=chat_result['conversation_id'],
            stream=True
        )
        print("\n")
        
    except Exception as e:
        print(f"❌ Chat failed: {e}")
    
    # 5. Document ingestion
    print("\n5️⃣ Document Ingestion")
    print("="*60)
    try:
        documents = [
            {
                "title": "RAG Best Practices",
                "content": "When building RAG systems, focus on chunk size optimization, embedding model selection, and caching strategies."
            },
            {
                "title": "Vector Search Optimization",
                "content": "Optimize vector search by using appropriate index configurations, filtering strategies, and reranking."
            }
        ]
        
        ingestion_result = await client.ingest_documents(
            documents=documents,
            collection="demo_collection"
        )
        print(f"Task ID: {ingestion_result['task_id']}")
        print(f"Status: {ingestion_result['status']}")
        print(f"Document count: {ingestion_result['document_count']}")
        print(f"Message: {ingestion_result['message']}")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")

async def demonstrate_batch_operations():
    """Demonstrate batch operations for efficiency"""
    print("\n\n📦 BATCH OPERATIONS DEMONSTRATION")
    print("="*60)
    
    client = RAGAPIClient()
    
    # Batch embeddings
    texts = [
        "MongoDB Atlas provides cloud database services",
        "Vector search enables semantic search capabilities",
        "RAG systems combine retrieval and generation"
    ]
    
    print("Generating embeddings for multiple texts...")
    start_time = asyncio.get_event_loop().time()
    
    # Concurrent embedding generation
    tasks = [client.generate_embedding(text) for text in texts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    elapsed = asyncio.get_event_loop().time() - start_time
    
    successful = sum(1 for r in results if not isinstance(r, Exception))
    print(f"✅ Generated {successful}/{len(texts)} embeddings in {elapsed:.2f}s")
    print(f"   Average time per embedding: {elapsed/len(texts):.3f}s")

async def demonstrate_error_handling():
    """Demonstrate error handling"""
    print("\n\n⚠️  ERROR HANDLING DEMONSTRATION")
    print("="*60)
    
    client = RAGAPIClient()
    
    # Test with invalid API key
    print("Testing with invalid API key...")
    client.api_key = "invalid-key"
    client.headers["Authorization"] = f"Bearer {client.api_key}"
    
    try:
        await client.search("test query")
    except Exception as e:
        print(f"✅ Expected error caught: {e}")
    
    # Test with invalid input
    print("\nTesting with invalid input...")
    client.api_key = os.getenv("RAG_API_KEY", "test-api-key")
    client.headers["Authorization"] = f"Bearer {client.api_key}"
    
    try:
        await client.generate_embedding("")  # Empty text
    except Exception as e:
        print(f"✅ Expected validation error: {e}")

async def create_interactive_session():
    """Create an interactive chat session"""
    print("\n\n💬 INTERACTIVE CHAT SESSION")
    print("="*60)
    print("Type 'quit' to exit")
    print("-"*60)
    
    client = RAGAPIClient()
    conversation_id = None
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Send to API
            print("Assistant: ", end='', flush=True)
            
            result = await client.chat(
                message=user_input,
                conversation_id=conversation_id,
                stream=True
            )
            
            # Store conversation ID for continuity
            if not conversation_id and 'conversation_id' in result:
                conversation_id = result['conversation_id']
            
            print()  # New line after response
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    print("🎓 MongoDB RAG Course - FastAPI Client Examples\n")
    
    # Run demonstrations
    asyncio.run(demonstrate_api_usage())
    asyncio.run(demonstrate_batch_operations())
    asyncio.run(demonstrate_error_handling())
    
    # Uncomment to run interactive session
    # asyncio.run(create_interactive_session())
    
    print("\n✅ All demonstrations completed!")