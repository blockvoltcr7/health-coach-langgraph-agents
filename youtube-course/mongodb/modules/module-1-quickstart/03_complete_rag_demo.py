"""
Module 1.3: Complete RAG Demo
Time: 15 minutes
Goal: Build a complete Q&A system with document chunking and GPT integration
"""

import os
from pymongo import MongoClient
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
mongodb_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongodb_client[os.getenv("MONGODB_DATABASE", "rag_course")]
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Longer document for chunking demonstration
KNOWLEDGE_BASE = """
# MongoDB Vector Search Guide

## Introduction to Vector Search
Vector search is a powerful technique that enables semantic search capabilities in MongoDB. Unlike traditional keyword-based search, vector search understands the meaning and context of your queries. This makes it perfect for building intelligent applications that can answer questions, find similar content, and provide relevant recommendations.

## Setting Up MongoDB Atlas
To get started with vector search, you need a MongoDB Atlas cluster. Atlas is MongoDB's cloud database service that provides built-in vector search capabilities. Here's how to set it up:

1. Create a free MongoDB Atlas account
2. Deploy a free M0 cluster
3. Configure network access and database users
4. Connect to your cluster using the connection string

## Creating Vector Indexes
Vector indexes are essential for performing efficient semantic searches. In MongoDB Atlas, you can create vector indexes through the Atlas UI or programmatically. The index definition specifies the field containing embeddings, the number of dimensions, and the similarity metric (usually cosine similarity).

## Generating Embeddings
Embeddings are numerical representations of text that capture semantic meaning. You can generate embeddings using various models:
- OpenAI's text-embedding-ada-002 (1536 dimensions)
- Voyage AI's voyage-3-large (1024 dimensions)
- Cohere's embed-v3 models
- Open-source models like sentence-transformers

## Implementing RAG Systems
RAG (Retrieval-Augmented Generation) combines vector search with language models to create intelligent Q&A systems. The process involves:
1. Chunking documents into manageable pieces
2. Generating embeddings for each chunk
3. Storing chunks with embeddings in MongoDB
4. Searching for relevant chunks using vector search
5. Passing retrieved context to a language model
6. Generating accurate, contextual responses

## Best Practices
- Choose appropriate chunk sizes (typically 200-500 tokens)
- Add overlap between chunks to preserve context
- Include metadata for filtering and tracking
- Implement fallback search strategies
- Monitor embedding costs and optimize usage
- Use reranking for better relevance
"""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        # Create chunk with metadata
        chunk = {
            "content": chunk_text,
            "chunk_index": len(chunks),
            "start_index": i,
            "word_count": len(chunk_words)
        }
        chunks.append(chunk)
    
    print(f"📄 Created {len(chunks)} chunks from document")
    return chunks

def prepare_knowledge_base():
    """Prepare and store knowledge base with embeddings"""
    collection = db["rag_knowledge_base"]
    
    # Clear existing data
    collection.delete_many({})
    print("🧹 Cleared existing knowledge base")
    
    # Chunk the document
    chunks = chunk_text(KNOWLEDGE_BASE)
    
    # Generate embeddings for all chunks
    texts = [chunk["content"] for chunk in chunks]
    print(f"🧮 Generating embeddings for {len(texts)} chunks...")
    
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    
    # Add embeddings to chunks
    for chunk, embedding_data in zip(chunks, response.data):
        chunk["embedding"] = embedding_data.embedding
    
    # Insert into MongoDB
    result = collection.insert_many(chunks)
    print(f"✅ Inserted {len(result.inserted_ids)} chunks into knowledge base")
    
    return collection

def retrieve_context(collection, query: str, num_chunks: int = 3) -> List[Dict]:
    """Retrieve relevant context using vector search"""
    # Generate query embedding
    query_response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=[query]
    )
    query_embedding = query_response.data[0].embedding
    
    # Vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": num_chunks
            }
        },
        {
            "$project": {
                "content": 1,
                "chunk_index": 1,
                "score": {"$meta": "vectorSearchScore"},
                "_id": 0
            }
        }
    ]
    
    # Execute search
    results = list(collection.aggregate(pipeline))
    
    print(f"\n🔍 Retrieved {len(results)} relevant chunks for: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"   Chunk {i}: Score {result['score']:.4f}")
    
    return results

def generate_answer(query: str, context_chunks: List[Dict]) -> str:
    """Generate answer using GPT with retrieved context"""
    # Combine context
    context = "\n\n---\n\n".join([chunk["content"] for chunk in context_chunks])
    
    # Create prompt
    system_prompt = """You are a helpful MongoDB expert. Answer questions based on the provided context. 
    If the answer cannot be found in the context, say so clearly."""
    
    user_prompt = f"""Context:
{context}

Question: {query}

Please provide a clear and concise answer based on the context above."""
    
    # Generate response
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content

def rag_qa_system(collection, query: str):
    """Complete RAG Q&A pipeline"""
    print(f"\n{'='*60}")
    print(f"💬 Question: {query}")
    print(f"{'='*60}")
    
    # Step 1: Retrieve relevant context
    context_chunks = retrieve_context(collection, query)
    
    # Step 2: Generate answer
    print("\n🤖 Generating answer...")
    answer = generate_answer(query, context_chunks)
    
    print(f"\n📝 Answer:\n{answer}")
    
    return answer

def interactive_demo(collection):
    """Interactive Q&A demonstration"""
    print("\n" + "="*60)
    print("🎯 RAG Q&A SYSTEM DEMO")
    print("="*60)
    
    # Pre-defined questions for demo
    demo_questions = [
        "What is vector search and how does it work?",
        "How do I create a vector index in MongoDB?",
        "What are the best practices for RAG systems?",
        "What embedding models can I use?",
        "How should I chunk my documents?"
    ]
    
    print("\n📋 Demo Questions:")
    for i, q in enumerate(demo_questions, 1):
        print(f"{i}. {q}")
    
    # Process each question
    for question in demo_questions[:3]:  # Demo first 3 questions
        rag_qa_system(collection, question)
        input("\n⏸️  Press Enter to continue...")
    
    # Allow custom questions
    print("\n\n🎤 Try your own question!")
    while True:
        user_question = input("\nYour question (or 'quit'): ").strip()
        if user_question.lower() == 'quit':
            break
        
        if not user_question:
            print("❌ Please enter a question or type 'quit' to exit.")
            continue
        
        try:
            rag_qa_system(collection, user_question)
        except Exception as e:
            print(f"\n❌ Error processing question: {e}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    print("🎓 MongoDB RAG Course - Complete RAG System\n")
    
    try:
        # Step 1: Prepare knowledge base
        print("📚 Preparing knowledge base...")
        collection = prepare_knowledge_base()
        
        # Step 2: Create index reminder
        print("\n⚠️  IMPORTANT: Vector search requires an index in MongoDB Atlas!")
        print("\n📝 To create the index:")
        print("1. Go to MongoDB Atlas > Your Cluster > Atlas Search")
        print("2. Click 'Create Search Index'")
        print("3. Choose 'JSON Editor' and paste this configuration:")
        print("""
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
}""")
        print(f"\n4. Name it: 'vector_index'")
        print(f"5. Select database: '{db.name}' and collection: '{collection.name}'")
        print("6. Click 'Create' and wait 1-2 minutes for it to become active")
        
        input("\n⏸️  Press Enter when index is created and shows 'READY' status...")
        
        # Step 3: Run interactive demo
        interactive_demo(collection)
        
        print("\n\n🎉 Congratulations! You've built a complete RAG system!")
        print("📈 What you've learned:")
        print("   ✅ Document chunking strategies")
        print("   ✅ Embedding generation at scale")
        print("   ✅ Vector search implementation")
        print("   ✅ Context retrieval")
        print("   ✅ LLM integration for answers")
        print("\n🚀 Ready for Module 2: Deep dive into production patterns!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("🔧 Troubleshooting tips:")
        print("   - Check MongoDB connection")
        print("   - Verify vector index exists")
        print("   - Ensure API keys are valid")