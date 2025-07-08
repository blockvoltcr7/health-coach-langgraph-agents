"""
Module 1.2: Your First Vector Search
Time: 10 minutes
Goal: Create collection, generate embeddings, and run semantic search
"""

import os
from pymongo import MongoClient
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
mongodb_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongodb_client[os.getenv("MONGODB_DATABASE", "rag_course")]
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Sample documents about AI concepts
SAMPLE_DOCUMENTS = [
    {
        "title": "What is RAG?",
        "content": "RAG (Retrieval-Augmented Generation) combines the power of retrieval systems with generative AI models. It retrieves relevant information from a knowledge base and uses it to generate more accurate and contextual responses.",
        "category": "concepts"
    },
    {
        "title": "Vector Embeddings Explained",
        "content": "Vector embeddings are numerical representations of text that capture semantic meaning. Similar concepts have vectors that are close together in the embedding space, enabling semantic search.",
        "category": "concepts"
    },
    {
        "title": "MongoDB Atlas Setup",
        "content": "MongoDB Atlas is a cloud database service that supports vector search. You can create a free cluster, build vector indexes, and perform semantic searches on your data.",
        "category": "tutorial"
    },
    {
        "title": "Implementing Semantic Search",
        "content": "Semantic search goes beyond keyword matching. It understands the meaning and context of queries, finding relevant results even when exact words don't match.",
        "category": "tutorial"
    },
    {
        "title": "Voyage AI Benefits",
        "content": "Voyage AI provides specialized embeddings that outperform general-purpose models. Their embeddings are more accurate for domain-specific content and cost 70% less than OpenAI.",
        "category": "comparison"
    }
]

def create_collection():
    """Create MongoDB collection for our documents"""
    collection_name = "quickstart_docs"
    
    # Drop existing collection if it exists
    if collection_name in db.list_collection_names():
        db[collection_name].drop()
        print(f"♻️  Dropped existing '{collection_name}' collection")
    
    # Create new collection
    collection = db[collection_name]
    print(f"✅ Created collection: '{collection_name}'")
    
    return collection

def generate_embeddings(texts):
    """Generate embeddings using OpenAI"""
    print(f"🧮 Generating embeddings for {len(texts)} texts...")
    
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    
    embeddings = [item.embedding for item in response.data]
    print(f"✅ Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")
    
    return embeddings

def insert_documents_with_embeddings(collection):
    """Insert documents with their embeddings"""
    # Extract text for embedding
    texts = [doc["content"] for doc in SAMPLE_DOCUMENTS]
    
    # Generate embeddings
    embeddings = generate_embeddings(texts)
    
    # Add embeddings to documents
    documents_with_embeddings = []
    for doc, embedding in zip(SAMPLE_DOCUMENTS, embeddings):
        doc_with_embedding = doc.copy()
        doc_with_embedding["embedding"] = embedding
        documents_with_embeddings.append(doc_with_embedding)
    
    # Insert into MongoDB
    result = collection.insert_many(documents_with_embeddings)
    print(f"✅ Inserted {len(result.inserted_ids)} documents")
    
    return documents_with_embeddings

def create_vector_index(collection):
    """Create vector search index"""
    index_name = "vector_index"
    
    # Define the index
    index_definition = {
        "name": index_name,
        "type": "vectorSearch",
        "definition": {
            "mappings": {
                "dynamic": True,
                "fields": {
                    "embedding": {
                        "type": "knnVector",
                        "dimensions": 1536,  # OpenAI ada-002 dimensions
                        "similarity": "cosine"
                    }
                }
            }
        }
    }
    
    print(f"📊 Creating vector index: '{index_name}'")
    
    try:
        # Get list of existing search indexes
        existing_indexes = list(collection.list_search_indexes())
        index_exists = any(idx.get("name") == index_name for idx in existing_indexes)
        
        if index_exists:
            print(f"✅ Vector index '{index_name}' already exists")
        else:
            # Create the search index
            collection.create_search_index(index_definition)
            print(f"✅ Created vector index '{index_name}'")
            print("⏳ Waiting 60 seconds for index to become active...")
            import time
            time.sleep(60)
    except Exception as e:
        print(f"⚠️  Could not create index programmatically: {e}")
        print("📝 Please create the index manually in MongoDB Atlas:")
        print(json.dumps(index_definition["definition"], indent=2))
    
    return index_name

def vector_search(collection, query, limit=3):
    """Perform vector search"""
    print(f"\n🔍 Searching for: '{query}'")
    
    # Generate query embedding
    query_embedding = generate_embeddings([query])[0]
    
    # Vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": limit
            }
        },
        {
            "$project": {
                "title": 1,
                "content": 1,
                "category": 1,
                "score": {"$meta": "vectorSearchScore"},
                "_id": 0
            }
        }
    ]
    
    # Execute search
    results = list(collection.aggregate(pipeline))
    
    print(f"\n📊 Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   Category: {result['category']}")
        print(f"   Preview: {result['content'][:100]}...")
    
    return results

def demo_searches(collection):
    """Run demonstration searches"""
    test_queries = [
        "How do I implement RAG?",
        "What makes Voyage AI special?",
        "Setting up semantic search",
        "Cost-effective embedding solutions"
    ]
    
    print("\n" + "="*50)
    print("🎯 DEMONSTRATION SEARCHES")
    print("="*50)
    
    for query in test_queries:
        vector_search(collection, query)
        print("\n" + "-"*50)

if __name__ == "__main__":
    print("🎓 MongoDB RAG Course - First Vector Search\n")
    
    try:
        # Step 1: Create collection
        collection = create_collection()
        
        # Step 2: Insert documents with embeddings
        print("\n" + "="*50 + "\n")
        insert_documents_with_embeddings(collection)
        
        # Step 3: Create vector index (instructions)
        print("\n" + "="*50 + "\n")
        create_vector_index(collection)
        
        # Step 4: Perform searches
        demo_searches(collection)
        
        print("\n🎉 Congratulations! You've built your first vector search!")
        print("🚀 Ready for Module 1.3: Complete RAG System")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("🔧 Check your environment variables and MongoDB connection")