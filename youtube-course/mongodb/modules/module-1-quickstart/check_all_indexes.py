"""
Check all vector indexes across all collections
"""

import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Connect to MongoDB
client = MongoClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("MONGODB_DATABASE", "rag_course")]

print("🔍 Checking all MongoDB collections and indexes...\n")

# List all collections
collections = db.list_collection_names()
print(f"📁 Found {len(collections)} collections in database '{db.name}':\n")

for collection_name in collections:
    collection = db[collection_name]
    doc_count = collection.count_documents({})
    print(f"📄 Collection: {collection_name}")
    print(f"   Documents: {doc_count}")
    
    # Check for vector search indexes
    try:
        indexes = list(collection.list_search_indexes())
        if indexes:
            print(f"   ✅ Vector Indexes:")
            for idx in indexes:
                print(f"      - Name: {idx.get('name')}")
                print(f"        Status: {idx.get('status')}")
                if idx.get('status') != 'READY':
                    print(f"        ⏳ Index is still building...")
        else:
            print(f"   ❌ No vector indexes found")
            
            # Check if documents have embeddings
            sample = collection.find_one()
            if sample and "embedding" in sample:
                print(f"   ⚠️  Documents have embeddings but no vector index!")
                print(f"   💡 Create an index named 'vector_index' for this collection")
    except Exception as e:
        print(f"   ⚠️  Could not check indexes: {e}")
    
    print()

print("\n" + "="*60)
print("📝 To create a missing vector index:")
print("1. Go to MongoDB Atlas > Atlas Search")
print("2. Click 'Create Search Index'")
print("3. Choose the collection that needs an index")
print("4. Use the JSON configuration shown in the course")
print("5. Name it 'vector_index'")
print("="*60)