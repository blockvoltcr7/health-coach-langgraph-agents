"""
Quick script to verify MongoDB Atlas vector index setup
"""

import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Connect to MongoDB
client = MongoClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("MONGODB_DATABASE", "rag_course")]
collection = db["quickstart_docs"]

print("🔍 Checking MongoDB setup...\n")

# Check connection
print(f"✅ Connected to MongoDB")
print(f"📊 Database: {db.name}")
print(f"📁 Collection: {collection.name}")

# Check documents
doc_count = collection.count_documents({})
print(f"\n📄 Documents in collection: {doc_count}")

if doc_count > 0:
    # Show sample document
    sample = collection.find_one()
    print(f"\n📋 Sample document structure:")
    print(f"  - Fields: {list(sample.keys())}")
    if "embedding" in sample:
        print(f"  - Embedding dimension: {len(sample['embedding'])}")

# Check search indexes
print(f"\n🔍 Checking search indexes...")
try:
    indexes = list(collection.list_search_indexes())
    if indexes:
        print(f"✅ Found {len(indexes)} search index(es):")
        for idx in indexes:
            print(f"\n  Index: {idx.get('name', 'unnamed')}")
            print(f"  Status: {idx.get('status', 'unknown')}")
            print(f"  Type: {idx.get('type', 'unknown')}")
            
            # Check if it's ready
            if idx.get('status') == 'READY':
                print(f"  ✅ Index is active and ready for queries")
            else:
                print(f"  ⏳ Index is still building, please wait...")
    else:
        print("❌ No search indexes found!")
        print("\n📝 To create an index:")
        print("1. Go to MongoDB Atlas > Your Cluster > Search tab")
        print("2. Click 'Create Search Index'")
        print("3. Choose 'JSON Editor' and use this configuration:")
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
}
        """)
        print("4. Name it 'vector_index'")
        print(f"5. Select database: {db.name}, collection: {collection.name}")
        
except Exception as e:
    print(f"⚠️  Could not list search indexes: {e}")
    print("Note: This might be a permissions issue or older MongoDB version")

# Test a simple query
print("\n🧪 Testing basic query...")
try:
    results = list(collection.find().limit(1))
    if results:
        print("✅ Basic queries work fine")
    else:
        print("⚠️  No documents found")
except Exception as e:
    print(f"❌ Query failed: {e}")

print("\n" + "="*50)
print("💡 Next steps:")
print("1. Ensure vector index 'vector_index' exists and is READY")
print("2. Run 02_first_vector_search.py again")
print("3. If still no results, check that embeddings were generated correctly")
print("="*50)