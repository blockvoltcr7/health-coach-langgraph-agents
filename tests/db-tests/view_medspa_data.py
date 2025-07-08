#!/usr/bin/env python3
"""
Script to view the MedSpa data that was inserted into MongoDB.
"""

import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# MongoDB connection
mongo_password = os.getenv("MONGO_DB_PASSWORD")
if not mongo_password:
    print("❌ MONGO_DB_PASSWORD not found in environment")
    exit(1)

uri = f"mongodb+srv://health-coach-ai-sami:{mongo_password}@cluster0-health-coach-a.69bhzsd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0-health-coach-ai"
client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())

# Test connection
try:
    client.admin.command('ping')
    print("✅ Connected to MongoDB!")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    exit(1)

# Access the collection
db = client["health_coach_ai"]
collection = db["medspa_services"]

# Get document count
count = collection.count_documents({})
print(f"\n📊 Total documents in 'medspa_services' collection: {count}")

if count > 0:
    print("\n📄 Sample documents (first 3):")
    print("-" * 80)
    
    # Get first 3 documents
    for i, doc in enumerate(collection.find().limit(3), 1):
        print(f"\nDocument {i}:")
        print(f"ID: {doc['_id']}")
        print(f"Content preview: {doc['content'][:200]}...")
        print(f"Metadata:")
        for key, value in doc['metadata'].items():
            if key != 'embedding':
                print(f"  - {key}: {value}")
        print(f"Embedding dimensions: {len(doc.get('embedding', []))}")
        print("-" * 40)
    
    # Get statistics
    print("\n📈 Collection Statistics:")
    
    # Document types
    doc_types = list(collection.aggregate([
        {"$group": {"_id": "$metadata.doc_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]))
    
    print("\nDocument Types:")
    for dt in doc_types:
        print(f"  - {dt['_id']}: {dt['count']} documents")
    
    # Check for specific content
    print("\n🔍 Content Search (checking if key services exist):")
    
    services_to_check = [
        "CEO Drip",
        "Beauty & Glow",
        "Athletic Recovery",
        "NAD+ Anti-Aging",
        "Diamond Concierge"
    ]
    
    for service in services_to_check:
        count = collection.count_documents({"content": {"$regex": service, "$options": "i"}})
        print(f"  - '{service}': {count} documents")
    
    # Check indexes
    print("\n🔧 Collection Indexes:")
    for index in collection.list_indexes():
        print(f"  - {index['name']}: {index.get('key', 'N/A')}")
    
    # Show how to create vector search index
    print("\n💡 To enable vector search, create an Atlas Search index with this configuration:")
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
    print("Index name should be: 'vector_index'")
    
else:
    print("\n⚠️  No documents found in the collection")
    print("Run the test to insert data: uv run pytest tests/db-tests/test_mongo_medspa_data.py::TestMedSpaDataVectorSearch::test_process_medspa_data -v")

# Close connection
client.close()
print("\n✅ Done!")