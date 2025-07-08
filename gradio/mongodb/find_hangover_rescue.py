#!/usr/bin/env python3
"""
Find where Hangover Rescue is stored
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi

load_dotenv()

mongo_password = os.getenv("MONGO_DB_PASSWORD")
uri = f"mongodb+srv://health-coach-ai-sami:{mongo_password}@cluster0-health-coach-a.69bhzsd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0-health-coach-ai"
client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
db = client["health_coach_ai"]
collection = db["medspa_services"]

print("Searching for Hangover Rescue...")
print("="*60)

# Find documents containing Hangover Rescue
hangover_docs = list(collection.find(
    {"content": {"$regex": "Hangover Rescue", "$options": "i"}},
    {"content": 1, "metadata": 1}
))

print(f"Found {len(hangover_docs)} documents containing 'Hangover Rescue'")

for i, doc in enumerate(hangover_docs, 1):
    print(f"\nDocument {i}:")
    print(f"Metadata: {doc.get('metadata', {})}")
    print(f"Content preview: {doc['content'][:300]}...")
    
    # Check what section it's in
    content = doc['content']
    if "## IV Therapy Treatments" in content:
        print("✅ This document contains the IV Therapy Treatments section")
    else:
        print("❌ This document does NOT contain the IV Therapy Treatments section")

client.close()