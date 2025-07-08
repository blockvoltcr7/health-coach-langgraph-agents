#!/usr/bin/env python3
"""
Test what documents are retrieved for "all IV Therapy Treatments"
"""

import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mongodb_vector_search_app import search_context_for_chat, format_context_for_prompt, test_connection
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi

load_dotenv()

# Initialize
print("Initializing connection...")
success, msg, details = test_connection()

# Test query
query = "what are all of your IV Therapy Treatments?"
collection_name = "medspa_services"

print(f"\nQUERY: '{query}'")
print("="*60)

# Test with different top_k values
for top_k in [3, 5, 10, 20]:
    print(f"\n\nTesting with top_k={top_k}")
    print("-"*40)
    
    contexts = search_context_for_chat(query, collection_name, top_k=top_k)
    print(f"Found {len(contexts)} documents")
    
    # Check which treatments are found
    treatments_found = []
    for ctx in contexts:
        content = ctx.get('content', '')
        # Look for treatment names
        if "CEO Drip" in content:
            treatments_found.append("CEO Drip")
        if "Hangover Rescue" in content:
            treatments_found.append("Hangover Rescue")
        if "Athletic Recovery" in content:
            treatments_found.append("Athletic Recovery")
        if "Beauty & Glow" in content:
            treatments_found.append("Beauty & Glow")
        if "Immunity Shield" in content:
            treatments_found.append("Immunity Shield")
        if "Weight Loss Accelerator" in content:
            treatments_found.append("Weight Loss Accelerator")
        if "Migraine Relief" in content:
            treatments_found.append("Migraine Relief")
    
    print(f"Treatments found: {set(treatments_found)}")
    
    # Show first 200 chars of each document
    for i, ctx in enumerate(contexts, 1):
        print(f"\nDoc {i}: {ctx.get('content', '')[:200]}...")

# Now let's check how many IV treatments actually exist in the database
print("\n\n" + "="*60)
print("CHECKING TOTAL IV TREATMENTS IN DATABASE")
print("="*60)

mongo_password = os.getenv("MONGO_DB_PASSWORD")
uri = f"mongodb+srv://health-coach-ai-sami:{mongo_password}@cluster0-health-coach-a.69bhzsd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0-health-coach-ai"
client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
db = client["health_coach_ai"]
collection = db["medspa_services"]

# Find all documents containing "IV Therapy"
iv_therapy_docs = list(collection.find(
    {"content": {"$regex": "## IV Therapy Treatments", "$options": "i"}},
    {"content": 1}
))

print(f"Found {len(iv_therapy_docs)} documents with '## IV Therapy Treatments'")

# Extract all unique treatments
all_treatments = set()
for doc in collection.find({}, {"content": 1}):
    content = doc['content']
    lines = content.split('\n')
    for line in lines:
        if line.startswith('### ') and any(keyword in line for keyword in ['Drip', 'Therapy', 'IV', 'Shield', 'Relief', 'Recovery', 'Rescue']):
            treatment = line.strip('### ')
            all_treatments.add(treatment)

print(f"\nAll unique treatments found in database:")
for treatment in sorted(all_treatments):
    print(f"  - {treatment}")

client.close()