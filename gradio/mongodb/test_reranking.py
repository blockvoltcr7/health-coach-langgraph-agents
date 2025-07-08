#!/usr/bin/env python3
"""
Test Voyage AI reranking for IV therapy search
"""

import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mongodb_vector_search_app import search_context_for_chat, format_context_for_prompt, test_connection

load_dotenv()

# Initialize
print("Initializing connection...")
success, msg, details = test_connection()
print(f"Provider: {details.get('embedding_provider', 'Unknown')}")

# Test query
query = "what are all of your IV Therapy Treatments?"
collection_name = "medspa_services"

print(f"\nQUERY: '{query}'")
print("="*60)

# Test with different top_k values
for top_k in [5, 10, 15]:
    print(f"\n\nTesting with top_k={top_k} (with reranking)")
    print("-"*40)
    
    contexts = search_context_for_chat(query, collection_name, top_k=top_k)
    print(f"Returned {len(contexts)} documents after reranking")
    
    # Check which treatments are found
    treatments_found = set()
    for ctx in contexts:
        content = ctx.get('content', '')
        # Look for treatment names
        if "CEO Drip" in content:
            treatments_found.add("CEO Drip")
        if "Hangover Rescue" in content:
            treatments_found.add("Hangover Rescue")
        if "Athletic Recovery" in content:
            treatments_found.add("Athletic Recovery")
        if "Beauty & Glow" in content:
            treatments_found.add("Beauty & Glow")
        if "Immunity Shield" in content:
            treatments_found.add("Immunity Shield")
        if "Weight Loss Accelerator" in content:
            treatments_found.add("Weight Loss Accelerator")
        if "Migraine Relief" in content:
            treatments_found.add("Migraine Relief")
    
    print(f"Treatments found: {treatments_found}")
    print(f"Missing Hangover Rescue: {'Hangover Rescue' not in treatments_found}")
    
    # Show relevance scores from reranking
    print("\nReranked documents with scores:")
    for i, ctx in enumerate(contexts, 1):
        score = ctx.get('score', 0)
        content_preview = ctx.get('content', '')[:150].replace('\n', ' ')
        print(f"{i}. Score: {score:.3f} - {content_preview}...")

# Now let's specifically look for Hangover Rescue
print("\n\n" + "="*60)
print("SPECIFIC TEST: Finding Hangover Rescue")
print("="*60)

# First, let's see if it's in the database at all
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi

mongo_password = os.getenv("MONGO_DB_PASSWORD")
uri = f"mongodb+srv://health-coach-ai-sami:{mongo_password}@cluster0-health-coach-a.69bhzsd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0-health-coach-ai"
client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
db = client["health_coach_ai"]
collection = db["medspa_services"]

hangover_docs = list(collection.find(
    {"content": {"$regex": "Hangover Rescue", "$options": "i"}},
    {"content": 1, "metadata": 1}
))

print(f"Found {len(hangover_docs)} documents with Hangover Rescue in database")
if hangover_docs:
    print(f"First doc metadata: {hangover_docs[0].get('metadata', {})}")

client.close()