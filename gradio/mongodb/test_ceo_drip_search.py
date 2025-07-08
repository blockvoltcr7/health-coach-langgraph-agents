#!/usr/bin/env python3
"""
Test searching for CEO Drip specifically
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi

load_dotenv()

# Connect to MongoDB
mongo_password = os.getenv("MONGO_DB_PASSWORD")
uri = f"mongodb+srv://health-coach-ai-sami:{mongo_password}@cluster0-health-coach-a.69bhzsd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0-health-coach-ai"
client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())

db = client["health_coach_ai"]
collection = db["medspa_services"]

print("Testing search for 'ceo drip'")
print("=" * 50)

# Test 1: Case-insensitive search
print("\n1. Testing case-insensitive regex search:")
results = list(collection.find(
    {"content": {"$regex": "ceo drip", "$options": "i"}},
    {"content": 1}
).limit(5))

print(f"Found {len(results)} results")
for i, r in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(r['content'][:200] + "...")

# Test 2: Search for individual words
print("\n\n2. Testing search for 'ceo' OR 'drip':")
results = list(collection.find(
    {"content": {"$regex": "ceo|drip", "$options": "i"}},
    {"content": 1}
).limit(5))

print(f"Found {len(results)} results")
for i, r in enumerate(results, 1):
    print(f"\nResult {i}:")
    content = r['content'][:200] + "..."
    # Highlight matches
    if 'CEO' in content or 'Drip' in content:
        print("✅", content)
    else:
        print(content)

# Test 3: Check specific document
print("\n\n3. Checking if CEO Drip document exists:")
ceo_drip_doc = collection.find_one(
    {"content": {"$regex": "The CEO Drip.*\\$299", "$options": "i"}}
)


# Test 4: Count all documents with prices
print("\n\n4. Documents with prices:")
price_docs = collection.count_documents({"content": {"$regex": "\\$\\d+", "$options": "i"}})
print(f"Found {price_docs} documents with prices")

# Test 5: List all treatment names
print("\n\n5. All treatments found:")
all_docs = collection.find({}, {"content": 1}).limit(10)
treatments = []
for doc in all_docs:
    content = doc['content']
    # Look for treatment names (usually after ### or **)
    lines = content.split('\n')
    for line in lines:
        if '###' in line or '**' in line:
            if 'Drip' in line or 'Therapy' in line or 'Treatment' in line:
                treatments.append(line.strip('#* '))

print("Treatments found:")
for t in set(treatments):
    if t:
        print(f"  - {t}")

client.close()