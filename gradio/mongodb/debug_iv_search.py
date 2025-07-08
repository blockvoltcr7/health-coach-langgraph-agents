#!/usr/bin/env python3
"""
Debug why Hangover Rescue is not found
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

# The search query used in the code
print("Testing the exact search query from the code...")
print("="*60)

results = list(collection.find({
    "$or": [
        {"content": {"$regex": "IV Therapy", "$options": "i"}},
        {"content": {"$regex": "Drip|Shield|Recovery|Rescue", "$options": "i"}},
        {"metadata.Section": "IV Therapy Treatments"}
    ]
}, {"content": 1, "metadata": 1}).limit(30))

print(f"Found {len(results)} documents")

# Check if Hangover Rescue is in the results
hangover_found = False
for i, r in enumerate(results):
    if "Hangover Rescue" in r['content']:
        hangover_found = True
        print(f"\n✅ FOUND Hangover Rescue at position {i+1}")
        print(f"Metadata: {r['metadata']}")
        break

if not hangover_found:
    print("\n❌ Hangover Rescue NOT found in results")

# Now let's see all the treatments we DID find
print("\n\nAll treatments found:")
print("-"*40)
treatments = set()
for r in results:
    content = r['content']
    lines = content.split('\n')
    for line in lines:
        if line.startswith('### ') and any(keyword in line for keyword in ['Drip', 'Shield', 'Recovery', 'Rescue', 'IV']):
            treatment = line.strip('### ')
            treatments.add(treatment)

for t in sorted(treatments):
    print(f"  - {t}")

client.close()