#!/usr/bin/env python3
"""
Final test of chat with all IV treatments
"""

import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mongodb_vector_search_app import chat_with_rag, test_connection

load_dotenv()

# Initialize
print("Initializing connection...")
success, msg, details = test_connection()

# Test with different top_k values
query = "what are all of your IV Therapy Treatments?"

print(f"\nTesting: '{query}'")
print("="*60)

for top_k in [10, 15]:
    print(f"\n\nWith {top_k} context documents:")
    print("-"*40)
    
    response = chat_with_rag(
        message=query,
        history=[],
        collection_name="medspa_services",
        model_name="gpt-4o-mini",
        temperature=0.3,
        top_k=top_k
    )
    
    print("AI Response:")
    print(response)
    
    # Check which treatments are mentioned
    treatments_mentioned = []
    if "CEO Drip" in response:
        treatments_mentioned.append("CEO Drip")
    if "Hangover Rescue" in response:
        treatments_mentioned.append("Hangover Rescue")
    if "Athletic Recovery" in response:
        treatments_mentioned.append("Athletic Recovery")
    if "Beauty & Glow" in response:
        treatments_mentioned.append("Beauty & Glow")
    if "Immunity Shield" in response:
        treatments_mentioned.append("Immunity Shield")
    if "Weight Loss" in response:
        treatments_mentioned.append("Weight Loss")
    if "Migraine Relief" in response:
        treatments_mentioned.append("Migraine Relief")
    
    print(f"\n✅ Treatments mentioned: {treatments_mentioned}")
    print(f"❌ Missing Hangover Rescue: {'Hangover Rescue' not in treatments_mentioned}")