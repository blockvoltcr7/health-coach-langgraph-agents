#!/usr/bin/env python3
"""
Verify that documents from MongoDB are being passed to OpenAI
"""

import os
import sys
from dotenv import load_dotenv
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mongodb_vector_search_app import search_context_for_chat, format_context_for_prompt, test_connection

load_dotenv()

# Initialize
print("Initializing connection...")
success, msg, details = test_connection()

# Search for CEO Drip
query = "what is the ceo drip price?"
collection_name = "medspa_services"

print(f"\n1. SEARCHING MONGODB for: '{query}'")
contexts = search_context_for_chat(query, collection_name, top_k=3)
print(f"   Found {len(contexts)} documents")

print(f"\n2. DOCUMENTS RETRIEVED FROM MONGODB:")
for i, ctx in enumerate(contexts, 1):
    print(f"\n   Document {i}:")
    print(f"   - Content: {ctx.get('content', '')[:200]}...")
    print(f"   - Has CEO Drip info: {'$299' in ctx.get('content', '')}")

print(f"\n3. FORMATTED CONTEXT FOR OPENAI:")
formatted_context = format_context_for_prompt(contexts)
print(formatted_context)

print(f"\n4. WHAT OPENAI RECEIVES:")
messages = [
    {"role": "system", "content": "You are a helpful AI assistant with access to a knowledge base. Use the provided context to answer questions accurately."},
    {"role": "system", "content": f"Available Context:\n{formatted_context}"},
    {"role": "user", "content": query}
]

print(json.dumps(messages, indent=2))

print(f"\n5. VERIFICATION:")
if contexts and "$299" in formatted_context:
    print("✅ SUCCESS: CEO Drip document with $299 price IS being passed to OpenAI")
else:
    print("❌ FAIL: CEO Drip document NOT being passed to OpenAI")