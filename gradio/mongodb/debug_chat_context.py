#!/usr/bin/env python3
"""
Debug script to see what context is being sent to OpenAI
"""

import os
import sys
from dotenv import load_dotenv
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mongodb_vector_search_app import search_context_for_chat, format_context_for_prompt, test_connection

load_dotenv()

# Initialize connection
print("Initializing connection...")
success, msg, details = test_connection()
print(f"Connection: {msg}")

# Test query
query = "what is the ceo drop price?"
collection_name = "medspa_services"

print(f"\n{'='*60}")
print(f"Query: '{query}'")
print(f"Collection: {collection_name}")
print(f"{'='*60}")

# Get contexts
contexts = search_context_for_chat(query, collection_name, top_k=3)

print(f"\nFound {len(contexts)} contexts")
print("\nRaw contexts:")
for i, ctx in enumerate(contexts, 1):
    print(f"\n--- Context {i} ---")
    print(f"Score: {ctx.get('score', 'N/A')}")
    print(f"Content length: {len(ctx.get('content', ''))}")
    print(f"Content preview: {ctx.get('content', '')[:200]}...")
    print(f"Metadata: {ctx.get('metadata', {})}")

# Format contexts
formatted_context = format_context_for_prompt(contexts)

print(f"\n{'='*60}")
print("FORMATTED CONTEXT THAT WOULD BE SENT TO OPENAI:")
print(f"{'='*60}")
print(formatted_context)

# Show what the full system message would look like
system_prompt = """You are a helpful AI assistant with access to a knowledge base. 
Use the provided context to answer questions accurately. If the context doesn't contain 
relevant information, say so and provide the best answer you can based on your general knowledge.
Always cite which context you're using when providing specific information."""

print(f"\n{'='*60}")
print("FULL SYSTEM MESSAGE TO OPENAI:")
print(f"{'='*60}")
print(f"{system_prompt}\n\nAvailable Context:\n{formatted_context}")

# Show the actual messages that would be sent
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "system", "content": f"Available Context:\n{formatted_context}"},
    {"role": "user", "content": query}
]

print(f"\n{'='*60}")
print("COMPLETE MESSAGES ARRAY:")
print(f"{'='*60}")
print(json.dumps(messages, indent=2))