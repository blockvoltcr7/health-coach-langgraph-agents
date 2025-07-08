#!/usr/bin/env python3
"""
Compare search_documents vs search_context_for_chat
"""

import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mongodb_vector_search_app import search_documents, search_context_for_chat, test_connection

load_dotenv()

# Initialize connection
print("Initializing connection...")
success, msg, details = test_connection()
print(f"Connection: {msg}")

# Test queries
queries = [
    "ceo drip price",
    "what is the ceo drop price?",
    "How much is the CEO Drip?"
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Testing query: '{query}'")
    print(f"{'='*60}")
    
    # Test search_documents (used in Search tab)
    print("\n1. Using search_documents (Search tab):")
    result_df = search_documents(query, "medspa_services", top_k=3)
    if not result_df.empty and 'Content Preview' in result_df.columns:
        print(f"   Found {len(result_df)} results")
        for idx, row in result_df.iterrows():
            if 'CEO' in row['Content Preview']:
                print(f"   ✅ {row['Content Preview'][:100]}...")
    else:
        print(f"   ❌ No results or error: {result_df}")
    
    # Test search_context_for_chat (used in AI Chat)
    print("\n2. Using search_context_for_chat (AI Chat):")
    contexts = search_context_for_chat(query, "medspa_services", top_k=3)
    if contexts:
        print(f"   Found {len(contexts)} contexts")
        for i, ctx in enumerate(contexts, 1):
            content_preview = ctx.get('content', '')[:100]
            if 'CEO' in content_preview:
                print(f"   ✅ {content_preview}...")
            else:
                print(f"   ❌ {content_preview}...")
    else:
        print("   ❌ No contexts found!")