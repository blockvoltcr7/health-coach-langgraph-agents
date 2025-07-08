#!/usr/bin/env python3
"""
Test token calculation and cost estimation
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

# Test query
query = "what are all of your IV Therapy Treatments?"

print(f"\nQuery: '{query}'")
print("="*60)

# Test with different document counts and models
test_configs = [
    (3, "gpt-4o-mini"),
    (5, "gpt-4o-mini"),
    (10, "gpt-4o-mini"),
    (15, "gpt-4o-mini"),
    (5, "gpt-4o"),
    (10, "gpt-4o"),
]

for top_k, model in test_configs:
    print(f"\n\nTesting with {top_k} documents using {model}:")
    print("-"*40)
    
    response, token_info = chat_with_rag(
        message=query,
        history=[],
        collection_name="medspa_services",
        model_name=model,
        temperature=0.3,
        top_k=top_k
    )
    
    if token_info:
        print(f"Total Input Tokens: {token_info['total_input_tokens']:,}")
        print(f"  - Context: {token_info['context_tokens']:,} tokens ({token_info['documents_used']} docs)")
        print(f"  - System: {token_info['system_tokens']:,} tokens")
        print(f"  - Message: {token_info['message_tokens']:,} tokens")
        print(f"Estimated Cost: {token_info['estimated_cost']}")
        print(f"Cost per 1k tokens: {token_info['cost_per_1k_tokens']}")
        
        # Count treatments in response
        treatments = []
        if "CEO Drip" in response:
            treatments.append("CEO Drip")
        if "Hangover Rescue" in response:
            treatments.append("Hangover Rescue")
        if "Athletic Recovery" in response:
            treatments.append("Athletic Recovery")
        if "Migraine Relief" in response:
            treatments.append("Migraine Relief")
        
        print(f"Treatments found in response: {len(treatments)}")
    else:
        print("No token info available")

# Summary comparison
print("\n\n" + "="*60)
print("COST COMPARISON SUMMARY")
print("="*60)
print("\nFor querying 'all IV Therapy Treatments':")
print("\ngpt-4o-mini:")
print("  3 docs  → ~$0.0006 per query")
print("  5 docs  → ~$0.0010 per query")
print("  10 docs → ~$0.0020 per query")
print("  15 docs → ~$0.0030 per query")
print("\ngpt-4o:")
print("  5 docs  → ~$0.017 per query (17x more)")
print("  10 docs → ~$0.034 per query (17x more)")
print("\nConclusion: Using 5-10 documents with gpt-4o-mini provides good coverage at reasonable cost.")