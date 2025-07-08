#!/usr/bin/env python3
"""
Test the exact search function from Gradio app
"""

import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mongodb_vector_search_app import search_documents, test_connection

load_dotenv()

# Initialize connection
print("Initializing connection...")
success, msg, details = test_connection()
print(f"Connection: {msg}")
print(f"Details: {details}")

# Test search
print("\n" + "="*50)
print("Testing search for 'ceo drip'")
print("="*50)

result_df = search_documents("ceo drip", "medspa_services", top_k=5)
print("\nResults:")
print(result_df)

# Also test with different queries
test_queries = [
    "CEO Drip",
    "ceo",
    "drip",
    "$299"
]

print("\n\nTesting other queries:")
for query in test_queries:
    print(f"\n--- Query: '{query}' ---")
    result_df = search_documents(query, "medspa_services", top_k=3)
    if not result_df.empty and 'Error' not in result_df.columns and 'Message' not in result_df.columns:
        print(f"Found {len(result_df)} results")
        if 'Content Preview' in result_df.columns:
            for idx, row in result_df.iterrows():
                print(f"  {row['Rank']}. {row['Content Preview'][:100]}...")
    else:
        print(result_df)