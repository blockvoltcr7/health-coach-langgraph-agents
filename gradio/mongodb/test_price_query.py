#!/usr/bin/env python3
"""
Test price-based queries specifically
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

# Test price-related queries
price_queries = [
    "How much is the CEO Drip?",
    "CEO Drip price",
    "what's the cost of CEO Drip",
    "CEO Drip $299"
]

print("\n" + "="*50)
print("Testing price-related queries")
print("="*50)

for query in price_queries:
    print(f"\n--- Query: '{query}' ---")
    result_df = search_documents(query, "medspa_services", top_k=3)
    
    if not result_df.empty and 'Error' not in result_df.columns and 'Message' not in result_df.columns:
        print(f"Found {len(result_df)} results")
        if 'Content Preview' in result_df.columns:
            for idx, row in result_df.iterrows():
                # Check if CEO Drip is in the result
                if 'CEO Drip' in row['Content Preview']:
                    print(f"✅ {row['Rank']}. {row['Content Preview'][:100]}...")
                else:
                    print(f"  {row['Rank']}. {row['Content Preview'][:100]}...")
    else:
        print("❌ No results found")
        print(result_df)