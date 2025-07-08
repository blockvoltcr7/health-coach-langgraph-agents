#!/usr/bin/env python3
"""
Test the complete chat flow to verify OpenAI returns correct answer
"""

import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mongodb_vector_search_app import chat_with_rag, test_connection

load_dotenv()

# Initialize connection
print("Initializing connection...")
success, msg, details = test_connection()
print(f"Connection: {msg}\n")

# Test queries
test_queries = [
    "what is the ceo drop price?",
    "How much is the CEO Drip?",
    "Tell me about the CEO Drip pricing"
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"USER: {query}")
    print(f"{'='*60}")
    
    # Get AI response
    response = chat_with_rag(
        message=query,
        history=[],  # Empty history for first message
        collection_name="medspa_services",
        model_name="gpt-4o-mini",  # Using mini for faster response
        temperature=0.3,  # Lower temperature for more factual responses
        top_k=3
    )
    
    print(f"\nASSISTANT: {response}")
    
    # Check if the response contains the correct price
    if "$299" in response:
        print("\n✅ CORRECT: Response contains the correct price ($299)")
    else:
        print("\n❌ INCORRECT: Response does not contain the correct price")