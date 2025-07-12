#!/usr/bin/env python3
"""Test Tavily research functionality only"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_tavily_direct():
    """Test Tavily API directly"""
    print("Testing Tavily API...")
    
    API_KEY = os.getenv("TAVILY_API_KEY")
    if not API_KEY:
        print("❌ TAVILY_API_KEY not found!")
        return
    
    print(f"✅ Found API key: {API_KEY[:10]}...")
    
    try:
        # Try the tavily package first
        from tavily import TavilyClient
        
        tavily = TavilyClient(api_key=API_KEY)
        results = tavily.search("what are AI agents and MCP tools", max_results=2)
        
        print("\n✅ Tavily package works!")
        print(f"Results: {results.get('answer', 'No answer')[:200]}...")
        
    except ImportError:
        print("\n⚠️ Tavily package not installed, trying langchain...")
        
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            
            search = TavilySearchResults()
            results = search.invoke({"query": "what are AI agents and MCP tools"})
            
            print("\n✅ Langchain Tavily works!")
            print(f"Found {len(results)} results")
            if results:
                print(f"First result: {results[0].get('snippet', 'No snippet')[:200]}...")
                
        except Exception as e:
            print(f"\n❌ Langchain Tavily error: {e}")

if __name__ == "__main__":
    test_tavily_direct()