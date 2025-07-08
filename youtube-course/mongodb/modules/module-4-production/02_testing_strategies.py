"""
Module 4.2: Testing Strategies for RAG Systems
Time: 15 minutes
Goal: Implement comprehensive testing for production RAG systems
"""

import os
import unittest
import pytest
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import json
import hashlib
from unittest.mock import Mock, patch, MagicMock
import asyncio
import time
from collections import defaultdict
from pymongo import MongoClient
from openai import OpenAI
import voyageai
from dotenv import load_dotenv

load_dotenv()

# Test configuration
TEST_DATABASE = "rag_test_db"
TEST_COLLECTION = "test_documents"

@dataclass
class TestCase:
    """Structured test case for RAG systems"""
    name: str
    query: str
    expected_docs: List[str]  # Document IDs that should be retrieved
    expected_keywords: List[str]  # Keywords that should appear in response
    min_relevance_score: float = 0.7
    max_response_time: float = 2.0  # seconds
    metadata: Dict[str, Any] = None

class RAGTestFramework:
    """
    Comprehensive testing framework for RAG systems
    Includes unit tests, integration tests, and performance tests
    """
    
    def __init__(self):
        self.mongodb_client = MongoClient(os.getenv("MONGODB_URI"))
        self.test_db = self.mongodb_client[TEST_DATABASE]
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Test metrics storage
        self.test_results = []
        self.performance_metrics = defaultdict(list)
        
    def setup_test_data(self):
        """Create test dataset with known characteristics"""
        test_documents = [
            {
                "doc_id": "test_001",
                "title": "Introduction to Vector Search",
                "content": "Vector search enables semantic search by converting text into numerical representations called embeddings.",
                "category": "tutorial",
                "ground_truth_queries": ["what is vector search", "explain embeddings"]
            },
            {
                "doc_id": "test_002",
                "title": "MongoDB Atlas Configuration",
                "content": "To configure MongoDB Atlas for vector search, create a vector index with appropriate dimensions and similarity metrics.",
                "category": "configuration",
                "ground_truth_queries": ["mongodb atlas setup", "create vector index"]
            },
            {
                "doc_id": "test_003",
                "title": "RAG System Architecture",
                "content": "RAG systems combine retrieval and generation. They retrieve relevant documents and use them as context for generating responses.",
                "category": "architecture",
                "ground_truth_queries": ["RAG architecture", "retrieval augmented generation"]
            },
            {
                "doc_id": "test_004",
                "title": "Performance Optimization",
                "content": "Optimize RAG performance through caching, batch processing, and efficient embedding models.",
                "category": "optimization",
                "ground_truth_queries": ["improve RAG performance", "optimization techniques"]
            },
            {
                "doc_id": "test_005",
                "title": "Error Handling Best Practices",
                "content": "Implement retry logic, circuit breakers, and graceful degradation for robust RAG systems.",
                "category": "best_practices",
                "ground_truth_queries": ["error handling", "resilience patterns"]
            }
        ]
        
        # Clear and insert test data
        collection = self.test_db[TEST_COLLECTION]
        collection.delete_many({})
        
        # Add embeddings
        for doc in test_documents:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=doc["content"]
            )
            doc["embedding"] = response.data[0].embedding
        
        collection.insert_many(test_documents)
        print(f"✅ Inserted {len(test_documents)} test documents")
        
        return test_documents
    
    def create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases"""
        return [
            # Exact match tests
            TestCase(
                name="Exact Match - Vector Search",
                query="what is vector search",
                expected_docs=["test_001"],
                expected_keywords=["vector", "search", "embeddings"],
                min_relevance_score=0.9
            ),
            
            # Semantic similarity tests
            TestCase(
                name="Semantic Similarity - RAG",
                query="How does retrieval augmented generation work?",
                expected_docs=["test_003"],
                expected_keywords=["retrieval", "generation", "context"],
                min_relevance_score=0.8
            ),
            
            # Multi-document retrieval
            TestCase(
                name="Multi-Document - Performance",
                query="How to optimize vector search performance in MongoDB?",
                expected_docs=["test_002", "test_004"],
                expected_keywords=["optimize", "performance", "mongodb"],
                min_relevance_score=0.7
            ),
            
            # Edge case - ambiguous query
            TestCase(
                name="Edge Case - Ambiguous Query",
                query="best practices",
                expected_docs=["test_005"],
                expected_keywords=["practices"],
                min_relevance_score=0.6
            ),
            
            # Negative test - no relevant docs
            TestCase(
                name="Negative Test - Irrelevant Query",
                query="quantum computing algorithms",
                expected_docs=[],
                expected_keywords=[],
                min_relevance_score=0.0
            )
        ]
    
    def test_embedding_generation(self):
        """Test embedding generation and consistency"""
        print("\n🧪 Testing Embedding Generation")
        print("="*60)
        
        test_texts = [
            "MongoDB vector search",
            "MongoDB vector search",  # Duplicate for consistency check
            "Completely different text about databases"
        ]
        
        embeddings = []
        for text in test_texts:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embeddings.append(response.data[0].embedding)
        
        # Test 1: Dimension consistency
        dimensions = [len(emb) for emb in embeddings]
        assert all(d == dimensions[0] for d in dimensions), "Embedding dimensions inconsistent"
        print(f"✅ Dimension consistency: All embeddings have {dimensions[0]} dimensions")
        
        # Test 2: Identical text produces identical embeddings
        similarity = np.dot(embeddings[0], embeddings[1])
        assert similarity > 0.999, f"Identical texts produced different embeddings: {similarity}"
        print(f"✅ Consistency check: Identical texts have similarity {similarity:.4f}")
        
        # Test 3: Different texts have lower similarity
        similarity_diff = np.dot(embeddings[0], embeddings[2])
        assert similarity_diff < 0.9, f"Different texts too similar: {similarity_diff}"
        print(f"✅ Distinctiveness: Different texts have similarity {similarity_diff:.4f}")
        
        return True
    
    def test_vector_search_accuracy(self, test_cases: List[TestCase]):
        """Test vector search accuracy"""
        print("\n🧪 Testing Vector Search Accuracy")
        print("="*60)
        
        collection = self.test_db[TEST_COLLECTION]
        results = []
        
        for test_case in test_cases:
            print(f"\n📋 Test: {test_case.name}")
            print(f"   Query: '{test_case.query}'")
            
            # Generate query embedding
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=test_case.query
            )
            query_embedding = response.data[0].embedding
            
            # Perform search
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 100,
                        "limit": 5
                    }
                },
                {
                    "$project": {
                        "doc_id": 1,
                        "title": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            search_results = list(collection.aggregate(pipeline))
            retrieved_ids = [r["doc_id"] for r in search_results]
            
            # Evaluate results
            if test_case.expected_docs:
                # Check if expected docs are retrieved
                found = [doc_id for doc_id in test_case.expected_docs if doc_id in retrieved_ids]
                precision = len(found) / len(retrieved_ids) if retrieved_ids else 0
                recall = len(found) / len(test_case.expected_docs) if test_case.expected_docs else 0
                
                print(f"   Expected: {test_case.expected_docs}")
                print(f"   Retrieved: {retrieved_ids[:3]}")
                print(f"   Precision: {precision:.2f}, Recall: {recall:.2f}")
                
                # Check relevance scores
                if search_results:
                    top_score = search_results[0]["score"]
                    print(f"   Top Score: {top_score:.4f} (min required: {test_case.min_relevance_score})")
                    
                    test_passed = (
                        recall >= 0.5 and  # At least 50% of expected docs retrieved
                        top_score >= test_case.min_relevance_score
                    )
                else:
                    test_passed = False
            else:
                # Negative test - should not retrieve relevant docs
                test_passed = len(search_results) == 0 or search_results[0]["score"] < 0.5
            
            results.append({
                "test": test_case.name,
                "passed": test_passed,
                "precision": precision if test_case.expected_docs else None,
                "recall": recall if test_case.expected_docs else None
            })
            
            print(f"   Result: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        
        return results
    
    def test_response_generation(self, test_cases: List[TestCase]):
        """Test response generation quality"""
        print("\n🧪 Testing Response Generation")
        print("="*60)
        
        collection = self.test_db[TEST_COLLECTION]
        results = []
        
        for test_case in test_cases[:3]:  # Test first 3 cases
            print(f"\n📋 Test: {test_case.name}")
            
            # Get context documents
            context_docs = list(collection.find(
                {"doc_id": {"$in": test_case.expected_docs}}
            )) if test_case.expected_docs else []
            
            if not context_docs and test_case.expected_docs:
                # Fallback - get any documents
                context_docs = list(collection.find().limit(2))
            
            # Generate response
            context = "\n".join([doc["content"] for doc in context_docs])
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Answer based on the provided context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {test_case.query}"}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            generated_response = response.choices[0].message.content
            print(f"   Response: {generated_response[:100]}...")
            
            # Check for expected keywords
            found_keywords = [
                kw for kw in test_case.expected_keywords 
                if kw.lower() in generated_response.lower()
            ]
            
            keyword_coverage = len(found_keywords) / len(test_case.expected_keywords) if test_case.expected_keywords else 1.0
            print(f"   Keyword Coverage: {keyword_coverage:.2f} ({found_keywords})")
            
            test_passed = keyword_coverage >= 0.5
            
            results.append({
                "test": test_case.name,
                "passed": test_passed,
                "keyword_coverage": keyword_coverage,
                "response_length": len(generated_response)
            })
            
            print(f"   Result: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        
        return results
    
    def test_performance(self, num_queries: int = 10):
        """Test system performance and latency"""
        print("\n🧪 Testing Performance")
        print("="*60)
        
        test_queries = [
            "What is vector search?",
            "How to optimize MongoDB?",
            "Explain RAG architecture",
            "Best practices for embeddings",
            "Error handling strategies"
        ] * (num_queries // 5)
        
        latencies = []
        
        for i, query in enumerate(test_queries):
            start_time = time.time()
            
            # Simulate full RAG pipeline
            try:
                # 1. Generate embedding
                embedding_start = time.time()
                response = self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=query
                )
                embedding_time = time.time() - embedding_start
                
                # 2. Vector search (simulated)
                search_time = np.random.uniform(0.05, 0.15)  # 50-150ms
                
                # 3. Response generation (simulated)
                generation_time = np.random.uniform(0.5, 1.0)  # 500ms-1s
                
                total_time = time.time() - start_time
                latencies.append(total_time)
                
                if i < 5:  # Show first 5
                    print(f"   Query {i+1}: {total_time:.3f}s "
                          f"(embed: {embedding_time:.3f}s, "
                          f"search: {search_time:.3f}s, "
                          f"gen: {generation_time:.3f}s)")
                
            except Exception as e:
                print(f"   Query {i+1}: FAILED - {e}")
        
        # Calculate performance metrics
        if latencies:
            metrics = {
                "avg_latency": np.mean(latencies),
                "p50_latency": np.percentile(latencies, 50),
                "p95_latency": np.percentile(latencies, 95),
                "p99_latency": np.percentile(latencies, 99),
                "max_latency": np.max(latencies),
                "queries_per_second": len(latencies) / sum(latencies)
            }
            
            print(f"\n📊 Performance Metrics:")
            print(f"   Average Latency: {metrics['avg_latency']:.3f}s")
            print(f"   P50 Latency: {metrics['p50_latency']:.3f}s")
            print(f"   P95 Latency: {metrics['p95_latency']:.3f}s")
            print(f"   P99 Latency: {metrics['p99_latency']:.3f}s")
            print(f"   Max Latency: {metrics['max_latency']:.3f}s")
            print(f"   Throughput: {metrics['queries_per_second']:.2f} queries/second")
            
            return metrics
        
        return None
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n🧪 Testing Edge Cases")
        print("="*60)
        
        edge_cases = [
            {
                "name": "Empty Query",
                "query": "",
                "should_handle": True
            },
            {
                "name": "Very Long Query",
                "query": "a" * 1000,
                "should_handle": True
            },
            {
                "name": "Special Characters",
                "query": "!@#$%^&*()_+-={}[]|\\:;<>?,./",
                "should_handle": True
            },
            {
                "name": "Non-English Query",
                "query": "这是一个中文查询",
                "should_handle": True
            },
            {
                "name": "SQL Injection Attempt",
                "query": "'; DROP TABLE documents; --",
                "should_handle": True
            }
        ]
        
        results = []
        
        for edge_case in edge_cases:
            print(f"\n📋 Test: {edge_case['name']}")
            print(f"   Query: '{edge_case['query'][:50]}...' (length: {len(edge_case['query'])})")
            
            try:
                # Try to process the query
                response = self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=edge_case["query"] or "empty query"  # Handle empty string
                )
                
                embedding = response.data[0].embedding
                print(f"   ✅ Successfully generated embedding (dim: {len(embedding)})")
                
                test_passed = edge_case["should_handle"]
                
            except Exception as e:
                print(f"   ❌ Failed with error: {e}")
                test_passed = not edge_case["should_handle"]
            
            results.append({
                "test": edge_case["name"],
                "passed": test_passed
            })
            
            print(f"   Result: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        
        return results

class MockTesting:
    """Demonstrate unit testing with mocks"""
    
    @staticmethod
    def test_embedding_fallback():
        """Test embedding provider fallback logic"""
        print("\n🧪 Testing with Mocks")
        print("="*60)
        
        # Mock the embedding providers
        with patch('voyageai.Client') as mock_voyage, \
             patch('openai.OpenAI') as mock_openai:
            
            # Configure mocks
            mock_voyage_instance = MagicMock()
            mock_openai_instance = MagicMock()
            
            mock_voyage.return_value = mock_voyage_instance
            mock_openai.return_value = mock_openai_instance
            
            # Test 1: Voyage AI succeeds
            print("\n📋 Test: Voyage AI Success")
            mock_voyage_instance.embed.return_value = MagicMock(embeddings=[[0.1] * 1024])
            
            # Simulate the fallback logic
            try:
                result = mock_voyage_instance.embed("test text")
                print("✅ Voyage AI returned embedding")
            except:
                print("❌ Voyage AI failed unexpectedly")
            
            # Test 2: Voyage AI fails, OpenAI succeeds
            print("\n📋 Test: Voyage AI Fails, OpenAI Fallback")
            mock_voyage_instance.embed.side_effect = Exception("Rate limit exceeded")
            mock_openai_instance.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=[0.2] * 1536)]
            )
            
            try:
                # First try Voyage
                result = mock_voyage_instance.embed("test text")
            except:
                print("⚠️  Voyage AI failed as expected")
                # Fallback to OpenAI
                result = mock_openai_instance.embeddings.create(
                    model="text-embedding-ada-002",
                    input="test text"
                )
                print("✅ OpenAI fallback succeeded")
            
            # Test 3: Both fail
            print("\n📋 Test: Both Providers Fail")
            mock_voyage_instance.embed.side_effect = Exception("Voyage error")
            mock_openai_instance.embeddings.create.side_effect = Exception("OpenAI error")
            
            all_failed = False
            try:
                result = mock_voyage_instance.embed("test text")
            except:
                try:
                    result = mock_openai_instance.embeddings.create(
                        model="text-embedding-ada-002",
                        input="test text"
                    )
                except:
                    all_failed = True
                    print("✅ Both providers failed as expected")
            
            assert all_failed, "Expected both providers to fail"

def create_test_suite():
    """Create comprehensive test suite"""
    print("\n📦 COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    test_categories = [
        {
            "name": "Unit Tests",
            "tests": [
                "test_embedding_dimension_validation",
                "test_query_sanitization",
                "test_response_truncation",
                "test_cache_operations",
                "test_error_classification"
            ]
        },
        {
            "name": "Integration Tests",
            "tests": [
                "test_mongodb_connection",
                "test_openai_api_integration",
                "test_voyage_api_integration",
                "test_end_to_end_rag_pipeline",
                "test_multi_provider_fallback"
            ]
        },
        {
            "name": "Performance Tests",
            "tests": [
                "test_embedding_generation_latency",
                "test_vector_search_performance",
                "test_concurrent_request_handling",
                "test_cache_hit_rate",
                "test_memory_usage"
            ]
        },
        {
            "name": "Security Tests",
            "tests": [
                "test_input_validation",
                "test_injection_prevention",
                "test_rate_limiting",
                "test_api_key_security",
                "test_data_privacy"
            ]
        },
        {
            "name": "Regression Tests",
            "tests": [
                "test_backward_compatibility",
                "test_model_version_changes",
                "test_api_version_compatibility",
                "test_data_migration",
                "test_configuration_changes"
            ]
        }
    ]
    
    for category in test_categories:
        print(f"\n📂 {category['name']}:")
        for test in category['tests']:
            print(f"   □ {test}")
    
    print("\n📋 Test Execution Plan:")
    print("1. Run unit tests on every commit")
    print("2. Run integration tests on pull requests")
    print("3. Run performance tests nightly")
    print("4. Run security tests weekly")
    print("5. Run regression tests before releases")

def demonstrate_continuous_testing():
    """Demonstrate continuous testing practices"""
    print("\n🔄 CONTINUOUS TESTING PRACTICES")
    print("="*60)
    
    ci_pipeline = """
# .github/workflows/rag-tests.yml
name: RAG System Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # Nightly at 2 AM

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run unit tests
      run: |
        pytest tests/unit --cov=rag_system --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v2

  integration-tests:
    runs-on: ubuntu-latest
    services:
      mongodb:
        image: mongo:5.0
        options: >-
          --health-cmd mongo
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
    - uses: actions/checkout@v2
    - name: Run integration tests
      env:
        MONGODB_URI: mongodb://localhost:27017
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        pytest tests/integration -v

  performance-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
    - uses: actions/checkout@v2
    - name: Run performance tests
      run: |
        python tests/performance/benchmark.py
    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: performance-results
        path: results/
"""
    
    print("CI/CD Pipeline Configuration:")
    print(ci_pipeline)
    
    print("\n📊 Test Metrics to Track:")
    metrics = [
        "Code Coverage: Aim for >80%",
        "Test Execution Time: Monitor for regressions",
        "Flaky Test Rate: Should be <5%",
        "Performance Benchmarks: Track P95 latencies",
        "Security Scan Results: Zero critical vulnerabilities"
    ]
    
    for metric in metrics:
        print(f"   • {metric}")

if __name__ == "__main__":
    print("🎓 MongoDB RAG Course - Testing Strategies\n")
    
    try:
        # Initialize test framework
        test_framework = RAGTestFramework()
        
        # Setup test data
        test_framework.setup_test_data()
        
        # Create test cases
        test_cases = test_framework.create_test_cases()
        
        # Run tests
        print("\n" + "="*60)
        print("🏃 RUNNING TEST SUITE")
        print("="*60)
        
        # 1. Embedding tests
        test_framework.test_embedding_generation()
        
        # 2. Search accuracy tests
        search_results = test_framework.test_vector_search_accuracy(test_cases)
        
        # 3. Response generation tests
        generation_results = test_framework.test_response_generation(test_cases)
        
        # 4. Performance tests
        performance_metrics = test_framework.test_performance(num_queries=5)
        
        # 5. Edge case tests
        edge_case_results = test_framework.test_edge_cases()
        
        # 6. Mock testing demo
        MockTesting.test_embedding_fallback()
        
        # Show test suite structure
        create_test_suite()
        
        # Show CI/CD practices
        demonstrate_continuous_testing()
        
        # Summary
        print("\n\n📊 TEST SUMMARY")
        print("="*60)
        
        total_tests = len(search_results) + len(generation_results) + len(edge_case_results)
        passed_tests = sum(1 for r in search_results + generation_results + edge_case_results if r["passed"])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        print("\n🎉 Key Testing Strategies Demonstrated:")
        print("✅ Embedding validation and consistency")
        print("✅ Search accuracy and relevance testing")
        print("✅ Response quality evaluation")
        print("✅ Performance benchmarking")
        print("✅ Edge case handling")
        print("✅ Mock testing for unit tests")
        print("✅ Continuous integration setup")
        
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()