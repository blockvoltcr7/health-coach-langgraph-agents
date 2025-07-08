"""
Module 2.2: MongoDB Vector Search Deep Dive
Time: 20 minutes
Goal: Master MongoDB vector search indexes, pipelines, and optimization
"""

import os
from pymongo import MongoClient
from openai import OpenAI
import time
from typing import List, Dict, Optional
from datetime import datetime
import json
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
mongodb_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongodb_client[os.getenv("MONGODB_DATABASE", "rag_course")]
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class VectorSearchOptimizer:
    """Advanced MongoDB vector search implementation"""
    
    def __init__(self, collection_name: str = "optimized_search"):
        self.collection = db[collection_name]
        
    def create_optimized_index(self):
        """Create production-ready vector index configuration"""
        
        # Index configurations for different use cases
        index_configs = {
            "standard": {
                "name": "vector_index",
                "definition": {
                    "mappings": {
                        "dynamic": True,
                        "fields": {
                            "embedding": {
                                "type": "knnVector",
                                "dimensions": 1536,
                                "similarity": "cosine"
                            }
                        }
                    }
                }
            },
            "filtered": {
                "name": "filtered_vector_index",
                "definition": {
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "embedding": {
                                "type": "knnVector",
                                "dimensions": 1536,
                                "similarity": "cosine"
                            },
                            "category": {
                                "type": "string",
                                "searchable": True,
                                "filterable": True
                            },
                            "timestamp": {
                                "type": "date",
                                "filterable": True
                            },
                            "tags": {
                                "type": "string",
                                "searchable": True,
                                "filterable": True
                            },
                            "importance": {
                                "type": "number",
                                "filterable": True
                            }
                        }
                    }
                }
            },
            "hybrid": {
                "name": "hybrid_search_index",
                "definition": {
                    "mappings": {
                        "dynamic": True,
                        "fields": {
                            "embedding": {
                                "type": "knnVector",
                                "dimensions": 1536,
                                "similarity": "cosine"
                            },
                            "content": {
                                "type": "string",
                                "searchable": True,
                                "analyzer": "lucene.standard"
                            },
                            "title": {
                                "type": "string",
                                "searchable": True,
                                "analyzer": "lucene.standard"
                            }
                        }
                    }
                }
            }
        }
        
        print("📊 Vector Index Configurations:")
        for config_type, config in index_configs.items():
            print(f"\n{config_type.upper()} Index Configuration:")
            print(json.dumps(config["definition"], indent=2))
        
        return index_configs
    
    def advanced_search_pipeline(
        self,
        query_embedding: List[float],
        filters: Optional[Dict] = None,
        num_candidates: int = 200,
        limit: int = 10,
        include_metadata: bool = True
    ) -> List[Dict]:
        """Build advanced search pipeline with filtering and optimization"""
        
        # Base vector search stage
        vector_search_stage = {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": num_candidates,
                "limit": limit * 3  # Get more for post-filtering
            }
        }
        
        # Add filters if provided
        if filters:
            vector_search_stage["$vectorSearch"]["filter"] = filters
        
        # Build complete pipeline
        pipeline = [vector_search_stage]
        
        # Add scoring and metadata
        if include_metadata:
            pipeline.append({
                "$addFields": {
                    "search_score": {"$meta": "vectorSearchScore"},
                    "search_timestamp": datetime.utcnow()
                }
            })
        
        # Project relevant fields
        pipeline.append({
            "$project": {
                "title": 1,
                "content": 1,
                "category": 1,
                "tags": 1,
                "importance": 1,
                "search_score": 1,
                "search_timestamp": 1,
                "_id": 1
            }
        })
        
        # Sort by score and limit
        pipeline.extend([
            {"$sort": {"search_score": -1}},
            {"$limit": limit}
        ])
        
        return pipeline
    
    def demonstrate_filtering(self):
        """Show different filtering strategies"""
        print("\n🔍 FILTERING STRATEGIES\n")
        
        # Sample data with metadata
        sample_docs = [
            {
                "title": "MongoDB Basics",
                "content": "Introduction to MongoDB and document databases",
                "category": "tutorial",
                "tags": ["mongodb", "basics", "database"],
                "importance": 3,
                "timestamp": datetime(2024, 1, 15)
            },
            {
                "title": "Advanced Vector Search",
                "content": "Implementing production vector search with filters",
                "category": "advanced",
                "tags": ["vector-search", "mongodb", "production"],
                "importance": 5,
                "timestamp": datetime(2024, 1, 20)
            },
            {
                "title": "Embedding Strategies",
                "content": "Choosing the right embedding model for your use case",
                "category": "tutorial",
                "tags": ["embeddings", "ai", "optimization"],
                "importance": 4,
                "timestamp": datetime(2024, 1, 18)
            }
        ]
        
        # Generate embeddings
        print("🧮 Generating embeddings for sample documents...")
        for doc in sample_docs:
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=doc["content"]
            )
            doc["embedding"] = response.data[0].embedding
        
        # Clear and insert
        self.collection.delete_many({})
        self.collection.insert_many(sample_docs)
        print(f"✅ Inserted {len(sample_docs)} documents")
        
        # Example filters
        filters = [
            {
                "name": "Category Filter",
                "filter": {"category": "tutorial"}
            },
            {
                "name": "Importance Filter",
                "filter": {"importance": {"$gte": 4}}
            },
            {
                "name": "Tag Filter",
                "filter": {"tags": {"$in": ["mongodb", "production"]}}
            },
            {
                "name": "Date Range Filter",
                "filter": {
                    "timestamp": {
                        "$gte": datetime(2024, 1, 17),
                        "$lte": datetime(2024, 1, 25)
                    }
                }
            },
            {
                "name": "Combined Filter",
                "filter": {
                    "$and": [
                        {"category": "tutorial"},
                        {"importance": {"$gte": 3}}
                    ]
                }
            }
        ]
        
        # Test query
        query = "How to implement vector search?"
        query_response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_embedding = query_response.data[0].embedding
        
        print(f"\n🔍 Query: '{query}'")
        print("=" * 60)
        
        for filter_config in filters:
            print(f"\n📋 {filter_config['name']}:")
            print(f"Filter: {json.dumps(filter_config['filter'], default=str)}")
            
            # Note: In production, you would apply these filters
            # For demo, we'll show the pipeline
            pipeline = self.advanced_search_pipeline(
                query_embedding,
                filters=filter_config['filter'],
                limit=3
            )
            print(f"Pipeline stages: {len(pipeline)}")
    
    def optimization_techniques(self):
        """Demonstrate search optimization techniques"""
        print("\n⚡ OPTIMIZATION TECHNIQUES\n")
        
        techniques = [
            {
                "name": "Candidate Optimization",
                "description": "Balance between accuracy and performance",
                "examples": [
                    {"candidates": 100, "use_case": "Real-time search", "latency": "~50ms"},
                    {"candidates": 500, "use_case": "Quality-focused", "latency": "~150ms"},
                    {"candidates": 1000, "use_case": "Maximum accuracy", "latency": "~300ms"}
                ]
            },
            {
                "name": "Index Compound Strategy",
                "description": "Combine vector and text search",
                "pipeline": [
                    {"$vectorSearch": "..."},
                    {"$match": {"$text": {"$search": "query terms"}}},
                    {"$group": {"_id": "$_id", "combined_score": {"$sum": ["$vector_score", "$text_score"]}}}
                ]
            },
            {
                "name": "Result Caching",
                "description": "Cache frequent queries",
                "implementation": """
# Cache in MongoDB
cache_collection.insert_one({
    "query_hash": hash(query),
    "results": search_results,
    "timestamp": datetime.utcnow(),
    "ttl": 3600  # 1 hour
})
                """
            },
            {
                "name": "Batch Processing",
                "description": "Process multiple queries efficiently",
                "tip": "Group similar queries and process together"
            }
        ]
        
        for technique in techniques:
            print(f"\n🔧 {technique['name']}")
            print(f"   {technique['description']}")
            
            if "examples" in technique:
                for example in technique["examples"]:
                    print(f"   • {example['candidates']} candidates: {example['use_case']} ({example['latency']})")
            
            if "implementation" in technique:
                print(f"   Implementation:")
                print(technique["implementation"])
    
    def performance_monitoring(self):
        """Show how to monitor vector search performance"""
        print("\n📈 PERFORMANCE MONITORING\n")
        
        # Simulated search with timing
        queries = [
            "How to optimize MongoDB queries?",
            "Vector search best practices",
            "Scaling MongoDB Atlas"
        ]
        
        results = []
        for query in queries:
            start_time = time.time()
            
            # Generate embedding
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            embedding_time = time.time() - start_time
            
            # Simulate search (would be actual search in production)
            search_start = time.time()
            # pipeline = self.advanced_search_pipeline(response.data[0].embedding)
            # results = list(self.collection.aggregate(pipeline))
            search_time = 0.05  # Simulated
            
            total_time = time.time() - start_time
            
            results.append({
                "query": query,
                "embedding_time": embedding_time,
                "search_time": search_time,
                "total_time": total_time
            })
        
        # Display metrics
        print("Query Performance Metrics:")
        print("-" * 80)
        print(f"{'Query':<40} {'Embedding':<12} {'Search':<12} {'Total':<12}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['query'][:40]:<40} "
                  f"{result['embedding_time']*1000:>8.1f} ms  "
                  f"{result['search_time']*1000:>8.1f} ms  "
                  f"{result['total_time']*1000:>8.1f} ms")
        
        # Performance tips
        print("\n💡 Performance Tips:")
        tips = [
            "Use connection pooling for MongoDB",
            "Implement embedding caching for repeated queries",
            "Consider using smaller embedding models for speed",
            "Use appropriate numCandidates based on dataset size",
            "Monitor Atlas Performance Advisor recommendations"
        ]
        
        for i, tip in enumerate(tips, 1):
            print(f"{i}. {tip}")

def demonstrate_index_strategies():
    """Show different index strategies for various use cases"""
    print("\n🏗️ INDEX STRATEGY GUIDE\n")
    
    strategies = {
        "Simple RAG": {
            "index_type": "Basic vector index",
            "dimensions": 1536,
            "filters": "None",
            "use_case": "Simple Q&A, documentation search"
        },
        "E-commerce Search": {
            "index_type": "Filtered vector index",
            "dimensions": 1024,
            "filters": "category, price_range, availability",
            "use_case": "Product search with faceted filtering"
        },
        "Time-series Analysis": {
            "index_type": "Compound index with date",
            "dimensions": 768,
            "filters": "timestamp, source, severity",
            "use_case": "Log analysis, event correlation"
        },
        "Multi-modal Search": {
            "index_type": "Multiple vector fields",
            "dimensions": "text: 1536, image: 512",
            "filters": "content_type, language",
            "use_case": "Combined text and image search"
        }
    }
    
    for name, strategy in strategies.items():
        print(f"\n📋 {name}:")
        for key, value in strategy.items():
            print(f"   {key}: {value}")

if __name__ == "__main__":
    print("🎓 MongoDB RAG Course - Vector Search Deep Dive\n")
    
    try:
        # Initialize optimizer
        optimizer = VectorSearchOptimizer()
        
        # Demo 1: Index configurations
        print("📊 Demo 1: Index Configurations")
        optimizer.create_optimized_index()
        
        # Demo 2: Filtering strategies
        print("\n📊 Demo 2: Filtering Strategies")
        optimizer.demonstrate_filtering()
        
        # Demo 3: Optimization techniques
        print("\n📊 Demo 3: Optimization Techniques")
        optimizer.optimization_techniques()
        
        # Demo 4: Performance monitoring
        print("\n📊 Demo 4: Performance Monitoring")
        optimizer.performance_monitoring()
        
        # Demo 5: Index strategies
        demonstrate_index_strategies()
        
        print("\n\n🎉 Key Takeaways:")
        print("✅ Choose the right index configuration for your use case")
        print("✅ Use filters to improve relevance and performance")
        print("✅ Monitor and optimize based on real usage patterns")
        print("✅ Consider hybrid approaches for best results")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("🔧 Ensure MongoDB connection and index exist")