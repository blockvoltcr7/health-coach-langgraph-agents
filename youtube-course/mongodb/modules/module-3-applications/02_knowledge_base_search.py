"""
Module 3.2: Knowledge Base Search System
Time: 20 minutes
Goal: Build a production knowledge base with faceted search and relevance scoring
"""

import os
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from pymongo import MongoClient, ASCENDING, TEXT
from openai import OpenAI
import voyageai
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
mongodb_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongodb_client[os.getenv("MONGODB_DATABASE", "rag_course")]
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_AI_API_KEY"))

@dataclass
class SearchResult:
    """Enhanced search result with metadata"""
    doc_id: str
    title: str
    content: str
    score: float
    category: str
    tags: List[str]
    last_updated: datetime
    view_count: int = 0
    helpful_count: int = 0
    relevance_score: Optional[float] = None
    highlights: List[str] = field(default_factory=list)
    related_docs: List[str] = field(default_factory=list)

@dataclass
class SearchQuery:
    """Structured search query with filters"""
    text: str
    categories: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_score: float = 0.0
    limit: int = 10
    include_related: bool = True
    
class KnowledgeBaseSearch:
    """
    Production knowledge base search with:
    - Multi-modal search (vector + text)
    - Faceted filtering
    - Result highlighting
    - Related document suggestions
    - Search analytics
    - Relevance feedback
    """
    
    def __init__(self, collection_name: str = "knowledge_base"):
        self.collection = db[collection_name]
        self.search_history = db["search_history"]
        self.feedback_collection = db["search_feedback"]
        
        # Create compound indexes
        self._setup_indexes()
        
        # Search configuration
        self.vector_weight = 0.7
        self.text_weight = 0.3
        self.rerank_candidates = 3
        
    def _setup_indexes(self):
        """Setup MongoDB indexes for optimal performance"""
        # Text index for full-text search
        self.collection.create_index([("title", TEXT), ("content", TEXT)])
        
        # Compound indexes for filtering
        self.collection.create_index([("category", ASCENDING), ("last_updated", ASCENDING)])
        self.collection.create_index([("tags", ASCENDING)])
        
        # Analytics indexes
        self.search_history.create_index([("timestamp", ASCENDING)])
        self.search_history.create_index([("user_id", ASCENDING)])
        
        print("✅ Indexes created for optimal search performance")
    
    def ingest_documents(self, documents: List[Dict], batch_size: int = 50):
        """Ingest documents with batch processing"""
        print(f"📚 Ingesting {len(documents)} documents...")
        
        # Clear existing documents
        self.collection.delete_many({})
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            processed_batch = []
            
            # Generate embeddings for batch
            texts = [doc["content"] for doc in batch]
            
            try:
                # Try Voyage AI
                result = voyage_client.embed(
                    texts=texts,
                    model="voyage-3-large",
                    input_type="document"
                )
                embeddings = result.embeddings
            except:
                # Fallback to OpenAI
                embeddings = []
                for text in texts:
                    response = openai_client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=text
                    )
                    embeddings.append(response.data[0].embedding)
            
            # Process documents
            for doc, embedding in zip(batch, embeddings):
                processed_doc = {
                    **doc,
                    "embedding": embedding,
                    "doc_id": f"doc_{i + batch.index(doc):04d}",
                    "ingested_at": datetime.utcnow(),
                    "last_updated": doc.get("last_updated", datetime.utcnow()),
                    "view_count": 0,
                    "helpful_count": 0,
                    "tags": doc.get("tags", []),
                    "category": doc.get("category", "General")
                }
                processed_batch.append(processed_doc)
            
            # Insert batch
            self.collection.insert_many(processed_batch)
            print(f"  Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        print(f"✅ Ingested {len(documents)} documents successfully")
    
    def search(self, query: SearchQuery) -> Tuple[List[SearchResult], Dict]:
        """Perform multi-modal search with filters"""
        start_time = datetime.utcnow()
        
        # Step 1: Build filter stage
        filter_stage = self._build_filter(query)
        
        # Step 2: Perform vector search
        vector_results = self._vector_search(query.text, filter_stage, query.limit * 3)
        
        # Step 3: Perform text search
        text_results = self._text_search(query.text, filter_stage, query.limit * 2)
        
        # Step 4: Combine and score results
        combined_results = self._combine_results(vector_results, text_results)
        
        # Step 5: Rerank if available
        if os.getenv("VOYAGE_AI_API_KEY") and len(combined_results) > query.limit:
            combined_results = self._rerank_results(query.text, combined_results, query.limit)
        
        # Step 6: Apply final filtering and limit
        final_results = [r for r in combined_results if r.score >= query.min_score][:query.limit]
        
        # Step 7: Find related documents if requested
        if query.include_related:
            for result in final_results:
                result.related_docs = self._find_related_docs(result.doc_id)
        
        # Step 8: Add highlights
        for result in final_results:
            result.highlights = self._generate_highlights(result.content, query.text)
        
        # Track search
        search_time = (datetime.utcnow() - start_time).total_seconds()
        self._track_search(query, final_results, search_time)
        
        # Generate search metadata
        metadata = {
            "total_results": len(final_results),
            "search_time": f"{search_time:.2f}s",
            "vector_results": len(vector_results),
            "text_results": len(text_results),
            "filters_applied": bool(filter_stage),
            "reranked": os.getenv("VOYAGE_AI_API_KEY") is not None
        }
        
        return final_results, metadata
    
    def _build_filter(self, query: SearchQuery) -> Dict:
        """Build MongoDB filter from query parameters"""
        filter_conditions = []
        
        if query.categories:
            filter_conditions.append({"category": {"$in": query.categories}})
        
        if query.tags:
            filter_conditions.append({"tags": {"$in": query.tags}})
        
        if query.date_from or query.date_to:
            date_filter = {}
            if query.date_from:
                date_filter["$gte"] = query.date_from
            if query.date_to:
                date_filter["$lte"] = query.date_to
            filter_conditions.append({"last_updated": date_filter})
        
        if filter_conditions:
            return {"$and": filter_conditions} if len(filter_conditions) > 1 else filter_conditions[0]
        
        return {}
    
    def _vector_search(self, query_text: str, filter_stage: Dict, limit: int) -> List[SearchResult]:
        """Perform vector search"""
        # Generate query embedding
        try:
            result = voyage_client.embed(
                texts=[query_text],
                model="voyage-3-large",
                input_type="query"
            )
            query_embedding = result.embeddings[0]
        except:
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=query_text
            )
            query_embedding = response.data[0].embedding
        
        # Build pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit,
                    "filter": filter_stage if filter_stage else None
                }
            },
            {
                "$project": {
                    "doc_id": 1,
                    "title": 1,
                    "content": 1,
                    "category": 1,
                    "tags": 1,
                    "last_updated": 1,
                    "view_count": 1,
                    "helpful_count": 1,
                    "vector_score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(self.collection.aggregate(pipeline))
        
        return [
            SearchResult(
                doc_id=r["doc_id"],
                title=r["title"],
                content=r["content"],
                score=r["vector_score"] * self.vector_weight,
                category=r["category"],
                tags=r.get("tags", []),
                last_updated=r["last_updated"],
                view_count=r.get("view_count", 0),
                helpful_count=r.get("helpful_count", 0)
            )
            for r in results
        ]
    
    def _text_search(self, query_text: str, filter_stage: Dict, limit: int) -> List[SearchResult]:
        """Perform text search"""
        # Build text search query
        search_query = {"$text": {"$search": query_text}}
        
        # Combine with filters
        if filter_stage:
            search_query = {"$and": [search_query, filter_stage]}
        
        # Execute search
        results = self.collection.find(
            search_query,
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
        
        return [
            SearchResult(
                doc_id=r["doc_id"],
                title=r["title"],
                content=r["content"],
                score=r["score"] * self.text_weight,
                category=r["category"],
                tags=r.get("tags", []),
                last_updated=r["last_updated"],
                view_count=r.get("view_count", 0),
                helpful_count=r.get("helpful_count", 0)
            )
            for r in results
        ]
    
    def _combine_results(self, vector_results: List[SearchResult], text_results: List[SearchResult]) -> List[SearchResult]:
        """Combine and deduplicate results"""
        # Create a map to combine scores
        result_map = {}
        
        # Add vector results
        for result in vector_results:
            result_map[result.doc_id] = result
        
        # Combine with text results
        for result in text_results:
            if result.doc_id in result_map:
                # Combine scores
                result_map[result.doc_id].score += result.score
            else:
                result_map[result.doc_id] = result
        
        # Sort by combined score
        combined = list(result_map.values())
        combined.sort(key=lambda x: x.score, reverse=True)
        
        return combined
    
    def _rerank_results(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Rerank results using Voyage AI"""
        try:
            documents = [r.content for r in results]
            
            reranking = voyage_client.rerank(
                query=query,
                documents=documents,
                model="rerank-2-lite",
                top_k=top_k
            )
            
            reranked_results = []
            for rank_result in reranking.results:
                result = results[rank_result.index]
                result.relevance_score = rank_result.relevance_score
                reranked_results.append(result)
            
            return reranked_results
        except:
            return results[:top_k]
    
    def _find_related_docs(self, doc_id: str, limit: int = 3) -> List[str]:
        """Find related documents based on embeddings"""
        # Get document
        doc = self.collection.find_one({"doc_id": doc_id})
        if not doc or "embedding" not in doc:
            return []
        
        # Find similar documents
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": doc["embedding"],
                    "numCandidates": 50,
                    "limit": limit + 1,  # +1 to exclude self
                    "filter": {"doc_id": {"$ne": doc_id}}
                }
            },
            {
                "$project": {
                    "doc_id": 1,
                    "title": 1
                }
            }
        ]
        
        similar = list(self.collection.aggregate(pipeline))
        return [f"{s['doc_id']}: {s['title']}" for s in similar if s['doc_id'] != doc_id][:limit]
    
    def _generate_highlights(self, content: str, query: str, context_size: int = 50) -> List[str]:
        """Generate highlighted snippets"""
        highlights = []
        query_words = query.lower().split()
        content_lower = content.lower()
        
        for word in query_words:
            if word in content_lower:
                pos = content_lower.find(word)
                start = max(0, pos - context_size)
                end = min(len(content), pos + len(word) + context_size)
                
                snippet = content[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(content):
                    snippet = snippet + "..."
                
                highlights.append(snippet)
        
        return highlights[:3]  # Return top 3 highlights
    
    def _track_search(self, query: SearchQuery, results: List[SearchResult], search_time: float):
        """Track search for analytics"""
        self.search_history.insert_one({
            "timestamp": datetime.utcnow(),
            "query_text": query.text,
            "filters": {
                "categories": query.categories,
                "tags": query.tags,
                "date_range": {
                    "from": query.date_from,
                    "to": query.date_to
                }
            },
            "results_count": len(results),
            "result_ids": [r.doc_id for r in results[:5]],  # Top 5
            "search_time": search_time
        })
    
    def update_document_analytics(self, doc_id: str, action: str):
        """Update document analytics (views, helpful, etc.)"""
        if action == "view":
            self.collection.update_one(
                {"doc_id": doc_id},
                {"$inc": {"view_count": 1}}
            )
        elif action == "helpful":
            self.collection.update_one(
                {"doc_id": doc_id},
                {"$inc": {"helpful_count": 1}}
            )
    
    def get_search_analytics(self) -> Dict:
        """Get search analytics and insights"""
        # Popular searches
        popular_pipeline = [
            {"$group": {"_id": "$query_text", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        popular_searches = list(self.search_history.aggregate(popular_pipeline))
        
        # Search volume over time
        time_pipeline = [
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$timestamp"
                        }
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id": 1}},
            {"$limit": 30}
        ]
        search_volume = list(self.search_history.aggregate(time_pipeline))
        
        # Most viewed documents
        most_viewed = list(self.collection.find().sort("view_count", -1).limit(10))
        
        # Most helpful documents
        most_helpful = list(self.collection.find().sort("helpful_count", -1).limit(10))
        
        return {
            "popular_searches": [
                {"query": s["_id"], "count": s["count"]}
                for s in popular_searches
            ],
            "search_volume": [
                {"date": s["_id"], "count": s["count"]}
                for s in search_volume
            ],
            "most_viewed": [
                {"title": d["title"], "views": d["view_count"]}
                for d in most_viewed
            ],
            "most_helpful": [
                {"title": d["title"], "helpful": d["helpful_count"]}
                for d in most_helpful
            ]
        }
    
    def get_facets(self) -> Dict:
        """Get available facets for filtering"""
        # Get categories
        categories = self.collection.distinct("category")
        
        # Get tags with counts
        tag_pipeline = [
            {"$unwind": "$tags"},
            {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 20}
        ]
        tags = list(self.collection.aggregate(tag_pipeline))
        
        return {
            "categories": categories,
            "tags": [{"tag": t["_id"], "count": t["count"]} for t in tags]
        }

def create_sample_knowledge_base() -> List[Dict]:
    """Create comprehensive sample knowledge base"""
    return [
        # MongoDB Content
        {
            "title": "Introduction to MongoDB Atlas Vector Search",
            "content": "MongoDB Atlas Vector Search enables semantic search capabilities by allowing you to search based on meaning rather than exact text matches. It uses vector embeddings to find similar content.",
            "category": "MongoDB",
            "tags": ["vector-search", "atlas", "introduction"],
            "last_updated": datetime.utcnow() - timedelta(days=5)
        },
        {
            "title": "Creating Vector Search Indexes in MongoDB",
            "content": "To create a vector search index: 1. Navigate to Atlas Search in your cluster. 2. Click Create Index. 3. Use the JSON editor to define your vector field configuration with dimensions and similarity metric.",
            "category": "MongoDB",
            "tags": ["vector-search", "indexing", "tutorial"],
            "last_updated": datetime.utcnow() - timedelta(days=3)
        },
        
        # RAG Content
        {
            "title": "Building RAG Systems: Complete Guide",
            "content": "RAG (Retrieval-Augmented Generation) combines the power of semantic search with language models. This guide covers architecture, implementation, and optimization strategies for production RAG systems.",
            "category": "RAG",
            "tags": ["rag", "architecture", "guide"],
            "last_updated": datetime.utcnow() - timedelta(days=7)
        },
        {
            "title": "RAG Chunking Strategies",
            "content": "Effective chunking is crucial for RAG performance. Consider: 1. Token-based chunking for consistent sizes. 2. Semantic chunking for coherent segments. 3. Document-aware chunking for structured content.",
            "category": "RAG",
            "tags": ["rag", "chunking", "optimization"],
            "last_updated": datetime.utcnow() - timedelta(days=1)
        },
        
        # Embeddings Content
        {
            "title": "Comparing Embedding Models: OpenAI vs Voyage AI",
            "content": "OpenAI's ada-002 offers 1536 dimensions with general-purpose embeddings. Voyage AI's voyage-3-large provides 1024 dimensions with specialized embeddings for better domain-specific performance at lower cost.",
            "category": "Embeddings",
            "tags": ["embeddings", "comparison", "voyage-ai", "openai"],
            "last_updated": datetime.utcnow() - timedelta(days=2)
        },
        {
            "title": "Embedding Dimension Optimization",
            "content": "Choosing the right embedding dimensions affects both performance and cost. Higher dimensions capture more nuance but increase storage and computation. Consider your use case when selecting models.",
            "category": "Embeddings",
            "tags": ["embeddings", "optimization", "dimensions"],
            "last_updated": datetime.utcnow() - timedelta(days=4)
        },
        
        # Best Practices
        {
            "title": "Vector Search Best Practices",
            "content": "Key practices: 1. Normalize your embeddings. 2. Use appropriate similarity metrics (cosine for most cases). 3. Implement caching for frequent queries. 4. Monitor and optimize based on usage patterns.",
            "category": "Best Practices",
            "tags": ["best-practices", "vector-search", "optimization"],
            "last_updated": datetime.utcnow()
        },
        {
            "title": "Production RAG Deployment Checklist",
            "content": "Before deploying: ✓ Test with real data ✓ Implement monitoring ✓ Set up fallback mechanisms ✓ Configure rate limiting ✓ Plan for scaling ✓ Document API endpoints",
            "category": "Best Practices",
            "tags": ["deployment", "production", "checklist"],
            "last_updated": datetime.utcnow() - timedelta(hours=12)
        }
    ]

def demonstrate_knowledge_base_search():
    """Demonstrate the knowledge base search system"""
    print("🔍 KNOWLEDGE BASE SEARCH DEMO\n")
    
    # Initialize search system
    kb_search = KnowledgeBaseSearch()
    
    # Ingest sample documents
    documents = create_sample_knowledge_base()
    kb_search.ingest_documents(documents)
    
    print("\n" + "="*60)
    print("🔍 Performing Searches")
    print("="*60)
    
    # Demo searches
    test_queries = [
        SearchQuery(
            text="How to create vector search index?",
            limit=3
        ),
        SearchQuery(
            text="RAG optimization strategies",
            categories=["RAG", "Best Practices"],
            limit=3
        ),
        SearchQuery(
            text="Voyage AI embeddings",
            tags=["embeddings"],
            date_from=datetime.utcnow() - timedelta(days=7),
            limit=3
        )
    ]
    
    for query in test_queries:
        print(f"\n\n{'='*60}")
        print(f"🔍 Query: '{query.text}'")
        if query.categories:
            print(f"   Categories: {query.categories}")
        if query.tags:
            print(f"   Tags: {query.tags}")
        print(f"{'='*60}")
        
        results, metadata = kb_search.search(query)
        
        print(f"\n📊 Search Metadata:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
        
        print(f"\n📄 Results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.title}")
            print(f"   Category: {result.category}")
            print(f"   Tags: {', '.join(result.tags)}")
            print(f"   Score: {result.score:.4f}")
            if result.relevance_score:
                print(f"   Rerank Score: {result.relevance_score:.4f}")
            print(f"   Views: {result.view_count} | Helpful: {result.helpful_count}")
            
            if result.highlights:
                print(f"   Highlights:")
                for highlight in result.highlights:
                    print(f"     • {highlight}")
            
            if result.related_docs:
                print(f"   Related:")
                for related in result.related_docs:
                    print(f"     • {related}")
    
    # Show facets
    print(f"\n\n{'='*60}")
    print("📊 Available Facets")
    print(f"{'='*60}")
    
    facets = kb_search.get_facets()
    print(f"\nCategories: {', '.join(facets['categories'])}")
    print(f"\nTop Tags:")
    for tag_info in facets['tags'][:10]:
        print(f"  • {tag_info['tag']} ({tag_info['count']} docs)")
    
    # Simulate document interactions
    print(f"\n\n{'='*60}")
    print("📈 Simulating User Interactions")
    print(f"{'='*60}")
    
    # Simulate some views and helpful votes
    if results:
        kb_search.update_document_analytics(results[0].doc_id, "view")
        kb_search.update_document_analytics(results[0].doc_id, "view")
        kb_search.update_document_analytics(results[0].doc_id, "helpful")
        print(f"✅ Updated analytics for: {results[0].title}")

def show_search_analytics():
    """Display search analytics"""
    print(f"\n\n{'='*60}")
    print("📊 SEARCH ANALYTICS")
    print(f"{'='*60}")
    
    kb_search = KnowledgeBaseSearch()
    analytics = kb_search.get_search_analytics()
    
    print("\n🔥 Popular Searches:")
    for search in analytics["popular_searches"][:5]:
        print(f"  • '{search['query']}' ({search['count']} searches)")
    
    print("\n📈 Search Volume (Last 7 Days):")
    for day in analytics["search_volume"][-7:]:
        print(f"  • {day['date']}: {day['count']} searches")
    
    print("\n👀 Most Viewed Documents:")
    for doc in analytics["most_viewed"][:5]:
        print(f"  • {doc['title']} ({doc['views']} views)")
    
    print("\n👍 Most Helpful Documents:")
    for doc in analytics["most_helpful"][:5]:
        print(f"  • {doc['title']} ({doc['helpful']} helpful votes)")

if __name__ == "__main__":
    print("🎓 MongoDB RAG Course - Knowledge Base Search\n")
    
    try:
        # Run demonstration
        demonstrate_knowledge_base_search()
        
        # Show analytics
        show_search_analytics()
        
        print("\n\n🎉 Key Features Demonstrated:")
        print("✅ Multi-modal search (vector + text)")
        print("✅ Faceted filtering")
        print("✅ Result highlighting")
        print("✅ Related document suggestions")
        print("✅ Search analytics")
        print("✅ User interaction tracking")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("🔧 Ensure MongoDB connection and indexes are configured")