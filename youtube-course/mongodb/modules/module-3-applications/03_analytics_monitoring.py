"""
Module 3.3: Analytics & Monitoring System
Time: 20 minutes
Goal: Build comprehensive analytics and monitoring for RAG systems
"""

import os
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from pymongo import MongoClient, ASCENDING, DESCENDING
from openai import OpenAI
import time
import hashlib
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
mongodb_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongodb_client[os.getenv("MONGODB_DATABASE", "rag_course")]
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class TokenUsage:
    """Track token usage and costs"""
    timestamp: datetime
    operation: str  # "embedding", "completion", "rerank"
    model: str
    tokens: int
    cost: float
    provider: str  # "openai", "voyage", "anthropic"
    request_id: str
    user_id: Optional[str] = None

@dataclass
class PerformanceMetric:
    """Track system performance"""
    timestamp: datetime
    operation: str
    latency_ms: float
    success: bool
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class QualityMetric:
    """Track search and response quality"""
    timestamp: datetime
    query: str
    relevance_score: float
    user_feedback: Optional[int] = None  # 1-5 rating
    clicked_results: List[str] = field(default_factory=list)
    session_id: str = ""

class RAGAnalyticsMonitor:
    """
    Comprehensive analytics and monitoring for RAG systems:
    - Token usage and cost tracking
    - Performance monitoring
    - Quality metrics
    - User behavior analytics
    - System health monitoring
    - Cost optimization recommendations
    """
    
    # Pricing configuration (per 1K tokens)
    PRICING = {
        "openai": {
            "text-embedding-ada-002": {"input": 0.0001, "output": 0},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-4": {"input": 0.03, "output": 0.06}
        },
        "voyage": {
            "voyage-3-large": {"input": 0.00012, "output": 0},
            "rerank-2-lite": {"input": 0.00005, "output": 0}
        }
    }
    
    def __init__(self):
        # Collections
        self.token_usage = db["analytics_token_usage"]
        self.performance_metrics = db["analytics_performance"]
        self.quality_metrics = db["analytics_quality"]
        self.user_sessions = db["analytics_sessions"]
        self.system_health = db["analytics_health"]
        
        # Create indexes
        self._setup_indexes()
        
        # Initialize monitoring
        self.monitoring_interval = 60  # seconds
        self.alert_thresholds = {
            "error_rate": 0.05,  # 5%
            "latency_p95": 2000,  # 2 seconds
            "cost_per_hour": 10.0  # $10/hour
        }
    
    def _setup_indexes(self):
        """Setup indexes for analytics collections"""
        # Token usage indexes
        self.token_usage.create_index([("timestamp", DESCENDING)])
        self.token_usage.create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])
        self.token_usage.create_index([("provider", ASCENDING), ("model", ASCENDING)])
        
        # Performance indexes
        self.performance_metrics.create_index([("timestamp", DESCENDING)])
        self.performance_metrics.create_index([("operation", ASCENDING), ("timestamp", DESCENDING)])
        
        # Quality indexes
        self.quality_metrics.create_index([("timestamp", DESCENDING)])
        self.quality_metrics.create_index([("session_id", ASCENDING)])
        
        print("✅ Analytics indexes created")
    
    def track_token_usage(
        self,
        operation: str,
        model: str,
        tokens: int,
        provider: str = "openai",
        user_id: Optional[str] = None
    ) -> str:
        """Track token usage and calculate cost"""
        # Calculate cost
        pricing = self.PRICING.get(provider, {}).get(model, {"input": 0, "output": 0})
        
        if operation == "embedding":
            cost = (tokens / 1000) * pricing["input"]
        else:
            # For completions, assume 70% input, 30% output (rough estimate)
            input_tokens = int(tokens * 0.7)
            output_tokens = tokens - input_tokens
            cost = (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]
        
        # Create usage record
        request_id = hashlib.md5(f"{datetime.utcnow().isoformat()}_{model}_{tokens}".encode()).hexdigest()[:12]
        
        usage = TokenUsage(
            timestamp=datetime.utcnow(),
            operation=operation,
            model=model,
            tokens=tokens,
            cost=cost,
            provider=provider,
            request_id=request_id,
            user_id=user_id
        )
        
        # Store in database
        self.token_usage.insert_one(usage.__dict__)
        
        return request_id
    
    def track_performance(
        self,
        operation: str,
        latency_ms: float,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Track performance metrics"""
        metric = PerformanceMetric(
            timestamp=datetime.utcnow(),
            operation=operation,
            latency_ms=latency_ms,
            success=success,
            error=error,
            metadata=metadata or {}
        )
        
        self.performance_metrics.insert_one(metric.__dict__)
        
        # Check for alerts
        self._check_performance_alerts(operation, latency_ms, success)
    
    def track_quality(
        self,
        query: str,
        relevance_score: float,
        session_id: str,
        clicked_results: Optional[List[str]] = None,
        user_feedback: Optional[int] = None
    ):
        """Track search and response quality"""
        metric = QualityMetric(
            timestamp=datetime.utcnow(),
            query=query,
            relevance_score=relevance_score,
            user_feedback=user_feedback,
            clicked_results=clicked_results or [],
            session_id=session_id
        )
        
        self.quality_metrics.insert_one(metric.__dict__)
    
    def get_token_analytics(self, time_range: timedelta = timedelta(days=7)) -> Dict:
        """Get comprehensive token usage analytics"""
        start_time = datetime.utcnow() - time_range
        
        # Total usage and cost
        pipeline = [
            {"$match": {"timestamp": {"$gte": start_time}}},
            {
                "$group": {
                    "_id": None,
                    "total_tokens": {"$sum": "$tokens"},
                    "total_cost": {"$sum": "$cost"},
                    "request_count": {"$sum": 1}
                }
            }
        ]
        
        total_stats = list(self.token_usage.aggregate(pipeline))
        
        # Usage by provider
        provider_pipeline = [
            {"$match": {"timestamp": {"$gte": start_time}}},
            {
                "$group": {
                    "_id": "$provider",
                    "tokens": {"$sum": "$tokens"},
                    "cost": {"$sum": "$cost"},
                    "requests": {"$sum": 1}
                }
            }
        ]
        
        provider_stats = list(self.token_usage.aggregate(provider_pipeline))
        
        # Usage by model
        model_pipeline = [
            {"$match": {"timestamp": {"$gte": start_time}}},
            {
                "$group": {
                    "_id": {"provider": "$provider", "model": "$model"},
                    "tokens": {"$sum": "$tokens"},
                    "cost": {"$sum": "$cost"},
                    "requests": {"$sum": 1}
                }
            },
            {"$sort": {"cost": -1}}
        ]
        
        model_stats = list(self.token_usage.aggregate(model_pipeline))
        
        # Cost over time
        time_pipeline = [
            {"$match": {"timestamp": {"$gte": start_time}}},
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$timestamp"
                        }
                    },
                    "daily_cost": {"$sum": "$cost"},
                    "daily_tokens": {"$sum": "$tokens"}
                }
            },
            {"$sort": {"_id": 1}}
        ]
        
        daily_costs = list(self.token_usage.aggregate(time_pipeline))
        
        # Top users by cost
        user_pipeline = [
            {"$match": {"timestamp": {"$gte": start_time}, "user_id": {"$ne": None}}},
            {
                "$group": {
                    "_id": "$user_id",
                    "total_cost": {"$sum": "$cost"},
                    "total_tokens": {"$sum": "$tokens"},
                    "request_count": {"$sum": 1}
                }
            },
            {"$sort": {"total_cost": -1}},
            {"$limit": 10}
        ]
        
        top_users = list(self.token_usage.aggregate(user_pipeline))
        
        return {
            "summary": {
                "total_tokens": total_stats[0]["total_tokens"] if total_stats else 0,
                "total_cost": f"${total_stats[0]['total_cost']:.2f}" if total_stats else "$0.00",
                "total_requests": total_stats[0]["request_count"] if total_stats else 0,
                "time_range_days": time_range.days
            },
            "by_provider": [
                {
                    "provider": stat["_id"],
                    "tokens": f"{stat['tokens']:,}",
                    "cost": f"${stat['cost']:.2f}",
                    "requests": stat["requests"]
                }
                for stat in provider_stats
            ],
            "by_model": [
                {
                    "provider": stat["_id"]["provider"],
                    "model": stat["_id"]["model"],
                    "tokens": f"{stat['tokens']:,}",
                    "cost": f"${stat['cost']:.2f}",
                    "requests": stat["requests"]
                }
                for stat in model_stats[:10]
            ],
            "daily_costs": [
                {
                    "date": day["_id"],
                    "cost": f"${day['daily_cost']:.2f}",
                    "tokens": f"{day['daily_tokens']:,}"
                }
                for day in daily_costs
            ],
            "top_users": [
                {
                    "user_id": user["_id"],
                    "cost": f"${user['total_cost']:.2f}",
                    "tokens": f"{user['total_tokens']:,}",
                    "requests": user["request_count"]
                }
                for user in top_users
            ]
        }
    
    def get_performance_analytics(self, time_range: timedelta = timedelta(hours=24)) -> Dict:
        """Get performance analytics"""
        start_time = datetime.utcnow() - time_range
        
        # Overall performance
        perf_pipeline = [
            {"$match": {"timestamp": {"$gte": start_time}}},
            {
                "$group": {
                    "_id": "$operation",
                    "count": {"$sum": 1},
                    "success_count": {"$sum": {"$cond": ["$success", 1, 0]}},
                    "avg_latency": {"$avg": "$latency_ms"},
                    "p95_latency": {"$percentile": {"input": "$latency_ms", "p": [0.95], "method": "approximate"}}
                }
            }
        ]
        
        # Note: $percentile requires MongoDB 7.0+
        # Fallback to simpler aggregation if not available
        try:
            perf_stats = list(self.performance_metrics.aggregate(perf_pipeline))
        except:
            # Simplified pipeline without percentiles
            perf_pipeline = [
                {"$match": {"timestamp": {"$gte": start_time}}},
                {
                    "$group": {
                        "_id": "$operation",
                        "count": {"$sum": 1},
                        "success_count": {"$sum": {"$cond": ["$success", 1, 0]}},
                        "avg_latency": {"$avg": "$latency_ms"},
                        "max_latency": {"$max": "$latency_ms"}
                    }
                }
            ]
            perf_stats = list(self.performance_metrics.aggregate(perf_pipeline))
        
        # Error analysis
        error_pipeline = [
            {"$match": {"timestamp": {"$gte": start_time}, "success": False}},
            {
                "$group": {
                    "_id": {"operation": "$operation", "error": "$error"},
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        
        error_stats = list(self.performance_metrics.aggregate(error_pipeline))
        
        # Latency over time
        latency_pipeline = [
            {"$match": {"timestamp": {"$gte": start_time}}},
            {
                "$group": {
                    "_id": {
                        "time": {
                            "$dateToString": {
                                "format": "%Y-%m-%d %H:00",
                                "date": "$timestamp"
                            }
                        },
                        "operation": "$operation"
                    },
                    "avg_latency": {"$avg": "$latency_ms"},
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id.time": 1}}
        ]
        
        latency_trends = list(self.performance_metrics.aggregate(latency_pipeline))
        
        return {
            "operations": [
                {
                    "operation": stat["_id"],
                    "total_requests": stat["count"],
                    "success_rate": f"{(stat['success_count'] / stat['count'] * 100):.1f}%",
                    "avg_latency_ms": f"{stat['avg_latency']:.0f}",
                    "p95_latency_ms": f"{stat.get('p95_latency', [stat.get('max_latency', 0)])[0]:.0f}"
                }
                for stat in perf_stats
            ],
            "errors": [
                {
                    "operation": err["_id"]["operation"],
                    "error": err["_id"]["error"] or "Unknown",
                    "count": err["count"]
                }
                for err in error_stats
            ],
            "latency_trends": latency_trends
        }
    
    def get_quality_analytics(self, time_range: timedelta = timedelta(days=7)) -> Dict:
        """Get quality metrics analytics"""
        start_time = datetime.utcnow() - time_range
        
        # Average relevance scores
        relevance_pipeline = [
            {"$match": {"timestamp": {"$gte": start_time}}},
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$timestamp"
                        }
                    },
                    "avg_relevance": {"$avg": "$relevance_score"},
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id": 1}}
        ]
        
        relevance_trends = list(self.quality_metrics.aggregate(relevance_pipeline))
        
        # User feedback stats
        feedback_pipeline = [
            {"$match": {"timestamp": {"$gte": start_time}, "user_feedback": {"$ne": None}}},
            {
                "$group": {
                    "_id": "$user_feedback",
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id": 1}}
        ]
        
        feedback_stats = list(self.quality_metrics.aggregate(feedback_pipeline))
        
        # Click-through rates
        ctr_pipeline = [
            {"$match": {"timestamp": {"$gte": start_time}}},
            {
                "$project": {
                    "has_clicks": {"$gt": [{"$size": "$clicked_results"}, 0]}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "with_clicks": {"$sum": {"$cond": ["$has_clicks", 1, 0]}}
                }
            }
        ]
        
        ctr_stats = list(self.quality_metrics.aggregate(ctr_pipeline))
        
        # Popular queries
        query_pipeline = [
            {"$match": {"timestamp": {"$gte": start_time}}},
            {
                "$group": {
                    "_id": "$query",
                    "count": {"$sum": 1},
                    "avg_relevance": {"$avg": "$relevance_score"}
                }
            },
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        
        popular_queries = list(self.quality_metrics.aggregate(query_pipeline))
        
        return {
            "relevance_trends": [
                {
                    "date": trend["_id"],
                    "avg_relevance": f"{trend['avg_relevance']:.3f}",
                    "queries": trend["count"]
                }
                for trend in relevance_trends
            ],
            "user_feedback": {
                f"rating_{stat['_id']}": stat["count"]
                for stat in feedback_stats
            },
            "click_through_rate": f"{(ctr_stats[0]['with_clicks'] / ctr_stats[0]['total'] * 100):.1f}%" if ctr_stats else "0%",
            "popular_queries": [
                {
                    "query": q["_id"],
                    "count": q["count"],
                    "avg_relevance": f"{q['avg_relevance']:.3f}"
                }
                for q in popular_queries
            ]
        }
    
    def get_cost_optimization_recommendations(self) -> List[Dict]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        # Analyze recent usage
        recent_usage = self.get_token_analytics(timedelta(days=7))
        
        # Check for expensive models
        if recent_usage["by_model"]:
            top_model = recent_usage["by_model"][0]
            if "gpt-4" in top_model["model"]:
                recommendations.append({
                    "priority": "high",
                    "category": "model_selection",
                    "recommendation": "Consider using GPT-3.5-turbo instead of GPT-4 for non-critical queries",
                    "potential_savings": "Up to 90% cost reduction",
                    "implementation": "Add query classification to route simple queries to cheaper models"
                })
        
        # Check for Voyage AI usage
        voyage_usage = sum(1 for p in recent_usage["by_provider"] if p["provider"] == "voyage")
        if voyage_usage == 0:
            recommendations.append({
                "priority": "high",
                "category": "embedding_provider",
                "recommendation": "Switch to Voyage AI for embeddings",
                "potential_savings": "20-30% cost reduction with better quality",
                "implementation": "Configure Voyage AI API key and update embedding provider"
            })
        
        # Check for caching opportunities
        perf_data = self.get_performance_analytics(timedelta(hours=24))
        embedding_ops = next((op for op in perf_data["operations"] if op["operation"] == "embedding"), None)
        
        if embedding_ops and embedding_ops["total_requests"] > 1000:
            recommendations.append({
                "priority": "medium",
                "category": "caching",
                "recommendation": "Implement embedding caching for frequent queries",
                "potential_savings": "Reduce embedding API calls by 40-60%",
                "implementation": "Add Redis or MongoDB caching layer with TTL"
            })
        
        # Check for batch processing
        hourly_costs = recent_usage["daily_costs"]
        if hourly_costs and len(hourly_costs) > 1:
            cost_variance = np.std([float(d["cost"].replace("$", "")) for d in hourly_costs])
            if cost_variance > 10:
                recommendations.append({
                    "priority": "medium",
                    "category": "batch_processing",
                    "recommendation": "Implement batch processing for embeddings",
                    "potential_savings": "10-20% through better API utilization",
                    "implementation": "Queue embeddings and process in batches of 100"
                })
        
        return recommendations
    
    def _check_performance_alerts(self, operation: str, latency_ms: float, success: bool):
        """Check for performance alerts"""
        # Calculate recent error rate
        recent_errors = self.performance_metrics.count_documents({
            "operation": operation,
            "success": False,
            "timestamp": {"$gte": datetime.utcnow() - timedelta(minutes=5)}
        })
        
        recent_total = self.performance_metrics.count_documents({
            "operation": operation,
            "timestamp": {"$gte": datetime.utcnow() - timedelta(minutes=5)}
        })
        
        if recent_total > 10:
            error_rate = recent_errors / recent_total
            if error_rate > self.alert_thresholds["error_rate"]:
                self._send_alert(
                    "high",
                    f"High error rate for {operation}",
                    f"Error rate: {error_rate:.1%} (threshold: {self.alert_thresholds['error_rate']:.1%})"
                )
        
        # Check latency
        if latency_ms > self.alert_thresholds["latency_p95"]:
            self._send_alert(
                "medium",
                f"High latency for {operation}",
                f"Latency: {latency_ms:.0f}ms (threshold: {self.alert_thresholds['latency_p95']}ms)"
            )
    
    def _send_alert(self, severity: str, title: str, message: str):
        """Send alert (placeholder for actual alerting)"""
        alert = {
            "timestamp": datetime.utcnow(),
            "severity": severity,
            "title": title,
            "message": message,
            "acknowledged": False
        }
        
        self.system_health.insert_one(alert)
        
        # In production, integrate with:
        # - PagerDuty
        # - Slack
        # - Email
        # - SMS
        print(f"\n🚨 ALERT [{severity.upper()}]: {title}\n   {message}")

def demonstrate_analytics_monitoring():
    """Demonstrate the analytics and monitoring system"""
    print("📊 ANALYTICS & MONITORING DEMO\n")
    
    # Initialize monitor
    monitor = RAGAnalyticsMonitor()
    
    # Simulate various operations
    print("🔄 Simulating RAG operations...")
    
    # Simulate token usage
    operations = [
        # Embeddings
        {"op": "embedding", "model": "text-embedding-ada-002", "tokens": 150, "provider": "openai"},
        {"op": "embedding", "model": "voyage-3-large", "tokens": 120, "provider": "voyage"},
        {"op": "embedding", "model": "text-embedding-ada-002", "tokens": 200, "provider": "openai"},
        
        # Completions
        {"op": "completion", "model": "gpt-3.5-turbo", "tokens": 500, "provider": "openai"},
        {"op": "completion", "model": "gpt-4", "tokens": 800, "provider": "openai"},
        {"op": "completion", "model": "gpt-3.5-turbo", "tokens": 300, "provider": "openai"},
        
        # Reranking
        {"op": "rerank", "model": "rerank-2-lite", "tokens": 1000, "provider": "voyage"},
    ]
    
    for op in operations * 3:  # Simulate multiple requests
        request_id = monitor.track_token_usage(
            op["op"],
            op["model"],
            op["tokens"] + np.random.randint(-50, 50),
            op["provider"],
            f"user_{np.random.randint(1, 4)}"
        )
        
        # Simulate performance
        latency = np.random.normal(
            200 if op["op"] == "embedding" else 500,
            50 if op["op"] == "embedding" else 200
        )
        
        success = np.random.random() > 0.05  # 95% success rate
        
        monitor.track_performance(
            op["op"],
            max(10, latency),
            success,
            "Timeout" if not success and np.random.random() > 0.5 else None
        )
        
        # Simulate quality metrics for searches
        if op["op"] == "embedding" and np.random.random() > 0.5:
            monitor.track_quality(
                f"Sample query {np.random.randint(1, 10)}",
                np.random.uniform(0.6, 0.95),
                f"session_{np.random.randint(1, 20)}",
                [f"doc_{i}" for i in range(np.random.randint(0, 3))],
                np.random.choice([None, 3, 4, 5], p=[0.6, 0.1, 0.2, 0.1])
            )
    
    print("✅ Simulated operations complete\n")
    
    # Display analytics
    print("="*60)
    print("💰 TOKEN USAGE ANALYTICS")
    print("="*60)
    
    token_analytics = monitor.get_token_analytics(timedelta(days=1))
    
    print(f"\n📊 Summary (Last {token_analytics['summary']['time_range_days']} days):")
    print(f"   Total Tokens: {token_analytics['summary']['total_tokens']:,}")
    print(f"   Total Cost: {token_analytics['summary']['total_cost']}")
    print(f"   Total Requests: {token_analytics['summary']['total_requests']}")
    
    print(f"\n💳 Cost by Provider:")
    for provider in token_analytics["by_provider"]:
        print(f"   {provider['provider']}: {provider['cost']} ({provider['tokens']} tokens)")
    
    print(f"\n🏷️ Top Models by Cost:")
    for model in token_analytics["by_model"][:5]:
        print(f"   {model['provider']}/{model['model']}: {model['cost']}")
    
    print(f"\n👥 Top Users by Cost:")
    for user in token_analytics["top_users"][:3]:
        print(f"   {user['user_id']}: {user['cost']} ({user['requests']} requests)")
    
    # Performance analytics
    print(f"\n\n{'='*60}")
    print("⚡ PERFORMANCE ANALYTICS")
    print("="*60)
    
    perf_analytics = monitor.get_performance_analytics(timedelta(hours=1))
    
    print(f"\n📈 Operation Performance:")
    for op in perf_analytics["operations"]:
        print(f"\n   {op['operation']}:")
        print(f"      Requests: {op['total_requests']}")
        print(f"      Success Rate: {op['success_rate']}")
        print(f"      Avg Latency: {op['avg_latency_ms']}ms")
        print(f"      P95 Latency: {op['p95_latency_ms']}ms")
    
    if perf_analytics["errors"]:
        print(f"\n❌ Recent Errors:")
        for err in perf_analytics["errors"][:5]:
            print(f"   {err['operation']}: {err['error']} ({err['count']} times)")
    
    # Quality analytics
    print(f"\n\n{'='*60}")
    print("🎯 QUALITY ANALYTICS")
    print("="*60)
    
    quality_analytics = monitor.get_quality_analytics(timedelta(days=1))
    
    print(f"\n📊 Search Quality:")
    print(f"   Click-through Rate: {quality_analytics['click_through_rate']}")
    
    if quality_analytics["user_feedback"]:
        print(f"\n⭐ User Ratings:")
        for rating, count in sorted(quality_analytics["user_feedback"].items()):
            print(f"   {rating}: {'⭐' * int(rating.split('_')[1])} ({count} ratings)")
    
    if quality_analytics["popular_queries"]:
        print(f"\n🔍 Popular Queries:")
        for query in quality_analytics["popular_queries"][:5]:
            print(f"   '{query['query']}' ({query['count']} times, relevance: {query['avg_relevance']})")
    
    # Cost optimization
    print(f"\n\n{'='*60}")
    print("💡 COST OPTIMIZATION RECOMMENDATIONS")
    print("="*60)
    
    recommendations = monitor.get_cost_optimization_recommendations()
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. [{rec['priority'].upper()}] {rec['recommendation']}")
        print(f"   Category: {rec['category']}")
        print(f"   Savings: {rec['potential_savings']}")
        print(f"   How: {rec['implementation']}")

def create_monitoring_dashboard():
    """Create a simple monitoring dashboard summary"""
    print(f"\n\n{'='*60}")
    print("📊 MONITORING DASHBOARD")
    print("="*60)
    
    monitor = RAGAnalyticsMonitor()
    
    # System health check
    recent_alerts = monitor.system_health.count_documents({
        "timestamp": {"$gte": datetime.utcnow() - timedelta(hours=1)},
        "acknowledged": False
    })
    
    print(f"\n🏥 System Health:")
    print(f"   Active Alerts: {recent_alerts}")
    print(f"   Status: {'⚠️  WARNING' if recent_alerts > 0 else '✅ HEALTHY'}")
    
    # Quick stats
    last_hour = datetime.utcnow() - timedelta(hours=1)
    
    recent_requests = monitor.token_usage.count_documents({"timestamp": {"$gte": last_hour}})
    recent_errors = monitor.performance_metrics.count_documents({
        "timestamp": {"$gte": last_hour},
        "success": False
    })
    
    hourly_cost = monitor.token_usage.aggregate([
        {"$match": {"timestamp": {"$gte": last_hour}}},
        {"$group": {"_id": None, "total": {"$sum": "$cost"}}}
    ])
    
    hourly_cost_value = next(hourly_cost, {}).get("total", 0)
    
    print(f"\n📈 Last Hour Stats:")
    print(f"   Requests: {recent_requests}")
    print(f"   Errors: {recent_errors}")
    print(f"   Cost: ${hourly_cost_value:.2f}")
    print(f"   Projected Daily: ${hourly_cost_value * 24:.2f}")

if __name__ == "__main__":
    print("🎓 MongoDB RAG Course - Analytics & Monitoring\n")
    
    try:
        # Run demonstration
        demonstrate_analytics_monitoring()
        
        # Show dashboard
        create_monitoring_dashboard()
        
        print("\n\n🎉 Key Features Demonstrated:")
        print("✅ Token usage and cost tracking")
        print("✅ Performance monitoring with alerts")
        print("✅ Quality metrics and user feedback")
        print("✅ Cost optimization recommendations")
        print("✅ Real-time monitoring dashboard")
        print("✅ Comprehensive analytics")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Ensure MongoDB connection is configured")