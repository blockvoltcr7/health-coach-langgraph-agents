"""
Module 4.1: Error Handling & Resilience
Time: 15 minutes
Goal: Build robust RAG systems that handle failures gracefully
"""

import os
import time
import asyncio
from typing import Optional, Dict, Any, List, Callable, TypeVar, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import wraps
import json
import logging
from enum import Enum
import traceback
from collections import defaultdict
import numpy as np
from pymongo import MongoClient
from openai import OpenAI
import voyageai
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')

class ErrorType(Enum):
    """Categories of errors in RAG systems"""
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    NETWORK_ERROR = "network_error"
    DATABASE_ERROR = "database_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class ErrorContext:
    """Context for error handling"""
    error_type: ErrorType
    error_message: str
    operation: str
    timestamp: datetime
    retry_count: int = 0
    metadata: Dict[str, Any] = None

class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    Prevents cascading failures by stopping requests to failing services
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise Exception(f"Circuit breaker is open. Service unavailable.")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset the circuit"""
        return (
            self.last_failure_time and
            datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

class RetryStrategy:
    """
    Configurable retry strategy with exponential backoff
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        exponential_base: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.exponential_base = exponential_base
        self.max_delay = max_delay
        self.jitter = jitter
    
    def get_delay(self, retry_count: int) -> float:
        """Calculate delay for given retry count"""
        delay = min(
            self.initial_delay * (self.exponential_base ** retry_count),
            self.max_delay
        )
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            delay = delay * (0.5 + np.random.random() * 0.5)
        
        return delay

def with_retry(
    retry_strategy: Optional[RetryStrategy] = None,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Decorator for adding retry logic to functions
    """
    if retry_strategy is None:
        retry_strategy = RetryStrategy()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for retry_count in range(retry_strategy.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if retry_count < retry_strategy.max_retries:
                        delay = retry_strategy.get_delay(retry_count)
                        
                        if on_retry:
                            on_retry(e, retry_count, delay)
                        
                        logger.warning(
                            f"Retry {retry_count + 1}/{retry_strategy.max_retries} "
                            f"for {func.__name__} after {delay:.2f}s delay. "
                            f"Error: {str(e)}"
                        )
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All retries exhausted for {func.__name__}. "
                            f"Final error: {str(e)}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator

class ResilientRAGSystem:
    """
    Production-ready RAG system with comprehensive error handling
    """
    
    def __init__(self, mongodb_uri: str, database_name: str):
        # Initialize with resilient connections
        self.mongodb_client = self._create_resilient_mongo_client(mongodb_uri)
        self.db = self.mongodb_client[database_name]
        
        # Initialize API clients with circuit breakers
        self.openai_circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        self.voyage_circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_AI_API_KEY"))
        
        # Error tracking
        self.error_collection = self.db["system_errors"]
        self.health_collection = self.db["system_health"]
        
        # Fallback configurations
        self.fallback_responses = {
            "general": "I'm having trouble accessing the information right now. Please try again in a moment.",
            "rate_limit": "We're experiencing high demand. Please wait a moment before trying again.",
            "database": "I'm having trouble accessing my knowledge base. Our team has been notified.",
            "api": "I'm temporarily unable to process your request. Please try again shortly."
        }
    
    def _create_resilient_mongo_client(self, uri: str) -> MongoClient:
        """Create MongoDB client with connection pooling and timeouts"""
        return MongoClient(
            uri,
            maxPoolSize=50,
            minPoolSize=10,
            maxIdleTimeMS=30000,
            waitQueueTimeoutMS=5000,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=5000,
            retryWrites=True,
            retryReads=True
        )
    
    @with_retry(
        retry_strategy=RetryStrategy(max_retries=3, initial_delay=0.5),
        retryable_exceptions=(Exception,)
    )
    def generate_embedding_with_fallback(self, text: str) -> Tuple[List[float], str]:
        """Generate embedding with automatic provider fallback"""
        # Try Voyage AI first
        if os.getenv("VOYAGE_AI_API_KEY"):
            try:
                result = self.voyage_circuit.call(
                    self._voyage_embed,
                    text
                )
                return result, "voyage"
            except Exception as e:
                logger.warning(f"Voyage AI embedding failed: {e}")
        
        # Fallback to OpenAI
        try:
            result = self.openai_circuit.call(
                self._openai_embed,
                text
            )
            return result, "openai"
        except Exception as e:
            logger.error(f"All embedding providers failed: {e}")
            raise
    
    def _voyage_embed(self, text: str) -> List[float]:
        """Generate embedding using Voyage AI"""
        result = self.voyage_client.embed(
            texts=[text],
            model="voyage-3-large",
            input_type="document"
        )
        return result.embeddings[0]
    
    def _openai_embed(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    
    async def search_with_resilience(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        timeout: float = 10.0
    ) -> List[Dict]:
        """Perform search with comprehensive error handling"""
        try:
            # Set operation timeout
            search_task = asyncio.create_task(
                self._async_search(query, collection_name, limit)
            )
            
            results = await asyncio.wait_for(search_task, timeout=timeout)
            
            # Log successful search
            self._log_operation_health("search", True)
            
            return results
            
        except asyncio.TimeoutError:
            self._handle_error(
                ErrorType.TIMEOUT_ERROR,
                f"Search timeout after {timeout}s",
                "search",
                {"query": query, "collection": collection_name}
            )
            return self._get_cached_results(query) or []
            
        except Exception as e:
            self._handle_error(
                ErrorType.UNKNOWN_ERROR,
                str(e),
                "search",
                {"query": query, "collection": collection_name}
            )
            return []
    
    async def _async_search(self, query: str, collection_name: str, limit: int) -> List[Dict]:
        """Async search implementation"""
        # This would be your actual async search implementation
        # For demo, using sync version wrapped in async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_search, query, collection_name, limit)
    
    def _sync_search(self, query: str, collection_name: str, limit: int) -> List[Dict]:
        """Synchronous search implementation"""
        collection = self.db[collection_name]
        
        # Generate embedding
        embedding, provider = self.generate_embedding_with_fallback(query)
        
        # Adjust dimensions based on provider
        if provider == "voyage" and len(embedding) == 1024:
            index_name = "voyage_vector_index"
        else:
            index_name = "vector_index"
        
        # Perform search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": limit * 10,
                    "limit": limit
                }
            }
        ]
        
        return list(collection.aggregate(pipeline))
    
    def generate_response_with_fallback(
        self,
        query: str,
        context: List[Dict],
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Generate response with multiple fallback strategies"""
        try:
            # Try primary response generation
            response = self._generate_gpt_response(query, context, conversation_history)
            self._log_operation_health("generation", True)
            return response
            
        except Exception as e:
            error_type = self._classify_error(e)
            self._handle_error(error_type, str(e), "generation", {"query": query})
            
            # Try fallback strategies
            if error_type == ErrorType.RATE_LIMIT:
                # Wait and retry with smaller model
                time.sleep(5)
                try:
                    return self._generate_gpt_response(
                        query, context, conversation_history,
                        model="gpt-3.5-turbo"  # Fallback to cheaper model
                    )
                except:
                    pass
            
            # Use template-based fallback
            return self._generate_template_response(query, context)
    
    def _generate_gpt_response(
        self,
        query: str,
        context: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
        model: str = "gpt-3.5-turbo"
    ) -> str:
        """Generate response using GPT"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."}
        ]
        
        if conversation_history:
            messages.extend(conversation_history[-5:])  # Last 5 messages
        
        context_text = "\n\n".join([doc.get("content", "") for doc in context])
        messages.append({
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {query}"
        })
        
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def _generate_template_response(self, query: str, context: List[Dict]) -> str:
        """Generate response using templates (ultimate fallback)"""
        if not context:
            return self.fallback_responses["general"]
        
        # Simple template-based response
        top_result = context[0]
        return f"Based on my knowledge base: {top_result.get('content', 'No information available.')}"
    
    def _handle_error(
        self,
        error_type: ErrorType,
        error_message: str,
        operation: str,
        metadata: Optional[Dict] = None
    ):
        """Centralized error handling"""
        error_context = ErrorContext(
            error_type=error_type,
            error_message=error_message,
            operation=operation,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Log to database
        self.error_collection.insert_one({
            "type": error_type.value,
            "message": error_message,
            "operation": operation,
            "timestamp": error_context.timestamp,
            "metadata": metadata
        })
        
        # Log to file/console
        logger.error(f"[{error_type.value}] {operation}: {error_message}")
        
        # Send alerts for critical errors
        if error_type in [ErrorType.DATABASE_ERROR, ErrorType.API_ERROR]:
            self._send_alert(error_context)
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type based on exception"""
        error_str = str(error).lower()
        
        if "rate limit" in error_str or "429" in error_str:
            return ErrorType.RATE_LIMIT
        elif "timeout" in error_str:
            return ErrorType.TIMEOUT_ERROR
        elif "connection" in error_str or "network" in error_str:
            return ErrorType.NETWORK_ERROR
        elif "database" in error_str or "mongo" in error_str:
            return ErrorType.DATABASE_ERROR
        elif "validation" in error_str or "invalid" in error_str:
            return ErrorType.VALIDATION_ERROR
        elif "api" in error_str:
            return ErrorType.API_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR
    
    def _log_operation_health(self, operation: str, success: bool):
        """Log operation health metrics"""
        self.health_collection.insert_one({
            "operation": operation,
            "success": success,
            "timestamp": datetime.utcnow()
        })
    
    def _get_cached_results(self, query: str) -> Optional[List[Dict]]:
        """Get cached results for query (if available)"""
        # Implement caching logic here
        # For demo, returning None
        return None
    
    def _send_alert(self, error_context: ErrorContext):
        """Send alert for critical errors"""
        # In production, integrate with:
        # - PagerDuty
        # - Slack
        # - Email
        # - SMS
        logger.critical(f"ALERT: {error_context.error_type.value} - {error_context.error_message}")
    
    def get_system_health(self) -> Dict:
        """Get overall system health metrics"""
        # Calculate health metrics for last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        
        # Success rates by operation
        pipeline = [
            {"$match": {"timestamp": {"$gte": one_hour_ago}}},
            {
                "$group": {
                    "_id": "$operation",
                    "total": {"$sum": 1},
                    "success": {"$sum": {"$cond": ["$success", 1, 0]}}
                }
            }
        ]
        
        health_stats = list(self.health_collection.aggregate(pipeline))
        
        # Recent errors
        recent_errors = self.error_collection.count_documents({
            "timestamp": {"$gte": one_hour_ago}
        })
        
        # Circuit breaker states
        circuit_states = {
            "openai": self.openai_circuit.state,
            "voyage": self.voyage_circuit.state
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "operations": {
                stat["_id"]: {
                    "success_rate": (stat["success"] / stat["total"] * 100) if stat["total"] > 0 else 0,
                    "total_calls": stat["total"]
                }
                for stat in health_stats
            },
            "recent_errors": recent_errors,
            "circuit_breakers": circuit_states,
            "status": "healthy" if recent_errors < 10 else "degraded" if recent_errors < 50 else "unhealthy"
        }

def demonstrate_error_handling():
    """Demonstrate error handling and resilience patterns"""
    print("🛡️ ERROR HANDLING & RESILIENCE DEMO\n")
    
    # Initialize resilient system
    rag_system = ResilientRAGSystem(
        mongodb_uri=os.getenv("MONGODB_URI"),
        database_name=os.getenv("MONGODB_DATABASE", "rag_course")
    )
    
    # Demo 1: Retry with exponential backoff
    print("📊 Demo 1: Retry with Exponential Backoff")
    print("="*60)
    
    @with_retry(
        retry_strategy=RetryStrategy(max_retries=3, initial_delay=1.0),
        on_retry=lambda e, r, d: print(f"  Retry {r+1}: Waiting {d:.2f}s after error: {e}")
    )
    def flaky_operation():
        """Simulated flaky operation"""
        if np.random.random() < 0.7:  # 70% failure rate
            raise Exception("Random failure occurred")
        return "Success!"
    
    try:
        result = flaky_operation()
        print(f"✅ Operation succeeded: {result}")
    except Exception as e:
        print(f"❌ Operation failed after all retries: {e}")
    
    # Demo 2: Circuit breaker pattern
    print(f"\n\n📊 Demo 2: Circuit Breaker Pattern")
    print("="*60)
    
    circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=5)
    
    def unreliable_service():
        """Simulated unreliable service"""
        if np.random.random() < 0.8:  # 80% failure rate
            raise Exception("Service unavailable")
        return "Service response"
    
    for i in range(10):
        try:
            result = circuit.call(unreliable_service)
            print(f"  Call {i+1}: ✅ Success - {result}")
        except Exception as e:
            print(f"  Call {i+1}: ❌ Failed - {e} (Circuit: {circuit.state})")
        
        if i == 5:
            print("  ⏸️  Waiting for circuit recovery...")
            time.sleep(6)
    
    # Demo 3: Embedding fallback
    print(f"\n\n📊 Demo 3: Embedding Provider Fallback")
    print("="*60)
    
    test_text = "MongoDB vector search enables semantic search capabilities"
    
    try:
        embedding, provider = rag_system.generate_embedding_with_fallback(test_text)
        print(f"✅ Generated embedding using {provider}")
        print(f"   Dimensions: {len(embedding)}")
    except Exception as e:
        print(f"❌ All embedding providers failed: {e}")
    
    # Demo 4: Async search with timeout
    print(f"\n\n📊 Demo 4: Search with Timeout Protection")
    print("="*60)
    
    async def demo_timeout_search():
        """Demo async search with timeout"""
        try:
            results = await rag_system.search_with_resilience(
                "How to handle errors in RAG?",
                "knowledge_base",
                timeout=2.0
            )
            print(f"✅ Search completed: {len(results)} results")
        except Exception as e:
            print(f"❌ Search failed: {e}")
    
    # Run async demo
    asyncio.run(demo_timeout_search())
    
    # Demo 5: System health monitoring
    print(f"\n\n📊 Demo 5: System Health Monitoring")
    print("="*60)
    
    # Simulate some operations
    for _ in range(5):
        rag_system._log_operation_health("search", True)
    for _ in range(2):
        rag_system._log_operation_health("search", False)
    
    health = rag_system.get_system_health()
    print(f"System Status: {health['status'].upper()}")
    print(f"Recent Errors: {health['recent_errors']}")
    print(f"Circuit Breakers: {health['circuit_breakers']}")
    
    if health['operations']:
        print("\nOperation Success Rates:")
        for op, stats in health['operations'].items():
            print(f"  {op}: {stats['success_rate']:.1f}% ({stats['total_calls']} calls)")

def demonstrate_graceful_degradation():
    """Demonstrate graceful degradation strategies"""
    print(f"\n\n🔄 GRACEFUL DEGRADATION STRATEGIES")
    print("="*60)
    
    strategies = [
        {
            "name": "Cached Response Fallback",
            "description": "Use cached results when primary search fails",
            "implementation": """
# Cache recent searches
search_cache = {}

def search_with_cache(query):
    cache_key = hashlib.md5(query.encode()).hexdigest()
    
    try:
        # Try primary search
        results = perform_search(query)
        # Update cache
        search_cache[cache_key] = {
            'results': results,
            'timestamp': datetime.utcnow()
        }
        return results
    except Exception:
        # Use cached results if available
        if cache_key in search_cache:
            cached = search_cache[cache_key]
            age = datetime.utcnow() - cached['timestamp']
            if age < timedelta(hours=1):
                return cached['results']
        raise
"""
        },
        {
            "name": "Quality Degradation",
            "description": "Use simpler models when premium ones fail",
            "implementation": """
# Tiered model fallback
models = [
    {'name': 'gpt-4', 'quality': 'high', 'cost': 'high'},
    {'name': 'gpt-3.5-turbo', 'quality': 'medium', 'cost': 'medium'},
    {'name': 'template', 'quality': 'low', 'cost': 'free'}
]

def generate_with_fallback(prompt):
    for model in models:
        try:
            if model['name'] == 'template':
                return generate_template_response(prompt)
            else:
                return call_llm(model['name'], prompt)
        except Exception as e:
            logger.warning(f"Model {model['name']} failed: {e}")
            continue
    
    return "Service temporarily unavailable"
"""
        },
        {
            "name": "Feature Degradation",
            "description": "Disable non-essential features during high load",
            "implementation": """
# Load-based feature toggling
class FeatureFlags:
    def __init__(self):
        self.flags = {
            'reranking': True,
            'related_docs': True,
            'analytics': True,
            'spell_correction': True
        }
    
    def check_load(self):
        # Check system load
        if get_current_load() > 0.8:
            self.flags['reranking'] = False
            self.flags['related_docs'] = False
        
        if get_current_load() > 0.9:
            self.flags['analytics'] = False
            self.flags['spell_correction'] = False
    
    def is_enabled(self, feature):
        return self.flags.get(feature, False)
"""
        }
    ]
    
    for strategy in strategies:
        print(f"\n📋 {strategy['name']}")
        print(f"   {strategy['description']}")
        print(f"\nImplementation:")
        print(strategy['implementation'])

if __name__ == "__main__":
    print("🎓 MongoDB RAG Course - Error Handling & Resilience\n")
    
    try:
        # Run demonstrations
        demonstrate_error_handling()
        demonstrate_graceful_degradation()
        
        print("\n\n🎉 Key Resilience Patterns Demonstrated:")
        print("✅ Retry with exponential backoff")
        print("✅ Circuit breaker pattern")
        print("✅ Provider fallback strategies")
        print("✅ Timeout protection")
        print("✅ Health monitoring")
        print("✅ Graceful degradation")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        traceback.print_exc()