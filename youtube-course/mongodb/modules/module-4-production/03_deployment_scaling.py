"""
Module 4.3: Deployment & Scaling
Time: 15 minutes  
Goal: Deploy and scale RAG systems for production workloads
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
from datetime import datetime
import json
import redis
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from functools import lru_cache
import hashlib
import numpy as np
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DeploymentConfig:
    """Configuration for RAG deployment"""
    environment: str  # dev, staging, production
    api_replicas: int
    worker_processes: int
    cache_ttl: int  # seconds
    max_batch_size: int
    request_timeout: int  # seconds
    auto_scale_min: int
    auto_scale_max: int
    cpu_threshold: float  # 0-1
    memory_threshold: float  # 0-1

class CacheManager:
    """
    Distributed caching for RAG systems
    Reduces API calls and improves response time
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()
            self.enabled = True
            print("✅ Redis cache connected")
        except:
            self.redis_client = None
            self.enabled = False
            self.local_cache = {}
            print("⚠️  Redis unavailable, using local cache")
    
    def get_embedding_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for embeddings"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding:{model}:{text_hash}"
    
    def get_search_cache_key(self, query: str, filters: Optional[Dict] = None) -> str:
        """Generate cache key for search results"""
        filter_str = json.dumps(filters, sort_keys=True) if filters else ""
        combined = f"{query}:{filter_str}"
        query_hash = hashlib.md5(combined.encode()).hexdigest()
        return f"search:{query_hash}"
    
    async def get_cached_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding if available"""
        if not self.enabled:
            return self.local_cache.get(self.get_embedding_cache_key(text, model))
        
        try:
            key = self.get_embedding_cache_key(text, model)
            cached = self.redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Cache get error: {e}")
        
        return None
    
    async def cache_embedding(self, text: str, model: str, embedding: List[float], ttl: int = 3600):
        """Cache embedding with TTL"""
        key = self.get_embedding_cache_key(text, model)
        
        if not self.enabled:
            self.local_cache[key] = embedding
            return
        
        try:
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(embedding)
            )
        except Exception as e:
            print(f"Cache set error: {e}")
    
    async def get_cached_search(self, query: str, filters: Optional[Dict] = None) -> Optional[List[Dict]]:
        """Get cached search results"""
        if not self.enabled:
            return None
        
        try:
            key = self.get_search_cache_key(query, filters)
            cached = self.redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Cache get error: {e}")
        
        return None
    
    async def cache_search_results(
        self,
        query: str,
        results: List[Dict],
        filters: Optional[Dict] = None,
        ttl: int = 300
    ):
        """Cache search results with shorter TTL"""
        if not self.enabled:
            return
        
        try:
            key = self.get_search_cache_key(query, filters)
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(results)
            )
        except Exception as e:
            print(f"Cache set error: {e}")

class BatchProcessor:
    """
    Batch processing for efficient API usage
    Reduces costs and improves throughput
    """
    
    def __init__(self, max_batch_size: int = 100, max_wait_time: float = 0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.request_futures = {}
        self.processing = False
        
    async def process_embedding_request(self, text: str) -> List[float]:
        """Add request to batch and wait for result"""
        request_id = hashlib.md5(f"{text}{datetime.utcnow()}".encode()).hexdigest()
        
        # Create future for this request
        future = asyncio.Future()
        self.request_futures[request_id] = future
        
        # Add to pending requests
        self.pending_requests.append({
            "id": request_id,
            "text": text,
            "timestamp": datetime.utcnow()
        })
        
        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_batch())
        
        # Wait for result
        return await future
    
    async def _process_batch(self):
        """Process accumulated requests as a batch"""
        self.processing = True
        
        # Wait for more requests or timeout
        await asyncio.sleep(self.max_wait_time)
        
        # Get requests to process
        batch = self.pending_requests[:self.max_batch_size]
        self.pending_requests = self.pending_requests[self.max_batch_size:]
        
        if not batch:
            self.processing = False
            return
        
        try:
            # Simulate batch API call
            texts = [req["text"] for req in batch]
            print(f"📦 Processing batch of {len(texts)} embeddings")
            
            # In production, this would be actual API call
            embeddings = await self._generate_batch_embeddings(texts)
            
            # Resolve futures
            for req, embedding in zip(batch, embeddings):
                if req["id"] in self.request_futures:
                    self.request_futures[req["id"]].set_result(embedding)
                    del self.request_futures[req["id"]]
        
        except Exception as e:
            # Reject all futures on error
            for req in batch:
                if req["id"] in self.request_futures:
                    self.request_futures[req["id"]].set_exception(e)
                    del self.request_futures[req["id"]]
        
        finally:
            # Continue processing if more requests
            if self.pending_requests:
                asyncio.create_task(self._process_batch())
            else:
                self.processing = False
    
    async def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch (simulated)"""
        # In production, use actual embedding API
        await asyncio.sleep(0.5)  # Simulate API latency
        return [[np.random.random() for _ in range(1536)] for _ in texts]

class LoadBalancer:
    """
    Load balancing for distributed RAG system
    """
    
    def __init__(self, workers: List[str], strategy: str = "round_robin"):
        self.workers = workers
        self.strategy = strategy
        self.current_index = 0
        self.worker_loads = {worker: 0 for worker in workers}
        self.worker_health = {worker: True for worker in workers}
    
    def get_next_worker(self) -> Optional[str]:
        """Get next available worker based on strategy"""
        healthy_workers = [w for w in self.workers if self.worker_health[w]]
        
        if not healthy_workers:
            return None
        
        if self.strategy == "round_robin":
            worker = healthy_workers[self.current_index % len(healthy_workers)]
            self.current_index += 1
            return worker
        
        elif self.strategy == "least_loaded":
            # Sort by load
            sorted_workers = sorted(
                healthy_workers,
                key=lambda w: self.worker_loads[w]
            )
            return sorted_workers[0]
        
        elif self.strategy == "random":
            import random
            return random.choice(healthy_workers)
        
        return healthy_workers[0]
    
    def update_worker_load(self, worker: str, load: int):
        """Update worker load for least_loaded strategy"""
        self.worker_loads[worker] = load
    
    def mark_worker_unhealthy(self, worker: str):
        """Mark worker as unhealthy"""
        self.worker_health[worker] = False
        print(f"❌ Worker {worker} marked unhealthy")
    
    def mark_worker_healthy(self, worker: str):
        """Mark worker as healthy"""
        self.worker_health[worker] = True
        print(f"✅ Worker {worker} marked healthy")

class AutoScaler:
    """
    Auto-scaling for RAG systems based on metrics
    """
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.current_replicas = config.api_replicas
        self.scaling_history = []
        self.last_scale_time = datetime.utcnow()
        self.cooldown_period = 300  # 5 minutes
    
    def should_scale(self, metrics: Dict[str, float]) -> Tuple[bool, str, int]:
        """Determine if scaling is needed based on metrics"""
        current_time = datetime.utcnow()
        
        # Check cooldown period
        if (current_time - self.last_scale_time).seconds < self.cooldown_period:
            return False, "cooldown", self.current_replicas
        
        # Check CPU usage
        if metrics.get("cpu_usage", 0) > self.config.cpu_threshold:
            if self.current_replicas < self.config.auto_scale_max:
                new_replicas = min(
                    self.current_replicas + 1,
                    self.config.auto_scale_max
                )
                return True, "scale_up_cpu", new_replicas
        
        # Check memory usage
        if metrics.get("memory_usage", 0) > self.config.memory_threshold:
            if self.current_replicas < self.config.auto_scale_max:
                new_replicas = min(
                    self.current_replicas + 1,
                    self.config.auto_scale_max
                )
                return True, "scale_up_memory", new_replicas
        
        # Check if we can scale down
        if (metrics.get("cpu_usage", 0) < self.config.cpu_threshold * 0.5 and
            metrics.get("memory_usage", 0) < self.config.memory_threshold * 0.5):
            if self.current_replicas > self.config.auto_scale_min:
                new_replicas = max(
                    self.current_replicas - 1,
                    self.config.auto_scale_min
                )
                return True, "scale_down", new_replicas
        
        return False, "no_change", self.current_replicas
    
    def scale(self, new_replicas: int, reason: str):
        """Execute scaling action"""
        old_replicas = self.current_replicas
        self.current_replicas = new_replicas
        self.last_scale_time = datetime.utcnow()
        
        self.scaling_history.append({
            "timestamp": self.last_scale_time,
            "from_replicas": old_replicas,
            "to_replicas": new_replicas,
            "reason": reason
        })
        
        print(f"🔄 Scaling: {old_replicas} → {new_replicas} replicas (reason: {reason})")

def create_docker_config():
    """Create Docker configuration for RAG deployment"""
    
    dockerfile = """# Multi-stage build for optimal size
FROM python:3.9-slim as builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Copy application code
COPY . .

# Non-root user for security
RUN useradd -m -u 1000 raguser && chown -R raguser:raguser /app
USER raguser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
"""
    
    docker_compose = """version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MONGODB_URI=${MONGODB_URI}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - VOYAGE_AI_API_KEY=${VOYAGE_AI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - mongo
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

  mongo:
    image: mongo:5.0
    ports:
      - "27017:27017"
    volumes:
      - mongo-data:/data/db
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=password

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
    depends_on:
      - rag-api

volumes:
  redis-data:
  mongo-data:
"""
    
    nginx_config = """events {
    worker_connections 1024;
}

http {
    upstream rag_backend {
        least_conn;
        server rag-api_1:8000 max_fails=3 fail_timeout=30s;
        server rag-api_2:8000 max_fails=3 fail_timeout=30s;
        server rag-api_3:8000 max_fails=3 fail_timeout=30s;
    }

    server {
        listen 80;
        server_name api.example.com;
        
        # Rate limiting
        limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
        limit_req zone=api_limit burst=20 nodelay;
        
        # Caching
        proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=api_cache:10m max_size=1g inactive=60m;
        
        location / {
            proxy_pass http://rag_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
            
            # Cache GET requests
            proxy_cache api_cache;
            proxy_cache_valid 200 5m;
            proxy_cache_valid 404 1m;
        }
        
        location /health {
            proxy_pass http://rag_backend/health;
            access_log off;
        }
    }
}
"""
    
    print("🐳 Docker Configuration:")
    print("\n📄 Dockerfile:")
    print(dockerfile)
    print("\n📄 docker-compose.yml:")
    print(docker_compose)
    print("\n📄 nginx.conf:")
    print(nginx_config)

def create_kubernetes_config():
    """Create Kubernetes configuration for RAG deployment"""
    
    deployment_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  namespace: rag-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: your-registry/rag-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MONGODB_URI
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: mongodb-uri
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
  namespace: rag-system
spec:
  selector:
    app: rag-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
"""
    
    print("\n☸️  Kubernetes Configuration:")
    print(deployment_yaml)

def demonstrate_deployment_strategies():
    """Demonstrate various deployment strategies"""
    print("\n🚀 DEPLOYMENT STRATEGIES")
    print("="*60)
    
    strategies = [
        {
            "name": "Blue-Green Deployment",
            "description": "Zero-downtime deployment with instant rollback",
            "steps": [
                "1. Deploy new version (green) alongside current (blue)",
                "2. Test green environment thoroughly",
                "3. Switch traffic from blue to green",
                "4. Keep blue as rollback option",
                "5. Remove blue after validation period"
            ]
        },
        {
            "name": "Canary Deployment",
            "description": "Gradual rollout with risk mitigation",
            "steps": [
                "1. Deploy new version to small subset (5-10%)",
                "2. Monitor metrics and errors",
                "3. Gradually increase traffic (25%, 50%, 100%)",
                "4. Rollback if issues detected",
                "5. Complete rollout if successful"
            ]
        },
        {
            "name": "Rolling Deployment",
            "description": "Sequential update with no downtime",
            "steps": [
                "1. Update one instance at a time",
                "2. Health check before proceeding",
                "3. Maintain minimum healthy instances",
                "4. Pause on errors",
                "5. Complete when all instances updated"
            ]
        }
    ]
    
    for strategy in strategies:
        print(f"\n📋 {strategy['name']}")
        print(f"   {strategy['description']}")
        print("\n   Steps:")
        for step in strategy['steps']:
            print(f"   {step}")

async def demonstrate_scaling():
    """Demonstrate scaling scenarios"""
    print("\n📊 SCALING DEMONSTRATION")
    print("="*60)
    
    # Initialize components
    config = DeploymentConfig(
        environment="production",
        api_replicas=3,
        worker_processes=4,
        cache_ttl=3600,
        max_batch_size=100,
        request_timeout=30,
        auto_scale_min=2,
        auto_scale_max=10,
        cpu_threshold=0.7,
        memory_threshold=0.8
    )
    
    cache_manager = CacheManager()
    batch_processor = BatchProcessor(max_batch_size=50)
    load_balancer = LoadBalancer(
        workers=["worker1", "worker2", "worker3"],
        strategy="least_loaded"
    )
    auto_scaler = AutoScaler(config)
    
    # Simulate load scenarios
    print("\n🔄 Simulating Load Scenarios:")
    
    scenarios = [
        {"name": "Normal Load", "cpu": 0.4, "memory": 0.5, "requests": 100},
        {"name": "High CPU", "cpu": 0.85, "memory": 0.6, "requests": 500},
        {"name": "High Memory", "cpu": 0.5, "memory": 0.9, "requests": 300},
        {"name": "Low Load", "cpu": 0.2, "memory": 0.3, "requests": 50}
    ]
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"   CPU: {scenario['cpu']*100:.0f}%, Memory: {scenario['memory']*100:.0f}%")
        
        # Check if scaling needed
        should_scale, reason, new_replicas = auto_scaler.should_scale({
            "cpu_usage": scenario['cpu'],
            "memory_usage": scenario['memory']
        })
        
        if should_scale:
            auto_scaler.scale(new_replicas, reason)
        else:
            print(f"   No scaling needed ({reason})")
        
        # Simulate batch processing
        print(f"   Processing {scenario['requests']} requests...")
        
        # Demonstrate caching benefit
        cache_hits = int(scenario['requests'] * 0.3)  # 30% cache hit rate
        print(f"   Cache hits: {cache_hits} ({cache_hits/scenario['requests']*100:.0f}%)")
        
        # Wait before next scenario
        await asyncio.sleep(1)
    
    # Show scaling history
    print("\n📈 Scaling History:")
    for event in auto_scaler.scaling_history:
        print(f"   {event['timestamp'].strftime('%H:%M:%S')}: "
              f"{event['from_replicas']} → {event['to_replicas']} "
              f"({event['reason']})")

def create_monitoring_setup():
    """Create monitoring configuration"""
    print("\n📊 MONITORING SETUP")
    print("="*60)
    
    prometheus_config = """# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'rag-api'
    static_configs:
      - targets: ['rag-api:8000']
    metrics_path: '/metrics'
    
  - job_name: 'mongodb'
    static_configs:
      - targets: ['mongo-exporter:9216']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
"""
    
    grafana_dashboard = """
{
  "dashboard": {
    "title": "RAG System Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time P95",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket)"
          }
        ]
      },
      {
        "title": "Token Usage",
        "targets": [
          {
            "expr": "sum(rate(tokens_used_total[5m])) by (provider)"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(cache_hits_total[5m]) / rate(cache_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])"
          }
        ]
      }
    ]
  }
}
"""
    
    alerts_config = """# alerts.yml
groups:
  - name: rag_alerts
    rules:
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value | humanizePercentage }}"
        
    - alert: HighTokenCost
      expr: sum(rate(token_cost_dollars[1h])) > 10
      for: 15m
      labels:
        severity: warning
      annotations:
        summary: "High token usage cost"
        description: "Hourly cost: ${{ $value }}"
        
    - alert: LowCacheHitRate
      expr: rate(cache_hits_total[5m]) / rate(cache_requests_total[5m]) < 0.2
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "Low cache hit rate"
        description: "Cache hit rate: {{ $value | humanizePercentage }}"
"""
    
    print("📊 Prometheus Configuration:")
    print(prometheus_config)
    print("\n📊 Grafana Dashboard:")
    print(json.dumps(json.loads(grafana_dashboard), indent=2))
    print("\n🚨 Alert Rules:")
    print(alerts_config)

if __name__ == "__main__":
    print("🎓 MongoDB RAG Course - Deployment & Scaling\n")
    
    try:
        # Show deployment configurations
        create_docker_config()
        create_kubernetes_config()
        
        # Demonstrate deployment strategies
        demonstrate_deployment_strategies()
        
        # Run scaling demonstration
        asyncio.run(demonstrate_scaling())
        
        # Show monitoring setup
        create_monitoring_setup()
        
        print("\n\n🎉 Key Deployment & Scaling Concepts:")
        print("✅ Containerization with Docker")
        print("✅ Orchestration with Kubernetes")
        print("✅ Caching strategies")
        print("✅ Batch processing")
        print("✅ Load balancing")
        print("✅ Auto-scaling")
        print("✅ Deployment strategies")
        print("✅ Monitoring and alerting")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()