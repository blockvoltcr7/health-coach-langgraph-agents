"""
FastAPI RAG Service Deployment Configuration
Production deployment setup and configurations
"""

import os
from typing import Dict, List, Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    api_title: str = Field(default="RAG API Service", env="API_TITLE")
    api_version: str = Field(default="1.0.0", env="API_VERSION")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # Security
    api_keys: List[str] = Field(default=[], env="VALID_API_KEYS")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Database
    mongodb_uri: str = Field(..., env="MONGODB_URI")
    mongodb_database: str = Field(default="rag_production", env="MONGODB_DATABASE")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    
    # AI Providers
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    voyage_ai_api_key: Optional[str] = Field(default=None, env="VOYAGE_AI_API_KEY")
    
    # Performance
    max_batch_size: int = Field(default=100, env="MAX_BATCH_SIZE")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    max_concurrent_requests: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Environment
    environment: str = Field(default="production", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

def create_dockerfile():
    """Create production Dockerfile"""
    
    dockerfile = '''# Multi-stage build for production
FROM python:3.9-slim as builder

# Build dependencies
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production image
FROM python:3.9-slim

# Security: Create non-root user
RUN useradd -m -u 1000 -s /bin/bash raguser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies from builder
COPY --from=builder /root/.local /home/raguser/.local

# Set up application directory
WORKDIR /app
COPY --chown=raguser:raguser . .

# Environment
ENV PATH=/home/raguser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER raguser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "rag_api_service:app", "--host", "0.0.0.0", "--port", "8000"]'''
    
    return dockerfile

def create_docker_compose():
    """Create docker-compose configuration"""
    
    docker_compose = '''version: '3.8'

services:
  rag-api:
    build:
      context: .
      dockerfile: Dockerfile
    image: rag-api:latest
    container_name: rag-api
    ports:
      - "${API_PORT:-8000}:8000"
    environment:
      - MONGODB_URI=${MONGODB_URI}
      - MONGODB_DATABASE=${MONGODB_DATABASE:-rag_production}
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - VOYAGE_AI_API_KEY=${VOYAGE_AI_API_KEY}
      - VALID_API_KEYS=${VALID_API_KEYS}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - API_WORKERS=${API_WORKERS:-4}
    depends_on:
      redis:
        condition: service_healthy
      mongodb:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    networks:
      - rag-network

  redis:
    image: redis:7-alpine
    container_name: rag-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: >
      redis-server
      --appendonly yes
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks:
      - rag-network

  mongodb:
    image: mongo:6.0
    container_name: rag-mongodb
    ports:
      - "27017:27017"
    environment:
      - MONGO_INITDB_ROOT_USERNAME=${MONGO_USERNAME:-admin}
      - MONGO_INITDB_ROOT_PASSWORD=${MONGO_PASSWORD:-password}
      - MONGO_INITDB_DATABASE=${MONGODB_DATABASE:-rag_production}
    volumes:
      - mongo-data:/data/db
      - ./mongo-init.js:/docker-entrypoint-initdb.d/init.js:ro
    healthcheck:
      test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks:
      - rag-network

  nginx:
    image: nginx:alpine
    container_name: rag-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/certs:/etc/nginx/certs:ro
      - nginx-cache:/var/cache/nginx
    depends_on:
      - rag-api
    restart: unless-stopped
    networks:
      - rag-network

  prometheus:
    image: prom/prometheus:latest
    container_name: rag-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    restart: unless-stopped
    networks:
      - rag-network

  grafana:
    image: grafana/grafana:latest
    container_name: rag-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - rag-network

volumes:
  redis-data:
  mongo-data:
  prometheus-data:
  grafana-data:
  nginx-cache:

networks:
  rag-network:
    driver: bridge'''
    
    return docker_compose

def create_kubernetes_deployment():
    """Create Kubernetes deployment configuration"""
    
    k8s_deployment = '''apiVersion: v1
kind: Namespace
metadata:
  name: rag-system
---
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
  namespace: rag-system
type: Opaque
stringData:
  mongodb-uri: "mongodb://username:password@mongodb:27017"
  openai-api-key: "your-openai-api-key"
  voyage-ai-api-key: "your-voyage-api-key"
  valid-api-keys: "key1,key2,key3"
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-system
data:
  API_TITLE: "RAG API Service"
  API_VERSION: "1.0.0"
  MONGODB_DATABASE: "rag_production"
  REDIS_URL: "redis://redis-service:6379"
  LOG_LEVEL: "INFO"
  CACHE_TTL: "3600"
  MAX_BATCH_SIZE: "100"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  namespace: rag-system
  labels:
    app: rag-api
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
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
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
        - name: VOYAGE_AI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: voyage-ai-api-key
        - name: VALID_API_KEYS
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: valid-api-keys
        envFrom:
        - configMapRef:
            name: rag-config
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
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: logs
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
  namespace: rag-system
  labels:
    app: rag-api
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: rag-api
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
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
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
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 60
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-api-ingress
  namespace: rag-system
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: rag-api-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-api-service
            port:
              number: 80'''
    
    return k8s_deployment

def create_nginx_config():
    """Create NGINX configuration"""
    
    nginx_config = '''events {
    worker_connections 4096;
}

http {
    # Basic settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 10M;

    # Logging
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss application/rss+xml application/atom+xml image/svg+xml;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=chat_limit:10m rate=2r/s;

    # Upstream configuration
    upstream rag_backend {
        least_conn;
        server rag-api:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }

    # Cache configuration
    proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=api_cache:10m max_size=1g inactive=60m use_temp_path=off;

    # Server configuration
    server {
        listen 80;
        server_name _;

        # Redirect to HTTPS
        return 301 https://$host$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name api.yourdomain.com;

        # SSL configuration
        ssl_certificate /etc/nginx/certs/fullchain.pem;
        ssl_certificate_key /etc/nginx/certs/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # Security headers
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        add_header X-Frame-Options "DENY" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;

        # Health check endpoint (no rate limit)
        location /health {
            proxy_pass http://rag_backend;
            proxy_set_header Host $host;
            access_log off;
        }

        # Metrics endpoint (internal only)
        location /metrics {
            allow 10.0.0.0/8;
            deny all;
            proxy_pass http://rag_backend;
        }

        # Chat endpoint (stricter rate limit)
        location /api/v1/chat {
            limit_req zone=chat_limit burst=5 nodelay;
            
            proxy_pass http://rag_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Streaming support
            proxy_buffering off;
            proxy_cache off;
            proxy_read_timeout 300s;
        }

        # API endpoints
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://rag_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Connection settings
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
            
            # Cache GET requests
            proxy_cache api_cache;
            proxy_cache_methods GET HEAD;
            proxy_cache_valid 200 5m;
            proxy_cache_valid 404 1m;
            proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
            proxy_cache_background_update on;
            proxy_cache_lock on;
            
            # Add cache status header
            add_header X-Cache-Status $upstream_cache_status;
        }

        # Default location
        location / {
            return 404;
        }
    }
}'''
    
    return nginx_config

def create_monitoring_config():
    """Create monitoring configuration"""
    
    prometheus_config = '''# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'rag-production'
    
# Alerting configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

# Rule files
rule_files:
  - "alerts/*.yml"

# Scrape configurations
scrape_configs:
  # RAG API metrics
  - job_name: 'rag-api'
    static_configs:
      - targets: ['rag-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  # Redis metrics
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    
  # MongoDB metrics
  - job_name: 'mongodb'
    static_configs:
      - targets: ['mongodb-exporter:9216']
      
  # NGINX metrics
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']'''
    
    alerts_config = '''# alerts/rag_alerts.yml
groups:
  - name: rag_api_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          (
            sum(rate(rag_requests_total{status=~"5.."}[5m]))
            /
            sum(rate(rag_requests_total[5m]))
          ) > 0.05
        for: 5m
        labels:
          severity: critical
          service: rag-api
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"
          
      # High latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(rag_request_duration_seconds_bucket[5m])) by (endpoint, le)
          ) > 2
        for: 5m
        labels:
          severity: warning
          service: rag-api
        annotations:
          summary: "High latency on {{ $labels.endpoint }}"
          description: "95th percentile latency is {{ $value }}s"
          
      # High token cost
      - alert: HighTokenCost
        expr: |
          sum(increase(rag_token_usage_total[1h])) * 0.002 > 50
        for: 15m
        labels:
          severity: warning
          service: rag-api
        annotations:
          summary: "High token usage cost"
          description: "Estimated hourly cost: ${{ $value }}"
          
      # Low cache hit rate
      - alert: LowCacheHitRate
        expr: |
          (
            sum(rate(rag_cache_hits_total[5m]))
            /
            (sum(rate(rag_cache_hits_total[5m])) + sum(rate(rag_cache_misses_total[5m])))
          ) < 0.3
        for: 10m
        labels:
          severity: warning
          service: rag-api
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is {{ $value | humanizePercentage }}"
          
      # Service down
      - alert: ServiceDown
        expr: up{job="rag-api"} == 0
        for: 1m
        labels:
          severity: critical
          service: rag-api
        annotations:
          summary: "RAG API service is down"
          description: "The service has been down for more than 1 minute"'''
    
    return {
        "prometheus": prometheus_config,
        "alerts": alerts_config
    }

if __name__ == "__main__":
    print("🚀 RAG API DEPLOYMENT CONFIGURATIONS\n")
    
    # Display configurations
    print("📄 Dockerfile:")
    print("-" * 60)
    print(create_dockerfile())
    
    print("\n\n📄 docker-compose.yml:")
    print("-" * 60)
    print(create_docker_compose())
    
    print("\n\n📄 kubernetes-deployment.yaml:")
    print("-" * 60)
    print(create_kubernetes_deployment())
    
    print("\n\n📄 nginx.conf:")
    print("-" * 60)
    print(create_nginx_config())
    
    monitoring = create_monitoring_config()
    print("\n\n📄 prometheus.yml:")
    print("-" * 60)
    print(monitoring["prometheus"])
    
    print("\n\n📄 alerts.yml:")
    print("-" * 60)
    print(monitoring["alerts"])
    
    print("\n\n✅ All deployment configurations generated!")