# BONUS Module: FastAPI RAG Service

## 🎯 Module Overview
Transform everything you've learned into a production-ready API service. This bonus module shows you how to build, deploy, and scale a complete RAG system as a FastAPI service with professional features like authentication, caching, monitoring, and auto-scaling.

## 📚 Learning Objectives
By the end of this module, you will:
- ✅ Build a complete FastAPI RAG service
- ✅ Implement authentication and rate limiting
- ✅ Add caching and performance optimization
- ✅ Deploy with Docker and Kubernetes
- ✅ Set up monitoring and alerting
- ✅ Create client SDKs for easy integration

## 🎬 Module Structure

### Part 1: API Design (10 minutes)
**Focus**: RESTful design, endpoints, and models

**Topics Covered**:
- API versioning strategy
- Endpoint design patterns
- Request/response models
- Error handling standards
- OpenAPI documentation

### Part 2: Service Implementation (20 minutes)
**Focus**: Core functionality and features

**Topics Covered**:
- Async request handling
- Multi-provider embeddings
- Batch processing
- Streaming responses
- Background tasks
- WebSocket support

### Part 3: Deploy to Production (15 minutes)
**Focus**: Deployment and scaling

**Topics Covered**:
- Docker containerization
- Kubernetes orchestration
- Load balancing
- Auto-scaling
- Monitoring setup
- CI/CD pipelines

## 🛠️ Setup Instructions

### Prerequisites
```bash
# Core dependencies
pip install fastapi uvicorn motor redis pymongo
pip install openai voyageai prometheus-client
pip install python-multipart aiofiles

# Development dependencies
pip install pytest httpx pytest-asyncio
pip install black isort mypy
```

### Environment Configuration
```bash
# Create .env file
cat > .env << EOF
# API Configuration
API_TITLE=RAG API Service
API_VERSION=1.0.0
API_PORT=8000
API_WORKERS=4

# Security
VALID_API_KEYS=dev-key-1,dev-key-2
CORS_ORIGINS=["http://localhost:3000"]

# Database
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=rag_production
REDIS_URL=redis://localhost:6379

# AI Providers
OPENAI_API_KEY=your_openai_key
VOYAGE_AI_API_KEY=your_voyage_key

# Performance
MAX_BATCH_SIZE=100
REQUEST_TIMEOUT=30
CACHE_TTL=3600

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO
EOF
```

## 🚀 Running the Service

### Local Development
```bash
# Run with auto-reload
uvicorn rag_api_service:app --reload --port 8000

# Run with multiple workers
uvicorn rag_api_service:app --workers 4 --port 8000
```

### Docker Deployment
```bash
# Build image
docker build -t rag-api:latest .

# Run container
docker run -d \
  --name rag-api \
  -p 8000:8000 \
  --env-file .env \
  rag-api:latest

# Using docker-compose
docker-compose up -d
```

### Kubernetes Deployment
```bash
# Create namespace
kubectl create namespace rag-system

# Apply configurations
kubectl apply -f k8s/

# Check deployment
kubectl get pods -n rag-system
kubectl get svc -n rag-system
```

## 📊 API Endpoints

### Core Endpoints

#### Health Check
```http
GET /health
```
Returns service health status and component states.

#### Generate Embeddings
```http
POST /api/v1/embeddings
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "text": "Your text here",
  "model": "voyage-3-large"
}
```

#### Vector Search
```http
POST /api/v1/search
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "query": "Your search query",
  "collection": "documents",
  "limit": 5,
  "filters": {"category": "tutorial"},
  "rerank": true
}
```

#### RAG Chat
```http
POST /api/v1/chat
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "message": "Your question here",
  "conversation_id": "optional-id",
  "model": "gpt-3.5-turbo",
  "stream": false
}
```

#### Document Ingestion
```http
POST /api/v1/ingest
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "documents": [
    {
      "title": "Document Title",
      "content": "Document content..."
    }
  ],
  "collection": "documents",
  "embedding_model": "voyage-3-large"
}
```

### Monitoring Endpoints

#### Prometheus Metrics
```http
GET /metrics
```

#### Custom Analytics
```http
GET /api/v1/analytics/usage
Authorization: Bearer {api_key}
```

## 💡 Implementation Best Practices

### Security
1. **API Key Management**
   ```python
   # Rotate keys regularly
   # Use different keys per environment
   # Implement key scoping
   ```

2. **Rate Limiting**
   ```python
   # Configure per endpoint
   # Implement user-based limits
   # Add burst allowances
   ```

3. **Input Validation**
   ```python
   # Use Pydantic models
   # Validate all inputs
   # Sanitize user content
   ```

### Performance
1. **Caching Strategy**
   ```python
   # Cache embeddings (1 hour)
   # Cache search results (5 minutes)
   # Use cache warming
   ```

2. **Batch Processing**
   ```python
   # Batch embedding requests
   # Process documents in chunks
   # Use async operations
   ```

3. **Connection Pooling**
   ```python
   # MongoDB connection pool
   # Redis connection pool
   # HTTP client pooling
   ```

### Monitoring
1. **Metrics to Track**
   - Request rate and latency
   - Token usage and costs
   - Cache hit rates
   - Error rates
   - Resource utilization

2. **Alerting Rules**
   - High error rate (>5%)
   - High latency (>2s P95)
   - High cost rate (>$50/hour)
   - Low cache hit rate (<30%)
   - Service downtime

## 🧪 Testing

### Unit Tests
```python
# Test individual components
pytest tests/unit/

# Test with coverage
pytest tests/unit/ --cov=rag_api_service
```

### Integration Tests
```python
# Test API endpoints
pytest tests/integration/

# Test with real services
pytest tests/integration/ --env=staging
```

### Load Testing
```bash
# Using locust
locust -f tests/load/locustfile.py --host=http://localhost:8000

# Using k6
k6 run tests/load/script.js
```

## 📈 Scaling Strategies

### Horizontal Scaling
```yaml
# Kubernetes HPA
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70
```

### Vertical Scaling
```yaml
# Resource optimization
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### Database Scaling
- MongoDB sharding for large datasets
- Redis clustering for cache
- Read replicas for search

## 🆘 Troubleshooting

### Common Issues

**High Latency**
- Check cache hit rates
- Monitor embedding generation time
- Review database query performance
- Analyze reranking overhead

**Memory Issues**
- Implement streaming for large responses
- Use pagination for results
- Clear unused cache entries
- Monitor connection pools

**Rate Limit Errors**
- Implement exponential backoff
- Use request queuing
- Add provider fallbacks
- Monitor usage patterns

### Debug Commands
```bash
# Check logs
docker logs rag-api -f

# Monitor metrics
curl http://localhost:8000/metrics

# Test endpoints
curl -X POST http://localhost:8000/api/v1/embeddings \
  -H "Authorization: Bearer test-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "test"}'

# Check health
curl http://localhost:8000/health
```

## 📝 Exercises

1. **Add Authentication Methods**
   - Implement OAuth2
   - Add JWT tokens
   - Create user management

2. **Extend Search Features**
   - Add fuzzy search
   - Implement faceted search
   - Add search suggestions

3. **Build Admin Dashboard**
   - Usage statistics
   - User management
   - System configuration

4. **Create SDK**
   - Python client library
   - JavaScript/TypeScript SDK
   - CLI tool

## 🎯 Module Completion Checklist
- [ ] Built complete FastAPI service
- [ ] Implemented all endpoints
- [ ] Added authentication and rate limiting
- [ ] Set up caching and optimization
- [ ] Deployed with Docker/Kubernetes
- [ ] Configured monitoring and alerts
- [ ] Created client examples

## 📚 Additional Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MongoDB Motor Async Driver](https://motor.readthedocs.io/)
- [Redis Python Guide](https://redis-py.readthedocs.io/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/workloads/)
- [Prometheus Monitoring](https://prometheus.io/docs/introduction/overview/)

## 🎉 Congratulations!
You've built a production-ready RAG API service! This implementation can handle real-world workloads and scale to meet demand. The patterns and practices you've learned here apply to any AI-powered API service.

### What's Next?
- Deploy to cloud providers (AWS, GCP, Azure)
- Add more AI providers (Anthropic, Cohere)
- Implement multi-tenancy
- Build a SaaS product
- Open source your improvements

Happy building! 🚀