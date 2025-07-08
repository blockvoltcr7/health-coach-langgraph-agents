# Module 4: Production Patterns

## 🎯 Module Overview
Master the patterns and practices needed to run RAG systems in production. This module covers error handling, testing strategies, and deployment at scale - the difference between a demo and a production system.

## 📚 Learning Objectives
By the end of this module, you will:
- ✅ Implement robust error handling and resilience patterns
- ✅ Create comprehensive test suites for RAG systems
- ✅ Deploy and scale RAG applications efficiently
- ✅ Monitor and optimize production systems
- ✅ Handle real-world failure scenarios gracefully

## 🎬 Video Structure

### Video 4.1: Error Handling & Resilience (15 minutes)
**File**: `01_error_handling_resilience.py`

**What you'll learn**:
- Retry strategies with exponential backoff
- Circuit breaker pattern implementation
- Provider fallback mechanisms
- Graceful degradation strategies
- Error classification and handling
- System health monitoring

**Key Patterns**:
- Automatic retries with jitter
- Circuit breaker state management
- Multi-provider fallback chains
- Timeout protection
- Centralized error logging

**Real-World Scenarios**:
- API rate limit handling
- Network failures
- Provider outages
- Database connection issues

### Video 4.2: Testing Strategies (15 minutes)
**File**: `02_testing_strategies.py`

**What you'll learn**:
- Unit testing with mocks
- Integration testing strategies
- Performance benchmarking
- Edge case testing
- Continuous integration setup
- Test data management

**Key Components**:
- Embedding validation tests
- Search accuracy metrics
- Response quality evaluation
- Load testing frameworks
- Mock provider testing

**Testing Pyramid**:
- Unit tests (fast, isolated)
- Integration tests (API level)
- End-to-end tests (full pipeline)
- Performance tests (load/stress)

### Video 4.3: Deployment & Scaling (15 minutes)
**File**: `03_deployment_scaling.py`

**What you'll learn**:
- Containerization with Docker
- Orchestration with Kubernetes
- Caching strategies
- Batch processing
- Load balancing
- Auto-scaling patterns

**Key Technologies**:
- Docker multi-stage builds
- Kubernetes deployments
- Redis caching
- NGINX load balancing
- Prometheus monitoring
- Grafana dashboards

**Deployment Strategies**:
- Blue-green deployment
- Canary releases
- Rolling updates

## 🛠️ Setup Instructions

### Prerequisites
```bash
# Core requirements
pip install pymongo openai voyageai redis numpy pytest

# Additional for production
pip install asyncio prometheus-client pytest-asyncio
```

### Environment Configuration
```bash
# API Keys
OPENAI_API_KEY=your_openai_key
VOYAGE_AI_API_KEY=your_voyage_key
MONGODB_URI=your_mongodb_connection

# Production settings
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
MAX_WORKERS=4
REQUEST_TIMEOUT=30
```

### Infrastructure Setup

**Redis Cache**:
```bash
# Local Redis
docker run -d -p 6379:6379 redis:7-alpine

# Redis with persistence
docker run -d -p 6379:6379 \
  -v redis-data:/data \
  redis:7-alpine redis-server --appendonly yes
```

**Monitoring Stack**:
```bash
# Prometheus
docker run -d -p 9090:9090 \
  -v prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Grafana
docker run -d -p 3000:3000 grafana/grafana
```

## 🚀 Running the Examples

### Error Handling Demo
```bash
python 01_error_handling_resilience.py
```
- See retry mechanisms in action
- Watch circuit breakers open/close
- Test provider fallbacks
- Monitor system health

### Testing Suite
```bash
# Run all tests
python 02_testing_strategies.py

# Run with pytest
pytest tests/ -v --cov=rag_system
```

### Deployment Demo
```bash
# Build Docker image
docker build -t rag-api:latest .

# Run with docker-compose
docker-compose up -d

# Deploy to Kubernetes
kubectl apply -f k8s/
```

## 📊 Production Architecture

### High-Level Architecture
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│ Load Balancer│────▶│   API Gateway│
└─────────────┘     └─────────────┘     └─────────────┘
                                                │
                    ┌───────────────────────────┴───────────────┐
                    │                                           │
            ┌───────▼────────┐                         ┌───────▼────────┐
            │   RAG API #1   │                         │   RAG API #2   │
            └───────┬────────┘                         └───────┬────────┘
                    │                                           │
                    └─────────────┬─────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
            ┌───────▼────────┐         ┌───────▼────────┐
            │     Redis      │         │    MongoDB     │
            │    (Cache)     │         │  (Persistent)  │
            └────────────────┘         └────────────────┘
```

### Component Responsibilities

**API Gateway**:
- Rate limiting
- Authentication
- Request routing
- Response caching

**RAG API Instances**:
- Embedding generation
- Vector search
- Response generation
- Error handling

**Redis Cache**:
- Embedding cache
- Search result cache
- Session storage
- Rate limit tracking

**MongoDB**:
- Document storage
- Vector indexes
- Analytics data
- System logs

## 💡 Production Best Practices

### Error Handling
1. **Always Have Fallbacks**
   ```python
   try:
       result = primary_service()
   except ServiceError:
       result = fallback_service()
   ```

2. **Use Circuit Breakers**
   ```python
   if circuit.is_open():
       return cached_response()
   ```

3. **Implement Timeouts**
   ```python
   async with timeout(30):
       result = await slow_operation()
   ```

### Testing
1. **Test Coverage Goals**
   - Unit tests: >80%
   - Integration tests: Critical paths
   - Performance tests: Regular benchmarks

2. **Mock External Services**
   ```python
   @patch('openai.OpenAI')
   def test_embedding_generation(mock_openai):
       # Test without API calls
   ```

### Deployment
1. **Resource Limits**
   ```yaml
   resources:
     requests:
       memory: "1Gi"
       cpu: "500m"
     limits:
       memory: "2Gi"
       cpu: "1000m"
   ```

2. **Health Checks**
   ```python
   @app.get("/health")
   async def health_check():
       return {
           "status": "healthy",
           "timestamp": datetime.utcnow(),
           "checks": await run_health_checks()
       }
   ```

## 🆘 Troubleshooting

### Common Issues

**High Latency**:
- Check cache hit rates
- Monitor API rate limits
- Review batch processing
- Analyze slow queries

**Memory Issues**:
- Implement connection pooling
- Use streaming for large responses
- Clear unused caches
- Monitor memory leaks

**Scaling Problems**:
- Review auto-scaling metrics
- Check resource limits
- Monitor network I/O
- Analyze bottlenecks

### Debug Commands
```bash
# Check pod resources
kubectl top pods

# View logs
kubectl logs -f deployment/rag-api

# Monitor Redis
redis-cli monitor

# MongoDB slow queries
db.currentOp({"secs_running": {$gte: 3}})
```

## 📝 Exercises

1. **Implement Custom Retry Strategy**
   - Add custom backoff algorithm
   - Implement retry budgets
   - Add retry metrics

2. **Create Load Test Suite**
   - Simulate concurrent users
   - Test rate limits
   - Measure breaking points

3. **Build Monitoring Dashboard**
   - Track key metrics
   - Set up alerts
   - Create SLO dashboard

4. **Implement Feature Flags**
   - Dynamic feature toggling
   - A/B testing support
   - Gradual rollouts

## 🎯 Module Completion Checklist
- [ ] Implemented all resilience patterns
- [ ] Created comprehensive test suite
- [ ] Deployed with Docker/Kubernetes
- [ ] Set up monitoring and alerts
- [ ] Tested failure scenarios

## 📚 Additional Resources
- [The Site Reliability Workbook](https://sre.google/workbook/table-of-contents/)
- [Kubernetes Patterns](https://www.oreilly.com/library/view/kubernetes-patterns/9781492050278/)
- [Testing Microservices](https://martinfowler.com/articles/microservice-testing/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)

Ready for the Bonus Module? Let's build a FastAPI service! 🚀