# RAG API Deployment Templates

This directory contains production-ready deployment templates and configurations for deploying the RAG API service across various platforms.

## 📁 Directory Structure

```
templates/
├── .env.example                 # Environment variables template
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions CI/CD pipeline
├── Dockerfile                  # Multi-stage production Docker build
├── docker-compose.yml          # Full stack local development
├── kubernetes/
│   ├── deployment.yaml         # K8s deployment with HPA, PDB
│   └── ingress.yaml           # Ingress with TLS and rate limiting
└── cloud/
    ├── aws-terraform/         # AWS ECS Fargate deployment
    ├── gcp-terraform/         # Google Cloud Run deployment
    └── azure-terraform/       # Azure App Service deployment
```

## 🚀 Quick Start

### Local Development with Docker Compose

1. Copy environment variables:
```bash
cp .env.example .env
# Edit .env with your values
```

2. Start the stack:
```bash
docker-compose up -d
```

3. Access services:
- API: http://localhost:8000
- Redis: localhost:6379
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### Kubernetes Deployment

1. Create namespace:
```bash
kubectl create namespace rag-system
```

2. Create secrets:
```bash
kubectl create secret generic rag-secrets \
  --from-literal=mongodb-uri="$MONGODB_URI" \
  --from-literal=openai-api-key="$OPENAI_API_KEY" \
  --from-literal=voyage-ai-api-key="$VOYAGE_AI_API_KEY" \
  -n rag-system
```

3. Apply manifests:
```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/ingress.yaml
```

### Cloud Deployments

#### AWS (ECS Fargate)
```bash
cd cloud/aws-terraform
terraform init
terraform plan -var="environment=production"
terraform apply
```

#### Google Cloud (Cloud Run)
```bash
cd cloud/gcp-terraform
terraform init
terraform plan -var="project_id=your-project"
terraform apply
```

#### Azure (App Service)
```bash
cd cloud/azure-terraform
terraform init
terraform plan -var="environment=production"
terraform apply
```

## 🔧 Configuration

### Environment Variables

Key environment variables (see `.env.example` for full list):

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URI` | MongoDB connection string | Required |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `VOYAGE_AI_API_KEY` | Voyage AI API key | Optional |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `API_WORKERS` | Number of API workers | `4` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Docker Build

Build the production image:
```bash
docker build -t rag-api:latest .
```

Multi-architecture build:
```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t your-registry/rag-api:latest \
  --push .
```

### GitHub Actions

The CI/CD pipeline includes:
- Code quality checks (linting, type checking, security)
- Unit and integration tests
- Docker image building and scanning
- Automated deployments to staging/production
- Post-deployment health checks

Required GitHub secrets:
- `MONGODB_URI_TEST`
- `OPENAI_API_KEY`
- `VOYAGE_AI_API_KEY`
- `DOCKER_USERNAME` / `DOCKER_PASSWORD`
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
- `KUBE_CONFIG_STAGING` / `KUBE_CONFIG_PRODUCTION`
- `SLACK_WEBHOOK`

## 🔒 Security Considerations

1. **Secrets Management**:
   - Never commit secrets to version control
   - Use cloud provider secret managers
   - Rotate API keys regularly

2. **Network Security**:
   - Enable TLS/SSL for all endpoints
   - Use network policies in Kubernetes
   - Configure firewall rules

3. **Container Security**:
   - Run containers as non-root user
   - Use minimal base images
   - Scan images for vulnerabilities

4. **API Security**:
   - Implement rate limiting
   - Use API key authentication
   - Enable CORS appropriately

## 📊 Monitoring

### Metrics
- Prometheus metrics exposed on `/metrics`
- Custom business metrics for cost tracking
- Performance metrics (latency, throughput)

### Logging
- Structured JSON logging
- Log aggregation with ELK/CloudWatch/Stackdriver
- Request/response logging with sampling

### Alerting
- CPU/Memory usage alerts
- API error rate alerts
- Cost threshold alerts
- SLA violation alerts

## 🆘 Troubleshooting

### Common Issues

1. **MongoDB Connection Failed**
   - Check connection string format
   - Verify network access (IP whitelist)
   - Check authentication credentials

2. **Out of Memory Errors**
   - Increase container memory limits
   - Check for memory leaks
   - Optimize batch sizes

3. **High Latency**
   - Enable Redis caching
   - Optimize embedding batch size
   - Use connection pooling

4. **Rate Limiting**
   - Adjust rate limit thresholds
   - Implement request queuing
   - Use horizontal scaling

### Debug Commands

```bash
# Check pod logs
kubectl logs -f deployment/rag-api -n rag-system

# Exec into container
kubectl exec -it deployment/rag-api -n rag-system -- /bin/bash

# Check service status
kubectl get all -n rag-system

# Test API health
curl https://api.yourdomain.com/health
```

## 📚 Additional Resources

- [MongoDB Atlas Setup Guide](https://docs.atlas.mongodb.com/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Docker Security Checklist](https://docs.docker.com/develop/security-best-practices/)
- [Terraform Documentation](https://www.terraform.io/docs/)

## 🤝 Contributing

When adding new deployment configurations:
1. Follow existing naming conventions
2. Include comprehensive comments
3. Add security scanning steps
4. Update this README
5. Test in staging before production

## 📝 License

These templates are part of the RAG API YouTube course and are provided as-is for educational purposes.