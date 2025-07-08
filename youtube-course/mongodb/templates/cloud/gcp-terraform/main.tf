# Google Cloud Platform Terraform configuration for RAG API deployment

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
  
  backend "gcs" {
    bucket = "your-terraform-state-bucket"
    prefix = "terraform/state/rag-api"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name"
  default     = "production"
}

variable "app_name" {
  description = "Application name"
  default     = "rag-api"
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "artifactregistry.googleapis.com",
    "run.googleapis.com",
    "redis.googleapis.com",
    "secretmanager.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "certificatemanager.googleapis.com"
  ])
  
  service = each.value
  disable_on_destroy = false
}

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = "${var.app_name}-${var.environment}-vpc"
  auto_create_subnetworks = false
  project                 = var.project_id
  
  depends_on = [google_project_service.apis]
}

resource "google_compute_subnetwork" "subnet" {
  name          = "${var.app_name}-${var.environment}-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.vpc.id
  
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.1.0.0/16"
  }
  
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.2.0.0/16"
  }
  
  private_ip_google_access = true
}

# Cloud NAT for outbound connectivity
resource "google_compute_router" "router" {
  name    = "${var.app_name}-${var.environment}-router"
  region  = var.region
  network = google_compute_network.vpc.id
}

resource "google_compute_router_nat" "nat" {
  name                               = "${var.app_name}-${var.environment}-nat"
  router                             = google_compute_router.router.name
  region                             = var.region
  nat_ip_allocate_option            = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

# Artifact Registry for Docker images
resource "google_artifact_registry_repository" "docker" {
  location      = var.region
  repository_id = "${var.app_name}-${var.environment}"
  description   = "Docker repository for ${var.app_name}"
  format        = "DOCKER"
  
  depends_on = [google_project_service.apis]
}

# Cloud Run Service
resource "google_cloud_run_v2_service" "app" {
  name     = "${var.app_name}-${var.environment}"
  location = var.region
  
  template {
    scaling {
      min_instance_count = 1
      max_instance_count = 100
    }
    
    vpc_access {
      connector = google_vpc_access_connector.connector.id
      egress    = "ALL_TRAFFIC"
    }
    
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}/${var.app_name}:latest"
      
      ports {
        name           = "http1"
        container_port = 8000
      }
      
      env {
        name  = "API_TITLE"
        value = "RAG API Service"
      }
      
      env {
        name  = "API_VERSION"
        value = "1.0.0"
      }
      
      env {
        name  = "MONGODB_DATABASE"
        value = "rag_production"
      }
      
      env {
        name  = "LOG_LEVEL"
        value = "INFO"
      }
      
      env {
        name = "MONGODB_URI"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.mongodb_uri.secret_id
            version = "latest"
          }
        }
      }
      
      env {
        name = "OPENAI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.openai_api_key.secret_id
            version = "latest"
          }
        }
      }
      
      env {
        name = "VOYAGE_AI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.voyage_ai_api_key.secret_id
            version = "latest"
          }
        }
      }
      
      env {
        name  = "REDIS_HOST"
        value = google_redis_instance.cache.host
      }
      
      env {
        name  = "REDIS_PORT"
        value = "6379"
      }
      
      resources {
        limits = {
          cpu    = "2000m"
          memory = "2Gi"
        }
        
        cpu_idle = true
      }
      
      startup_probe {
        initial_delay_seconds = 10
        timeout_seconds       = 3
        period_seconds        = 10
        failure_threshold     = 3
        
        http_get {
          path = "/health"
          port = 8000
        }
      }
      
      liveness_probe {
        initial_delay_seconds = 30
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 3
        
        http_get {
          path = "/health"
          port = 8000
        }
      }
    }
    
    service_account = google_service_account.run_sa.email
    execution_environment = "EXECUTION_ENVIRONMENT_GEN2"
    
    annotations = {
      "autoscaling.knative.dev/minScale" = "1"
      "autoscaling.knative.dev/maxScale" = "100"
      "run.googleapis.com/cpu-throttling" = "false"
    }
  }
  
  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
  
  depends_on = [
    google_project_service.apis,
    google_vpc_access_connector.connector
  ]
}

# VPC Connector for Cloud Run
resource "google_vpc_access_connector" "connector" {
  name          = "${var.app_name}-${var.environment}-connector"
  region        = var.region
  ip_cidr_range = "10.8.0.0/28"
  network       = google_compute_network.vpc.name
  
  depends_on = [google_project_service.apis]
}

# Cloud Run IAM binding for public access
resource "google_cloud_run_service_iam_member" "public" {
  service  = google_cloud_run_v2_service.app.name
  location = google_cloud_run_v2_service.app.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Service Account for Cloud Run
resource "google_service_account" "run_sa" {
  account_id   = "${var.app_name}-${var.environment}-run-sa"
  display_name = "Service Account for ${var.app_name} Cloud Run"
}

# Grant necessary permissions to service account
resource "google_project_iam_member" "run_sa_roles" {
  for_each = toset([
    "roles/secretmanager.secretAccessor",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/cloudtrace.agent"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.run_sa.email}"
}

# Memorystore Redis Instance
resource "google_redis_instance" "cache" {
  name           = "${var.app_name}-${var.environment}-redis"
  tier           = "STANDARD_HA"
  memory_size_gb = 1
  
  location_id             = var.zone
  alternative_location_id = "${var.region}-b"
  
  authorized_network = google_compute_network.vpc.id
  connect_mode       = "PRIVATE_SERVICE_ACCESS"
  
  redis_version = "REDIS_7_0"
  display_name  = "${var.app_name} Redis Cache"
  
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 3
        minutes = 0
        seconds = 0
        nanos   = 0
      }
    }
  }
  
  depends_on = [google_project_service.apis]
}

# Secret Manager Secrets
resource "google_secret_manager_secret" "mongodb_uri" {
  secret_id = "${var.app_name}-${var.environment}-mongodb-uri"
  
  replication {
    auto {}
  }
  
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret" "openai_api_key" {
  secret_id = "${var.app_name}-${var.environment}-openai-api-key"
  
  replication {
    auto {}
  }
  
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret" "voyage_ai_api_key" {
  secret_id = "${var.app_name}-${var.environment}-voyage-ai-api-key"
  
  replication {
    auto {}
  }
  
  depends_on = [google_project_service.apis]
}

# Load Balancer with Cloud Armor
resource "google_compute_global_address" "default" {
  name = "${var.app_name}-${var.environment}-ip"
}

resource "google_compute_region_network_endpoint_group" "cloudrun_neg" {
  name                  = "${var.app_name}-${var.environment}-neg"
  network_endpoint_type = "SERVERLESS"
  region                = var.region
  
  cloud_run {
    service = google_cloud_run_v2_service.app.name
  }
}

resource "google_compute_backend_service" "default" {
  name                  = "${var.app_name}-${var.environment}-backend"
  protocol              = "HTTP"
  port_name             = "http"
  timeout_sec           = 30
  enable_cdn            = true
  
  backend {
    group = google_compute_region_network_endpoint_group.cloudrun_neg.id
  }
  
  cdn_policy {
    cache_mode                   = "CACHE_ALL_STATIC"
    default_ttl                  = 3600
    client_ttl                   = 7200
    max_ttl                      = 86400
    negative_caching             = true
    serve_while_stale            = 86400
    signed_url_cache_max_age_sec = 7200
  }
  
  security_policy = google_compute_security_policy.default.id
  
  log_config {
    enable      = true
    sample_rate = 1.0
  }
}

# Cloud Armor Security Policy
resource "google_compute_security_policy" "default" {
  name = "${var.app_name}-${var.environment}-security-policy"
  
  # Default rule
  rule {
    action   = "allow"
    priority = "2147483647"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    description = "Default allow all"
  }
  
  # Rate limiting rule
  rule {
    action   = "rate_based_ban"
    priority = "1000"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"
      enforce_on_key = "IP"
      
      rate_limit_threshold {
        count        = 100
        interval_sec = 60
      }
      
      ban_duration_sec = 600
    }
    description = "Rate limiting"
  }
  
  # Block malicious requests
  rule {
    action   = "deny(403)"
    priority = "500"
    match {
      expr {
        expression = "evaluatePreconfiguredExpr('xss-stable') || evaluatePreconfiguredExpr('sqli-stable')"
      }
    }
    description = "Block XSS and SQL injection"
  }
}

# URL Map
resource "google_compute_url_map" "default" {
  name            = "${var.app_name}-${var.environment}-url-map"
  default_service = google_compute_backend_service.default.id
}

# HTTPS Proxy
resource "google_compute_target_https_proxy" "default" {
  name             = "${var.app_name}-${var.environment}-https-proxy"
  url_map          = google_compute_url_map.default.id
  ssl_certificates = [google_compute_managed_ssl_certificate.default.id]
}

# SSL Certificate
resource "google_compute_managed_ssl_certificate" "default" {
  name = "${var.app_name}-${var.environment}-cert"
  
  managed {
    domains = ["api.yourdomain.com"]
  }
}

# Global Forwarding Rule
resource "google_compute_global_forwarding_rule" "default" {
  name                  = "${var.app_name}-${var.environment}-forwarding-rule"
  ip_protocol           = "TCP"
  port_range            = "443"
  target                = google_compute_target_https_proxy.default.id
  ip_address            = google_compute_global_address.default.address
  load_balancing_scheme = "EXTERNAL"
}

# HTTP to HTTPS redirect
resource "google_compute_url_map" "https_redirect" {
  name = "${var.app_name}-${var.environment}-https-redirect"
  
  default_url_redirect {
    https_redirect         = true
    redirect_response_code = "MOVED_PERMANENTLY_DEFAULT"
    strip_query            = false
  }
}

resource "google_compute_target_http_proxy" "https_redirect" {
  name    = "${var.app_name}-${var.environment}-http-proxy"
  url_map = google_compute_url_map.https_redirect.id
}

resource "google_compute_global_forwarding_rule" "https_redirect" {
  name                  = "${var.app_name}-${var.environment}-http-forwarding-rule"
  ip_protocol           = "TCP"
  port_range            = "80"
  target                = google_compute_target_http_proxy.https_redirect.id
  ip_address            = google_compute_global_address.default.address
  load_balancing_scheme = "EXTERNAL"
}

# Monitoring
resource "google_monitoring_uptime_check_config" "https" {
  display_name = "${var.app_name}-${var.environment}-uptime-check"
  timeout      = "10s"
  period       = "60s"
  
  http_check {
    path         = "/health"
    port         = "443"
    use_ssl      = true
    validate_ssl = true
  }
  
  monitored_resource {
    type = "uptime_url"
    labels = {
      project_id = var.project_id
      host       = "api.yourdomain.com"
    }
  }
}

# Outputs
output "service_url" {
  description = "URL of the Cloud Run service"
  value       = google_cloud_run_v2_service.app.uri
}

output "load_balancer_ip" {
  description = "IP address of the load balancer"
  value       = google_compute_global_address.default.address
}

output "artifact_registry_url" {
  description = "URL of the Artifact Registry"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}"
}

output "redis_host" {
  description = "Redis instance host"
  value       = google_redis_instance.cache.host
}