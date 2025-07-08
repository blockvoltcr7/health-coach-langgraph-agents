# Azure Terraform configuration for RAG API deployment

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.85"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~> 2.47"
    }
  }
  
  backend "azurerm" {
    resource_group_name  = "terraform-state-rg"
    storage_account_name = "tfstateragapi"
    container_name       = "tfstate"
    key                  = "rag-api/terraform.tfstate"
  }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

provider "azuread" {}

# Variables
variable "location" {
  description = "Azure region"
  default     = "East US"
}

variable "environment" {
  description = "Environment name"
  default     = "production"
}

variable "app_name" {
  description = "Application name"
  default     = "rag-api"
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "${var.app_name}-${var.environment}-rg"
  location = var.location
  
  tags = {
    Environment = var.environment
    Application = var.app_name
    ManagedBy   = "Terraform"
  }
}

# Virtual Network
resource "azurerm_virtual_network" "main" {
  name                = "${var.app_name}-${var.environment}-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  
  tags = azurerm_resource_group.main.tags
}

resource "azurerm_subnet" "app" {
  name                 = "app-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.1.0/24"]
  
  delegation {
    name = "app-delegation"
    
    service_delegation {
      name    = "Microsoft.Web/serverFarms"
      actions = ["Microsoft.Network/virtualNetworks/subnets/action"]
    }
  }
}

resource "azurerm_subnet" "redis" {
  name                 = "redis-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.2.0/24"]
}

# Container Registry
resource "azurerm_container_registry" "main" {
  name                = "${replace(var.app_name, "-", "")}${var.environment}acr"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Premium"
  admin_enabled       = false
  
  georeplications {
    location                = "West US"
    zone_redundancy_enabled = true
    tags                    = azurerm_resource_group.main.tags
  }
  
  network_rule_set {
    default_action = "Allow"
  }
  
  retention_policy {
    days    = 30
    enabled = true
  }
  
  trust_policy {
    enabled = true
  }
  
  tags = azurerm_resource_group.main.tags
}

# Azure Cache for Redis
resource "azurerm_redis_cache" "main" {
  name                = "${var.app_name}-${var.environment}-redis"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  capacity            = 1
  family              = "P"
  sku_name            = "Premium"
  
  redis_configuration {
    enable_authentication = true
    maxmemory_policy     = "allkeys-lru"
  }
  
  subnet_id = azurerm_subnet.redis.id
  
  zones = ["1", "2"]
  
  tags = azurerm_resource_group.main.tags
}

# Key Vault
resource "azurerm_key_vault" "main" {
  name                        = "${replace(var.app_name, "-", "")}${var.environment}kv"
  location                    = azurerm_resource_group.main.location
  resource_group_name         = azurerm_resource_group.main.name
  enabled_for_disk_encryption = false
  tenant_id                   = data.azurerm_client_config.current.tenant_id
  soft_delete_retention_days  = 7
  purge_protection_enabled    = true
  sku_name                    = "standard"
  
  network_acls {
    default_action = "Allow"
    bypass         = "AzureServices"
  }
  
  tags = azurerm_resource_group.main.tags
}

data "azurerm_client_config" "current" {}

# Key Vault Access Policy for Terraform
resource "azurerm_key_vault_access_policy" "terraform" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = data.azurerm_client_config.current.object_id
  
  secret_permissions = [
    "Get", "List", "Set", "Delete", "Purge", "Recover"
  ]
}

# Key Vault Secrets
resource "azurerm_key_vault_secret" "mongodb_uri" {
  name         = "mongodb-uri"
  value        = "mongodb+srv://username:password@cluster.mongodb.net/rag_production?retryWrites=true"
  key_vault_id = azurerm_key_vault.main.id
  
  depends_on = [azurerm_key_vault_access_policy.terraform]
}

resource "azurerm_key_vault_secret" "openai_api_key" {
  name         = "openai-api-key"
  value        = "your-openai-api-key"
  key_vault_id = azurerm_key_vault.main.id
  
  depends_on = [azurerm_key_vault_access_policy.terraform]
}

resource "azurerm_key_vault_secret" "voyage_ai_api_key" {
  name         = "voyage-ai-api-key"
  value        = "your-voyage-api-key"
  key_vault_id = azurerm_key_vault.main.id
  
  depends_on = [azurerm_key_vault_access_policy.terraform]
}

# App Service Plan
resource "azurerm_service_plan" "main" {
  name                = "${var.app_name}-${var.environment}-plan"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = "P1v3"
  
  tags = azurerm_resource_group.main.tags
}

# Container App (using Web App for Containers)
resource "azurerm_linux_web_app" "main" {
  name                = "${var.app_name}-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_service_plan.main.location
  service_plan_id     = azurerm_service_plan.main.id
  
  site_config {
    always_on                               = true
    container_registry_use_managed_identity = true
    vnet_route_all_enabled                  = true
    
    application_stack {
      docker_image_name        = "${azurerm_container_registry.main.login_server}/${var.app_name}:latest"
      docker_registry_url      = "https://${azurerm_container_registry.main.login_server}"
    }
    
    health_check_path                 = "/health"
    health_check_eviction_time_in_min = 10
    
    app_command_line = "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4"
    
    cors {
      allowed_origins = ["*"]
      support_credentials = false
    }
    
    ip_restriction {
      action     = "Allow"
      priority   = 100
      name       = "Allow all"
      ip_address = "0.0.0.0/0"
    }
  }
  
  app_settings = {
    # Application settings
    API_TITLE     = "RAG API Service"
    API_VERSION   = "1.0.0"
    MONGODB_DATABASE = "rag_production"
    LOG_LEVEL     = "INFO"
    
    # Redis connection
    REDIS_HOST = azurerm_redis_cache.main.hostname
    REDIS_PORT = "6380"
    REDIS_SSL  = "true"
    REDIS_PASSWORD = azurerm_redis_cache.main.primary_access_key
    
    # Key Vault references
    MONGODB_URI = "@Microsoft.KeyVault(SecretUri=${azurerm_key_vault_secret.mongodb_uri.id})"
    OPENAI_API_KEY = "@Microsoft.KeyVault(SecretUri=${azurerm_key_vault_secret.openai_api_key.id})"
    VOYAGE_AI_API_KEY = "@Microsoft.KeyVault(SecretUri=${azurerm_key_vault_secret.voyage_ai_api_key.id})"
    
    # Docker settings
    DOCKER_REGISTRY_SERVER_URL = "https://${azurerm_container_registry.main.login_server}"
    WEBSITES_ENABLE_APP_SERVICE_STORAGE = false
    DOCKER_ENABLE_CI = true
  }
  
  identity {
    type = "SystemAssigned"
  }
  
  virtual_network_subnet_id = azurerm_subnet.app.id
  
  logs {
    detailed_error_messages = true
    failed_request_tracing  = true
    http_logs {
      file_system {
        retention_in_days = 30
        retention_in_mb   = 100
      }
    }
    application_logs {
      file_system_level = "Information"
    }
  }
  
  tags = azurerm_resource_group.main.tags
}

# Grant App Service access to Key Vault
resource "azurerm_key_vault_access_policy" "app" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = azurerm_linux_web_app.main.identity[0].tenant_id
  object_id    = azurerm_linux_web_app.main.identity[0].principal_id
  
  secret_permissions = ["Get", "List"]
}

# Grant App Service access to Container Registry
resource "azurerm_role_assignment" "acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_linux_web_app.main.identity[0].principal_id
}

# Application Gateway (Load Balancer)
resource "azurerm_public_ip" "main" {
  name                = "${var.app_name}-${var.environment}-pip"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  allocation_method   = "Static"
  sku                 = "Standard"
  zones               = ["1", "2", "3"]
  
  tags = azurerm_resource_group.main.tags
}

resource "azurerm_subnet" "appgw" {
  name                 = "appgw-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.3.0/24"]
}

resource "azurerm_application_gateway" "main" {
  name                = "${var.app_name}-${var.environment}-appgw"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  
  sku {
    name     = "WAF_v2"
    tier     = "WAF_v2"
    capacity = 2
  }
  
  gateway_ip_configuration {
    name      = "gateway-ip-configuration"
    subnet_id = azurerm_subnet.appgw.id
  }
  
  frontend_port {
    name = "http"
    port = 80
  }
  
  frontend_port {
    name = "https"
    port = 443
  }
  
  frontend_ip_configuration {
    name                 = "frontend-ip"
    public_ip_address_id = azurerm_public_ip.main.id
  }
  
  backend_address_pool {
    name = "backend-pool"
    fqdns = [azurerm_linux_web_app.main.default_hostname]
  }
  
  backend_http_settings {
    name                  = "backend-http-settings"
    cookie_based_affinity = "Disabled"
    port                  = 443
    protocol              = "Https"
    request_timeout       = 30
    pick_host_name_from_backend_address = true
    
    probe_name = "health-probe"
  }
  
  http_listener {
    name                           = "http-listener"
    frontend_ip_configuration_name = "frontend-ip"
    frontend_port_name             = "http"
    protocol                       = "Http"
  }
  
  http_listener {
    name                           = "https-listener"
    frontend_ip_configuration_name = "frontend-ip"
    frontend_port_name             = "https"
    protocol                       = "Https"
    ssl_certificate_name           = "ssl-cert"
  }
  
  request_routing_rule {
    name                       = "http-to-https-redirect"
    rule_type                  = "Basic"
    http_listener_name         = "http-listener"
    redirect_configuration_name = "http-to-https-redirect"
    priority                   = 100
  }
  
  request_routing_rule {
    name                       = "https-routing"
    rule_type                  = "Basic"
    http_listener_name         = "https-listener"
    backend_address_pool_name  = "backend-pool"
    backend_http_settings_name = "backend-http-settings"
    priority                   = 200
  }
  
  redirect_configuration {
    name                 = "http-to-https-redirect"
    redirect_type        = "Permanent"
    target_listener_name = "https-listener"
    include_path         = true
    include_query_string = true
  }
  
  probe {
    name                = "health-probe"
    protocol            = "Https"
    path                = "/health"
    interval            = 30
    timeout             = 30
    unhealthy_threshold = 3
    pick_host_name_from_backend_http_settings = true
  }
  
  ssl_certificate {
    name     = "ssl-cert"
    data     = filebase64("path/to/your/certificate.pfx")
    password = "certificate-password"
  }
  
  waf_configuration {
    enabled                  = true
    firewall_mode            = "Prevention"
    rule_set_type            = "OWASP"
    rule_set_version         = "3.2"
    request_body_check       = true
    max_request_body_size_kb = 128
  }
  
  autoscale_configuration {
    min_capacity = 2
    max_capacity = 10
  }
  
  zones = ["1", "2", "3"]
  
  tags = azurerm_resource_group.main.tags
}

# Auto-scaling for App Service
resource "azurerm_monitor_autoscale_setting" "main" {
  name                = "${var.app_name}-${var.environment}-autoscale"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  target_resource_id  = azurerm_service_plan.main.id
  
  profile {
    name = "default"
    
    capacity {
      default = 2
      minimum = 2
      maximum = 10
    }
    
    rule {
      metric_trigger {
        metric_name        = "CpuPercentage"
        metric_resource_id = azurerm_service_plan.main.id
        time_grain         = "PT1M"
        statistic          = "Average"
        time_window        = "PT5M"
        time_aggregation   = "Average"
        operator           = "GreaterThan"
        threshold          = 70
      }
      
      scale_action {
        direction = "Increase"
        type      = "ChangeCount"
        value     = "1"
        cooldown  = "PT5M"
      }
    }
    
    rule {
      metric_trigger {
        metric_name        = "CpuPercentage"
        metric_resource_id = azurerm_service_plan.main.id
        time_grain         = "PT1M"
        statistic          = "Average"
        time_window        = "PT5M"
        time_aggregation   = "Average"
        operator           = "LessThan"
        threshold          = 30
      }
      
      scale_action {
        direction = "Decrease"
        type      = "ChangeCount"
        value     = "1"
        cooldown  = "PT10M"
      }
    }
    
    rule {
      metric_trigger {
        metric_name        = "MemoryPercentage"
        metric_resource_id = azurerm_service_plan.main.id
        time_grain         = "PT1M"
        statistic          = "Average"
        time_window        = "PT5M"
        time_aggregation   = "Average"
        operator           = "GreaterThan"
        threshold          = 80
      }
      
      scale_action {
        direction = "Increase"
        type      = "ChangeCount"
        value     = "1"
        cooldown  = "PT5M"
      }
    }
  }
  
  tags = azurerm_resource_group.main.tags
}

# Application Insights
resource "azurerm_application_insights" "main" {
  name                = "${var.app_name}-${var.environment}-insights"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  application_type    = "web"
  
  tags = azurerm_resource_group.main.tags
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "main" {
  name                = "${var.app_name}-${var.environment}-logs"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  
  tags = azurerm_resource_group.main.tags
}

# Outputs
output "app_service_url" {
  description = "URL of the App Service"
  value       = "https://${azurerm_linux_web_app.main.default_hostname}"
}

output "application_gateway_ip" {
  description = "Public IP of the Application Gateway"
  value       = azurerm_public_ip.main.ip_address
}

output "container_registry_url" {
  description = "URL of the Container Registry"
  value       = azurerm_container_registry.main.login_server
}

output "redis_connection_string" {
  description = "Redis connection string"
  value       = azurerm_redis_cache.main.primary_connection_string
  sensitive   = true
}

output "application_insights_key" {
  description = "Application Insights instrumentation key"
  value       = azurerm_application_insights.main.instrumentation_key
  sensitive   = true
}