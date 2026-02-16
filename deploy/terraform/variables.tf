# =============================================================================
# Project Configuration
# =============================================================================

variable "project_id" {
  type        = string
  description = "GCP Project ID"
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "GCP Region for all resources"
}

variable "environment" {
  type        = string
  default     = "prod"
  description = "Environment name (prod, staging)"
  validation {
    condition     = contains(["prod", "staging"], var.environment)
    error_message = "Environment must be 'prod' or 'staging'."
  }
}

variable "industry" {
  type        = string
  default     = "agro"
  description = "Industry identifier (agro, vending)"
  validation {
    condition     = contains(["agro", "vending"], var.industry)
    error_message = "Industry must be 'agro' or 'vending'."
  }
}

# =============================================================================
# Cloud Run Configuration
# =============================================================================

variable "cloudrun_cpu" {
  type        = string
  default     = "4"
  description = "CPU allocation for ML Worker (4 vCPU for ML inference)"
}

variable "cloudrun_memory" {
  type        = string
  default     = "8Gi"
  description = "Memory allocation for ML Worker (8GB for model loading)"
}

variable "cloudrun_min_instances" {
  type        = number
  default     = 0
  description = "Minimum Cloud Run instances (0 for scale-to-zero)"
}

variable "cloudrun_max_instances" {
  type        = number
  default     = 10
  description = "Maximum Cloud Run instances"
}

variable "cloudrun_timeout" {
  type        = string
  default     = "1800s"
  description = "Request timeout (30 min for ML inference)"
}

variable "mlworker_deployed" {
  type        = bool
  default     = false
  description = "Set to true after initial Cloud Run deployment to enable data sources"
}

# =============================================================================
# Cloud Tasks Configuration
# =============================================================================

variable "tasks_rate_limit" {
  type        = number
  default     = 10
  description = "Maximum dispatches per second per queue"
}

variable "tasks_concurrent" {
  type        = number
  default     = 5
  description = "Maximum concurrent dispatches per queue"
}

# =============================================================================
# Database Configuration
# =============================================================================

variable "db_connection_name" {
  type        = string
  description = "Cloud SQL connection name (project:region:instance)"
}

variable "db_name" {
  type        = string
  default     = "demeter"
  description = "Database name"
}

variable "db_user" {
  type        = string
  default     = "demeter_app"
  description = "Database user"
}

variable "db_password" {
  type        = string
  sensitive   = true
  description = "Database password"
}

# =============================================================================
# Storage Configuration
# =============================================================================

variable "gcs_bucket" {
  type        = string
  description = "Cloud Storage bucket for images"
}

variable "model_path" {
  type        = string
  default     = "gs://demeter-models/yolo"
  description = "GCS path to ML models"
}

# =============================================================================
# Backend Integration
# =============================================================================

variable "backend_url" {
  type        = string
  description = "Demeter backend URL for callbacks (e.g., https://demeter-backend-prod.run.app)"
}
