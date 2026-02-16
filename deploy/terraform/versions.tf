# =============================================================================
# Terraform & Provider Versions
# =============================================================================

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 6.0"
    }
  }

  # Backend configuration for remote state (uncomment for production)
  # backend "gcs" {
  #   bucket = "demeter-terraform-state"
  #   prefix = "mlworker"
  # }
}
