# =============================================================================
# StockControl ML Worker - Terraform Configuration
# =============================================================================
# Infrastructure for background ML processing with Cloud Tasks + Cloud Run

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# =============================================================================
# Enable Required APIs
# =============================================================================

resource "google_project_service" "cloudtasks" {
  service            = "cloudtasks.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudrun" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "secretmanager" {
  service            = "secretmanager.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "sqladmin" {
  service            = "sqladmin.googleapis.com"
  disable_on_destroy = false
}

# =============================================================================
# Data Sources
# =============================================================================

# Get Cloud Run service URL after deployment (for Cloud Tasks target)
data "google_cloud_run_v2_service" "mlworker" {
  count    = var.mlworker_deployed ? 1 : 0
  name     = google_cloud_run_v2_service.mlworker.name
  location = var.region
}
