# =============================================================================
# Service Accounts & IAM
# =============================================================================

# =============================================================================
# Cloud Tasks Invoker Service Account
# =============================================================================
# Used by Backend to create tasks in queues

resource "google_service_account" "cloudtasks_invoker" {
  account_id   = "cloudtasks-invoker-${var.environment}"
  display_name = "Cloud Tasks Invoker (${var.environment})"
  description  = "Service account for creating Cloud Tasks from Backend"
}

# Allow creating tasks in all queues
resource "google_cloud_tasks_queue_iam_member" "photo_processing_enqueuer" {
  name     = google_cloud_tasks_queue.photo_processing.name
  location = var.region
  role     = "roles/cloudtasks.enqueuer"
  member   = "serviceAccount:${google_service_account.cloudtasks_invoker.email}"
}

resource "google_cloud_tasks_queue_iam_member" "image_compress_enqueuer" {
  name     = google_cloud_tasks_queue.image_compress.name
  location = var.region
  role     = "roles/cloudtasks.enqueuer"
  member   = "serviceAccount:${google_service_account.cloudtasks_invoker.email}"
}

resource "google_cloud_tasks_queue_iam_member" "reports_enqueuer" {
  name     = google_cloud_tasks_queue.reports.name
  location = var.region
  role     = "roles/cloudtasks.enqueuer"
  member   = "serviceAccount:${google_service_account.cloudtasks_invoker.email}"
}

# =============================================================================
# ML Worker Runner Service Account
# =============================================================================
# Used by Cloud Run to run ML Worker

resource "google_service_account" "mlworker_runner" {
  account_id   = "mlworker-runner-${var.environment}"
  display_name = "ML Worker Runner (${var.environment})"
  description  = "Service account for ML Worker Cloud Run service"
}

# Cloud SQL Client access
resource "google_project_iam_member" "mlworker_cloudsql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.mlworker_runner.email}"
}

# Cloud Storage access (read/write images)
resource "google_project_iam_member" "mlworker_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.mlworker_runner.email}"
}

# Secret Manager access
resource "google_project_iam_member" "mlworker_secrets" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.mlworker_runner.email}"
}

# =============================================================================
# OIDC Authentication: Cloud Tasks → ML Worker
# =============================================================================
# Cloud Tasks uses OIDC token to authenticate requests to Cloud Run

# Allow Cloud Tasks service agent to create tokens for invoker
resource "google_service_account_iam_member" "cloudtasks_token_creator" {
  service_account_id = google_service_account.mlworker_runner.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:service-${data.google_project.current.number}@gcp-sa-cloudtasks.iam.gserviceaccount.com"
}

# Allow ML Worker runner to invoke Cloud Run
resource "google_cloud_run_v2_service_iam_member" "mlworker_invoker" {
  location = var.region
  name     = google_cloud_run_v2_service.mlworker.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.mlworker_runner.email}"
}

# =============================================================================
# Data Sources
# =============================================================================

data "google_project" "current" {
  project_id = var.project_id
}
