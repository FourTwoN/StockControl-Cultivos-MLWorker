# =============================================================================
# Terraform Outputs
# =============================================================================

# =============================================================================
# Cloud Run Outputs
# =============================================================================

output "mlworker_url" {
  description = "ML Worker Cloud Run service URL"
  value       = google_cloud_run_v2_service.mlworker.uri
}

output "mlworker_service_account" {
  description = "ML Worker service account email"
  value       = google_service_account.mlworker_runner.email
}

# =============================================================================
# Cloud Tasks Outputs
# =============================================================================

output "queue_photo_processing" {
  description = "Photo processing queue name"
  value       = google_cloud_tasks_queue.photo_processing.name
}

output "queue_image_compress" {
  description = "Image compression queue name"
  value       = google_cloud_tasks_queue.image_compress.name
}

output "queue_reports" {
  description = "Reports queue name"
  value       = google_cloud_tasks_queue.reports.name
}

output "cloudtasks_invoker_email" {
  description = "Cloud Tasks invoker service account email (for Backend)"
  value       = google_service_account.cloudtasks_invoker.email
}

# =============================================================================
# Integration Info (for Backend configuration)
# =============================================================================

output "backend_integration" {
  description = "Configuration values for Backend integration"
  value = {
    cloudtasks_invoker_sa = google_service_account.cloudtasks_invoker.email
    mlworker_url          = google_cloud_run_v2_service.mlworker.uri
    queues = {
      photo_processing = google_cloud_tasks_queue.photo_processing.name
      image_compress   = google_cloud_tasks_queue.image_compress.name
      reports          = google_cloud_tasks_queue.reports.name
    }
  }
}
