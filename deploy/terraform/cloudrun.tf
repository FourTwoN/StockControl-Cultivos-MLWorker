# =============================================================================
# Cloud Run v2 - ML Worker Service
# =============================================================================

resource "google_cloud_run_v2_service" "mlworker" {
  name     = "demeter-mlworker-${var.environment}"
  location = var.region

  template {
    service_account = google_service_account.mlworker_runner.email

    scaling {
      min_instance_count = var.cloudrun_min_instances
      max_instance_count = var.cloudrun_max_instances
    }

    # Cloud SQL connection via Unix socket
    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [var.db_connection_name]
      }
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/demeter/mlworker:latest"

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = var.cloudrun_cpu
          memory = var.cloudrun_memory
        }
        cpu_idle          = true    # Scale to zero when idle
        startup_cpu_boost = true    # Faster cold starts
      }

      # =======================================================================
      # Environment Variables
      # =======================================================================

      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }

      env {
        name  = "INDUSTRY"
        value = var.industry
      }

      env {
        name  = "GCS_BUCKET"
        value = var.gcs_bucket
      }

      env {
        name  = "MODEL_PATH"
        value = var.model_path
      }

      env {
        name  = "BACKEND_URL"
        value = var.backend_url
      }

      env {
        name  = "DB_CONNECTION_NAME"
        value = var.db_connection_name
      }

      # =======================================================================
      # Secrets from Secret Manager
      # =======================================================================

      env {
        name = "DB_USER"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.mlworker_db_user.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "DB_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.mlworker_db_password.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "DB_NAME"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.mlworker_db_name.secret_id
            version = "latest"
          }
        }
      }

      # =======================================================================
      # Volume Mounts
      # =======================================================================

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }

      # =======================================================================
      # Health Probes
      # =======================================================================

      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 10
        period_seconds        = 5
        failure_threshold     = 30    # Allow 2.5 min for model loading
        timeout_seconds       = 5
      }

      liveness_probe {
        http_get {
          path = "/health/live"
          port = 8080
        }
        period_seconds    = 30
        timeout_seconds   = 5
        failure_threshold = 3
      }
    }

    # ML inference can be slow - 30 minute timeout
    timeout = var.cloudrun_timeout
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_project_service.cloudrun,
    google_secret_manager_secret_version.mlworker_db_user,
    google_secret_manager_secret_version.mlworker_db_password,
    google_secret_manager_secret_version.mlworker_db_name,
    google_secret_manager_secret_iam_member.mlworker_db_user_access,
    google_secret_manager_secret_iam_member.mlworker_db_password_access,
    google_secret_manager_secret_iam_member.mlworker_db_name_access,
    google_project_iam_member.mlworker_cloudsql,
  ]
}

# =============================================================================
# Note: Cloud Run with GPU (for production)
# =============================================================================
# To enable GPU, use google-beta provider and add:
#
# provider = google-beta
#
# In containers block:
#   resources {
#     limits = {
#       cpu    = "4"
#       memory = "16Gi"
#       "nvidia.com/gpu" = "1"
#     }
#   }
#
# And add annotation:
#   annotations = {
#     "run.googleapis.com/gpu-type" = "nvidia-tesla-t4"
#   }
