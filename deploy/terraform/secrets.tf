# =============================================================================
# Secret Manager - ML Worker Secrets
# =============================================================================

# =============================================================================
# Database Secrets
# =============================================================================

resource "google_secret_manager_secret" "mlworker_db_user" {
  secret_id = "mlworker-db-user-${var.environment}"

  replication {
    auto {}
  }

  depends_on = [google_project_service.secretmanager]
}

resource "google_secret_manager_secret_version" "mlworker_db_user" {
  secret      = google_secret_manager_secret.mlworker_db_user.id
  secret_data = var.db_user
}

resource "google_secret_manager_secret" "mlworker_db_password" {
  secret_id = "mlworker-db-password-${var.environment}"

  replication {
    auto {}
  }

  depends_on = [google_project_service.secretmanager]
}

resource "google_secret_manager_secret_version" "mlworker_db_password" {
  secret      = google_secret_manager_secret.mlworker_db_password.id
  secret_data = var.db_password
}

resource "google_secret_manager_secret" "mlworker_db_name" {
  secret_id = "mlworker-db-name-${var.environment}"

  replication {
    auto {}
  }

  depends_on = [google_project_service.secretmanager]
}

resource "google_secret_manager_secret_version" "mlworker_db_name" {
  secret      = google_secret_manager_secret.mlworker_db_name.id
  secret_data = var.db_name
}

# =============================================================================
# IAM - Secret Access for ML Worker
# =============================================================================

resource "google_secret_manager_secret_iam_member" "mlworker_db_user_access" {
  secret_id = google_secret_manager_secret.mlworker_db_user.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.mlworker_runner.email}"
}

resource "google_secret_manager_secret_iam_member" "mlworker_db_password_access" {
  secret_id = google_secret_manager_secret.mlworker_db_password.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.mlworker_runner.email}"
}

resource "google_secret_manager_secret_iam_member" "mlworker_db_name_access" {
  secret_id = google_secret_manager_secret.mlworker_db_name.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.mlworker_runner.email}"
}
