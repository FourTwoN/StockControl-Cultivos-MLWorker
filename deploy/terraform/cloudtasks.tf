# =============================================================================
# Cloud Tasks Queues
# =============================================================================
# Three queues for different processing types:
# - photo-processing: ML inference (detection, segmentation, classification)
# - image-compress: Image compression and thumbnail generation
# - reports: Report generation

# =============================================================================
# Photo Processing Queue (ML Inference)
# =============================================================================

resource "google_cloud_tasks_queue" "photo_processing" {
  name     = "${var.industry}-photo-processing"
  location = var.region

  rate_limits {
    max_dispatches_per_second = var.tasks_rate_limit
    max_concurrent_dispatches = var.tasks_concurrent
  }

  retry_config {
    max_attempts       = 5
    min_backoff        = "10s"
    max_backoff        = "300s"
    max_doublings      = 4
    max_retry_duration = "3600s"  # 1 hour max retry window
  }

  stackdriver_logging_config {
    sampling_ratio = 1.0  # Log all tasks
  }

  depends_on = [google_project_service.cloudtasks]
}

# =============================================================================
# Image Compression Queue
# =============================================================================

resource "google_cloud_tasks_queue" "image_compress" {
  name     = "${var.industry}-image-compress"
  location = var.region

  rate_limits {
    max_dispatches_per_second = 20  # Higher rate for lighter tasks
    max_concurrent_dispatches = 10
  }

  retry_config {
    max_attempts       = 3
    min_backoff        = "5s"
    max_backoff        = "60s"
    max_doublings      = 3
    max_retry_duration = "600s"  # 10 min max retry window
  }

  stackdriver_logging_config {
    sampling_ratio = 0.5  # Sample 50% for cost savings
  }

  depends_on = [google_project_service.cloudtasks]
}

# =============================================================================
# Reports Queue
# =============================================================================

resource "google_cloud_tasks_queue" "reports" {
  name     = "${var.industry}-reports"
  location = var.region

  rate_limits {
    max_dispatches_per_second = 5
    max_concurrent_dispatches = 3
  }

  retry_config {
    max_attempts       = 3
    min_backoff        = "30s"
    max_backoff        = "600s"
    max_doublings      = 3
    max_retry_duration = "7200s"  # 2 hour max retry window
  }

  stackdriver_logging_config {
    sampling_ratio = 1.0
  }

  depends_on = [google_project_service.cloudtasks]
}
