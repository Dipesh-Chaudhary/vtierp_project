from typing import Dict, Any, List

# This dictionary will store the status of PDF processing tasks.
# It should be defined here, and other modules should import it from here.
# Key: pdf_id (str)
# Value: Dict containing status, filename, message, title, page_count, processed_at, processing_time_ms
TASK_STATUS: Dict[str, Dict[str, Any]] = {}

# Tracks PDF IDs uploaded in the current "session" or batch
CURRENT_SESSION_BATCH_PDF_IDS: List[str] = []
