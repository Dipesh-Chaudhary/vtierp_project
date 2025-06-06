# vtierp_project_custom/app/core/app_state.py

# This dictionary will store the status of PDF processing tasks.
# Key: pdf_id (str)
# Value: Dict containing status, filename, message, title, page_count, processed_at, processing_time_ms
TASK_STATUS: dict = {}
