from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class PDFProcessRequest(BaseModel):
    # Used if we allow processing by providing a URL or existing path on server
    # For upload, this is not directly used by client.
    file_path: Optional[str] = None
    pdf_id: Optional[str] = None # If ID is pre-generated

class UploadResponse(BaseModel):
    pdf_id: str
    filename: str
    message: str
    status_check_url: str

class PDFStatusResponse(BaseModel):
    pdf_id: str
    filename: str
    status: str # e.g., "PENDING", "PROCESSING", "COMPLETED", "FAILED"
    message: Optional[str] = None
    page_count: Optional[int] = None
    title: Optional[str] = None # Extracted title
    processed_at: Optional[str] = None # Timestamp

class QueryRequest(BaseModel):
    pdf_id: str
    question: str
    # Advanced options if needed:
    # max_text_results: Optional[int] = 7
    # max_image_results: Optional[int] = 5

class RetrievedContextDoc(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    pdf_id: str
    question: str
    answer: str
    # Optionally return some context for display/debugging
    # These are lists of Document objects, convert to dicts for Pydantic
    retrieved_text_context_sample: Optional[List[RetrievedContextDoc]] = Field(default_factory=list)
    retrieved_image_context_sample: Optional[List[RetrievedContextDoc]] = Field(default_factory=list)
    # In image context metadata, 'image_path_relative_to_pdf_data' will be key for UI to build URL