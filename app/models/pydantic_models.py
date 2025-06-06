# vtierp_project_custom/app/models/pydantic_models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class PDFProcessRequest(BaseModel): # No change
    file_path: Optional[str] = None
    pdf_id: Optional[str] = None

class UploadResponse(BaseModel): # No change
    pdf_id: str
    filename: str
    message: str
    status_check_url: str

class PDFStatusResponse(BaseModel): # No change
    pdf_id: str
    filename: str
    status: str 
    message: Optional[str] = None
    page_count: Optional[int] = None
    title: Optional[str] = None 
    processed_at: Optional[str] = None 
    processing_time_ms: Optional[float] = None

# --- ADD ChatMessage and UPDATE QueryRequest ---
class ChatMessage(BaseModel):
    role: str # "user" or "assistant"
    content: str

class QueryRequest(BaseModel):
    query_scope: str 
    pdf_id: Optional[str] = None 
    current_corpus_pdf_ids: Optional[List[str]] = None 
    question: str
    chat_history: Optional[List[ChatMessage]] = Field(default_factory=list) # Add chat history

class QueryResponse(BaseModel): # No change from previous correct version
    answer: str
    query_processing_time_ms: Optional[float] = None
    llm_generation_time_ms: Optional[float] = None

class RetrievedContextDoc(BaseModel): # No change
    page_content: str
    metadata: Dict[str, Any]
