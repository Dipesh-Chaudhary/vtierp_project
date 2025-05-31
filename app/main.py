import os
import uuid
import shutil
import logging
from datetime import datetime
from typing import List 

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Path as FastApiPath
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings, get_pdf_upload_dir, get_pdf_specific_data_dir, get_vector_store_base_dir
from app.models.pydantic_models import (
    UploadResponse, PDFStatusResponse, QueryRequest, QueryResponse, RetrievedContextDoc
)
from app.services.pdf_processor import process_single_pdf_custom
from app.services.vector_store_manager import create_or_load_vector_stores_for_pdf, check_if_pdf_processed
from app.services.rag_agent import run_rag_query, get_compiled_rag_agent # To pre-compile graph

# Configure logging
logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VTIERP -(VisioTextual Insight Engine for Research Papers) ",
    description="API for processing research papers (PDFs) and answering questions using RAG, adapted from user's notebook.",
    version="1.1.0" # Based on notebook v11.5 logic
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Persistent Task Status (simple in-memory, replace with DB/Redis for production) ---
# This is a simplified way to track status. For production, use Celery/Redis or a DB.
TASK_STATUS = {} # pdf_id: {"status": "...", "filename": "...", "message": "...", "title": "N/A", "page_count": 0, "processed_at": None}

# --- Static File Serving for Extracted Images ---
# Serve images from the pdf_id specific subdirectories within the vector_stores main dir
# Example: /app/data/vector_stores/<pdf_id>/extracted_images/image.png
# will be accessible via /static_data/<pdf_id>/extracted_images/image.png
# The `get_vector_store_base_dir()` returns `/app/data/vector_stores`
app.mount("/static_data", StaticFiles(directory=get_vector_store_base_dir()), name="static_data")


# --- Background PDF Processing Task ---
def background_pdf_processing_task(pdf_id: str, temp_file_path: str, original_filename: str):
    logger.info(f"Background task started for PDF ID: {pdf_id}, Filename: {original_filename}")
    TASK_STATUS[pdf_id] = {
        "status": "PROCESSING", "filename": original_filename, "message": "Extracting content...",
        "title": "N/A", "page_count": 0, "processed_at": None
    }

    try:
        # 1. Process PDF to extract elements (text, image descriptions, images saved to disk)
        extracted_docs, pdf_meta_summary = process_single_pdf_custom(temp_file_path, pdf_id)
        
        TASK_STATUS[pdf_id].update({
            "message": "Creating vector stores...",
            "title": pdf_meta_summary.get("title", "N/A"),
            "page_count": pdf_meta_summary.get("page_count", 0)
        })

        if not extracted_docs:
            logger.warning(f"No documents extracted for PDF ID: {pdf_id}. Processing may have failed or PDF is empty.")
            TASK_STATUS[pdf_id].update({"status": "FAILED", "message": "No content extracted from PDF."})
            # No need to clean up temp_file_path here, done in 'finally'
            return

        # 2. Create vector stores from extracted documents
        # force_recreate=True ensures fresh stores for this new upload/processing
        create_or_load_vector_stores_for_pdf(pdf_id, extracted_docs, force_recreate=True)

        TASK_STATUS[pdf_id].update({
            "status": "COMPLETED",
            "message": "PDF processed successfully. Ready for querying.",
            "processed_at": datetime.utcnow().isoformat()
        })
        logger.info(f"Successfully processed and created vector stores for PDF ID: {pdf_id}")

    except Exception as e:
        logger.error(f"Error processing PDF ID {pdf_id} in background: {e}", exc_info=True)
        TASK_STATUS[pdf_id].update({"status": "FAILED", "message": f"Error during processing: {str(e)}"})
        # Clean up pdf_id specific data directory if processing failed badly
        pdf_data_dir = get_pdf_specific_data_dir(pdf_id)
        if os.path.exists(pdf_data_dir):
            try:
                shutil.rmtree(pdf_data_dir)
                logger.info(f"Cleaned up data directory for failed PDF ID: {pdf_id} at {pdf_data_dir}")
            except Exception as cleanup_e:
                 logger.error(f"Error cleaning up data for failed PDF ID {pdf_id}: {cleanup_e}")
    finally:
        # Clean up the temporary uploaded PDF file from 'uploads' dir
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Removed temporary uploaded file: {temp_file_path}")
            except OSError as e_remove:
                logger.error(f"Error removing temporary file {temp_file_path}: {e_remove}")


# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    # Pre-compile LangGraph agent on startup (optional, can be lazy-loaded)
    try:
        get_compiled_rag_agent()
        logger.info("RAG Agent graph pre-compiled successfully.")
    except Exception as e:
        logger.error(f"Failed to pre-compile RAG agent: {e}")
    # You can also pre-initialize LLMs here if not done in llm_config.py
    # from app.dependencies_config.llm_config import get_rag_llm, get_aux_llm, get_embeddings_model
    # try:
    #     get_rag_llm()
    #     get_aux_llm()
    #     get_embeddings_model()
    #     logger.info("LLM and Embedding models pre-initialized.")
    # except Exception as e:
    #     logger.error(f"Failed to pre-initialize LLMs/Embeddings: {e}")


@app.post("/upload-multiple-pdfs/", response_model=List[UploadResponse]) # Response is a list
async def upload_multiple_pdfs_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...) # Expect a list of files
):
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    responses = []
    for file_upload_item in files:
        if not file_upload_item.filename.lower().endswith(".pdf"):
            # Skip non-PDFs or return an error for the batch
            logger.warning(f"Skipping non-PDF file: {file_upload_item.filename}")
            # Or add an error response to the list:
            # responses.append(UploadResponse(pdf_id="N/A", filename=file_upload_item.filename, message="Invalid file type, only PDF accepted.", status_check_url="N/A"))
            continue

        pdf_id = str(uuid.uuid4())
        original_filename = file_upload_item.filename
        upload_dir = get_pdf_upload_dir()
        temp_file_path = os.path.join(upload_dir, f"{pdf_id}_{original_filename}")

        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file_upload_item.file, buffer)
            logger.info(f"PDF '{original_filename}' part of batch uploaded as '{temp_file_path}', ID: {pdf_id}")
        except Exception as e:
            logger.error(f"Could not save uploaded file {original_filename} from batch: {e}")
            # Add a failure response for this file
            responses.append(UploadResponse(pdf_id="N/A", filename=original_filename, message=f"Failed to save: {str(e)}", status_check_url="N/A"))
            continue # to next file in batch
        finally:
            await file_upload_item.close() # Close each file

        TASK_STATUS[pdf_id] = {"status": "PENDING", "filename": original_filename, "message": "Awaiting processing."}
        background_tasks.add_task(background_pdf_processing_task, pdf_id, temp_file_path, original_filename)
        
        status_url = app.url_path_for("get_pdf_status_endpoint", pdf_id=pdf_id)
        responses.append(UploadResponse(
            pdf_id=pdf_id,
            filename=original_filename,
            message="PDF upload accepted, processing started.",
            status_check_url=str(status_url)
        ))
    
    if not responses: # If all files were invalid type
         raise HTTPException(status_code=400, detail="No valid PDF files found in the upload.")
    return responses

@app.get("/pdf-status/{pdf_id}", response_model=PDFStatusResponse, name="get_pdf_status_endpoint")
async def get_pdf_status_endpoint(pdf_id: str = FastApiPath(..., title="The ID of the PDF to check status for")):
    status_info = TASK_STATUS.get(pdf_id)
    if not status_info:
        # Fallback: check if processed by looking for vector store (if server restarted and TASK_STATUS lost)
        if check_if_pdf_processed(pdf_id):
             return PDFStatusResponse(pdf_id=pdf_id, filename="Unknown (restarted)", status="COMPLETED", message="PDF was processed (server might have restarted).")
        raise HTTPException(status_code=404, detail=f"PDF with ID '{pdf_id}' not found or status not available.")
    
    return PDFStatusResponse(
        pdf_id=pdf_id,
        filename=status_info.get("filename", "N/A"),
        status=status_info.get("status", "UNKNOWN"),
        message=status_info.get("message"),
        page_count=status_info.get("page_count"),
        title=status_info.get("title"),
        processed_at=status_info.get("processed_at")
    )

@app.post("/query/", response_model=QueryResponse)
async def query_pdf_endpoint(request: QueryRequest):
    logger.info(f"Query received for PDF ID: {request.pdf_id}, Question: '{request.question[:50]}...'")
    
    pdf_status = TASK_STATUS.get(request.pdf_id)
    if not pdf_status or pdf_status.get("status") != "COMPLETED":
        # Secondary check, in case TASK_STATUS was lost (e.g. server restart)
        if not check_if_pdf_processed(request.pdf_id):
            raise HTTPException(status_code=400, detail=f"PDF with ID '{request.pdf_id}' is not yet processed or processing failed. Current status: {pdf_status.get('status', 'UNKNOWN') if pdf_status else 'NOT FOUND'}")
        else: # It is processed on disk, but TASK_STATUS might be missing.
            logger.warning(f"Querying PDF {request.pdf_id} which is processed on disk, but in-memory status was not 'COMPLETED'.")


    # Prepare the PDF-specific summary (title + abstract) for the RAG agent
    pdf_title = pdf_status.get("title", "N/A") if pdf_status else "Title not available"
    # For abstract, we'd need to fetch it if not in TASK_STATUS.
    # Simplification: For now, use title. A more robust way would be to store/retrieve abstract.
    # The process_single_pdf_custom returns this, could be stored with TASK_STATUS or a small metadata file.
    # Let's assume for now the RAG agent's first context doc (if it's title/abstract) handles this.
    # A better way for pdf_summary_for_llm:
    # Try to get title/abstract from TASK_STATUS. If not available, load first few docs from vector store.
    pdf_summary_for_llm = f"Document Title: {pdf_title}"
    # (If abstract was stored in TASK_STATUS, append it here)

    try:
        agent_result = run_rag_query(
            pdf_id=request.pdf_id,
            question=request.question,
            pdf_summary_for_llm=pdf_summary_for_llm
        )

        # Convert Langchain Document objects in context to Pydantic models for response
        text_context_out = [RetrievedContextDoc(page_content=doc.page_content, metadata=doc.metadata)
                            for doc in agent_result.get("final_text_context", [])[:3]] # Sample of 3
        image_context_out = [RetrievedContextDoc(page_content=doc.page_content, metadata=doc.metadata)
                             for doc in agent_result.get("final_image_context", [])[:3]] # Sample of 3


        return QueryResponse(
            pdf_id=request.pdf_id,
            question=request.question,
            answer=agent_result["answer"],
            retrieved_text_context_sample=text_context_out,
            retrieved_image_context_sample=image_context_out
        )
    except FileNotFoundError as e: # ChromaDB files might be missing if data dir was cleared
        logger.error(f"Data not found for PDF ID {request.pdf_id} during query: {e}")
        raise HTTPException(status_code=404, detail=f"Data for PDF ID '{request.pdf_id}' not found. It might have failed processing or been deleted.")
    except Exception as e:
        logger.error(f"Error during query for PDF ID {request.pdf_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while processing your query: {str(e)}")

@app.get("/")
async def read_root():
    return {"message": "Welcome to VTIERP API (Custom Notebook Adaptation). Use /docs for API documentation."}