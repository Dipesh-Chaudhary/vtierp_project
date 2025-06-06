import os
import uuid
import shutil
import logging
from datetime import datetime
import time
from typing import List, Dict, Optional, Any
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Path as FastApiPath
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings, get_pdf_upload_dir, get_pdf_specific_data_dir, get_vector_store_base_dir
from app.models.pydantic_models import ( # QueryRequest now includes chat_history
    UploadResponse, PDFStatusResponse, QueryRequest, QueryResponse, ChatMessage
)
from app.services.pdf_processor import process_single_pdf_custom # This import should now work
from app.services.vector_store_manager import (
    create_or_load_vector_stores_for_pdf,
    check_if_pdf_processed,
    generate_current_batch_corpus_summary
)
from app.services.rag_agent import run_rag_query, get_compiled_rag_agent
from app.core.state_manager import TASK_STATUS, CURRENT_SESSION_BATCH_PDF_IDS # Import from new central location


logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VTIERP - Custom PDF Analysis Engine",
    description="API for processing research papers (PDFs) and answering questions using RAG.",
    version="1.3.6" # Incremented version
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

STATIC_FILES_DIR = get_vector_store_base_dir()
app.mount("/static_data", StaticFiles(directory=STATIC_FILES_DIR), name="static_data")

async def background_pdf_processing_task(pdf_id: str, original_filename: str, file_content_bytes: bytes): 
    logger.info(f"Background task starting for PDF ID: {pdf_id}, Filename: {original_filename}")
    process_start_time = time.perf_counter()
    upload_dir = get_pdf_upload_dir() 
    temp_file_path = os.path.join(upload_dir, f"{pdf_id}_{original_filename}")
    try:
        with open(temp_file_path, "wb") as buffer: buffer.write(file_content_bytes)
        logger.info(f"PDF '{original_filename}' saved to '{temp_file_path}' for processing.")
        TASK_STATUS[pdf_id] = {"status": "PROCESSING", "filename": original_filename, "message": "Extracting content and metadata...", "title": original_filename, "abstract": "N/A", "page_count": 0, "processed_at": None, "processing_time_ms": 0 }
        
        # --- CRITICAL FIX: Pass processed_captions_globally_for_this_pdf to extract_visual_elements_from_page ---
        # The logic inside process_single_pdf_custom is responsible for `processed_captions_globally_for_this_pdf`
        # and passing it correctly. So the call here should be fine if `process_single_pdf_custom` handles it.
        # This call should remain as-is, as process_single_pdf_custom has its own internal global tracking logic.
        extracted_docs, pdf_meta_summary = process_single_pdf_custom(pdf_file_path=temp_file_path, pdf_id=pdf_id, original_pdf_filename_for_metadata=original_filename)
        
        TASK_STATUS[pdf_id].update({"message": "Creating vector stores...", "title": pdf_meta_summary.get("title", original_filename), "abstract": pdf_meta_summary.get("abstract", "N/A"), "page_count": pdf_meta_summary.get("page_count", 0)})
        if not extracted_docs: logger.warning(f"No documents extracted for PDF ID: {pdf_id}. Marking as FAILED."); TASK_STATUS[pdf_id].update({"status": "FAILED", "message": "No content could be extracted from the PDF."}); return
        text_vs, image_vs = create_or_load_vector_stores_for_pdf(pdf_id, extracted_docs, force_recreate=True)
        if not text_vs and not image_vs and extracted_docs: logger.error(f"Vector store creation failed for PDF ID: {pdf_id} despite having extracted documents."); TASK_STATUS[pdf_id].update({"status": "FAILED", "message": "Failed to create vector stores for extracted content."}); return
        process_end_time = time.perf_counter(); processing_duration_ms = (process_end_time - process_start_time) * 1000
        TASK_STATUS[pdf_id].update({"status": "COMPLETED", "message": "PDF processed successfully and is ready for querying.", "processed_at": datetime.utcnow().isoformat(), "processing_time_ms": processing_duration_ms})
        logger.info(f"Successfully processed PDF ID: {pdf_id} ('{TASK_STATUS[pdf_id]['title']}') in {processing_duration_ms:.2f} ms.")
    except Exception as e:
        logger.error(f"Critical error processing PDF ID {pdf_id} ('{original_filename}') in background: {e}", exc_info=True)
        if pdf_id in TASK_STATUS: TASK_STATUS[pdf_id].update({"status": "FAILED", "message": f"Critical processing error: {str(e)}"})
        else: TASK_STATUS[pdf_id] = {"status": "FAILED", "filename": original_filename, "message": f"Critical processing error: {str(e)}"}
        pdf_data_dir = get_pdf_specific_data_dir(pdf_id)
        if os.path.exists(pdf_data_dir):
            try: shutil.rmtree(pdf_data_dir); logger.info(f"Cleaned up data directory for failed PDF ID: {pdf_id} at {pdf_data_dir}")
            except Exception as cleanup_e: logger.error(f"Error cleaning up data for failed PDF ID {pdf_id}: {cleanup_e}")
    finally:
        if os.path.exists(temp_file_path):
            try: os.remove(temp_file_path); logger.info(f"Removed temporary uploaded file: {temp_file_path}")
            except OSError as e_remove: logger.error(f"Error removing temporary file {temp_file_path}: {e_remove}")

@app.on_event("startup")
async def startup_event():
    try: get_compiled_rag_agent(); logger.info("RAG Agent graph pre-compiled successfully.")
    except Exception as e: logger.error(f"Failed to pre-compile RAG agent on startup: {e}", exc_info=True)
    logger.info("FastAPI application startup sequence complete.")


@app.post("/upload-multiple-pdfs/", response_model=List[UploadResponse])
async def upload_multiple_pdfs_endpoint(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    global CURRENT_SESSION_BATCH_PDF_IDS # Now importing this global variable
    if not files: raise HTTPException(status_code=400, detail="No files were provided for upload.")
    CURRENT_SESSION_BATCH_PDF_IDS.clear() # Clear the global list
    upload_responses: List[UploadResponse] = []
    for file_upload_item in files:
        original_filename = file_upload_item.filename
        if not original_filename.lower().endswith(".pdf"): logger.warning(f"Skipping non-PDF file: {original_filename}"); upload_responses.append(UploadResponse(pdf_id="N/A", filename=original_filename, message="Invalid file type (must be .pdf)", status_check_url="N/A")); continue
        pdf_id = str(uuid.uuid4())
        try: file_content_bytes = await file_upload_item.read()
        except Exception as e: logger.error(f"Failed to read file content for {original_filename}: {e}"); upload_responses.append(UploadResponse(pdf_id="N/A", filename=original_filename, message=f"Error reading file: {str(e)}", status_check_url="N/A")); await file_upload_item.close(); continue
        finally: await file_upload_item.close()
        TASK_STATUS[pdf_id] = {"status": "PENDING", "filename": original_filename, "message": "Queued for processing."}
        background_tasks.add_task(background_pdf_processing_task, pdf_id, original_filename, file_content_bytes)
        CURRENT_SESSION_BATCH_PDF_IDS.append(pdf_id)
        status_url = app.url_path_for("get_pdf_status_endpoint", pdf_id=pdf_id)
        upload_responses.append(UploadResponse(pdf_id=pdf_id, filename=original_filename, message="Upload accepted, processing queued.", status_check_url=str(status_url)))
    if not upload_responses and files: raise HTTPException(status_code=400, detail="No valid PDF files were processed from the upload.")
    if not upload_responses and not files: raise HTTPException(status_code=400, detail="No files were uploaded.")
    logger.info(f"Batch upload processed. Current session PDF IDs for processing: {CURRENT_SESSION_BATCH_PDF_IDS}")
    return upload_responses

@app.get("/pdf-status/{pdf_id}", response_model=PDFStatusResponse, name="get_pdf_status_endpoint")
async def get_pdf_status_endpoint(pdf_id: str = FastApiPath(..., title="The ID of the PDF to check status for")):
    status_info = TASK_STATUS.get(pdf_id)
    if not status_info:
        if check_if_pdf_processed(pdf_id):
             return PDFStatusResponse(pdf_id=pdf_id, filename=f"PDF (ID: {pdf_id[:8]}...)", status="COMPLETED", message="PDF was processed (status from disk, details may be missing due to server state).", title=f"PDF (ID: {pdf_id[:8]}...)", page_count=0, processed_at=None, processing_time_ms=None)
        raise HTTPException(status_code=404, detail=f"Status for PDF with ID '{pdf_id}' not found.")
    response_data = {"pdf_id": pdf_id, "filename": status_info.get("filename", "N/A"), "status": status_info.get("status", "UNKNOWN"), "message": status_info.get("message"), "page_count": status_info.get("page_count"), "title": status_info.get("title"), "processed_at": status_info.get("processed_at"), "processing_time_ms": status_info.get("processing_time_ms")}
    return PDFStatusResponse(**response_data)

@app.post("/query/", response_model=QueryResponse)
async def query_pdf_endpoint(request: QueryRequest):
    logger.info(f"Query received: Scope='{request.query_scope}', PDF_ID='{request.pdf_id}', "
                f"Corpus_IDs='{request.current_corpus_pdf_ids}', "
                f"ChatHistoryLen='{len(request.chat_history) if request.chat_history else 0}', "
                f"Q: '{request.question[:50]}...'")
    
    summary_for_rag_agent: Optional[str] = None
    query_pdf_id_for_agent_logic: Optional[str] = None 
    corpus_ids_for_agent_logic: Optional[List[str]] = None 

    if request.query_scope == "corpus":
        active_corpus_ids = request.current_corpus_pdf_ids
        if not active_corpus_ids: 
            if not CURRENT_SESSION_BATCH_PDF_IDS:
                raise HTTPException(status_code=400, detail="Corpus query: No PDFs in current batch.")
            active_corpus_ids = CURRENT_SESSION_BATCH_PDF_IDS
            logger.info("Corpus query: Using server-side batch IDs.")
        if not active_corpus_ids: raise HTTPException(status_code=400, detail="Corpus query: No PDF IDs specified.")
        corpus_ids_for_agent_logic = [pid for pid in active_corpus_ids if TASK_STATUS.get(pid, {}).get("status") == "COMPLETED" or check_if_pdf_processed(pid)]
        if not corpus_ids_for_agent_logic: raise HTTPException(status_code=400, detail="Corpus query: No specified PDFs are ready.")
        query_pdf_id_for_agent_logic = "current_corpus" 
        corpus_summary_doc = generate_current_batch_corpus_summary(corpus_ids_for_agent_logic, TASK_STATUS)
        summary_for_rag_agent = corpus_summary_doc.page_content if corpus_summary_doc else "Batch summary not generated."
        logger.info(f"Corpus query for {len(corpus_ids_for_agent_logic)} PDFs: {corpus_ids_for_agent_logic}")
    elif request.query_scope == "single":
        if not request.pdf_id: raise HTTPException(status_code=400, detail="Single PDF query: 'pdf_id' must be provided.")
        pdf_status = TASK_STATUS.get(request.pdf_id, {})
        if not (pdf_status.get("status") == "COMPLETED" or check_if_pdf_processed(request.pdf_id)):
            raise HTTPException(status_code=400, detail=f"PDF '{request.pdf_id}' not processed. Status: {pdf_status.get('status', 'NOT_FOUND')}")
        query_pdf_id_for_agent_logic = request.pdf_id
        single_pdf_title = pdf_status.get('title', 'Title Not Available')
        single_pdf_abstract = pdf_status.get('abstract', 'Abstract Not Available')
        summary_for_rag_agent = f"Document Title: {single_pdf_title}\nDocument Abstract: {single_pdf_abstract}"
    else:
        raise HTTPException(status_code=400, detail=f"Invalid query_scope: '{request.query_scope}'. Must be 'single' or 'corpus'.")

    try:
        history_for_agent = [{"role": msg.role, "content": msg.content} for msg in request.chat_history] if request.chat_history else []
        agent_result = run_rag_query(
            query_scope=request.query_scope,
            pdf_id=query_pdf_id_for_agent_logic,
            question=request.question,
            summary_for_context=summary_for_rag_agent, 
            current_corpus_pdf_ids=corpus_ids_for_agent_logic,
            chat_history=history_for_agent 
        )
    except Exception as e_rag:
        logger.error(f"Error during RAG agent execution: {e_rag}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during query processing: {str(e_rag)}")
    
    if not isinstance(agent_result, dict):
        logger.error(f"RAG query did not return a dictionary. Result: {agent_result}")
        raise HTTPException(status_code=500, detail="Internal server error: RAG agent returned unexpected data type.")

    return QueryResponse(
        answer=agent_result.get("answer", "Error: No answer was generated by the agent."),
        query_processing_time_ms=agent_result.get("processing_time_ms"),
        llm_generation_time_ms=agent_result.get("llm_generation_time_ms")
    )

@app.get("/")
async def read_root(): return {"message": "Welcome to VTIERP API (Custom Logic). Use /docs for API documentation."}
