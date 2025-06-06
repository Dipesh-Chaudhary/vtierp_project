import streamlit as st
import requests
import time
import os
import json
import io

# --- Configuration ---
FASTAPI_BASE_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
# INCREASED TIMEOUT FOR QUERY REQUESTS
QUERY_REQUEST_TIMEOUT_SECONDS = 1200 # 20 minutes

st.set_page_config(page_title="VTIERP Custom", layout="wide", initial_sidebar_state="expanded")

def safe_extract_error_detail(response_text: str) -> str:
    # ... (same as before)
    try:
        error_data = json.loads(response_text)
        if isinstance(error_data, dict) and "detail" in error_data:
            detail = error_data["detail"]
            if isinstance(detail, list): 
                error_messages = []
                for err_item in detail:
                    if isinstance(err_item, dict):
                        loc = " -> ".join(map(str, err_item.get("loc", [])))
                        msg = err_item.get("msg", "Unknown error detail")
                        error_messages.append(f"Field: '{loc}', Message: '{msg}'")
                    else: error_messages.append(str(err_item))
                return "; ".join(error_messages)
            return str(detail)
    except json.JSONDecodeError: pass
    except Exception: pass
    return response_text

# ... (Session state, Sidebar, PDF upload form - same as your last correct version) ...
# --- Session State Initialization (same as before) ---
if "processed_pdfs_info" not in st.session_state: st.session_state.processed_pdfs_info = {}
if "current_batch_pdf_ids" not in st.session_state: st.session_state.current_batch_pdf_ids = []
if "current_query_scope" not in st.session_state: st.session_state.current_query_scope = "corpus" 
if "selected_single_pdf_id" not in st.session_state: st.session_state.selected_single_pdf_id = None
if "chat_history" not in st.session_state: st.session_state.chat_history = [] 
if "api_error" not in st.session_state: st.session_state.api_error = None
if "st_uploaded_files_buffer" not in st.session_state: st.session_state.st_uploaded_files_buffer = []

st.title("🔬 VTIERP - Custom PDF Analysis Engine")
st.markdown("Welcome! Upload research PDFs. Once processed, query individual PDFs or the current uploaded batch.")

with st.sidebar:
    st.header("📄 PDF Management")
    with st.form("pdf_upload_form", clear_on_submit=True): 
        uploaded_files_in_form = st.file_uploader("1. Choose PDF files", type="pdf", accept_multiple_files=True, key="pdf_uploader_in_form")
        submitted_process_button = st.form_submit_button("Process Selected PDFs", type="primary")
    if submitted_process_button: 
        if not uploaded_files_in_form: st.warning("Please select at least one PDF file before processing.")
        else:
            st.session_state.api_error = None; st.session_state.chat_history = []
            st.session_state.current_query_scope = "corpus"; st.session_state.processed_pdfs_info = {}; st.session_state.current_batch_pdf_ids = []; st.session_state.selected_single_pdf_id = None 
            files_to_send = [('files', (f.name, io.BytesIO(f.getvalue()), "application/pdf")) for f in uploaded_files_in_form]
            upload_progress_bar = None 
            try:
                upload_progress_bar = st.progress(0, text="Uploading files...")
                upload_url = f"{FASTAPI_BASE_URL}/upload-multiple-pdfs/"; response = requests.post(upload_url, files=files_to_send, timeout=300) 
                if response.ok: upload_progress_bar.progress(10, text="Upload complete. Initiating processing...")
                else: upload_progress_bar.progress(100, text="Upload failed."); response.raise_for_status() 
                batch_responses = response.json(); st.info(f"Batch upload accepted by server. Processing {len(batch_responses)} PDFs in the background.")
                for pdf_resp in batch_responses:
                    pdf_id = pdf_resp.get("pdf_id")
                    if pdf_id and pdf_id != "N/A":
                        st.session_state.processed_pdfs_info[pdf_id] = {"filename": pdf_resp.get("filename"), "status": "PENDING", "message": pdf_resp.get("message"), "status_check_url": f"{FASTAPI_BASE_URL}{pdf_resp.get('status_check_url')}"}
                        st.session_state.current_batch_pdf_ids.append(pdf_id)
                    else: st.warning(f"Could not queue file '{pdf_resp.get('filename')}': {pdf_resp.get('message')}")
                if hasattr(upload_progress_bar, 'empty'): upload_progress_bar.empty() 
                if st.session_state.current_batch_pdf_ids: 
                    processing_progress_bar = st.progress(0, text="Overall Processing Progress: 0%")
                    all_processed_in_batch = False; polling_iterations = 0; max_polling_iterations = 360 * 2 
                    poll_interval_seconds = 5; status_request_timeout_seconds = 60
                    while not all_processed_in_batch and polling_iterations < max_polling_iterations:
                        all_processed_in_batch_current_check = True; current_pdfs_done_count = 0
                        total_pdfs_in_this_batch = len(st.session_state.current_batch_pdf_ids)
                        if total_pdfs_in_this_batch == 0: all_processed_in_batch = True; break 
                        for pdf_id_to_check in st.session_state.current_batch_pdf_ids:
                            pdf_info_current = st.session_state.processed_pdfs_info.get(pdf_id_to_check, {})
                            if pdf_info_current.get("status") not in ["COMPLETED", "FAILED"]:
                                try:
                                    status_resp = requests.get(pdf_info_current["status_check_url"], timeout=status_request_timeout_seconds)
                                    status_resp.raise_for_status(); status_data_from_api = status_resp.json()
                                    st.session_state.processed_pdfs_info[pdf_id_to_check].update(status_data_from_api)
                                    if status_data_from_api.get("status") == "COMPLETED" and status_data_from_api.get("title") and status_data_from_api.get("title") != "N/A" and status_data_from_api.get("title") != pdf_info_current.get("filename"):
                                        st.session_state.processed_pdfs_info[pdf_id_to_check]["filename"] = status_data_from_api.get("title")
                                    if status_data_from_api.get("status") in ["COMPLETED", "FAILED"]: current_pdfs_done_count += 1
                                except requests.exceptions.RequestException as req_poll_e: st.warning(f"Polling status error for '{pdf_info_current.get('filename','Unknown PDF')}': {str(req_poll_e)[:100]}..."); all_processed_in_batch_current_check = False;
                                except Exception as e_poll_generic: st.error(f"Unexpected error polling '{pdf_info_current.get('filename','Unknown PDF')}': {str(e_poll_generic)[:100]}..."); all_processed_in_batch_current_check = False;
                            else: current_pdfs_done_count += 1
                            if st.session_state.processed_pdfs_info.get(pdf_id_to_check, {}).get("status") not in ["COMPLETED", "FAILED"]: all_processed_in_batch_current_check = False
                        all_processed_in_batch = all_processed_in_batch_current_check
                        if all_processed_in_batch: break 
                        overall_progress_percentage = int((current_pdfs_done_count / total_pdfs_in_this_batch) * 100) if total_pdfs_in_this_batch > 0 else 0
                        if hasattr(processing_progress_bar, 'progress'): processing_progress_bar.progress(overall_progress_percentage, text=f"Overall Progress: {overall_progress_percentage}% ({current_pdfs_done_count}/{total_pdfs_in_this_batch} PDFs finalized)")
                        time.sleep(poll_interval_seconds); polling_iterations += 1
                    if not all_processed_in_batch: st.session_state.api_error = "⚠️ Some PDFs may still be processing or timed out status check."; st.warning(st.session_state.api_error)
                    else: st.success("🎉 All selected PDFs have completed processing (or marked as failed).")
                    if 'processing_progress_bar' in locals() and hasattr(processing_progress_bar, 'empty'): processing_progress_bar.empty()
            except requests.exceptions.HTTPError as e_http_upload: err_detail = safe_extract_error_detail(e_http_upload.response.text if e_http_upload.response else str(e_http_upload)); st.session_state.api_error = f"Upload/Processing HTTP Error: {e_http_upload.response.status_code if e_http_upload.response else 'N/A'} - {err_detail}"; st.error(st.session_state.api_error)
            except requests.exceptions.RequestException as e_req_upload: st.session_state.api_error = f"❌ Request Error during upload: {str(e_req_upload)}"; st.error(st.session_state.api_error)
            except Exception as e_gen_upload: st.session_state.api_error = f"❌ An unexpected error occurred during upload setup: {str(e_gen_upload)}"; st.error(st.session_state.api_error)
            finally:
                 if upload_progress_bar is not None and hasattr(upload_progress_bar, 'empty'): upload_progress_bar.empty()

    if st.session_state.current_batch_pdf_ids: 
        st.markdown("---"); st.subheader("Batch Processing Status:")
        for pdf_id_in_batch in st.session_state.current_batch_pdf_ids:
            info = st.session_state.processed_pdfs_info.get(pdf_id_in_batch, {})
            display_name = info.get("filename", f"PDF ID: {pdf_id_in_batch[:8]}...")
            status_emoji = "✅" if info.get("status") == "COMPLETED" else "⏳" if info.get("status") == "PROCESSING" else " PENDING " if info.get("status") == "PENDING" else "❌" if info.get("status") == "FAILED" else "❓"
            st.markdown(f"{status_emoji} **{display_name}**: {info.get('status', 'Unknown')} - *{info.get('message', 'No details')}*")
            if info.get("status") == "COMPLETED" and info.get("processing_time_ms"):
                 processing_time_sec = info.get('processing_time_ms', 0) / 1000.0
                 st.caption(f"Processing time: {processing_time_sec:.2f} seconds. Title: '{info.get('title', 'N/A')}'")
                 
    st.markdown("---"); st.subheader("2. Chat Scope") 
    query_scope_options = ["Query current batch", "Query a specific PDF from current batch"]
    default_radio_index = 0 if st.session_state.current_query_scope == "corpus" else 1
    selected_query_scope_display = st.radio("Choose query scope:", options=query_scope_options, index=default_radio_index, key="query_scope_selector")
    new_query_scope = "corpus" if selected_query_scope_display == "Query current batch" else "single_pdf" 
    if new_query_scope != st.session_state.current_query_scope:
        st.session_state.current_query_scope = new_query_scope; st.session_state.chat_history = []; st.rerun()
    if st.session_state.current_query_scope == "single_pdf": 
        available_pdfs_for_chat = {pid: info for pid, info in st.session_state.processed_pdfs_info.items() if pid in st.session_state.current_batch_pdf_ids and info.get("status") == "COMPLETED"}
        if not available_pdfs_for_chat: st.info("No PDFs from current batch are ready for single-document chat.")
        else:
            display_options_single_pdf = [f"{info.get('filename', pid)} (ID: {pid[:8]}...)" for pid, info in available_pdfs_for_chat.items()] 
            current_single_pdf_index = 0
            if st.session_state.selected_single_pdf_id in available_pdfs_for_chat:
                try: current_single_pdf_index = list(available_pdfs_for_chat.keys()).index(st.session_state.selected_single_pdf_id)
                except ValueError: st.session_state.selected_single_pdf_id = None; current_single_pdf_index = 0
            selected_display_option = st.selectbox("Select a PDF:", options=display_options_single_pdf, index=current_single_pdf_index, key="single_pdf_selector")
            if selected_display_option: 
                actual_selected_pdf_id = None
                for pid_key, info_val in available_pdfs_for_chat.items():
                    if f"{info_val.get('filename', pid_key)} (ID: {pid_key[:8]}...)" == selected_display_option:
                        actual_selected_pdf_id = pid_key; break
                if actual_selected_pdf_id and actual_selected_pdf_id != st.session_state.selected_single_pdf_id:
                    st.session_state.selected_single_pdf_id = actual_selected_pdf_id; st.session_state.chat_history = []; st.rerun()
    
    query_target_display = "Not Ready"; chat_enabled = False
    if st.session_state.current_query_scope == "corpus":
        if st.session_state.current_batch_pdf_ids and any(st.session_state.processed_pdfs_info.get(pid, {}).get("status") == "COMPLETED" for pid in st.session_state.current_batch_pdf_ids):
            query_target_display = f"Current Batch ({len(st.session_state.current_batch_pdf_ids)} PDFs)"; chat_enabled = True; st.success(f"Querying: {query_target_display}")
        else: st.info("Upload and process PDFs to query the current batch.")
    elif st.session_state.current_query_scope == "single_pdf": 
        if st.session_state.selected_single_pdf_id:
            pdf_info = st.session_state.processed_pdfs_info.get(st.session_state.selected_single_pdf_id)
            if pdf_info and pdf_info.get("status") == "COMPLETED":
                query_target_display = pdf_info.get('filename', f"PDF ID {st.session_state.selected_single_pdf_id[:8]}..."); chat_enabled = True; st.success(f"Querying: {query_target_display}")
            else: st.warning("Selected PDF not ready or status unknown.")
        else: st.info("Select a specific processed PDF from the current batch.")

st.header("💬 Chat with your PDF(s)")
if not chat_enabled: 
    st.info("☝️ Please upload PDF(s) & ensure processing is complete to enable chat.")
    if st.session_state.api_error: st.error(f"Last API Error: {st.session_state.api_error}")
else:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    
    if prompt := st.chat_input(f"Ask about {query_target_display}...", key="chat_input_main", disabled=not chat_enabled):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty(); message_placeholder.markdown("Thinking... 🧠")
            try:
                query_payload = {
                    "question": prompt, 
                    "query_scope": st.session_state.current_query_scope,
                    "chat_history": [msg for msg in st.session_state.chat_history[:-1] if isinstance(msg, dict) and "role" in msg and "content" in msg] # Ensure valid format
                }
                
                if st.session_state.current_query_scope == "corpus":
                    ready_corpus_ids = [pid for pid in st.session_state.current_batch_pdf_ids if st.session_state.processed_pdfs_info.get(pid, {}).get("status") == "COMPLETED"]
                    if not ready_corpus_ids: message_placeholder.error("No PDFs in batch are ready for corpus query."); st.stop() 
                    query_payload["current_corpus_pdf_ids"] = ready_corpus_ids
                elif st.session_state.current_query_scope == "single_pdf": 
                    if not st.session_state.selected_single_pdf_id: message_placeholder.error("No PDF selected for single query."); st.stop()
                    query_payload["pdf_id"] = st.session_state.selected_single_pdf_id
                
                # Use the increased timeout here
                query_response = requests.post(f"{FASTAPI_BASE_URL}/query/", json=query_payload, timeout=QUERY_REQUEST_TIMEOUT_SECONDS) 
                query_response.raise_for_status()
                response_data = query_response.json()
                assistant_reply_content = response_data.get("answer", "Sorry, I couldn't formulate a response.")
                
                timing_info_parts = []
                q_time_ms = response_data.get("query_processing_time_ms")
                llm_time_ms = response_data.get("llm_generation_time_ms")
                if q_time_ms is not None: timing_info_parts.append(f"Query processed in {q_time_ms / 1000.0:.2f}s")
                if llm_time_ms is not None: timing_info_parts.append(f"LLM: {llm_time_ms / 1000.0:.2f}s")
                if timing_info_parts: assistant_reply_content += f"\n\n---\n*" + " (".join(timing_info_parts) + ")*"

                message_placeholder.markdown(assistant_reply_content) 
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply_content})
            except requests.exceptions.ReadTimeout:
                timeout_msg = f"The query took too long (more than {QUERY_REQUEST_TIMEOUT_SECONDS / 60.0:.0f} minutes) and timed out. The server might still be processing. Please try a simpler query or check back."
                message_placeholder.error(timeout_msg); st.session_state.chat_history.append({"role": "assistant", "content": timeout_msg})
            except requests.exceptions.HTTPError as e_http_q:
                err_msg_q = f"API Error during query: {e_http_q.response.status_code if e_http_q.response else 'N/A'} - {safe_extract_error_detail(e_http_q.response.text if e_http_q.response else str(e_http_q))}"
                message_placeholder.error(err_msg_q); st.session_state.chat_history.append({"role": "assistant", "content": err_msg_q})
            except requests.exceptions.RequestException as e_req_q:
                err_msg_q_req = f"❌ Request Error during query: {str(e_req_q)}"; message_placeholder.error(err_msg_q_req); st.session_state.chat_history.append({"role": "assistant", "content": err_msg_q_req})
            except Exception as e_gen_q:
                unexp_err_msg_q = f"❌ An unexpected error occurred during query: {str(e_gen_q)}"; message_placeholder.error(unexp_err_msg_q); st.session_state.chat_history.append({"role": "assistant", "content": unexp_err_msg_q})

st.sidebar.markdown("---"); 
st.sidebar.info("VTIERP Custom Engine. Upload PDFs, then chat.")
