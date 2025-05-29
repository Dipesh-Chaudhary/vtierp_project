import streamlit as st
import requests
import time
import os
import json # For safely parsing JSON from error messages

# --- Configuration ---
# FASTAPI_URL is expected to be set as an environment variable,
# especially when running in Docker Compose.
FASTAPI_BASE_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

st.set_page_config(page_title="VTIERP Custom", layout="wide", initial_sidebar_state="expanded")

# --- Helper Functions ---
def safe_extract_error_detail(response_text: str) -> str:
    """Safely extracts 'detail' from a FastAPI JSON error response."""
    try:
        error_data = json.loads(response_text)
        if isinstance(error_data, dict) and "detail" in error_data:
            return str(error_data["detail"])
    except json.JSONDecodeError:
        pass # Not a JSON response
    return response_text # Fallback to full text

# --- Session State Initialization ---
if "current_pdf_id" not in st.session_state:
    st.session_state.current_pdf_id = None
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None
if "processing_status_message" not in st.session_state:
    st.session_state.processing_status_message = ""
if "pdf_is_ready" not in st.session_state:
    st.session_state.pdf_is_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # List of {"role": "user/assistant", "content": "...", "images": []}
if "api_error" not in st.session_state:
    st.session_state.api_error = None

# --- Page Layout ---
st.title("🔬 VTIERP - Custom PDF Analysis Engine")
st.markdown("""
Welcome! Upload a research PDF, wait for it to be processed, and then ask questions about its content.
The system uses AI to understand both text and visual elements (via their descriptions).
""")

# --- Sidebar for PDF Upload and Status ---
with st.sidebar:
    st.header("📄 PDF Management")

    uploaded_file = st.file_uploader("1. Upload your Research PDF", type="pdf", key="pdf_uploader")

    if uploaded_file is not None:
        if st.button("Process PDF", key="process_button", type="primary"):
            st.session_state.current_pdf_id = None
            st.session_state.pdf_is_ready = False
            st.session_state.chat_history = []
            st.session_state.api_error = None
            st.session_state.current_filename = uploaded_file.name
            st.session_state.processing_status_message = f"Uploading '{uploaded_file.name}'..."

            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            progress_bar = st.progress(0, text=st.session_state.processing_status_message)

            try:
                upload_url = f"{FASTAPI_BASE_URL}/upload-pdf/"
                response = requests.post(upload_url, files=files, timeout=30) # Increased timeout for upload
                progress_bar.progress(10, text="Upload complete. Initiating processing...")

                if response.status_code == 200:
                    data = response.json()
                    st.session_state.current_pdf_id = data["pdf_id"]
                    status_check_url = f"{FASTAPI_BASE_URL}{data['status_check_url']}" # Use relative URL from API
                    st.session_state.processing_status_message = f"PDF '{data['filename']}' received (ID: {st.session_state.current_pdf_id}). Processing..."
                    st.info(st.session_state.processing_status_message)

                    # Polling for status
                    polling_attempts = 0
                    max_polling_attempts = 120  # Poll for up to 10 minutes (120 * 5s)
                    poll_interval = 5 # seconds

                    while polling_attempts < max_polling_attempts:
                        progress_value = int(20 + (polling_attempts / max_polling_attempts) * 70)
                        progress_bar.progress(progress_value, text=f"Processing... Attempt {polling_attempts + 1}")

                        status_response = requests.get(status_check_url, timeout=10)
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            current_status = status_data.get("status", "UNKNOWN").upper()
                            st.session_state.processing_status_message = status_data.get("message", "Checking status...")
                            st.info(f"Status: {current_status} - {st.session_state.processing_status_message}")


                            if current_status == "COMPLETED":
                                st.session_state.pdf_is_ready = True
                                st.session_state.processing_status_message = f"✅ PDF '{st.session_state.current_filename}' (ID: {st.session_state.current_pdf_id}) processed successfully!"
                                st.success(st.session_state.processing_status_message)
                                progress_bar.progress(100, text="Processing Complete!")
                                # Store title if available
                                if status_data.get("title") and status_data.get("title") != "N/A":
                                    st.session_state.current_filename = status_data.get("title") # Update filename to title
                                break
                            elif current_status == "FAILED":
                                st.session_state.processing_status_message = f"❌ Processing FAILED for '{st.session_state.current_filename}'. Reason: {status_data.get('message', 'Unknown error')}"
                                st.error(st.session_state.processing_status_message)
                                progress_bar.progress(100, text="Processing Failed.")
                                break
                        else:
                            st.warning(f"Could not get processing status (HTTP {status_response.status_code}). Retrying...")

                        time.sleep(poll_interval)
                        polling_attempts += 1

                    if not st.session_state.pdf_is_ready and polling_attempts >= max_polling_attempts:
                        st.session_state.processing_status_message = "⚠️ PDF processing timed out. Please check the API or try again."
                        st.error(st.session_state.processing_status_message)
                        progress_bar.progress(100, text="Processing Timed Out.")
                else:
                    st.session_state.api_error = f"Upload failed: {response.status_code} - {safe_extract_error_detail(response.text)}"
                    st.error(st.session_state.api_error)
                    progress_bar.empty()

            except requests.exceptions.ConnectionError:
                st.session_state.api_error = f"❌ Connection Error: Could not connect to the API at {FASTAPI_BASE_URL}. Is the backend running?"
                st.error(st.session_state.api_error)
                if 'progress_bar' in locals(): progress_bar.empty()
            except requests.exceptions.Timeout:
                st.session_state.api_error = "❌ Timeout: The request to the API timed out."
                st.error(st.session_state.api_error)
                if 'progress_bar' in locals(): progress_bar.empty()
            except Exception as e:
                st.session_state.api_error = f"❌ An unexpected error occurred: {str(e)}"
                st.error(st.session_state.api_error)
                if 'progress_bar' in locals(): progress_bar.empty()


    st.markdown("---")
    if st.session_state.current_pdf_id and st.session_state.pdf_is_ready:
        st.success(f"**Ready to Chat About:**\n'{st.session_state.current_filename}'\n(ID: `{st.session_state.current_pdf_id}`)")
    elif st.session_state.current_pdf_id:
        st.info(f"**Current PDF:**\n'{st.session_state.current_filename}'\n(ID: `{st.session_state.current_pdf_id}`)\nStatus: {st.session_state.processing_status_message}")
    elif st.session_state.api_error:
         st.error(st.session_state.api_error)


# --- Main Chat Interface Area ---
st.header("💬 Chat with your PDF")

if not st.session_state.current_pdf_id or not st.session_state.pdf_is_ready:
    st.info("☝️ Please upload and process a PDF using the sidebar to begin chatting.")
else:
    # Display chat messages from history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "images" in message and message["images"]:
                # Display retrieved images (if any) associated with assistant's response
                # Assuming 'images' is a list of image contexts from the API
                # Each image context should have 'image_path_relative_to_pdf_data' and 'description'
                num_images = len(message["images"])
                if num_images > 0:
                    cols = st.columns(min(num_images, 3)) # Max 3 images per row
                    for i, img_data in enumerate(message["images"]):
                        # Construct full URL: FASTAPI_BASE_URL/static_data/<pdf_id>/extracted_images/filename.png
                        # The 'image_path_relative_to_pdf_data' should be like: '<pdf_id>/extracted_images/actual_image.png'
                        # OR it might just be 'extracted_images/actual_image.png' if the API is already prepending pdf_id to the StaticFiles path
                        # From main.py: StaticFiles(directory=get_vector_store_base_dir()), which is /app/data/vector_stores
                        # So the relative path is <pdf_id>/extracted_images/filename.png
                        img_relative_path = img_data.get("metadata", {}).get("image_path_relative_to_pdf_data")

                        if img_relative_path:
                            # Construct full image URL:
                            # FASTAPI_BASE_URL is http://api:8000 or http://localhost:8000
                            # Static mount is /static_data
                            # img_relative_path is <pdf_id>/extracted_images/....
                            # So, the metadata from API for image_path_relative_to_pdf_data should include the pdf_id part.
                            # Example: pdf_id_123/extracted_images/page1_fig1.png
                            full_image_url = f"{FASTAPI_BASE_URL}/static_data/{img_relative_path}"

                            with cols[i % 3]: # Cycle through columns
                                st.image(full_image_url, caption=f"Ref: {img_data.get('page_content', 'Visual Context')[:100]}...", use_column_width='auto')
                        else:
                             with cols[i % 3]:
                                st.caption(f"Context: {img_data.get('page_content', 'Visual Context')[:150]}... (Image path missing)")


    # User input
    if prompt := st.chat_input(f"Ask about '{st.session_state.current_filename}'...", key="chat_input_main"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call API for assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking... 🧠")
            try:
                query_url = f"{FASTAPI_BASE_URL}/query/"
                payload = {"pdf_id": st.session_state.current_pdf_id, "question": prompt}
                query_response = requests.post(query_url, json=payload, timeout=120) # Long timeout for RAG

                if query_response.status_code == 200:
                    response_data = query_response.json()
                    assistant_reply = response_data.get("answer", "Sorry, I couldn't formulate a response.")
                    retrieved_images_context = response_data.get("retrieved_image_context_sample", [])

                    message_placeholder.markdown(assistant_reply)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": assistant_reply,
                        "images": retrieved_images_context # Store for re-display
                    })

                    # Display images for the current response if any
                    if retrieved_images_context:
                        num_images = len(retrieved_images_context)
                        if num_images > 0:
                            st.markdown("--- \n**Relevant Visual Context:**")
                            cols = st.columns(min(num_images, 3))
                            for i, img_data in enumerate(retrieved_images_context):
                                img_relative_path = img_data.get("metadata", {}).get("image_path_relative_to_pdf_data")
                                if img_relative_path:
                                    full_image_url = f"{FASTAPI_BASE_URL}/static_data/{img_relative_path}"
                                    with cols[i % 3]:
                                        st.image(full_image_url, caption=f"Ref: {img_data.get('page_content', 'Visual Context')[:100]}...", use_column_width='auto')
                                else:
                                    with cols[i % 3]:
                                        st.caption(f"Context: {img_data.get('page_content', 'Visual Context')[:150]}... (Image path missing)")


                else:
                    error_detail = safe_extract_error_detail(query_response.text)
                    err_msg = f"Sorry, an error occurred: {query_response.status_code} - {error_detail}"
                    message_placeholder.error(err_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": err_msg})

            except requests.exceptions.ConnectionError:
                conn_err_msg = f"❌ Connection Error: Could not connect to the API at {FASTAPI_BASE_URL}."
                message_placeholder.error(conn_err_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": conn_err_msg})
            except requests.exceptions.Timeout:
                timeout_err_msg = "❌ Timeout: The query to the API timed out."
                message_placeholder.error(timeout_err_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": timeout_err_msg})
            except Exception as e:
                unexp_err_msg = f"❌ An unexpected error occurred: {str(e)}"
                message_placeholder.error(unexp_err_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": unexp_err_msg})

st.sidebar.markdown("---")
st.sidebar.info("VTIERP Custom Engine. Powered by Generative AI.")