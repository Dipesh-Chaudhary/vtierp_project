# vtierp_project_custom/app/services/rag_agent.py
import os
from typing import List, Dict, TypedDict, Optional, Any # Removed Set as it's not directly used in this file's definitions
from collections import defaultdict
import logging # Module-level import for logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.docstore.document import Document # Assuming Document is from langchain.docstore.document
from langgraph.graph import StateGraph, END

from app.dependencies_config.llm_config import get_rag_llm # get_aux_llm is not directly used here
from app.services.vector_store_manager import get_retrievers_for_pdf
from app.services.utils import image_to_base64 # For preparing image payload for LLM
from app.core.config import settings

# Consistent module-level logger
logger = logging.getLogger(__name__) # Use this logger throughout the module


# --- LangGraph State Definition ---
class AdvancedRAGState(TypedDict):
    pdf_id: str
    original_question: str
    transformed_question: str
    retrieved_text_docs: List[Document]
    retrieved_image_desc_docs: List[Document]
    final_text_context: List[Document]
    final_image_context: List[Document]
    images_for_llm_payload: List[Dict[str, Any]] # Base64 images for VLM
    answer: str
    pdf_specific_summary: Optional[str] # Title & Abstract for the current PDF


# --- Helper: Format documents for LLM Context ---
def format_docs_for_llm(docs: List[Document], doc_type_label: str) -> str:
    if not docs:
        return f"No relevant {doc_type_label} context found for this query."

    # Initialize the list to hold formatted parts at the beginning
    formatted_document_parts: List[str] = []

    if doc_type_label == "image descriptions":
        grouped_by_caption_id = defaultdict(list)
        for doc in docs:
            caption_id = doc.metadata.get("caption_id", f"ungrouped_img_desc_{os.path.basename(doc.metadata.get('image_path_on_server','no_path'))}")
            grouped_by_caption_id[caption_id].append(doc)
        
        for cap_id, desc_docs in grouped_by_caption_id.items():
            original_caption_text = desc_docs[0].metadata.get("original_caption", str(cap_id))
            if "ungrouped_img_desc" in str(cap_id) and original_caption_text == str(cap_id):
                original_caption_text = "Ungrouped Visual Element"
            formatted_document_parts.append(f"Visual Element Group: '{original_caption_text[:100]}...' ({len(desc_docs)} VLM description(s) retrieved)")
            for i, single_desc_doc in enumerate(desc_docs):
                img_path_hint = os.path.basename(str(single_desc_doc.metadata.get('image_path_on_server', 'N/A')))
                page_num_hint = single_desc_doc.metadata.get('page_number','N/A')
                formatted_document_parts.append(f"  VLM Description Part {i+1} (Source Hint: {img_path_hint}, Page: {page_num_hint}):\n    {single_desc_doc.page_content}")
    
    else: # For textual documents (text_chunks, summaries, text_table_content etc.)
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            source_file = os.path.basename(str(metadata.get('source_doc_name', 'N/A')))
            pg = metadata.get('page_number','N/A')
            doc_kind_raw = metadata.get('parser_source', metadata.get('type', 'N/A'))
            doc_kind_clean = str(doc_kind_raw).replace('text_', '').replace('_description', '').replace('_summary', '').replace('content', '').replace('pymupdf_meta', 'Metadata').replace('digital','Digital Text').replace('ocr','OCR Text')
            if doc_kind_clean == "chunk": doc_kind_clean = "Text Chunk"
            if doc_kind_clean == "corpus": doc_kind_clean = "Corpus Summary" # Ensure this matches type from pdf_processor
            
            importance_level = metadata.get('importance', '')
            header = f"Context Document {i+1} (Type: {doc_kind_clean}, Source: {source_file}, Page: {pg}{', Importance: '+importance_level if importance_level else ''}, CaptionID: {metadata.get('caption_id', 'N/A')} ):"
            
            content_for_llm = doc.page_content
            if metadata.get("type") == "text_table_content" and metadata.get("table_markdown"):
                content_for_llm += f"\n\n--- Replicated Table Structure (Markdown) for {metadata.get('caption_id', 'this table')} ---\n{metadata['table_markdown']}"
            
            formatted_document_parts.append(f"{header}\n{content_for_llm}\n")
            
    return "\n\n".join(formatted_document_parts)

# --- LangGraph Nodes ---
def query_transform_node(state: AdvancedRAGState) -> AdvancedRAGState:
    logger.info(f"--- RAG Node: Query Transform (PDF: {state.get('pdf_id')}) ---")
    state["transformed_question"] = state["original_question"]
    return state

def retrieve_documents_node(state: AdvancedRAGState) -> AdvancedRAGState:
    logger.info(f"--- RAG Node: Retrieve Documents (PDF: {state.get('pdf_id')}) ---")
    query = state["transformed_question"]
    pdf_id = state["pdf_id"]
    text_retriever, image_desc_retriever = get_retrievers_for_pdf(pdf_id)
    state["retrieved_text_docs"] = text_retriever.invoke(query) if text_retriever else []
    state["retrieved_image_desc_docs"] = image_desc_retriever.invoke(query) if image_desc_retriever else []
    logger.info(f"Retrieved {len(state['retrieved_text_docs'])} text docs, {len(state['retrieved_image_desc_docs'])} img_desc docs.")
    return state

def rerank_and_select_node(state: AdvancedRAGState) -> AdvancedRAGState:
    logger.info(f"--- RAG Node: Rerank and Select (PDF: {state.get('pdf_id')}) ---")
    retrieved_texts = state.get("retrieved_text_docs", [])
    retrieved_image_descs = state.get("retrieved_image_desc_docs", [])
    k_text_final = 7
    k_images_final = 5
    prioritized_texts = []
    critical_content_seen = set()

    # Add pdf_specific_summary (Title/Abstract for current PDF) to context if available
    if state.get("pdf_specific_summary"):
        summary_doc_content = state["pdf_specific_summary"]
        # Check if this exact summary content is already in retrieved_texts from a title/abstract doc
        # This avoids duplicating it if title/abstract extraction already created such docs.
        # However, it's simpler to just ensure it's at the top if provided.
        # For robust deduplication, you'd compare content prefixes.
        # For now, let's ensure it's prioritized if passed in `pdf_specific_summary`.
        if summary_doc_content[:200] not in critical_content_seen:
            summary_doc = Document(
                page_content=summary_doc_content,
                metadata={"source_pdf_id": state["pdf_id"], "source_doc_name": "Summary", "type": "pdf_summary_explicit", "importance": "critical"}
            )
            prioritized_texts.append(summary_doc)
            critical_content_seen.add(summary_doc_content[:200])

    for doc in retrieved_texts:
        is_critical = doc.metadata.get("importance") == "critical"
        content_prefix = doc.page_content[:200]
        if is_critical and content_prefix not in critical_content_seen:
            prioritized_texts.append(doc)
            critical_content_seen.add(content_prefix)
    for doc in retrieved_texts:
        if doc.metadata.get("importance") != "critical":
            content_prefix = doc.page_content[:200]
            if not any(p_doc.page_content[:200] == content_prefix for p_doc in prioritized_texts):
                prioritized_texts.append(doc)

    state["final_text_context"] = prioritized_texts[:k_text_final]
    state["final_image_context"] = retrieved_image_descs[:k_images_final]

    llm_image_payload = []
    for img_desc_doc in state["final_image_context"][:settings.max_images_to_llm_final]:
        image_path = img_desc_doc.metadata.get("image_path_on_server")
        if image_path and os.path.exists(image_path):
            b64_data, mime_type_str = image_to_base64(image_path)
            if b64_data and mime_type_str and mime_type_str.startswith("image/"):
                llm_image_payload.append({
                    "path": image_path,
                    "vlm_description": img_desc_doc.page_content, # This is the VLM description
                    "data": b64_data,
                    "mime_type": mime_type_str
                })
            else:
                logger.warning(f"Skipping image for LLM payload due to invalid b64/mime: Path='{image_path}', Mime='{mime_type_str}', B64_Exists={bool(b64_data)}")
        else:
            logger.warning(f"Image path not found or invalid for LLM payload: '{image_path}'")
            
    state["images_for_llm_payload"] = llm_image_payload
    logger.info(f"Selected {len(state['final_text_context'])} text, {len(state['final_image_context'])} img_desc. Prepared {len(llm_image_payload)} images for LLM.")
    return state

def generate_answer_node(state: AdvancedRAGState) -> AdvancedRAGState:
    logger.info(f"--- RAG Node: Generate Answer (PDF: {state.get('pdf_id')}) ---")
    rag_llm = get_rag_llm()
    if not rag_llm:
        state["answer"] = "RAG LLM not initialized. Cannot generate answer."
        logger.error("RAG LLM not available in generate_answer_node.")
        return state

    question = state["original_question"]
    text_context_str = format_docs_for_llm(state.get("final_text_context", []), "textual") # Corrected label
    image_desc_context_str = format_docs_for_llm(state.get("final_image_context", []), "image descriptions") # Corrected label
    actual_images_payload = state.get("images_for_llm_payload", [])
    
    # Get the PDF-specific summary (Title/Abstract) from the state
    pdf_summary_from_state = state.get("pdf_specific_summary", "No specific PDF summary (Title/Abstract) provided for this query.")

    system_prompt_content = (
        "You are an expert AI research assistant. Your task is to provide comprehensive, accurate, "
        "and well-structured answers to user questions based *only* on the provided documents as context.\n\n"
        "**Context Prioritization:** Prioritize context explicitly labeled 'DOCUMENT TITLE:', 'DOCUMENT ABSTRACT:', or 'PDF SUMMARY:'.\n\n"
        "**Figure/Table Description & Replication:**\n"
        "- When asked to 'describe' a specific figure/table (e.g., 'Figure 1', 'Table 2'):\n"
        "    - Look for context related to its 'CaptionID'.\n"
        "    - For figures (visuals), use their 'VLM Description' if available.\n"
        "    - For textual tables, if a 'Replicated Table Structure (Markdown)' is provided for that CaptionID, use it. Otherwise, describe based on the 'text_table_content'.\n"
        "- When asked to **'replicate' a table**: \n"
        "    - **If a 'Replicated Table Structure (Markdown)' is present in the context for the requested table (check CaptionID), reproduce that Markdown table exactly.**\n"
        "    - If no Markdown structure is available, state that you can describe the table's content based on the text but cannot replicate its exact visual structure.\n"
        "- For visual descriptions from images, explain key components, structure, any text within the figure, and its purpose/information conveyed.\n\n"
        "**General Guidance:**\n"
        "- Maintain a neutral, informative tone.\n"
        "- If information is not found in the provided context, state that clearly rather than hallucinating.\n"
        "- Format your answer clearly with headings or bullet points where appropriate.\n"
        "- Cite page numbers (P:X) or source document names (S:filename) if available in the context metadata. If a caption_id is available for a visual/table, mention it (e.g. Figure 1, Table 2).\n"
        "- You are answering questions about a single PDF document, identified by PDF ID: {pdf_id}. All context provided pertains to this document."
    )
    filled_system_prompt = system_prompt_content.format(pdf_id=state.get('pdf_id', 'UNKNOWN_PDF_ID'))

    human_message_parts = []
    human_message_parts.append({"type": "text", "text": f"--- PDF SUMMARY (Title/Abstract for PDF ID: {state.get('pdf_id', 'N/A')}) ---\n{pdf_summary_from_state}\n--- END PDF SUMMARY ---"})
    human_message_parts.append({"type": "text", "text": f"\n\n--- TEXTUAL & TEXTUAL TABLE CONTEXT ---\n{text_context_str}\n--- END TEXT CONTEXT ---"})
    human_message_parts.append({"type": "text", "text": f"\n\n--- VLM DESCRIPTIONS OF VISUAL ELEMENTS ---\n{image_desc_context_str}\n--- END VLM DESCRIPTIONS ---"})

    if actual_images_payload:
        human_message_parts.append({"type": "text", "text": f"\n\n--- ({len(actual_images_payload)}) ACTUAL IMAGES FOR YOUR REFERENCE ---"})
        logger.info(f"Constructing message with {len(actual_images_payload)} actual images.")
        for i, img_data in enumerate(actual_images_payload):
            vlm_desc_summary = img_data.get('vlm_description', img_data.get('caption_or_vlm_desc', 'N/A'))[:100]
            human_message_parts.append({"type": "text", "text": f"Image {i+1} Ref ({os.path.basename(img_data['path'])}). VLM Desc Summary: '{vlm_desc_summary}...'"})
            
            current_mime_type = img_data.get('mime_type')
            current_b64_data = img_data.get('data')

            logger.info(f"Image {i+1} for LLM: Path: {img_data['path']}")
            logger.info(f"Image {i+1} for LLM: MIME Type: '{current_mime_type}'")
            logger.info(f"Image {i+1} for LLM: Base64 Data Snippet (first 60 chars): '{current_b64_data[:60] if current_b64_data else 'None'}'...")
            
            if not (current_mime_type and current_b64_data and current_mime_type.startswith("image/")):
                logger.error(f"INVALID IMAGE DATA FOR LLM - Image {i+1} ({img_data['path']}): MIME='{current_mime_type}', B64_Exists={bool(current_b64_data)}. THIS IMAGE WILL LIKELY CAUSE AN ERROR.")
            
            image_url_for_llm = f"data:{current_mime_type};base64,{current_b64_data}"
            logger.info(f"Image {i+1} for LLM: Constructed URL (first 100 chars): '{image_url_for_llm[:100]}'...")

            human_message_parts.append({"type": "image_url", "image_url": {"url": image_url_for_llm}})
        human_message_parts.append({"type": "text", "text": "--- END ACTUAL IMAGES ---"})
    else:
        human_message_parts.append({"type": "text", "text": "\n\nNo actual images provided directly to the model for this query."})
    
    human_message_parts.append({"type": "text", "text": f"\n\n--- USER QUESTION ---\n{question}\n\n--- ANSWER BASED *ONLY* ON THE PROVIDED CONTEXT ---"})
    
    messages = [SystemMessage(content=filled_system_prompt), HumanMessage(content=human_message_parts)]
    
    final_answer = "Error generating answer (LLM not available or failed)."
    try:
        # logger.debug(f"Messages being sent to LLM: {messages}") # Can be very verbose
        response = rag_llm.invoke(messages)
        final_answer = response.content
    except ValueError as ve:
        logger.error(f"ValueError during LLM invocation for PDF {state.get('pdf_id', 'N/A')}, Q: '{question[:50]}...': {ve}", exc_info=True)
        final_answer = f"LLM Input Error: {ve}. One of the image data URIs was malformed. Check server logs for MIME Type and Base64 Data Snippet logs."
    except Exception as e:
        final_answer = f"LLM Error during answer generation: {e}"
        logger.error(f"LLM error for PDF {state.get('pdf_id', 'N/A')}, Q: '{question[:50]}...': {e}", exc_info=True)
        
    state["answer"] = final_answer
    return state

# --- Compile LangGraph Agent ---
_compiled_rag_agent_graph = None

def get_compiled_rag_agent():
    global _compiled_rag_agent_graph
    if _compiled_rag_agent_graph is None:
        graph_builder = StateGraph(AdvancedRAGState)
        graph_builder.add_node("query_transform", query_transform_node)
        graph_builder.add_node("retrieve_documents", retrieve_documents_node)
        graph_builder.add_node("rerank_and_select", rerank_and_select_node)
        graph_builder.add_node("generate_answer", generate_answer_node)

        graph_builder.set_entry_point("query_transform")
        graph_builder.add_edge("query_transform", "retrieve_documents")
        graph_builder.add_edge("retrieve_documents", "rerank_and_select")
        graph_builder.add_edge("rerank_and_select", "generate_answer")
        graph_builder.add_edge("generate_answer", END)

        _compiled_rag_agent_graph = graph_builder.compile()
        logger.info("Custom RAG Agent graph compiled successfully.") # Use module logger
    return _compiled_rag_agent_graph

def run_rag_query(pdf_id: str, question: str, pdf_summary_for_llm: str) -> Dict:
    agent_graph = get_compiled_rag_agent()
    
    # Initialize state dictionary robustly
    initial_state = {}
    try:
        for key_name, type_hint_ann in AdvancedRAGState.__annotations__.items():
            origin = getattr(type_hint_ann, '__origin__', None)
            args = getattr(type_hint_ann, '__args__', ())
            if origin is list or str(type_hint_ann).startswith("typing.List") or str(type_hint_ann).startswith("List["):
                initial_state[key_name] = []
            elif origin is dict or str(type_hint_ann).startswith("typing.Dict") or str(type_hint_ann).startswith("Dict["):
                initial_state[key_name] = {}
            elif origin is Optional or (hasattr(type_hint_ann, '__args__') and type(None) in args):
                initial_state[key_name] = None
            elif type_hint_ann is str:
                initial_state[key_name] = ""
            else: 
                initial_state[key_name] = None 
    except NameError:
        logger.critical("AdvancedRAGState not defined during initial_state creation in run_rag_query.")
        raise # Re-raise as this is a fundamental issue

    initial_state.update({
        "pdf_id": pdf_id,
        "original_question": question,
        "pdf_specific_summary": pdf_summary_for_llm,
        "transformed_question": question, # Default, can be overwritten
        # Other fields will default to their empty/None values from above
    })
    
    final_state_result = None
    try:
        final_state_result = agent_graph.invoke(initial_state, {"recursion_limit": 15})
    except Exception as e:
        logger.error(f"LangGraph execution error for PDF {pdf_id}, Q: '{question[:50]}...': {e}", exc_info=True)
        return {
            "answer": "Error during RAG agent execution. Check server logs for detailed LangGraph error.",
            "retrieved_text_context_sample": [],
            "retrieved_image_context_sample": []
        }
        
    # Ensure final_state_result is not None before accessing keys
    answer = "No answer generated or agent execution failed before answer."
    text_context = []
    image_context = []

    if final_state_result:
        answer = final_state_result.get("answer", answer)
        text_context = final_state_result.get("final_text_context", [])
        image_context = final_state_result.get("final_image_context", [])

    return {
        "answer": answer,
        "final_text_context": text_context, # Return full context for API to sample if needed
        "final_image_context": image_context
    }