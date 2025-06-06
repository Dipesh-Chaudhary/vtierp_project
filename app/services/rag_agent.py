import os
import time
from typing import List, Dict, TypedDict, Optional, Any 
import logging
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage 
from langchain.docstore.document import Document
from langgraph.graph import StateGraph, END

from app.dependencies_config.llm_config import get_rag_llm
from app.services.vector_store_manager import get_retrievers_for_pdf
from app.services.utils import clean_parsed_text, image_to_base64
from app.core.config import settings

logger = logging.getLogger(__name__)

class AdvancedRAGState(TypedDict): # No change
    query_scope: str; pdf_id: Optional[str]; current_corpus_pdf_ids: Optional[List[str]]
    question: str; original_question: str; transformed_question: str
    retrieved_text_docs: List[Document]; retrieved_image_desc_docs: List[Document]
    final_text_context: List[Document]; final_image_context: List[Document]
    images_for_llm_payload: List[Dict[str, Any]]; answer: str
    summary_for_context: Optional[str]; chat_history: Optional[List[Dict[str,str]]] 
    llm_generation_time_ms: Optional[float]; total_rag_time_ms: Optional[float]

def format_docs_for_llm(docs: List[Document], doc_type_label: str, query_scope: str) -> str: # No change needed from prev good version
    # ... (same as your previous correct version)
    if not docs: return f"No relevant {doc_type_label} context found for this query."
    formatted_document_parts: List[str] = []
    is_corpus_query_context = (query_scope == "corpus")
    if doc_type_label == "image descriptions":
        grouped_by_caption_id = defaultdict(list)
        for doc in docs:
            caption_id = doc.metadata.get("caption_id", f"ungrouped_vlm_desc_{os.path.basename(doc.metadata.get('image_path_on_server','no_path'))}")
            grouped_by_caption_id[caption_id].append(doc)
        for cap_id, desc_docs in grouped_by_caption_id.items():
            original_caption_text = desc_docs[0].metadata.get("original_caption", str(cap_id))
            if "ungrouped_vlm_desc" in str(cap_id) and original_caption_text == str(cap_id) : original_caption_text = "Ungrouped Visual Element"
            source_doc_name_for_group = desc_docs[0].metadata.get('source_doc_name', 'UnknownSource')
            element_subtype = desc_docs[0].metadata.get('element_subtype', 'visual element')
            group_type_indicator = " (VLM desc of Table Image)" if "table_image_vlm_description" in element_subtype else " (VLM desc of Figure/Visual)"
            header_citation = f" (From Doc: '{source_doc_name_for_group}')" if is_corpus_query_context else ""
            formatted_document_parts.append(f"Visual Element Group: '{original_caption_text[:100]}' {group_type_indicator}{header_citation} ({len(desc_docs)} VLM description(s) retrieved):")
            for i, single_desc_doc in enumerate(desc_docs):
                page_num_hint = single_desc_doc.metadata.get('page_number','N/A')
                formatted_document_parts.append(f"  VLM Description Part {i+1} (Page: {page_num_hint}):\n    {single_desc_doc.page_content}")
    else: 
        for i, doc in enumerate(docs):
            metadata = doc.metadata; source_file = os.path.basename(str(metadata.get('source_doc_name', 'N/A'))); pg = metadata.get('page_number','N/A')
            doc_kind_raw = metadata.get('type', 'N/A'); 
            doc_kind_clean = str(doc_kind_raw).replace('text_', '').replace('_summary', '').replace('_content', '').replace('description', ' desc')
            if doc_kind_clean == "chunk": doc_kind_clean = "Text Chunk"
            elif doc_kind_clean == "title_summary": doc_kind_clean = "Document Title"
            elif doc_kind_clean == "abstract_summary": doc_kind_clean = "Document Abstract"
            elif doc_kind_clean == "table_content": doc_kind_clean = "Table Textual Content"
            elif doc_kind_clean == "figure_desc": doc_kind_clean = "Figure Textual Desc"
            if metadata.get("type") == "corpus_summary" or "Corpus_Overview" in str(metadata.get("source_doc_name")):
                 doc_kind_clean = "Overall Corpus Summary"; citation_string = "" 
            elif is_corpus_query_context:
                citation_string = f" (From Doc: '{source_file}'" + (f", P: {pg}" if pg else "") + ")"
            else: citation_string = f" (P: {pg})" if pg else ""
            importance_level = metadata.get('importance', ''); identity_hint = f", ID: {metadata.get('caption_id')}" if metadata.get('caption_id') else ""
            header = f"Context Doc {i+1} (Type: {doc_kind_clean}{', Imp: '+importance_level if importance_level else ''}{identity_hint}){citation_string}:"
            content_for_llm = doc.page_content
            if metadata.get("type") == "text_table_content" and metadata.get("table_markdown"):
                content_for_llm += f"\n\n--- Replicated Table Structure (Markdown) for {metadata.get('caption_id', 'this table')} ---\n{metadata['table_markdown']}"
            formatted_document_parts.append(f"{header}\n{content_for_llm}\n")
    return "\n\n".join(formatted_document_parts)


def query_transform_node(state: AdvancedRAGState) -> AdvancedRAGState: # No change
    # ... (same as before)
    logger.debug(f"--- RAG Node: Query Transform (Scope: {state.get('query_scope')}, PDF: {state.get('pdf_id')}, Q: '{state.get('question')[:50]}...') ---")
    state["original_question"] = state["question"] 
    state["transformed_question"] = state["original_question"] 
    return state

def retrieve_documents_node(state: AdvancedRAGState) -> AdvancedRAGState: # No change
    # ... (same as before)
    logger.debug(f"--- RAG Node: Retrieve Documents (Scope: {state.get('query_scope')}, PDF: {state.get('pdf_id')}, Transformed Q: '{state.get('transformed_question')[:50]}...') ---")
    query = state["transformed_question"]; all_retrieved_texts: List[Document] = []; all_retrieved_images: List[Document] = []
    pdf_ids_to_query: List[str] = []
    if state.get("query_scope") == "corpus":
        pdf_ids_to_query = state.get("current_corpus_pdf_ids", [])
        if not pdf_ids_to_query: logger.warning("Corpus query: no current_corpus_pdf_ids. No retrieval.")
        else: logger.info(f"Corpus retrieval for {len(pdf_ids_to_query)} PDFs: {pdf_ids_to_query}")
    elif state.get("query_scope") == "single":
        single_pdf_id = state.get("pdf_id")
        if single_pdf_id and single_pdf_id != "current_corpus": pdf_ids_to_query = [single_pdf_id]; logger.info(f"Single PDF retrieval for '{single_pdf_id}'.")
        else: logger.warning(f"Single query: invalid pdf_id: '{single_pdf_id}'. No retrieval.")
    else: logger.error(f"Invalid query_scope ('{state.get('query_scope')}') for retrieval."); state["retrieved_text_docs"] = []; state["retrieved_image_desc_docs"] = []; return state
    for pdf_id_item in pdf_ids_to_query:
        text_retriever, image_desc_retriever = get_retrievers_for_pdf(pdf_id_item)
        if text_retriever:
            try: retrieved = text_retriever.invoke(query); [doc.metadata.update({"source_pdf_id_retrieved_from": pdf_id_item, "source_doc_name": doc.metadata.get("source_doc_name", pdf_id_item)}) for doc in retrieved]; all_retrieved_texts.extend(retrieved) 
            except Exception as e_retr_txt: logger.error(f"Error text retriever PDF {pdf_id_item}: {e_retr_txt}")
        if image_desc_retriever:
            try: retrieved_imgs = image_desc_retriever.invoke(query); [doc.metadata.update({"source_pdf_id_retrieved_from": pdf_id_item, "source_doc_name": doc.metadata.get("source_doc_name", pdf_id_item)}) for doc in retrieved_imgs]; all_retrieved_images.extend(retrieved_imgs)
            except Exception as e_retr_img: logger.error(f"Error image desc retriever PDF {pdf_id_item}: {e_retr_img}")
    logger.info(f"Total retrieved before selection: {len(all_retrieved_texts)} text docs, {len(all_retrieved_images)} image_desc docs.")
    state["retrieved_text_docs"] = all_retrieved_texts; state["retrieved_image_desc_docs"] = all_retrieved_images
    return state

def rerank_and_select_node(state: AdvancedRAGState) -> AdvancedRAGState: # No change
    # ... (same as before)
    logger.debug(f"--- RAG Node: Rerank and Select (Scope: {state.get('query_scope')}, PDF: {state.get('pdf_id')}) ---")
    retrieved_texts = state.get("retrieved_text_docs", []); retrieved_image_descs = state.get("retrieved_image_desc_docs", [])
    k_text_final = 7; k_images_final = 5; prioritized_texts: List[Document] = []; critical_content_identifiers_seen = set()
    if state.get("summary_for_context"):
        summary_content = state["summary_for_context"]; summary_source_name = "Single_PDF_Overview"; summary_type = "pdf_meta_summary"
        if state.get("query_scope") == "corpus": summary_source_name = "Current_Batch_Corpus_Overview"; summary_type = "corpus_summary"
        summary_doc = Document(page_content=summary_content, metadata={"source_doc_name": summary_source_name, "type": summary_type, "importance": "critical"})
        prioritized_texts.append(summary_doc); critical_content_identifiers_seen.add(summary_content[:200])
    for doc_type_list in [retrieved_texts]: 
        for doc in doc_type_list:
            is_critical = doc.metadata.get("importance") == "critical"
            content_id = (doc.metadata.get("source_pdf_id_retrieved_from", "unknown"), doc.metadata.get("page_number", "N/A"), doc.page_content[:200])
            if is_critical and content_id not in critical_content_identifiers_seen: prioritized_texts.append(doc); critical_content_identifiers_seen.add(content_id)
    for doc in retrieved_texts: 
        if doc.metadata.get("importance") != "critical":
            content_id = (doc.metadata.get("source_pdf_id_retrieved_from", "unknown"), doc.metadata.get("page_number", "N/A"), doc.page_content[:200])
            if not any(pid == content_id for pid in critical_content_identifiers_seen) and not any(p_doc.page_content[:200] == doc.page_content[:200] and p_doc.metadata.get("source_pdf_id_retrieved_from") == doc.metadata.get("source_pdf_id_retrieved_from") for p_doc in prioritized_texts):
                 prioritized_texts.append(doc)
    state["final_text_context"] = prioritized_texts[:k_text_final]; state["final_image_context"] = retrieved_image_descs[:k_images_final]
    llm_image_payload: List[Dict[str, Any]] = []; processed_image_paths_for_payload = set()
    for img_desc_doc in state["final_image_context"][:settings.max_images_to_llm_final]:
        image_path_on_server = img_desc_doc.metadata.get("image_path_on_server")
        if image_path_on_server and os.path.exists(image_path_on_server) and image_path_on_server not in processed_image_paths_for_payload:
            b64_data, mime_type_str = image_to_base64(image_path_on_server)
            if b64_data and mime_type_str and mime_type_str.startswith("image/"):
                llm_image_payload.append({"path": image_path_on_server, "vlm_description": img_desc_doc.page_content, "original_caption": img_desc_doc.metadata.get("original_caption", "N/A"), "caption_id": img_desc_doc.metadata.get("caption_id", "N/A"), "page_number": img_desc_doc.metadata.get("page_number", "N/A"), "source_doc_name": img_desc_doc.metadata.get("source_doc_name", "N/A"), "data": b64_data, "mime_type": mime_type_str})
                processed_image_paths_for_payload.add(image_path_on_server)
            else: logger.warning(f"Skipping image for LLM: invalid b64/mime. Path='{image_path_on_server}'")
        elif image_path_on_server in processed_image_paths_for_payload: logger.debug(f"Image {image_path_on_server} already in payload.")
        elif not image_path_on_server or not os.path.exists(image_path_on_server): logger.warning(f"Image path invalid for LLM: '{image_path_on_server}' from metadata: {img_desc_doc.metadata}")
    state["images_for_llm_payload"] = llm_image_payload
    logger.info(f"Rerank/Select: {len(state['final_text_context'])} text, {len(state['final_image_context'])} img_desc. {len(llm_image_payload)} unique images for LLM.")
    return state

# --- generate_answer_node (System Prompt Heavily Revised) ---
def generate_answer_node(state: AdvancedRAGState) -> AdvancedRAGState:
    logger.debug(f"--- RAG Node: Generate Answer (Scope: {state.get('query_scope')}, PDF: {state.get('pdf_id')}, Q: '{state.get('original_question')[:50]}...') ---")
    rag_llm = get_rag_llm()
    if not rag_llm: state["answer"] = "Error: RAG LLM not initialized."; state["llm_generation_time_ms"] = 0.0; return state
    
    question = state.get("original_question", "No question provided.")
    query_scope_for_formatting = state.get("query_scope", "single")
    chat_history_from_state = state.get("chat_history", []) 
    
    text_context_str = format_docs_for_llm(state.get("final_text_context", []), "textual", query_scope=query_scope_for_formatting)
    image_desc_context_str = format_docs_for_llm(state.get("final_image_context", []), "image descriptions", query_scope=query_scope_for_formatting)
    actual_images_payload = state.get("images_for_llm_payload", [])
    summary_for_prompt = state.get("summary_for_context", "No overall summary context was provided for this query.")

    system_prompt_content = (
        "You are an expert AI research assistant. Your primary goal is to answer user questions accurately and comprehensively, "
        "basing your answers *exclusively* on the provided contextual information. This context includes: PREVIOUS CONVERSATION, "
        "CONTEXT OVERVIEW (document title/abstract or batch summary), TEXTUAL & TABLE CONTEXT, and IMAGE DESCRIPTIONS.\n\n"
        "**Core Instructions:**\n"
        "1.  **Chat History First (CRITICAL):** If '--- PREVIOUS CONVERSATION ---' is provided, **you MUST review it carefully before addressing the 'CURRENT USER QUESTION'.** Your answer needs to be directly relevant to the ongoing dialogue. If the current question is a follow-up (e.g., 'Are you sure?', 'Tell me more about X'), address it in relation to the prior turn.\n"
        "2.  **Answer ONLY from Provided Context:** After considering chat history, use only information from CONTEXT OVERVIEW, TEXTUAL & TABLE CONTEXT, and IMAGE DESCRIPTIONS. If information isn't present, state that clearly.\n"
        "3.  **Document Specificity (VERY IMPORTANT for Corpus Queries & Specific Element Requests):**\n"
        "    *   If the user's query (or implied by chat history) specifies a particular document (e.g., 'AAG.pdf', 'Table 1 from AAG paper'), you MUST prioritize context with a matching 'From Doc: <filename>' in its citation header.\n"
        "    *   For corpus queries, CITE sources: '(From Document: <filename>, Page: <page_number>)' or '(From Document: <filename>)' if page is N/A.\n"
        "    *   For single PDF queries, cite page: '(Page: <page_number>)'.\n"
        "4.  **Definitions & Acronyms (e.g., 'full form of AAG'):**\n"
        "    *   **PRIORITIZE the 'CONTEXT OVERVIEW' section.** This usually contains the document title and abstract. Also check context documents labeled 'Document Title' or 'Document Abstract'.\n"
        "    *   **Example:** Query 'What is AAG?'. If CONTEXT OVERVIEW for 'AAG.pdf' includes 'Analogy-Augmented Generation (AAG) ...', answer 'Based on the overview of 'AAG.pdf', AAG stands for Analogy-Augmented Generation.' Cite appropriately.\n"
        "5.  **Table Replication & Information (NEW STRATEGY):**\n"
        "    *   When asked to 'replicate table X' or get info from 'table X' (e.g., 'Table 1 from AAG.pdf'):\n"
        "        a.  **Locate Context:** Find the 'Table Textual Content' document with 'ID: TableX' and matching 'From Doc: AAG.pdf' in its header.\n"
        "        b.  **Check for Markdown in Metadata:** If this document's context has a section like '--- Replicated Table Structure (Markdown) for TableX ---', YOU MUST use that Markdown directly to present the table. Format it as a Markdown table.\n"
        "        c.  **Use Extracted Raw Text:** If no such Markdown is found in that specific table's context, use the '--- Extracted Table Text ---' provided for that table. Present this raw text, trying to maintain its row structure as best as possible. State that you are presenting the extracted text.\n"
        "        d.  **Use VLM of Table Image:** If an 'IMAGE DESCRIPTION' for this table (labeled 'VLM desc of Table Image' for 'TableX') is also available, use its VLM description to understand the table's visual structure, content, and relationships, especially if the extracted text is sparse, unclear, or missing. You can synthesize information from both the text and the VLM description if both are informative. Clearly state if you are using information from an image description of the table.\n"
        "        e.  **If No Information:** If no specific context for 'Table X from AAG.pdf' is found (neither textual nor image description), state 'I could not find specific information for Table X from AAG.pdf in the provided context.' DO NOT invent table data or use a table from a different document unless explicitly asked.\n"
        "6.  **Figures:** Refer to figures by their ID (e.g., 'Figure 1 shows...'). Use their VLM descriptions from the 'IMAGE DESCRIPTIONS' section.\n"
        "7.  **Clarity & Conciseness:** Be clear, well-structured. Use bullet points if appropriate.\n"
        "8.  **Attached Images (Side Channel):** These are for your internal reference only. Rely on VLM descriptions primarily.\n\n"
        "**Query Scope Context Reminder:**\n" # No change to this section
        f"- Current query scope: '{query_scope_for_formatting}'.\n"
        f"- If 'single', all context pertains to one PDF (ID: {state.get('pdf_id', 'N/A')}).\n"
        f"- If 'corpus', context may span multiple PDFs (IDs: {state.get('current_corpus_pdf_ids', 'N/A')}). Document attribution is critical."
    )
    # ... (Human message construction - same as your last correct version) ...
    messages: List[SystemMessage | HumanMessage | AIMessage] = [SystemMessage(content=system_prompt_content)]
    if chat_history_from_state:
        logger.debug(f"Adding {len(chat_history_from_state)} turns from chat history to LLM prompt.")
        messages.append(HumanMessage(content="--- PREVIOUS CONVERSATION ---")) 
        for turn in chat_history_from_state:
            if turn.get("role") == "user": messages.append(HumanMessage(content=turn.get("content", "")))
            elif turn.get("role") == "assistant": messages.append(AIMessage(content=turn.get("content", ""))) 
        messages.append(HumanMessage(content="--- END PREVIOUS CONVERSATION ---"))
    current_human_message_parts: List[Dict[str, Any]] = [{"type": "text", "text": f"\n\n--- CONTEXT FOR CURRENT QUESTION ---"}] 
    current_human_message_parts.append({"type": "text", "text": f"\nCONTEXT OVERVIEW:\n{summary_for_prompt}\n--- END CONTEXT OVERVIEW ---"})
    current_human_message_parts.append({"type": "text", "text": f"\n\nTEXTUAL & TABLE CONTEXT:\n{text_context_str}\n--- END TEXTUAL & TABLE CONTEXT ---"})
    current_human_message_parts.append({"type": "text", "text": f"\n\nIMAGE DESCRIPTIONS (VLM-Generated):\n{image_desc_context_str}\n--- END IMAGE DESCRIPTIONS ---"})
    if actual_images_payload:
        current_human_message_parts.append({"type": "text", "text": f"\n\n--- ({len(actual_images_payload)}) ATTACHED IMAGES (Reference Only) ---"})
        for i, img_data in enumerate(actual_images_payload):
            img_citation = f"(From Doc: '{img_data.get('source_doc_name', 'N/A')}', P: {img_data.get('page_number', 'N/A')}, ID: {img_data.get('caption_id', 'N/A')})" if query_scope_for_formatting == "corpus" else f"(P: {img_data.get('page_number', 'N/A')}, ID: {img_data.get('caption_id', 'N/A')})"
            current_human_message_parts.append({"type": "text", "text": f"Attached Image {i+1} Ref: {img_data.get('original_caption', 'Image')} {img_citation}. VLM Desc Summary: '{img_data.get('vlm_description','N/A')[:70]}...'"})
            current_mime_type = img_data.get('mime_type'); current_b64_data = img_data.get('data')
            if not (current_mime_type and current_b64_data and current_mime_type.startswith("image/")): logger.error(f"INVALID IMAGE DATA FOR LLM ATTACHMENT - Image {i+1} ({img_data.get('path', 'N/A')})"); continue
            current_human_message_parts.append({"type": "image_url", "image_url": {"url": f"data:{current_mime_type};base64,{current_b64_data}"}})
    else: current_human_message_parts.append({"type": "text", "text": "\n\nNo actual images are attached for direct model reference for this query."})
    current_human_message_parts.append({"type": "text", "text": f"\n\n--- CURRENT USER QUESTION ---\n{question}\n\n--- ANSWER (Based ONLY on the provided context from all sections above, and previous conversation if relevant) ---"})
    messages.append(HumanMessage(content=current_human_message_parts)) 
    final_answer_content = "Error: Could not generate an answer."; llm_gen_start_time = time.perf_counter()
    try:
        response = rag_llm.invoke(messages); final_answer_content = response.content
    except Exception as e: logger.error(f"LLM invocation error for Q: '{question[:50]}...': {e}", exc_info=True); final_answer_content = f"\n[An error occurred while communicating with the language model: {str(e)}]"
    state["answer"] = final_answer_content; state["llm_generation_time_ms"] = (time.perf_counter() - llm_gen_start_time) * 1000
    logger.info(f"LLM generation took {state['llm_generation_time_ms']:.2f} ms.")
    return state

# ... (get_compiled_rag_agent and run_rag_query - same as your last correct version) ...
_compiled_rag_agent_graph = None
def get_compiled_rag_agent(): # No change
    global _compiled_rag_agent_graph
    if _compiled_rag_agent_graph is None:
        workflow = StateGraph(AdvancedRAGState)
        workflow.add_node("query_transform", query_transform_node)
        workflow.add_node("retrieve_documents", retrieve_documents_node)
        workflow.add_node("rerank_and_select", rerank_and_select_node)
        workflow.add_node("generate_answer", generate_answer_node) 
        workflow.set_entry_point("query_transform")
        workflow.add_edge("query_transform", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "rerank_and_select")
        workflow.add_edge("rerank_and_select", "generate_answer")
        workflow.add_edge("generate_answer", END)
        _compiled_rag_agent_graph = workflow.compile()
        logger.info("Custom RAG Agent graph (non-streaming for /query) compiled successfully.")
    return _compiled_rag_agent_graph

def run_rag_query(
    query_scope: str, question: str,
    summary_for_context: Optional[str], 
    chat_history: Optional[List[Dict[str,str]]] = None, 
    pdf_id: Optional[str] = None, 
    current_corpus_pdf_ids: Optional[List[str]] = None 
) -> Dict: # No change
    start_total_rag_time = time.perf_counter()
    initial_state: AdvancedRAGState = {
        "query_scope": query_scope, "pdf_id": pdf_id, 
        "current_corpus_pdf_ids": current_corpus_pdf_ids if query_scope == "corpus" else None,
        "question": question, "original_question": question, "transformed_question": question, 
        "retrieved_text_docs": [], "retrieved_image_desc_docs": [], 
        "final_text_context": [], "final_image_context": [],
        "images_for_llm_payload": [], "answer": "", 
        "summary_for_context": summary_for_context, 
        "chat_history": chat_history if chat_history else [], 
        "llm_generation_time_ms": None, "total_rag_time_ms": None,
    }
    agent_graph = get_compiled_rag_agent(); final_state_result: Optional[AdvancedRAGState] = None
    try: final_state_result = agent_graph.invoke(initial_state, {"recursion_limit": 15})
    except Exception as e:
        logger.error(f"LangGraph execution error for scope: {query_scope}, Q: '{question[:50]}...': {e}", exc_info=True)
        total_rag_time_ms_on_error = (time.perf_counter() - start_total_rag_time) * 1000
        return {"answer": "Error during RAG agent execution. Please check server logs.", "total_rag_time_ms": total_rag_time_ms_on_error, "llm_generation_time_ms": 0.0, "processing_time_ms": total_rag_time_ms_on_error}
    total_rag_time_ms = (time.perf_counter() - start_total_rag_time) * 1000
    if not isinstance(final_state_result, dict):
        logger.error(f"LangGraph invoke did not return a dictionary state. Type: {type(final_state_result)}")
        return {"answer": "Internal error: RAG agent state invalid.", "total_rag_time_ms": total_rag_time_ms, "llm_generation_time_ms": 0.0, "processing_time_ms": total_rag_time_ms}
    response_payload = {
        "answer": final_state_result.get("answer", "No answer was generated."),
        "processing_time_ms": total_rag_time_ms, 
        "llm_generation_time_ms": final_state_result.get("llm_generation_time_ms", 0.0)
    }
    return response_payload
