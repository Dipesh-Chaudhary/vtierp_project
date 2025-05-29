import os
from typing import List, Dict, TypedDict, Optional, Any
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.docstore.document import Document
from langgraph.graph import StateGraph, END

from app.dependencies_config.llm_config import get_rag_llm, get_aux_llm
from app.services.vector_store_manager import get_retrievers_for_pdf
from app.services.utils import image_to_base64 # For preparing image payload for LLM
from app.core.config import settings

# --- LangGraph State Definition (from your notebook) ---
class AdvancedRAGState(TypedDict):
    pdf_id: str # Added for context
    original_question: str
    transformed_question: str # Could be enhanced with query transformation logic
    retrieved_text_docs: List[Document]
    retrieved_image_desc_docs: List[Document]
    final_text_context: List[Document] # After reranking/selection
    final_image_context: List[Document] # After reranking/selection
    images_for_llm_payload: List[Dict[str, Any]] # Base64 images for VLM
    answer: str
    # This was global in notebook. For API, it's specific to the PDF's metadata.
    pdf_specific_summary: Optional[str]


# --- Helper: Format documents for LLM (from your notebook) ---
def format_docs_for_llm(docs: List[Document], doc_type: str) -> str:
    if not docs:
        return f"No relevant {doc_type} context found for this query."

    if doc_type == "image": # Image descriptions
        grouped = defaultdict(list)
        for doc in docs:
            # Use caption_id for grouping, fallback to image path if no caption_id
            cid = doc.metadata.get("caption_id", f"ungrouped_img_{os.path.basename(doc.metadata.get('image_path_on_server','no_path'))}")
            grouped[cid].append(doc)

        parts = []
        for grp_id, grp_docs in grouped.items():
            # original_caption might be long, or just be "Uncaptioned visual"
            cap_text = grp_docs[0].metadata.get("original_caption", str(grp_id))
            if "ungrouped_img" in str(grp_id) and cap_text == str(grp_id) : cap_text = "Ungrouped Visual"

            parts.append(f"Visual Element Group: '{cap_text[:100]}...' ({len(grp_docs)} description(s) found)")
            for i, d_g in enumerate(grp_docs):
                path_hint = os.path.basename(str(d_g.metadata.get('image_path_on_server', 'N/A')))
                page_num = d_g.metadata.get('page_number', 'N/A')
                # VLM description is d_g.page_content
                parts.append(f"  Description Part {i+1} (Source Hint: {path_hint}, Page: {page_num}): {d_g.page_content}")
        return "\n\n".join(parts)

    # For text documents
    parts = []
    for i, doc in enumerate(docs):
        m = doc.metadata
        src_doc_name = os.path.basename(str(m.get('source_doc_name', 'N/A')))
        pg = m.get('page_number', 'N/A')
        # Clean up type display
        type_raw = m.get('parser_source', m.get('type', 'N/A'))
        type_clean = str(type_raw).replace('text_', '').replace('_description', '').replace('_summary', '').replace('content', '')
        if type_clean == "chunk": type_clean = "Text Chunk"

        importance = m.get('importance', '')
        header = f"Context Document {i+1} (Source: {src_doc_name}, Page: {pg}, Type: {type_clean}{', Importance: '+importance if importance else ''}):"
        parts.append(f"{header}\n{doc.page_content}\n")
    return "\n\n".join(parts)


# --- LangGraph Nodes (adapted from your notebook) ---
def query_transform_node(state: AdvancedRAGState) -> AdvancedRAGState:
    # print(f"--- Node: query_transform_node (PDF: {state['pdf_id']}) ---")
    # For now, simple pass-through. Could involve query expansion, sub-queries etc.
    state["transformed_question"] = state["original_question"]
    return state

def retrieve_documents_node(state: AdvancedRAGState) -> AdvancedRAGState:
    # print(f"--- Node: retrieve_documents_node (PDF: {state['pdf_id']}) ---")
    query = state["transformed_question"]
    pdf_id = state["pdf_id"]

    text_retriever, image_desc_retriever = get_retrievers_for_pdf(pdf_id)

    state["retrieved_text_docs"] = text_retriever.invoke(query) if text_retriever else []
    state["retrieved_image_desc_docs"] = image_desc_retriever.invoke(query) if image_desc_retriever else []
    # print(f"Retrieved {len(state['retrieved_text_docs'])} text, {len(state['retrieved_image_desc_docs'])} img_desc docs.")
    return state

def rerank_and_select_node(state: AdvancedRAGState) -> AdvancedRAGState:
    # print(f"--- Node: rerank_and_select_node (PDF: {state['pdf_id']}) ---")
    # This implements the k-selection and critical doc prioritization from your notebook
    txt_docs = state.get("retrieved_text_docs", [])
    img_docs = state.get("retrieved_image_desc_docs", [])

    k_text_retrieved = 10 # default, from your notebook's retriever setup
    k_images_retrieved = 8 # default

    # For final context to LLM (can be smaller than retrieved k)
    k_text_final = 7    # From your notebook's rerank_and_select
    k_images_final = 5  # From your notebook's rerank_and_select

    # Prioritize critical documents (title, abstract, pdf_specific_summary)
    prioritized_texts = []
    critical_content_seen = set()

    # Add pdf_specific_summary if available (comes from initial state now)
    if state.get("pdf_specific_summary"):
        summary_doc = Document(
            page_content=state["pdf_specific_summary"],
            metadata={"source_pdf_id": state["pdf_id"], "type": "pdf_summary", "importance": "critical"}
        )
        prioritized_texts.append(summary_doc)
        critical_content_seen.add(state["pdf_specific_summary"][:200])


    for doc in txt_docs:
        is_critical = doc.metadata.get("importance") == "critical"
        content_prefix = doc.page_content[:200] # Simple dedupe based on prefix
        if is_critical and content_prefix not in critical_content_seen:
            prioritized_texts.append(doc)
            critical_content_seen.add(content_prefix)

    # Add remaining non-critical docs, avoiding duplicates
    for doc in txt_docs:
        if doc.metadata.get("importance") != "critical":
            content_prefix = doc.page_content[:200]
            if content_prefix not in critical_content_seen and not any(p_doc.page_content[:200] == content_prefix for p_doc in prioritized_texts):
                prioritized_texts.append(doc)
                # No need to add to critical_content_seen here, just ensuring it's not already in prioritized_texts

    state["final_text_context"] = prioritized_texts[:k_text_final]
    state["final_image_context"] = img_docs[:k_images_final] # Simpler selection for images for now

    # Prepare image payload for LLM (from your notebook)
    llm_image_payload = []
    for img_doc in state["final_image_context"][:settings.max_images_to_llm_final]: # Global limit from settings
        path = img_doc.metadata.get("image_path_on_server")
        if path and os.path.exists(path):
            b64, mime = image_to_base64(path)
            if b64 and mime:
                llm_image_payload.append({
                    "path": path,
                    "caption_or_vlm_desc": img_doc.page_content, # This is the VLM description
                    "data": b64,
                    "mime_type": mime
                })
    state["images_for_llm_payload"] = llm_image_payload
    # print(f"Selected {len(state['final_text_context'])} text, {len(state['final_image_context'])} img_desc for final context. {len(llm_image_payload)} images for LLM.")
    return state


def generate_answer_node(state: AdvancedRAGState) -> AdvancedRAGState:
    # print(f"--- Node: generate_answer_node (PDF: {state['pdf_id']}) ---")
    rag_llm = get_rag_llm()
    if not rag_llm:
        state["answer"] = "RAG LLM not initialized. Cannot generate answer."
        return state

    question = state["original_question"]
    text_context_str = format_docs_for_llm(state.get("final_text_context", []), "text")
    image_desc_context_str = format_docs_for_llm(state.get("final_image_context", []), "image")
    actual_images_payload = state.get("images_for_llm_payload", [])
    pdf_summary = state.get("pdf_specific_summary", "No specific PDF summary provided.")


    # System prompt from your notebook
    system_prompt_content = (
        "You are an expert AI research assistant. Your task is to provide comprehensive, accurate, "
        "and well-structured answers to user questions based *only* on the provided documents as context.\n\n"
        "**Context Prioritization:** Prioritize context explicitly labeled 'DOCUMENT TITLE:', 'DOCUMENT ABSTRACT:', or 'PDF SUMMARY:'.\n\n"
        "**Figure/Table Description:**\n"
        "- When asked to 'describe' a specific figure/table (e.g., 'Figure 1', 'Table 2'):\n"
        "    - **Prioritize 'image_description' documents or 'text_table_content' that have a matching 'caption_id'.** If multiple parts exist, consolidate their descriptions.\n"
        "    - **If no direct description is found but a 'text_figure_description' is, describe the figure based on that text.**\n"
        "    - **For visual descriptions from images, explain key components, structure, any text within the figure, and its purpose/information conveyed.**\n\n"
        "**General Guidance:**\n"
        "- Maintain a neutral, informative tone.\n"
        "- If information is not found in the provided context, state that clearly rather than hallucinating.\n"
        "- Format your answer clearly with headings or bullet points where appropriate.\n"
        "- Cite page numbers (P:X) or source document names (S:filename) if available in the context metadata. If a caption_id is available for a visual, mention it (e.g. Figure 1).\n"
        "- You are answering questions about a single PDF document, identified by PDF ID: {pdf_id}. All context provided pertains to this document."
    )


    human_message_parts = []
    human_message_parts.append({"type": "text", "text": f"--- PDF SUMMARY (Title/Abstract for PDF ID: {state['pdf_id']}) ---\n{pdf_summary}\n--- END PDF SUMMARY ---"})
    human_message_parts.append({"type": "text", "text": f"\n\n--- TEXT & TEXTUAL TABLE CONTEXT ---\n{text_context_str}\n--- END TEXT CONTEXT ---"})
    human_message_parts.append({"type": "text", "text": f"\n\n--- VLM DESCRIPTIONS OF VISUAL ELEMENTS ---\n{image_desc_context_str}\n--- END VLM DESCRIPTIONS ---"})

    if actual_images_payload:
        human_message_parts.append({"type": "text", "text": f"\n\n--- ({len(actual_images_payload)}) ACTUAL IMAGES FOR YOUR REFERENCE (DO NOT DESCRIBE THE IMAGES THEMSELVES UNLESS ASKED, USE THEIR VLM DESCRIPTIONS ABOVE) ---"})
        for i, img_data in enumerate(actual_images_payload):
            human_message_parts.append({"type": "text", "text": f"Image {i+1} Reference ({os.path.basename(img_data['path'])}). VLM Description Summary: '{img_data['caption_or_vlm_desc'][:100]}...'"})
            human_message_parts.append({"type": "image_url", "image_url": {"url": f"data:{img_data['mime_type']};base64,{img_data['data']}"}})
        human_message_parts.append({"type": "text", "text": "--- END ACTUAL IMAGES ---"})
    else:
        human_message_parts.append({"type": "text", "text": "\n\nNo actual images provided directly to the model for this query."})

    human_message_parts.append({"type": "text", "text": f"\n\n--- USER QUESTION ---\n{question}\n\n--- ANSWER BASED *ONLY* ON THE PROVIDED CONTEXT ---"})

    # Fill pdf_id in system prompt
    filled_system_prompt = system_prompt_content.format(pdf_id=state['pdf_id'])

    messages = [
        SystemMessage(content=filled_system_prompt),
        HumanMessage(content=human_message_parts)
    ]

    final_answer = "Error generating answer."
    try:
        response = rag_llm.invoke(messages)
        final_answer = response.content
    except Exception as e:
        final_answer = f"LLM Error during answer generation: {e}"
        print(f"LLM error for PDF {state['pdf_id']}, Q: '{question[:50]}...': {e}")

    state["answer"] = final_answer
    # print(f"Generated answer for PDF {state['pdf_id']}: {final_answer[:100]}...")
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
        print("Custom RAG Agent graph compiled.")
    return _compiled_rag_agent_graph

def run_rag_query(pdf_id: str, question: str, pdf_summary_for_llm: str) -> Dict:
    """
    Runs the RAG agent for a given PDF ID and question.
    `pdf_summary_for_llm` should be the title/abstract of the specific PDF.
    """
    agent_graph = get_compiled_rag_agent()

    # Initialize state (as in your notebook's main block)
    initial_state = {key: [] if isinstance(type_hint_ann, list) or str(type_hint_ann).startswith("List[") else
                          {} if isinstance(type_hint_ann, dict) or str(type_hint_ann).startswith("Dict[") else
                          None if getattr(type_hint_ann, '__origin__', None) is Optional or
                                  (hasattr(type_hint_ann, '__args__') and type(None) in type_hint_ann.__args__) else
                          "" for key, type_hint_ann in AdvancedRAGState.__annotations__.items()}

    initial_state.update({
        "pdf_id": pdf_id,
        "original_question": question,
        "pdf_specific_summary": pdf_summary_for_llm, # Use the specific PDF's summary
         # Ensure all keys in AdvancedRAGState are present
        "transformed_question": question, # Default, can be overwritten by transform_node
        "retrieved_text_docs": [],
        "retrieved_image_desc_docs": [],
        "final_text_context": [],
        "final_image_context": [],
        "images_for_llm_payload": [],
        "answer": ""
    })


    try:
        # Note on LangGraph input: If using .invoke(input_dict), ensure input_dict matches the graph's input schema.
        # If the entry point node takes the full state, then pass the full initial_state.
        final_state_result = agent_graph.invoke(initial_state, {"recursion_limit": 15}) # LangGraph handles state internally
    except Exception as e:
        print(f"LangGraph execution error for PDF {pdf_id}, Q: '{question[:50]}...': {e}")
        import traceback
        traceback.print_exc()
        return {
            "answer": "Error during RAG agent execution. Check server logs.",
            "retrieved_text_context_sample": [],
            "retrieved_image_context_sample": []
        }

    # Prepare output, similar to what Streamlit UI might need
    # Return samples of context for brevity or full if needed for debugging
    return {
        "answer": final_state_result.get("answer", "No answer generated."),
        "final_text_context": final_state_result.get("final_text_context", []), # List of Document objects
        "final_image_context": final_state_result.get("final_image_context", []) # List of Document objects
    }