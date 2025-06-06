import os
import shutil
from typing import List, Tuple, Optional, Any, Dict
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from app.dependencies_config.llm_config import get_embeddings_model, get_aux_llm
from app.core.config import settings, get_chroma_text_db_path, get_chroma_image_db_path, get_vector_store_base_dir
import logging
from app.services.utils import clean_parsed_text

logger = logging.getLogger(__name__)

TEXT_COLLECTION_NAME = f"texts_{settings.chroma_collection_version_tag}"
IMAGE_COLLECTION_NAME = f"images_{settings.chroma_collection_version_tag}"

def create_or_load_vector_stores_for_pdf(
    pdf_id: str,
    documents: Optional[List[Document]] = None, 
    force_recreate: bool = False
) -> Tuple[Optional[Chroma], Optional[Chroma]]:
    embeddings = get_embeddings_model()
    if not embeddings:
        logger.error(f"Embeddings model not available for PDF {pdf_id}.")
        return None, None

    text_db_persistent_path = get_chroma_text_db_path(pdf_id)
    image_db_persistent_path = get_chroma_image_db_path(pdf_id)
    
    text_collection_name_to_use = TEXT_COLLECTION_NAME 
    image_collection_name_to_use = IMAGE_COLLECTION_NAME

    if force_recreate:
        if os.path.exists(text_db_persistent_path): shutil.rmtree(text_db_persistent_path); logger.info(f"Removed existing text VS for {pdf_id}")
        if os.path.exists(image_db_persistent_path): shutil.rmtree(image_db_persistent_path); logger.info(f"Removed existing image VS for {pdf_id}")
        os.makedirs(text_db_persistent_path, exist_ok=True); os.makedirs(image_db_persistent_path, exist_ok=True)

    text_vectorstore = None; image_vectorstore = None

    text_docs_for_vs = [d for d in documents if d.metadata.get("type") in ["text_chunk", "title_summary", "abstract_summary", "text_table_content", "text_figure_description"]] if documents else [] 
    image_desc_docs_for_vs = [d for d in documents if d.metadata.get("type") == "image_description"] if documents else []
    
    if text_docs_for_vs or os.path.exists(os.path.join(text_db_persistent_path, "chroma.sqlite3")):
        if text_docs_for_vs and (force_recreate or not os.path.exists(os.path.join(text_db_persistent_path, "chroma.sqlite3"))):
            try:
                text_vectorstore = Chroma.from_documents(documents=text_docs_for_vs, embedding=embeddings, collection_name=text_collection_name_to_use, persist_directory=text_db_persistent_path)
                logger.info(f"Text vector store CREATED for PDF {pdf_id} with {text_vectorstore._collection.count()} entries.")
            except Exception as e: logger.error(f"Error creating text VS for {pdf_id}: {e}", exc_info=True)
        elif os.path.exists(os.path.join(text_db_persistent_path, "chroma.sqlite3")):
            try: 
                text_vectorstore = Chroma(persist_directory=text_db_persistent_path, embedding_function=embeddings, collection_name=text_collection_name_to_use)
                logger.info(f"Text vector store LOADED for PDF {pdf_id}. Collection count: {text_vectorstore._collection.count()}")
            except Exception as e: logger.error(f"Error loading text VS for {pdf_id}: {e}", exc_info=True); text_vectorstore = None
        else: logger.info(f"No text documents and no existing text store to load for PDF {pdf_id}.")

    if image_desc_docs_for_vs or os.path.exists(os.path.join(image_db_persistent_path, "chroma.sqlite3")):
        if image_desc_docs_for_vs and (force_recreate or not os.path.exists(os.path.join(image_db_persistent_path, "chroma.sqlite3"))):
            try: image_vectorstore = Chroma.from_documents(documents=image_desc_docs_for_vs, embedding=embeddings, collection_name=image_collection_name_to_use, persist_directory=image_db_persistent_path)
            except Exception as e: logger.error(f"Error creating image desc VS for {pdf_id}: {e}", exc_info=True)
        elif os.path.exists(os.path.join(image_db_persistent_path, "chroma.sqlite3")):
            try: image_vectorstore = Chroma(persist_directory=image_db_persistent_path, embedding_function=embeddings, collection_name=image_collection_name_to_use)
            except Exception as e: logger.error(f"Error loading image desc VS for {pdf_id}: {e}", exc_info=True); image_vectorstore = None
        else: logger.info(f"No image desc documents and no existing image store to load for PDF {pdf_id}.")

    return text_vectorstore, image_vectorstore

def get_retrievers_for_pdf(pdf_id: str, k_text: int = 10, k_images: int = 8) -> Tuple[Optional[Any], Optional[Any]]:
    text_vs_instance, image_vs_instance = create_or_load_vector_stores_for_pdf(pdf_id, documents=None, force_recreate=False)
    text_retriever = None; image_retriever = None
    if text_vs_instance:
        try:
            if text_vs_instance._collection.count() > 0:
                text_retriever = text_vs_instance.as_retriever(search_kwargs={"k": k_text})
                logger.debug(f"Text retriever successfully created for PDF {pdf_id} (k={k_text}, {text_vs_instance._collection.count()} docs).")
            else: logger.warning(f"Text vector store for PDF {pdf_id} is empty, no retriever created.")
        except Exception as e: logger.error(f"Error creating text retriever for PDF {pdf_id}: {e}", exc_info=True)
    else: logger.warning(f"Text vector store FAILED to load for PDF {pdf_id}, no retriever created.")
    
    if image_vs_instance:
        try:
            if image_vs_instance._collection.count() > 0:
                image_retriever = image_vs_instance.as_retriever(search_kwargs={"k": k_images})
                logger.debug(f"Image description retriever successfully created for PDF {pdf_id} (k={k_images}, {image_vs_instance._collection.count()} docs).")
            else: logger.warning(f"Image vector store for PDF {pdf_id} is empty, no image retriever created.")
        except Exception as e: logger.error(f"Error creating image retriever for PDF {pdf_id}: {e}", exc_info=True)
    else: logger.warning(f"Image description vector store FAILED to load for PDF {pdf_id}, no image retriever created.")
    return text_retriever, image_retriever

def check_if_pdf_processed(pdf_id: str) -> bool:
    text_db_path = get_chroma_text_db_path(pdf_id)
    return os.path.exists(os.path.join(text_db_path, "chroma.sqlite3"))

# --- generate_current_batch_corpus_summary (Refined prompt and metadata) ---
def generate_current_batch_corpus_summary(
    current_batch_pdf_ids: List[str], 
    task_status_dict: Dict[str, Any]
) -> Optional[Document]:
    if not current_batch_pdf_ids:
        logger.info("No PDF IDs in current batch to generate corpus summary for.")
        return None
        
    logger.info(f"Attempting to generate corpus summary for {len(current_batch_pdf_ids)} PDFs in current batch: {current_batch_pdf_ids}")
    
    aux_llm = get_aux_llm()
    if not aux_llm:
        logger.error("Cannot generate corpus summary: Auxiliary LLM (get_aux_llm) is not available.")
        return Document(page_content="Corpus summary generation failed: Auxiliary LLM not available.", 
                        metadata={"type":"corpus_summary_error", "source_doc_name": "Corpus_Overview_Error"})

    corpus_info_for_llm: List[str] = []
    successfully_processed_titles_in_batch: List[str] = []
    summarized_pdf_filenames_map: Dict[str, str] = {} # pdf_id -> original_filename

    for pdf_id in current_batch_pdf_ids:
        status_info = task_status_dict.get(pdf_id)
        if status_info and status_info.get("status") == "COMPLETED":
            pdf_title = status_info.get("title", f"PDF (ID: {pdf_id[:8]}...)")
            original_filename = status_info.get("filename", f"UnknownFile_{pdf_id[:8]}.pdf")
            pdf_abstract = status_info.get("abstract", "No abstract available for this PDF.")
            
            doc_info_str = f"Document Filename: '{original_filename}'\nTitle: {pdf_title}\nAbstract Snippet: {pdf_abstract[:300]}..."
            corpus_info_for_llm.append(doc_info_str)
            successfully_processed_titles_in_batch.append(pdf_title)
            summarized_pdf_filenames_map[pdf_id] = original_filename
        else:
            logger.warning(f"PDF {pdf_id} from current batch was not 'COMPLETED'. Skipping for corpus summary.")
    
    if not corpus_info_for_llm:
        logger.warning("No successfully processed documents in current batch for corpus summary.")
        return Document(page_content="No successfully processed documents in the current batch to summarize.", 
                        metadata={"type":"corpus_summary_insufficient_data", "source_doc_name": "Corpus_Overview_NoData"})

    summary_llm_input_text = "\n\n---\n\n".join(corpus_info_for_llm[:15]) 

    prompt_template = (
        "You are provided with titles, original filenames, and abstract snippets from a batch of recently processed research papers. "
        "Your task is to generate a concise, high-level thematic overview of THIS BATCH of documents. "
        "Identify the main research areas, core topics, or methodologies that appear to be common or prominent across these papers. "
        "If you mention a specific concept that seems to originate from one of these papers, try to mention the document filename. "
        "Avoid going into deep specifics of any single paper unless it's exceptionally illustrative of a shared theme. "
        "The goal is to give a user a quick understanding of what this particular collection of papers is generally about.\n\n"
        "INFORMATION FROM THE BATCH OF DOCUMENTS:\n{batch_doc_info}\n\n"
        "CONCISE THEMATIC OVERVIEW OF THIS BATCH (1-3 paragraphs):\n"
    )
    
    final_prompt_text = prompt_template.format(batch_doc_info=summary_llm_input_text)
    
    try:
        from langchain_core.messages import HumanMessage
        response = aux_llm.invoke([HumanMessage(content=final_prompt_text)])
        generated_summary_content = clean_parsed_text(response.content)
        logger.info(f"Successfully generated corpus summary for current batch (based on {len(successfully_processed_titles_in_batch)} PDFs): {generated_summary_content[:150]}...")
    except Exception as e:
        logger.error(f"LLM-based corpus summary generation failed for current batch: {e}", exc_info=True)
        generated_summary_content = f"Corpus summary generation for the current batch encountered an error: {str(e)}"
    
    corpus_summary_doc = Document(
        page_content=generated_summary_content,
        metadata={
            "source_doc_name": "Current_Batch_Corpus_Overview", 
            "type": "corpus_summary", "importance": "critical", 
            "summarized_pdf_titles": successfully_processed_titles_in_batch, 
            "summarized_pdf_ids_and_filenames": summarized_pdf_filenames_map # Store mapping for potential future use
        }
    )
    return corpus_summary_doc
    
    


        
