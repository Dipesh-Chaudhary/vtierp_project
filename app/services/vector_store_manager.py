import os
import shutil
from typing import List, Tuple, Optional, Any
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

from app.dependencies_config.llm_config import get_embeddings_model
from app.core.config import settings, get_chroma_text_db_path, get_chroma_image_db_path

def create_or_load_vector_stores_for_pdf(
    pdf_id: str,
    documents: Optional[List[Document]] = None,
    force_recreate: bool = False
) -> Tuple[Optional[Any], Optional[Any]]: # Returns (text_retriever, image_retriever)
    """
    Creates (if documents are provided) or loads vector stores for a given PDF ID.
    If force_recreate is True, existing stores for this pdf_id will be deleted.
    'documents' should be the full list of docs from process_single_pdf_custom.
    """
    embeddings = get_embeddings_model()
    if not embeddings:
        print(f"Error for PDF {pdf_id}: Embeddings model not available. Cannot create/load vector stores.")
        return None, None

    text_db_persistent_path = get_chroma_text_db_path(pdf_id)
    image_db_persistent_path = get_chroma_image_db_path(pdf_id)

    if force_recreate:
        if os.path.exists(text_db_persistent_path):
            shutil.rmtree(text_db_persistent_path)
            # print(f"Force recreate: Removed existing text vector store for PDF ID {pdf_id}")
        if os.path.exists(image_db_persistent_path):
            shutil.rmtree(image_db_persistent_path)
            # print(f"Force recreate: Removed existing image vector store for PDF ID {pdf_id}")
        # Re-create dirs as Chroma needs them
        os.makedirs(text_db_persistent_path, exist_ok=True)
        os.makedirs(image_db_persistent_path, exist_ok=True)


    text_vectorstore = None
    image_vectorstore = None

    # Filter documents for each store type
    text_docs_for_vs = []
    image_desc_docs_for_vs = []
    if documents:
        for d in documents:
            doc_type = d.metadata.get("type")
            if doc_type in ["text_chunk", "title_summary", "abstract_summary", "text_table_content", "text_figure_description"]:
                text_docs_for_vs.append(d)
            elif doc_type == "image_description": # Only VLM descriptions of rendered images
                image_desc_docs_for_vs.append(d)

    # Create/Load Text Vector Store
    if text_docs_for_vs and (force_recreate or not os.path.exists(os.path.join(text_db_persistent_path, "chroma.sqlite3"))):
        try:
            text_vectorstore = Chroma.from_documents(
                documents=text_docs_for_vs,
                embedding=embeddings,
                collection_name=f"text_{pdf_id}_{settings.chroma_collection_version_tag}",
                persist_directory=text_db_persistent_path
            )
            text_vectorstore.persist()
            print(f"Text vector store CREATED for PDF ID {pdf_id} with {len(text_docs_for_vs)} entries.")
        except Exception as e:
            print(f"Error creating text vector store for PDF {pdf_id}: {e}")
    elif os.path.exists(os.path.join(text_db_persistent_path, "chroma.sqlite3")):
        try:
            text_vectorstore = Chroma(
                persist_directory=text_db_persistent_path,
                embedding_function=embeddings,
                collection_name=f"text_{pdf_id}_{settings.chroma_collection_version_tag}" # Needs name if not default
            )
            print(f"Text vector store LOADED for PDF ID {pdf_id}. Count: {text_vectorstore._collection.count()}")
        except Exception as e:
            print(f"Error loading existing text vector store for PDF {pdf_id}: {e}")
    else:
        print(f"No text documents provided and no existing text store found for PDF ID {pdf_id}.")

    # Create/Load Image Description Vector Store
    if image_desc_docs_for_vs and (force_recreate or not os.path.exists(os.path.join(image_db_persistent_path, "chroma.sqlite3"))):
        try:
            image_vectorstore = Chroma.from_documents(
                documents=image_desc_docs_for_vs,
                embedding=embeddings,
                collection_name=f"image_desc_{pdf_id}_{settings.chroma_collection_version_tag}",
                persist_directory=image_db_persistent_path
            )
            image_vectorstore.persist()
            print(f"Image description vector store CREATED for PDF ID {pdf_id} with {len(image_desc_docs_for_vs)} entries.")
        except Exception as e:
            print(f"Error creating image desc vector store for PDF {pdf_id}: {e}")
    elif os.path.exists(os.path.join(image_db_persistent_path, "chroma.sqlite3")):
        try:
            image_vectorstore = Chroma(
                persist_directory=image_db_persistent_path,
                embedding_function=embeddings,
                collection_name=f"image_desc_{pdf_id}_{settings.chroma_collection_version_tag}"
            )
            print(f"Image description vector store LOADED for PDF ID {pdf_id}. Count: {image_vectorstore._collection.count()}")
        except Exception as e:
            print(f"Error loading existing image desc vector store for PDF {pdf_id}: {e}")

    else:
        print(f"No image description documents provided and no existing image store found for PDF ID {pdf_id}.")

    text_retriever = text_vectorstore.as_retriever(search_kwargs={"k": 10}) if text_vectorstore else None # k from notebook
    image_retriever = image_vectorstore.as_retriever(search_kwargs={"k": 8}) if image_vectorstore else None # k from notebook

    return text_retriever, image_retriever


def get_retrievers_for_pdf(pdf_id: str) -> Tuple[Optional[Any], Optional[Any]]:
    """Loads vector stores and returns retrievers for a given PDF ID if they exist."""
    # This function simply loads, does not create. Creation happens via process_single_pdf_custom -> create_or_load_vector_stores_for_pdf
    # It's essentially a read-only version for querying.
    return create_or_load_vector_stores_for_pdf(pdf_id, documents=None, force_recreate=False)


def check_if_pdf_processed(pdf_id: str) -> bool:
    """Checks if vector stores (at least text one) exist for a given PDF ID."""
    text_db_path = get_chroma_text_db_path(pdf_id)
    # A simple check for the presence of Chroma's main sqlite file.
    return os.path.exists(os.path.join(text_db_path, "chroma.sqlite3"))