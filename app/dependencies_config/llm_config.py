from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.core.config import settings

# Singleton instances for LLMs and Embeddings
_llm_rag_instance = None
_llm_aux_instance = None # For VLM descriptions, summaries, etc.
_embeddings_instance = None

def get_rag_llm() -> ChatGoogleGenerativeAI:
    """Returns the LLM instance for RAG answer generation."""
    global _llm_rag_instance
    if _llm_rag_instance is None:
        try:
            _llm_rag_instance = ChatGoogleGenerativeAI(
                model=settings.llm_rag_model_name,
                google_api_key=settings.google_api_key,
                temperature=0.0, # As per your notebook's llm_rag
                max_output_tokens=8192, # As per your notebook
                convert_system_message_to_human=True # Often good for Gemini
            )
        except Exception as e:
            print(f"CRITICAL Error initializing RAG LLM ({settings.llm_rag_model_name}): {e}")
            raise
    return _llm_rag_instance

def get_aux_llm() -> ChatGoogleGenerativeAI:
    """
    Returns the LLM instance for auxiliary tasks like VLM image description,
    corpus summarization etc. This is the one that can handle image inputs.
    """
    global _llm_aux_instance
    if _llm_aux_instance is None:
        try:
            # This model needs to be multimodal if used for VLM descriptions as in notebook
            # Your notebook used gemini-2.5-flash-preview-05-20 for llm_aux
            # Ensure settings.llm_aux_model_name is a multimodal model
            _llm_aux_instance = ChatGoogleGenerativeAI(
                model=settings.llm_aux_model_name,
                google_api_key=settings.google_api_key,
                temperature=0.1, # As per your notebook's llm_aux
                max_output_tokens=4096, # As per your notebook
                convert_system_message_to_human=True
            )
        except Exception as e:
            print(f"CRITICAL Error initializing AUX LLM/VLM ({settings.llm_aux_model_name}): {e}")
            raise
    return _llm_aux_instance

def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """Returns the embeddings model instance."""
    global _embeddings_instance
    if _embeddings_instance is None:
        try:
            _embeddings_instance = GoogleGenerativeAIEmbeddings(
                model=settings.embeddings_model_name,
                google_api_key=settings.google_api_key
            )
        except Exception as e:
            print(f"CRITICAL Error initializing Embeddings model ({settings.embeddings_model_name}): {e}")
            raise
    return _embeddings_instance

# Pre-initialize on module load to catch errors early (optional)
try:
    get_rag_llm()
    get_aux_llm()
    get_embeddings_model()
    print(f"LLM/Embedding models ({settings.llm_rag_model_name}, {settings.llm_aux_model_name}, {settings.embeddings_model_name}) configured.")
except Exception as e:
    print(f"Failed to pre-initialize LLM/Embedding models: {e}")