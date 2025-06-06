import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    llm_rag_model_name: str = os.getenv("LLM_RAG_MODEL", "gemini-1.5-flash-preview-05-20")
    llm_aux_model_name: str = os.getenv("LLM_AUX_MODEL", "gemini-1.5-flash-preview-05-20")
    embeddings_model_name: str = os.getenv("EMBEDDINGS_MODEL", "models/text-embedding-004")

    max_images_to_llm_final: int = int(os.getenv("MAX_IMAGES_TO_LLM_FINAL", 3))
    max_elements_for_vlm_description_per_pdf: int = int(os.getenv("MAX_ELEMENTS_FOR_VLM_DESCRIPTION_PER_PDF", 8))
    render_dpi_pymupdf: int = int(os.getenv("RENDER_DPI_PYMUPDF", 150))
    min_visual_width_pymupdf: int = int(os.getenv("MIN_VISUAL_WIDTH_PYMUPDF", 30))
    min_visual_height_pymupdf: int = int(os.getenv("MIN_VISUAL_HEIGHT_PYMUPDF", 30))
    ocr_dpi: int = int(os.getenv("OCR_DPI", 300))
    text_block_min_area_for_obstruction: int = 100
    drawing_cluster_max_dist_factor: float = 0.03
    min_ocr_text_length_for_scanned_pdf: int = 100

    base_data_path: str = "data"

    uploads_dir_name: str = "uploads"
    vector_stores_dir_name: str = "vector_stores"
    extracted_elements_dir_name: str = "extracted_document_elements"

    chroma_collection_version_tag: str = "v1_custom"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

def get_persistent_base_path() -> str: return settings.base_data_path
def get_pdf_upload_dir() -> str: path = os.path.join(get_persistent_base_path(), settings.uploads_dir_name); os.makedirs(path, exist_ok=True); return path
def get_vector_store_base_dir() -> str: path = os.path.join(get_persistent_base_path(), settings.vector_stores_dir_name); os.makedirs(path, exist_ok=True); return path
def get_pdf_specific_data_dir(pdf_id: str) -> str: path = os.path.join(get_vector_store_base_dir(), pdf_id); os.makedirs(path, exist_ok=True); return path
def get_pdf_extracted_images_dir(pdf_id: str) -> str: path = os.path.join(get_pdf_specific_data_dir(pdf_id), "extracted_images"); os.makedirs(path, exist_ok=True); return path
def get_chroma_text_db_path(pdf_id: str) -> str: path = os.path.join(get_pdf_specific_data_dir(pdf_id), f"chroma_text_{settings.chroma_collection_version_tag}"); os.makedirs(path, exist_ok=True); return path
def get_chroma_image_db_path(pdf_id: str) -> str: path = os.path.join(get_pdf_specific_data_dir(pdf_id), f"chroma_image_{settings.chroma_collection_version_tag}"); os.makedirs(path, exist_ok=True); return path

os.makedirs(get_pdf_upload_dir(), exist_ok=True)
os.makedirs(get_vector_store_base_dir(), exist_ok=True)
