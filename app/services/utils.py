import re
import base64
import io
import os
from PIL import Image
from typing import Tuple, Optional

try:
    from unstructured.cleaners.core import clean as unstruct_clean, clean_extra_whitespace as unstruct_clean_ws
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("Warning: Unstructured.io not found. Falling back to basic text cleaning.")

def clean_parsed_text(text: str) -> str:
    """Cleans text by removing extra whitespace and handling ligatures."""
    if not text:
        return ""
    if UNSTRUCTURED_AVAILABLE:
        try:
            # Note: Check unstructured documentation for current best practices on these flags
            text = unstruct_clean(text, bullets=False, extra_whitespace=False, dashes=False, trailing_punctuation=False)
            text = unstruct_clean_ws(text)
        except Exception: # Fallback if unstructured cleaning fails for some reason
            text = re.sub(r'\s+', ' ', text).strip()
    else:
        text = re.sub(r'\s+', ' ', text).strip()
    # Ligature replacement (from your notebook)
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return text.strip()

def image_to_base64(image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Converts an image file to a base64 string and returns its MIME type."""
    if not os.path.exists(image_path):
        print(f"Warning: Image path does not exist for base64 conversion: {image_path}")
        return None, None
    try:
        with Image.open(image_path) as img:
            fmt = img.format.lower() if img.format else 'png' # Default to png if format is None
            # Ensure format is one that BytesIO and PIL can handle well for saving
            save_format = fmt if fmt in ["jpeg", "png", "gif", "webp"] else "PNG"

            # Handle RGBA for JPEG
            if save_format.upper() == "JPEG" and img.mode == "RGBA":
                img = img.convert("RGB")

            bio = io.BytesIO()
            img.save(bio, format=save_format)
            img_bytes = bio.getvalue()

            mime_type = f"image/{save_format.lower()}"
            if save_format.upper() == "JPEG": # Ensure correct MIME for JPEG
                mime_type = "image/jpeg"

        return base64.b64encode(img_bytes).decode('utf-8'), mime_type
    except Exception as e:
        print(f"Error converting image {os.path.basename(image_path)} to base64: {e}")
        return None, None

def ocr_image_to_text_unstructured(image_path: str) -> str:
    """Performs OCR on an image file using Unstructured.io's partition if available."""
    if not UNSTRUCTURED_AVAILABLE:
        print("Unstructured.io not available for OCR.")
        return ""
    try:
        from unstructured.partition.auto import partition # Local import to keep it optional
        # Consider strategy "hi_res" or specific OCR engine if installed ("ocr_only" with model)
        # The 'yolox' model is for layout detection. OCR happens via tesseract by default if not specified.
        elements = partition(filename=image_path, strategy="ocr_only") # , hi_res_model_name="yolox" # Add if layout detection is used for OCR
        full_text = "\n".join([str(el.text) for el in elements if hasattr(el, 'text')])
        return clean_parsed_text(full_text)
    except Exception as e:
        print(f"Error during Unstructured OCR for {os.path.basename(image_path)}: {e}")
        return ""