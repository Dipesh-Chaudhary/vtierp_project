import fitz  # PyMuPDF
import os
import shutil
import uuid
import io
import re
import math
from PIL import Image
from tqdm import tqdm # Optional for command-line processing, less useful in API bg task
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any, Set

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import settings, get_pdf_extracted_images_dir
from app.dependencies_config.llm_config import get_aux_llm
from app.services.utils import clean_parsed_text, image_to_base64, ocr_image_to_text_unstructured

# Global counter for VLM descriptions per PDF processing call.
# This needs to be managed per processing task if tasks are concurrent.
# For a single background task per PDF, a simple counter reset at start is okay.
class VLMUsageTracker:
    def __init__(self, limit: int):
        self.count = 0
        self.limit = limit

    def increment(self):
        self.count += 1

    def can_use_vlm(self) -> bool:
        return self.count < self.limit

def generate_detailed_image_description(
    image_path: str,
    element_type: str,
    vlm_tracker: VLMUsageTracker
) -> str:
    """Generates a detailed description of an image using a multimodal LLM (VLM)."""
    aux_llm = get_aux_llm() # Get the VLM instance
    if not aux_llm:
        return f"VLM (aux_llm) not initialized for {os.path.basename(image_path)}."

    if not vlm_tracker.can_use_vlm():
        # print(f"VLM description limit reached for this PDF. Skipping VLM for {os.path.basename(image_path)}")
        return f"Visual Element ({element_type}): {os.path.basename(image_path)}. VLM description skipped due to PDF limit."

    try:
        img_base64, mime_type = image_to_base64(image_path)
        if not img_base64 or not mime_type:
            return f"Could not load image {os.path.basename(image_path)} for VLM."

        prompt_text = (
            f"Expert document analyst: Analyze this {element_type} from a research paper. "
            f"Describe its key visual components, structure, any text present within it, "
            f"and its apparent purpose or the information it conveys. "
            f"For Tables: describe columns, data types, and notable trends. "
            f"For Diagrams/Charts: describe type, axes, trends, and flow. "
            f"Provide a concise, comprehensive summary suitable for Q&A."
        )
        from langchain_core.messages import HumanMessage # Local import
        message = HumanMessage(content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}}
        ])

        response = aux_llm.invoke([message])
        vlm_tracker.increment()
        return clean_parsed_text(response.content)
    except Exception as e:
        print(f"Error generating VLM description for {os.path.basename(image_path)}: {e}")
        return f"Error in VLM description for {os.path.basename(image_path)}."


# --- PyMuPDF Helper functions from your notebook ---
def find_pymupdf_captions(page: fitz.Page) -> List[Dict]:
    captions = []
    # Using TEXTFLAGS_DICT is important if your notebook relied on it.
    # The flag combination from your notebook: fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES & ~fitz.TEXT_PRESERVE_IMAGES
    # For cleaner text extraction, often just page.get_text("dict") is fine.
    # Let's use a safe default, can be tuned.
    blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT)["blocks"]
    for block in blocks:
        if block['type'] == 0:  # Text block
            block_text_content = "".join(span['text'] for line in block['lines'] for span in line['spans'])
            cleaned_block_text = clean_parsed_text(block_text_content)

            # Regex from your notebook
            match = re.match(r"^(Figure|Fig\.?|Table)\s+([A-Za-z0-9]+\.?\d*)\s*[:\.]?\s*", cleaned_block_text, re.IGNORECASE)
            if match:
                caption_type = "figure" if "fig" in match.group(1).lower() else "table"
                caption_id_num = match.group(2)
                # Ensure unique caption_id format similar to notebook for consistency
                caption_id = f"{caption_type.capitalize()}{caption_id_num}"
                captions.append({
                    "text": cleaned_block_text,
                    "bbox": fitz.Rect(block['bbox']),
                    "type": caption_type,
                    "page_num": page.number, # 0-indexed
                    "id": caption_id
                })
    return captions

def get_pymupdf_drawing_clusters(page: fitz.Page) -> List[fitz.Rect]:
    drawings = page.get_drawings()
    path_rects = [d['rect'] for d in drawings if d['rect'].width > 1 and d['rect'].height > 1 and (d.get('type') != 'fill' or d.get('color') is not None)]
    if not path_rects:
        return []

    merged_rects = []
    path_rects.sort(key=lambda r: (r.y0, r.x0))

    page_diag = math.sqrt(page.rect.width**2 + page.rect.height**2) if page.rect.width > 0 and page.rect.height > 0 else 1000
    max_dist = page_diag * settings.drawing_cluster_max_dist_factor

    for r_obj in path_rects:
        r_rect = fitz.Rect(r_obj)
        if not merged_rects:
            merged_rects.append(r_rect)
        else:
            last = merged_rects[-1]
            expanded_last = fitz.Rect(last) + (-max_dist, -max_dist, max_dist, max_dist)
            if r_rect.intersects(expanded_last):
                merged_rects[-1] = last | r_rect
            else:
                merged_rects.append(r_rect)

    return [r_item for r_item in merged_rects if r_item.width > settings.min_visual_width_pymupdf / 2 and r_item.height > settings.min_visual_height_pymupdf / 2]

def find_table_content(page: fitz.Page, table_caption_bbox: fitz.Rect) -> str:
    # This function is quite heuristic, as acknowledged in your notebook.
    # It might need fine-tuning based on diverse PDF table structures.
    table_content_str = ""
    search_rect = fitz.Rect(page.rect.x0, table_caption_bbox.y1, page.rect.x1, table_caption_bbox.y1 + 300)
    text_blocks_in_roi = [
        block for block in page.get_text("blocks", flags=0)
        if fitz.Rect(block[:4]).intersects(search_rect) and block[6] == 0 # type 0 is text
    ]
    text_blocks_in_roi.sort(key=lambda b: (b[1], b[0])) # Sort by y0, then x0

    potential_table_lines = []
    last_table_line_y1 = table_caption_bbox.y1

    for block in text_blocks_in_roi:
        block_bbox = fitz.Rect(block[:4])
        # Check if block is reasonably close below the caption or last line
        if block_bbox.y0 > last_table_line_y1 - 5 and block_bbox.y0 < last_table_line_y1 + 70: # Allow some overlap or small gap
            # Check for horizontal overlap with caption (rough alignment)
            caption_x_mid = (table_caption_bbox.x0 + table_caption_bbox.x1) / 2
            block_x_mid = (block_bbox.x0 + block_bbox.x1) / 2
            # Ensure block isn't wildly offset horizontally
            if abs(caption_x_mid - block_x_mid) < page.rect.width * 0.3: # Block center within 30% of page width from caption center
                block_text_content = page.get_text(clip=block_bbox, sort=True) # Get sorted text from block
                lines = block_text_content.split('\n')
                # Filter out very short lines or lines that look like page numbers/footers
                lines = [line.strip() for line in lines if len(line.strip()) > 3 and not re.match(r"^\d+$", line.strip())]
                if lines:
                    potential_table_lines.extend(lines)
                    last_table_line_y1 = max(last_table_line_y1, block_bbox.y1) # Update last y position
        elif block_bbox.y0 >= last_table_line_y1 + 70: # If block is too far down, stop.
            break

    if potential_table_lines:
        table_content_str = "\n".join(potential_table_lines)

    return clean_parsed_text(table_content_str) if table_content_str else ""


def refine_roi_by_content_and_text(page: fitz.Page, initial_roi: fitz.Rect, is_likely_table: bool) -> Optional[fitz.Rect]:
    roi = fitz.Rect(initial_roi)
    page_rect = page.rect
    if not roi.intersects(page_rect) or roi.is_empty:
        return None

    # Logic from your notebook for ROI refinement based on content and text trimming
    content_bbox = fitz.Rect()
    has_explicit_visual_content = False

    for path in page.get_drawings():
        if path['rect'].intersects(initial_roi) and path['rect'].width > 1 and path['rect'].height > 1:
            content_bbox.include_rect(path['rect'])
            has_explicit_visual_content = True

    for img_info in page.get_images(full=True):
        try:
            img_bbox_cand = page.get_image_bbox(img_info, transform=False)
            if img_bbox_cand.intersects(initial_roi):
                content_bbox.include_rect(img_bbox_cand)
                has_explicit_visual_content = True
        except Exception: # PyMuPDF can sometimes error on specific images
            continue

    if has_explicit_visual_content and not content_bbox.is_empty and content_bbox.width > 5 and content_bbox.height > 5:
        roi = content_bbox.intersect(initial_roi)
        if roi.is_empty or roi.width < 5 or roi.height < 5:
            roi = fitz.Rect(initial_roi)
    elif not is_likely_table and (initial_roi.width < settings.min_visual_width_pymupdf or initial_roi.height < settings.min_visual_height_pymupdf):
        return None # Too small if no explicit content and not a table
    else: # Keep initial_roi if no explicit content but it's a table or large enough drawing cluster
        roi = fitz.Rect(initial_roi)


    text_blocks = [fitz.Rect(b[:4]) for b in page.get_text("blocks", flags=0) if b[6] == 0] # Type 0 is text

    # Trim from top
    current_top_texts = sorted([tb for tb in text_blocks if tb.y1 < roi.y0 + 15 and (max(tb.x0, roi.x0) < min(tb.x1, roi.x1))], key=lambda x: x.y1, reverse=True)
    for tb in current_top_texts:
        if tb.get_area() > settings.text_block_min_area_for_obstruction:
            roi.y0 = tb.y1 + 2
            break
    roi.intersect(page_rect)
    if roi.is_empty or roi.width < settings.min_visual_width_pymupdf or roi.height < settings.min_visual_height_pymupdf:
        return None

    # Trim from bottom
    current_bottom_texts = sorted([tb for tb in text_blocks if tb.y0 > roi.y1 - 15 and (max(tb.x0, roi.x0) < min(tb.x1, roi.x1))], key=lambda x: x.y0)
    for tb in current_bottom_texts:
        if tb.get_area() > settings.text_block_min_area_for_obstruction:
            roi.y1 = tb.y0 - 2
            break

    roi.intersect(page_rect)
    return roi if roi.width >= settings.min_visual_width_pymupdf and roi.height >= settings.min_visual_height_pymupdf else None


# --- Main Extraction Logic (Adapted from your notebook's v11.5) ---
def extract_visual_elements_from_page(
    doc_fitz: fitz.Document,
    page_num: int, # 0-indexed
    pdf_id: str, # For unique file naming and metadata
    vlm_tracker: VLMUsageTracker,
    processed_captions_for_this_pdf: Set[str]
) -> List[Document]:
    """
    Extracts visual elements (figures, visual tables) from a single PDF page.
    Corresponds to `extract_visual_elements_pymupdf_v11_5` from your notebook.
    Saves images to the PDF-specific image directory.
    """
    page = doc_fitz.load_page(page_num)
    visual_docs: List[Document] = []
    # Use pdf_id to get the correct image save directory for this PDF
    pdf_image_save_dir = get_pdf_extracted_images_dir(pdf_id)

    # 1. Identify primitive visuals (rasters, drawing clusters)
    primitive_visuals = []
    for img_idx, img_info_fitz in enumerate(page.get_images(full=True)):
        try:
            bbox = page.get_image_bbox(img_info_fitz, transform=False) # Use original coordinates
            # Basic check for image validity/size as in your notebook
            img_bytes_check = doc_fitz.extract_image(img_info_fitz[0])["image"]
            if not img_bytes_check: continue
            pil_img_check = Image.open(io.BytesIO(img_bytes_check))
            if pil_img_check.width > settings.min_visual_width_pymupdf / 2 and pil_img_check.height > settings.min_visual_height_pymupdf / 2:
                primitive_visuals.append({"bbox": bbox, "type": "raster", "id": f"p{page_num}_raster{img_idx}", "raw_info": img_info_fitz})
        except Exception: # Handle potential errors in image extraction
            continue

    for dc_idx, dc_bbox in enumerate(get_pymupdf_drawing_clusters(page)):
        is_within_raster = any(r_info["type"] == "raster" and r_info["bbox"].contains(dc_bbox) for r_info in primitive_visuals)
        if not is_within_raster:
            primitive_visuals.append({"bbox": dc_bbox, "type": "drawing_cluster", "id": f"p{page_num}_draw{dc_idx}"})

    # 2. Find captions and associate with primitives
    captions_on_page = find_pymupdf_captions(page)
    caption_to_visual_parts_map = defaultdict(list) # Maps caption_id to list of visual parts + caption meta
    unassigned_primitives = list(primitive_visuals) # Make a copy to modify
    processed_primitive_indices_this_page = set()

    for cap_info in captions_on_page:
        if cap_info["id"] in processed_captions_for_this_pdf:
            # print(f"Debug: Skipping already processed caption: {cap_info['id']} on page {page_num+1}")
            continue

        found_visual_for_caption = False
        potential_visuals_for_caption = [] # Tuples of (primitive_visual_dict, original_index)

        cap_info_bbox_center_y = (cap_info["bbox"].y0 + cap_info["bbox"].y1) / 2

        for i, prim_vis in enumerate(unassigned_primitives):
            if i in processed_primitive_indices_this_page:
                continue

            prim_vis_bbox_center_y = (prim_vis["bbox"].y0 + prim_vis["bbox"].y1) / 2
            v_dist = abs(prim_vis_bbox_center_y - cap_info_bbox_center_y)

            # Horizontal overlap calculation (from your notebook)
            h_overlap = max(0, min(prim_vis["bbox"].x1, cap_info["bbox"].x1) - max(prim_vis["bbox"].x0, cap_info["bbox"].x0))
            h_overlap_ratio = 0
            if prim_vis["bbox"].width > 0 and cap_info["bbox"].width > 0 :
                 h_overlap_ratio_prim = h_overlap / prim_vis["bbox"].width
                 h_overlap_ratio_cap = h_overlap / cap_info["bbox"].width
                 h_overlap_ratio = max(h_overlap_ratio_prim, h_overlap_ratio_cap)


            # Positioning logic (from your notebook)
            is_correctly_positioned = False
            max_v_dist_figure = 200 # From notebook
            max_v_dist_table = 100  # From notebook
            if cap_info["type"] == "figure":
                if ((prim_vis_bbox_center_y < cap_info_bbox_center_y and v_dist < max_v_dist_figure) or (abs(v_dist) < 40)):
                    is_correctly_positioned = True
            elif cap_info["type"] == "table": # This part implies tables can be visual (e.g. image of a table)
                if ((prim_vis_bbox_center_y > cap_info_bbox_center_y and v_dist < max_v_dist_table) or (abs(v_dist) < 40)):
                    is_correctly_positioned = True

            if h_overlap_ratio > 0.2 and is_correctly_positioned:
                potential_visuals_for_caption.append((prim_vis, i))

        if potential_visuals_for_caption:
            potential_visuals_for_caption.sort(key=lambda x: (
                0 if cap_info["type"] == "figure" and x[0]["type"] == "drawing_cluster" else 1,
                abs((x[0]["bbox"].y0 + x[0]["bbox"].y1)/2 - cap_info_bbox_center_y),
                -(x[0]["bbox"].width * x[0]["bbox"].height)
            ))
            best_prim_vis, best_prim_idx = potential_visuals_for_caption[0]

            target_list = caption_to_visual_parts_map[cap_info["id"]]
            if not any(item.get("is_caption_meta") for item in target_list):
                 target_list.append({"is_caption_meta": True, "caption_obj": cap_info})
            target_list.append(best_prim_vis)
            processed_primitive_indices_this_page.add(best_prim_idx)
            found_visual_for_caption = True
            # print(f"Debug: Matched caption {cap_info['id']} with visual {best_prim_vis['id']} on page {page_num+1}")


        # Handle textual captions (no DIRECT visual primitive match) - including textual table content
        if not found_visual_for_caption and cap_info["id"] not in processed_captions_for_this_pdf:
            doc_type = "text_table_content" if cap_info["type"] == "table" else "text_figure_description"
            content_to_add = cap_info["text"] # Start with the caption text itself

            if cap_info["type"] == "table":
                table_body_content = find_table_content(page, cap_info["bbox"])
                if table_body_content:
                    # Combine caption with extracted table body for the document content
                    content_to_add = f"{cap_info['text']}\n\n{table_body_content}"
                    # print(f"Debug: Extracted content for textual table {cap_info['id']} on page {page_num+1}: {table_body_content[:100]}...")

            visual_docs.append(Document(
                page_content=content_to_add,
                metadata={
                    "source_pdf_id": pdf_id, "source_doc_name": os.path.basename(doc_fitz.name),
                    "page_number": page_num + 1, # 1-indexed for user
                    "type": doc_type, # Differentiates from image_description
                    "original_caption": cap_info["text"],
                    "caption_id": cap_info["id"],
                    "element_subtype": "textual_content_with_caption"
                }
            ))
            processed_captions_for_this_pdf.add(cap_info["id"])
            # print(f"Debug: Added textual content for {doc_type}: {cap_info['id']} on page {page_num+1}")


    # 3. Handle unassigned primitives (large uncaptioned visuals)
    for i, prim_vis in enumerate(unassigned_primitives):
        if i in processed_primitive_indices_this_page:
            continue
        # Check if large enough (from your notebook)
        if prim_vis["bbox"].width > settings.min_visual_width_pymupdf * 2 and prim_vis["bbox"].height > settings.min_visual_height_pymupdf * 2:
            caption_to_visual_parts_map[prim_vis["id"]].append(prim_vis) # Use primitive's own ID as key
            # print(f"Debug: Found uncaptioned large visual: {prim_vis['id']} on page {page_num+1}")

    # 4. Process semantic visual groups (render images, get VLM descriptions)
    for semantic_id, group_elements in caption_to_visual_parts_map.items():
        # If this semantic_id was already processed as a purely textual caption, AND
        # this current group doesn't have its own caption_meta (meaning it's an uncaptioned visual that
        # happened to have the same ID as a textual one, which is unlikely but a safeguard), skip.
        # OR, if it has caption_meta but the ID is already processed, it means we are trying to re-process a visual.
        if semantic_id in processed_captions_for_this_pdf and \
           any(item.get("is_caption_meta") for item in group_elements): #This means it's a captioned visual we might have already processed the caption text for.
            pass # Let it proceed to render the visual part if it exists.
        elif semantic_id in processed_captions_for_this_pdf and not any(item.get("is_caption_meta") for item in group_elements):
             # This is an uncaptioned visual whose ID might clash with an already processed caption text.
             # This shouldn't happen if IDs are unique (pX_drawY vs FigureZ).
             # If it IS an uncaptioned visual, and its ID is already in processed_captions, it means we already logged it somehow.
             # This check might be too aggressive, let's refine.
             # If it's an uncaptioned visual, it should not be in `processed_captions_for_this_pdf` unless it was added as such.
             # The intent is to not re-render if `semantic_id` (which is caption_id or primitive_id) is fully done.
             # Let's ensure we only skip if the *visual rendering* for this ID was done.
             # The `processed_captions_for_this_pdf.add(semantic_id)` at the end of this loop handles this.
            if semantic_id in processed_captions_for_this_pdf: # If this semantic_id was already added (either as text or image), skip.
                continue

        visual_parts_data_list = [item for item in group_elements if not item.get("is_caption_meta")]
        caption_meta_item = next((item for item in group_elements if item.get("is_caption_meta")), None)

        if not visual_parts_data_list: # If no actual visual parts, skip (already handled as textual)
            continue

        composite_render_roi = fitz.Rect()
        final_caption_text = f"Uncaptioned Visual ({semantic_id})"
        final_semantic_type = "figure" # Default for uncaptioned

        if caption_meta_item:
            current_caption_info = caption_meta_item["caption_obj"]
            final_caption_text = current_caption_info["text"]
            final_semantic_type = current_caption_info["type"]
            composite_render_roi.include_rect(current_caption_info["bbox"]) # Include caption bbox in initial ROI

        for part_data in visual_parts_data_list:
            composite_render_roi.include_rect(part_data["bbox"])

        composite_render_roi.intersect(page.rect)

        if composite_render_roi.is_empty or composite_render_roi.width < settings.min_visual_width_pymupdf or composite_render_roi.height < settings.min_visual_height_pymupdf:
            continue

        is_table_visual = final_semantic_type == "table"
        refined_roi = refine_roi_by_content_and_text(page, composite_render_roi, is_table_visual)

        if not refined_roi:
            continue

        try:
            pix = page.get_pixmap(clip=refined_roi, dpi=settings.render_dpi_pymupdf, alpha=False)
            # Sanitize semantic_id for filename
            filename_part = re.sub(r'[^\w.-]', '_', str(semantic_id))
            base_filename = f"page{page_num+1}_SEMANTIC_{filename_part}.png"
            image_save_path = os.path.join(pdf_image_save_dir, base_filename)

            # Ensure unique filename if clashes occur (though semantic_id should be fairly unique per page)
            counter = 0
            while os.path.exists(image_save_path):
                counter += 1
                image_save_path = os.path.join(pdf_image_save_dir, f"page{page_num+1}_SEMANTIC_{filename_part}_{counter}.png")
            pix.save(image_save_path)

            vlm_description = generate_detailed_image_description(
                image_save_path,
                f"visual element ({final_semantic_type})",
                vlm_tracker
            )

            # The image_path stored should be relative to a common serving point if served via API
            # For now, store the full path as generated, API can make it relative.
            # Or, store path relative to 'get_pdf_specific_data_dir(pdf_id)'
            relative_image_path = os.path.relpath(image_save_path, get_pdf_specific_data_dir(pdf_id))


            visual_docs.append(Document(
                page_content=vlm_description, # VLM description or fallback
                metadata={
                    "source_pdf_id": pdf_id, "source_doc_name": os.path.basename(doc_fitz.name),
                    "page_number": page_num + 1,
                    "type": "image_description", # This type is for VLM descriptions of rendered images
                    "image_path_on_server": image_save_path, # Full path for server-side access
                    "image_path_relative_to_pdf_data": relative_image_path, # For API to construct URL
                    "original_caption": final_caption_text,
                    "element_subtype": f"pymupdf_semantic_{final_semantic_type}",
                    "caption_id": semantic_id # This is the figure/table ID or primitive ID
                }
            ))
            processed_captions_for_this_pdf.add(semantic_id) # Mark this semantic ID as processed (rendered)
            # print(f"Debug: Rendered and described visual: {semantic_id} on page {page_num+1} to {os.path.basename(image_save_path)}")

        except Exception as e:
            print(f"PyMuPDF: Error rendering or describing semantic visual '{semantic_id}' on page {page_num+1}: {e}")

    return visual_docs


def process_single_pdf_custom(
    pdf_file_path: str,
    pdf_id: str # Unique ID for this PDF processing job
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Processes a single PDF file using logic adapted from `extract_elements_from_file_hybrid_v11`.
    Returns a list of Langchain Documents and a metadata summary for the PDF.
    """
    print(f"\nProcessing PDF (ID: {pdf_id}): {os.path.basename(pdf_file_path)}")

    vlm_tracker = VLMUsageTracker(limit=settings.max_elements_for_vlm_description_per_pdf)
    processed_captions_globally_for_this_pdf = set() # Tracks caption IDs processed across all pages of this PDF

    final_documents: List[Document] = []
    doc_metadata_summary = {
        "pdf_id": pdf_id,
        "original_filename": os.path.basename(pdf_file_path),
        "title": "N/A", "abstract": "N/A",
        "is_scanned": False,
        "page_count": 0
    }

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300) # From your notebook

    try:
        doc_fitz = fitz.open(pdf_file_path)
        doc_metadata_summary["page_count"] = len(doc_fitz)

        # Step 1a: Extract Title/Abstract (from your notebook's logic)
        if len(doc_fitz) > 0:
            first_page_meta = doc_fitz.load_page(0)
            page_height = first_page_meta.rect.height
            title_candidates = []
            blocks = first_page_meta.get_text("dict", flags=fitz.TEXTFLAGS_DICT)["blocks"] # Simplified flags
            max_overall_font_size = 0
            eligible_title_blocks_data = []

            for block in blocks:
                if block['type'] == 0 and 0.03 * page_height < block['bbox'][1] < 0.40 * page_height:
                    for line in block['lines']:
                        for span in line['spans']:
                            if span['size'] > max_overall_font_size + 1e-3:
                                max_overall_font_size = span['size']
                            eligible_title_blocks_data.append({
                                'text': span['text'], 'size': span['size'],
                                'block_text': clean_parsed_text("".join(s['text'] for li in block['lines'] for s in li['spans']))
                            })
            pymupdf_title = ""
            if max_overall_font_size > 13: # Heuristic font size for title
                for data in eligible_title_blocks_data:
                    if abs(data['size'] - max_overall_font_size) < 1.0 and \
                       not re.match(r"arxiv:[\d\.]+", data['block_text'].lower(), re.I) and \
                       not re.search(r"\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}", data['block_text'].lower(), re.I) and \
                       len(data['text'].strip()) > 2:
                        title_candidates.append(data['text'].strip())

            if title_candidates: # Simpler title logic from common patterns
                # Assuming title is usually one of the first few largest font unique lines
                pymupdf_title = clean_parsed_text(" ".join(list(dict.fromkeys(title_candidates))[:2])) # Take first 2 unique candidates

            if pymupdf_title:
                doc_metadata_summary["title"] = pymupdf_title
                final_documents.append(Document(
                    page_content=f"DOCUMENT TITLE: {pymupdf_title}",
                    metadata={"source_pdf_id": pdf_id, "source_doc_name": os.path.basename(pdf_file_path), "page_number": 1, "type": "title_summary", "importance": "critical"}))

            # Abstract extraction (from your notebook)
            abs_text_sorted = first_page_meta.get_text("text", sort=True)
            abs_match = re.search(r"Abstract\s*\n(.*?)(?=\n\s*\n(1\.(?:\s|\n)|I\.(?:\s|\n)|Keywords|Introduction|Motivation)\b)", abs_text_sorted, re.I | re.S)
            if abs_match:
                pymupdf_abstract = clean_parsed_text(abs_match.group(1))
                if pymupdf_abstract and len(pymupdf_abstract) > 50:
                    doc_metadata_summary["abstract"] = pymupdf_abstract
                    final_documents.append(Document(
                        page_content=f"DOCUMENT ABSTRACT: {pymupdf_abstract}",
                        metadata={"source_pdf_id": pdf_id, "source_doc_name": os.path.basename(pdf_file_path), "page_number": 1, "type": "abstract_summary", "importance": "critical"}))

        # Check if PDF is scanned (from your notebook)
        digitally_extracted_text_sample = "".join(doc_fitz.load_page(i).get_text("text", sort=True) for i in range(min(3, len(doc_fitz))))
        if len(clean_parsed_text(digitally_extracted_text_sample)) < settings.min_ocr_text_length_for_scanned_pdf * min(3, len(doc_fitz)):
            doc_metadata_summary["is_scanned"] = True
            print(f"PDF '{os.path.basename(pdf_file_path)}' (ID: {pdf_id}) appears scanned. OCR will be used.")

        # Process each page
        for page_num in tqdm(range(len(doc_fitz)), desc=f"Processing pages for PDF {pdf_id}"):
            page = doc_fitz.load_page(page_num)
            page_text_for_chunking = ""
            parser_source_log = "pymupdf_digital"

            if doc_metadata_summary["is_scanned"]:
                parser_source_log = "pymupdf_ocr"
                # Save page as image for OCR
                temp_ocr_image_dir = get_pdf_extracted_images_dir(pdf_id) # Save temp ocr images here too
                ocr_img_path = os.path.join(temp_ocr_image_dir, f"page{page_num+1}_ocr_temp.png")
                pix = page.get_pixmap(dpi=settings.ocr_dpi, alpha=False)
                pix.save(ocr_img_path)
                pix = None # Release Pixmap memory

                page_text_for_chunking = ocr_image_to_text_unstructured(ocr_img_path)

                # Optional: Get VLM description for the full scanned page
                if vlm_tracker.can_use_vlm():
                    page_img_desc = generate_detailed_image_description(ocr_img_path, "scanned page", vlm_tracker)
                    relative_ocr_img_path = os.path.relpath(ocr_img_path, get_pdf_specific_data_dir(pdf_id))
                    final_documents.append(Document(
                        page_content=page_img_desc,
                        metadata={
                            "source_pdf_id": pdf_id, "source_doc_name": os.path.basename(pdf_file_path),
                            "page_number": page_num + 1, "type": "image_description",
                            "image_path_on_server": ocr_img_path,
                            "image_path_relative_to_pdf_data": relative_ocr_img_path,
                            "original_caption": f"Full Scanned Page {page_num+1}",
                            "element_subtype": "scanned_page_full_vlm", "caption_id": f"ScannedPage{page_num+1}_FullVLM"
                        }))
                # os.remove(ocr_img_path) # Clean up temp OCR image? Or keep if path is stored. Keep for now.
            else: # Digitally native PDF page
                page_text_for_chunking = page.get_text("text", sort=True)
                # Extract figures, visual tables etc.
                visual_docs_from_page = extract_visual_elements_from_page(
                    doc_fitz, page_num, pdf_id, vlm_tracker, processed_captions_globally_for_this_pdf
                )
                final_documents.extend(visual_docs_from_page)

            cleaned_page_text = clean_parsed_text(page_text_for_chunking)
            if cleaned_page_text:
                text_to_process = cleaned_page_text
                # Add title context if it's the first page and title wasn't in text (from notebook)
                if page_num == 0 and doc_metadata_summary["title"] != "N/A":
                    if not cleaned_page_text.strip().lower().startswith(doc_metadata_summary["title"].strip().lower()[:30]): # Check prefix
                         text_to_process = f"Document Title Context: {doc_metadata_summary['title']}\n\n{cleaned_page_text}"

                page_splits = text_splitter.create_documents(
                    [text_to_process],
                    metadatas=[{"source_pdf_id": pdf_id, "source_doc_name": os.path.basename(pdf_file_path), "page_number": page_num + 1, "type": "text_chunk", "parser_source": parser_source_log}]
                )
                final_documents.extend(page_splits)
        doc_fitz.close()

    except Exception as e:
        print(f"Error processing PDF {pdf_file_path} (ID: {pdf_id}): {e}")
        # Ensure Fitz document is closed if open
        if 'doc_fitz' in locals() and doc_fitz.is_open:
            doc_fitz.close()
        # Optionally, re-raise or handle more gracefully
        # For now, we return what we have and the error will be logged.

    # Deduplication logic from your notebook
    final_deduped_documents: List[Document] = []
    seen_keys = set()
    for doc_item in final_documents:
        key_parts = [
            doc_item.metadata.get("type"),
            doc_item.metadata.get("page_number"),
            doc_item.metadata.get("source_pdf_id")
        ]
        content_key_part = doc_item.page_content[:150]
        if doc_item.metadata.get("type") == "image_description":
            content_key_part = doc_item.metadata.get("image_path_on_server", content_key_part)
        elif doc_item.metadata.get("type") in ["text_table_content", "text_figure_description"]:
            content_key_part = doc_item.metadata.get("caption_id", content_key_part)

        key_parts.append(content_key_part)
        key = tuple(key_parts)
        if key not in seen_keys:
            final_deduped_documents.append(doc_item)
            seen_keys.add(key)

    print(f"Finished PDF (ID: {pdf_id}): Extracted {len(final_deduped_documents)} unique elements.")
    return final_deduped_documents, doc_metadata_summary