# vtierp_project_custom/app/services/pdf_processor.py
import logging
import fitz  # PyMuPDF
import os
import shutil
import uuid
import io
import re
import math
from PIL import Image
# from tqdm import tqdm # Removed for FastAPI
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any, Set

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import settings, get_pdf_extracted_images_dir, get_pdf_specific_data_dir
from app.dependencies_config.llm_config import get_aux_llm
from app.services.utils import clean_parsed_text, image_to_base64, ocr_image_to_text_unstructured

logger = logging.getLogger(__name__)

# --- VLM Usage Tracker ---
class VLMUsageTracker:
    def __init__(self, limit: int): self.count = 0; self.limit = limit
    def increment(self): self.count += 1
    def can_use_vlm(self) -> bool: return self.count < self.limit

# --- generate_detailed_image_description Function ---
def generate_detailed_image_description(image_path: str, element_type: str, vlm_tracker: VLMUsageTracker) -> str:
    aux_llm = get_aux_llm()
    if not aux_llm: logger.error(f"VLM (aux_llm) not initialized for {os.path.basename(image_path)}."); return f"VLM_NOT_INIT_{os.path.basename(image_path)}"
    if not vlm_tracker.can_use_vlm(): return f"VLM_LIMIT_REACHED_{os.path.basename(image_path)}"
    try:
        img_base64, mime_type = image_to_base64(image_path)
        if not img_base64 or not mime_type: logger.warning(f"Could not load/convert image {os.path.basename(image_path)} for VLM."); return f"VLM_LOAD_FAIL_{os.path.basename(image_path)}"
        prompt_text = (f"Expert document analyst: Analyze this {element_type} from a research paper. Describe its key visual components, structure, any text present within it, and its apparent purpose or the information it conveys. For Tables (if image): describe columns, data types, and notable trends. For Diagrams/Charts: describe type, axes, trends, and flow. Provide a concise, comprehensive summary suitable for Q&A.")
        from langchain_core.messages import HumanMessage
        message = HumanMessage(content=[ {"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}}])
        response = aux_llm.invoke([message]); vlm_tracker.increment()
        return clean_parsed_text(response.content)
    except Exception as e: logger.error(f"Error in VLM for {os.path.basename(image_path)}: {e}", exc_info=True); return f"VLM_ERROR_{os.path.basename(image_path)}"

# --- PyMuPDF Helper Functions ---
def find_pymupdf_captions(page: fitz.Page) -> List[Dict]:
    captions = []; blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES & ~fitz.TEXT_PRESERVE_IMAGES)["blocks"]
    for block in blocks:
        if block['type'] == 0:
            bt = "".join(s['text'] for l_val in block['lines'] for s in l_val['spans']); ct = clean_parsed_text(bt)
            m = re.match(r"^(Figure|Fig\.?|Table)\s+([A-Za-z0-9]+\.?\d*)\s*[:\.]?\s*", ct, re.I)
            if m: c_type = "figure" if "fig" in m.group(1).lower() else "table"; c_id = f"{c_type.capitalize()}{m.group(2)}"; captions.append({"text": ct, "bbox": fitz.Rect(block['bbox']), "type": c_type, "page_num": page.number, "id": c_id})
    return captions

# --- CORRECTED get_pymupdf_drawing_clusters ---
def get_pymupdf_drawing_clusters(page: fitz.Page) -> List[fitz.Rect]:
    drawings = page.get_drawings()
    path_rect_tuples = [
        (d['rect'].x0, d['rect'].y0, d['rect'].x1, d['rect'].y1)
        for d in drawings 
        if d['rect'].width > 1 and d['rect'].height > 1 and \
           (d.get('type') != 'fill' or d.get('color') is not None)
    ]
    if not path_rect_tuples: return []
    path_rects = sorted([fitz.Rect(r) for r in path_rect_tuples], key=lambda r_obj: (r_obj.y0, r_obj.x0))
    merged_rects: List[fitz.Rect] = []
    page_diagonal = math.sqrt(page.rect.width**2 + page.rect.height**2) if page.rect.width > 0 and page.rect.height > 0 else 1000.0
    max_merge_distance = page_diagonal * settings.drawing_cluster_max_dist_factor
    for current_rect in path_rects:
        if not merged_rects:
            merged_rects.append(current_rect)
        else:
            last_merged_rect = merged_rects[-1]
            expanded_last_rect = fitz.Rect(last_merged_rect) + \
                                 (-max_merge_distance, -max_merge_distance, 
                                  max_merge_distance, max_merge_distance)
            if current_rect.intersects(expanded_last_rect):
                merged_rects[-1] = last_merged_rect | current_rect
            else:
                merged_rects.append(current_rect)
    return [r_item for r_item in merged_rects if r_item.width > settings.min_visual_width_pymupdf / 2 and r_item.height > settings.min_visual_height_pymupdf / 2]

# This function goes into app/services/pdf_processor.py
# Ensure fitz, Optional, settings, and logger (if used inside) are available.

def refine_roi_by_content_and_text(page: fitz.Page, initial_roi: fitz.Rect, is_likely_table: bool) -> Optional[fitz.Rect]:
    """
    Refines the bounding box (ROI) of a visual element by considering its actual visual content
    (drawings, images) and trimming surrounding text.
    """
    # logger.debug(f"Refining ROI for initial_roi: {initial_roi}, is_table: {is_likely_table}")
    current_roi_base = fitz.Rect(initial_roi) # Start with a copy of the initial ROI
    page_rect = page.rect

    if not current_roi_base.intersects(page_rect) or current_roi_base.is_empty:
        # logger.debug("Initial ROI does not intersect page or is empty.")
        return None

    content_bbox = fitz.Rect()
    has_explicit_visual_content = False

    # 1. Accumulate bounds from drawing paths within the initial_roi
    for path in page.get_drawings():
        path_rect = fitz.Rect(path['rect']) # Ensure it's a Rect object
        if path_rect.intersects(current_roi_base) and path_rect.width > 1 and path_rect.height > 1:
            content_bbox.include_rect(path_rect)
            has_explicit_visual_content = True

    # 2. Accumulate bounds from images within the initial_roi
    for img_info in page.get_images(full=True):
        try:  # ***** START OF THE CRITICAL TRY BLOCK *****
            img_bbox_candidate = page.get_image_bbox(img_info, transform=False)
            if img_bbox_candidate.intersects(current_roi_base): # Check intersection with current_roi_base
                content_bbox.include_rect(img_bbox_candidate)
                has_explicit_visual_content = True
        except Exception as e:  # ***** MATCHING EXCEPT BLOCK *****
            # logger.debug(f"Could not get bbox for an image in refine_roi or other image error: {e}")
            continue # Skip to the next image if there's an error with this one

    # 3. Decide the working ROI based on content found
    if has_explicit_visual_content and not content_bbox.is_empty and content_bbox.width > 5 and content_bbox.height > 5:
        # Intersect the accumulated content_bbox with the initial_roi to constrain it
        working_roi = content_bbox.intersect(current_roi_base)
        if working_roi.is_empty or working_roi.width < 5 or working_roi.height < 5:
            # If intersection is too small, fall back to current_roi_base (which is initial_roi at this point)
            working_roi = fitz.Rect(current_roi_base)
    elif not is_likely_table and (current_roi_base.width < settings.min_visual_width_pymupdf or current_roi_base.height < settings.min_visual_height_pymupdf):
        # If no explicit visual content, not a table, and initial ROI is already too small, bail out.
        # logger.debug("No explicit visual content, not a table, and initial ROI too small.")
        return None
    else:
        # Use initial_roi (via current_roi_base) as the working_roi if no visual content but it's a table or large enough.
        working_roi = fitz.Rect(current_roi_base)

    # Ensure working_roi is still valid before text trimming
    if working_roi.is_empty or working_roi.width < settings.min_visual_width_pymupdf / 2 or working_roi.height < settings.min_visual_height_pymupdf / 2:
        # logger.debug(f"Working ROI too small before text trimming: {working_roi}")
        return None

    # 4. Trim based on surrounding text blocks
    final_trimmed_roi = fitz.Rect(working_roi) # Start trimming from this base
    text_blocks = [fitz.Rect(b[:4]) for b in page.get_text("blocks", flags=0) if b[6] == 0]

    # Trim from top
    text_above = [tb for tb in text_blocks if tb.y1 < final_trimmed_roi.y0 + 10 and tb.x1 > final_trimmed_roi.x0 - 5 and tb.x0 < final_trimmed_roi.x1 + 5] # Looser x overlap
    if text_above:
        lowest_text_above = max(text_above, key=lambda r_item: r_item.y1)
        if lowest_text_above.get_area() > settings.text_block_min_area_for_obstruction:
            if lowest_text_above.y1 < final_trimmed_roi.y1 - 5: # Ensure trimming doesn't invert rect
                 final_trimmed_roi.y0 = lowest_text_above.y1 + 2
            # else: logger.debug("Skipped top trim to avoid invalid rect.")


    # Trim from bottom
    text_below = [tb for tb in text_blocks if tb.y0 > final_trimmed_roi.y1 - 10 and tb.x1 > final_trimmed_roi.x0 - 5 and tb.x0 < final_trimmed_roi.x1 + 5] # Looser x overlap
    if text_below:
        highest_text_below = min(text_below, key=lambda r_item: r_item.y0)
        if highest_text_below.get_area() > settings.text_block_min_area_for_obstruction:
            if highest_text_below.y0 > final_trimmed_roi.y0 + 5: # Ensure trimming doesn't invert rect
                final_trimmed_roi.y1 = highest_text_below.y0 - 2
            # else: logger.debug("Skipped bottom trim to avoid invalid rect.")


    final_trimmed_roi.intersect(page_rect) # Ensure it's within page bounds

    if final_trimmed_roi.is_empty or \
       final_trimmed_roi.width < settings.min_visual_width_pymupdf or \
       final_trimmed_roi.height < settings.min_visual_height_pymupdf:
        # logger.debug(f"ROI too small after text trimming: {final_trimmed_roi}")
        return None

    # logger.debug(f"Refined ROI: {final_trimmed_roi}")
    return final_trimmed_roi





def find_table_content(page: fitz.Page, table_caption_info: Dict) -> Tuple[str, Optional[str]]:
    caption_bbox = fitz.Rect(table_caption_info["bbox"]); caption_id = table_caption_info["id"]; caption_text = table_caption_info["text"]
    raw_table_text_lines = []; markdown_table_str = None
    logger.debug(f"TABLE EXTRACTION: Attempting for '{caption_id}' on page {page.number + 1}, caption_bbox: {caption_bbox}")
    search_y_start = caption_bbox.y0 - 5 ; search_y_end = page.rect.height - 10
    search_x_start = page.rect.x0 + 5; search_x_end = page.rect.x1 - 5
    min_y_of_next_caption = page.rect.height
    for oc in find_pymupdf_captions(page):
        if oc["id"] != caption_id and fitz.Rect(oc["bbox"]).y0 > caption_bbox.y1: min_y_of_next_caption = min(min_y_of_next_caption, fitz.Rect(oc["bbox"]).y0)
    search_y_end_effective = min(search_y_end, min_y_of_next_caption - 2)
    search_rect_effective = fitz.Rect(search_x_start, search_y_start, search_x_end, search_y_end_effective)
    words_in_search_area = page.get_text("words", clip=search_rect_effective, sort=False)
    if not words_in_search_area: logger.debug(f"TABLE EXTRACTION: No words in search area for '{caption_id}'."); return "", None
    lines_dict = defaultdict(list); y_tolerance = 3.5
    for w_idx, w in enumerate(words_in_search_area):
        y_center = (w[1] + w[3]) / 2.0; found_line_key = None
        for key_y in lines_dict.keys():
            if abs(y_center - key_y) < y_tolerance: found_line_key = key_y; break
        if found_line_key is None: found_line_key = y_center
        lines_dict[found_line_key].append(w)
    sorted_line_keys = sorted(lines_dict.keys()); parsed_rows_for_markdown = []
    for y_key in sorted_line_keys:
        current_line_words = sorted(lines_dict[y_key], key=lambda w_item: w_item[0])
        if not current_line_words: continue
        line_text_raw = " ".join([word_item[4] for word_item in current_line_words])
        line_bbox = fitz.Rect(); [line_bbox.include_rect(fitz.Rect(w[:4])) for w in current_line_words]
        normalized_line_text = re.sub(r'\s+', ' ', line_text_raw.strip().lower()); normalized_caption_text = re.sub(r'\s+', ' ', caption_text.strip().lower())
        if line_bbox.y0 < caption_bbox.y1 + 10 and (normalized_line_text == normalized_caption_text or normalized_caption_text.startswith(normalized_line_text) or normalized_line_text.startswith(normalized_caption_text[:max(15,len(normalized_caption_text)//2 )]) ):
            if line_bbox.intersects(caption_bbox) or abs(line_bbox.y0 - caption_bbox.y0) < (caption_bbox.height + 5) : logger.debug(f"TABLE EXTRACTION: Skipping line as part of caption '{caption_id}': '{line_text_raw}'"); continue
        if line_bbox.y0 > caption_bbox.y1 + 200 and len(line_text_raw.split()) < 5: continue
        raw_table_text_lines.append(line_text_raw)
        row_cells = []; current_cell_text = current_line_words[0][4]
        avg_char_width_approx = sum((cw[2]-cw[0])/len(cw[4]) for cw in current_line_words if len(cw[4])>0 and (cw[2]-cw[0]) > 0) / len([cw for cw in current_line_words if len(cw[4])>0 and (cw[2]-cw[0])>0]) if any(len(cw[4])>0 and (cw[2]-cw[0])>0 for cw in current_line_words) else 5.0
        col_sep_threshold = avg_char_width_approx * 2.0
        for i in range(1, len(current_line_words)):
            gap = current_line_words[i][0] - current_line_words[i-1][2]
            if gap > col_sep_threshold: row_cells.append(current_cell_text.strip()); current_cell_text = current_line_words[i][4]
            else: current_cell_text += " " + current_line_words[i][4]
        row_cells.append(current_cell_text.strip()); parsed_rows_for_markdown.append([cell for cell in row_cells if cell])
    raw_table_text = "\n".join(raw_table_text_lines).strip()
    if not raw_table_text: logger.warning(f"TABLE EXTRACTION: No raw text body for table '{caption_id}'."); return "", None
    if parsed_rows_for_markdown:
        col_counts = defaultdict(int)
        for r_data in parsed_rows_for_markdown: col_counts[len(r_data)] += 1
        if not col_counts: return raw_table_text, None
        num_cols_candidate = max(col_counts, key=col_counts.get) if col_counts else 0
        filtered_parsed_rows = [row for row in parsed_rows_for_markdown if abs(len(row) - num_cols_candidate) <= 1 and len(row) > 0]
        if not filtered_parsed_rows : logger.info(f"TABLE EXTRACTION: No rows after col count filtering for '{caption_id}'."); return raw_table_text, None
        col_counts_filtered = defaultdict(int)
        for r_val in filtered_parsed_rows: col_counts_filtered[len(r_val)] += 1
        num_cols = max(col_counts_filtered, key=col_counts_filtered.get) if col_counts_filtered else 0
        if num_cols > 1:
            md_lines = []; header_row_idx = -1
            for i_r, r_r_val in enumerate(filtered_parsed_rows):
                if len(r_r_val) == num_cols: header_row_idx = i_r; break
            if header_row_idx == -1 : logger.warning(f"Could not find header for MD table '{caption_id}'."); return raw_table_text, None
            header_cells = list(filtered_parsed_rows[header_row_idx]); header_cells.extend([""] * (num_cols - len(header_cells)))
            md_lines.append("| " + " | ".join(h.replace("|", "\\|") for h in header_cells) + " |")
            md_lines.append("|" + " :-- |" * num_cols)
            for row_cells_list_val in filtered_parsed_rows[header_row_idx + 1:]:
                if len(row_cells_list_val) == num_cols:
                    row_cells_mutable = list(row_cells_list_val); md_lines.append("| " + " | ".join(rc.replace("|", "\\|") for rc in row_cells_mutable) + " |")
            if len(md_lines) > 2: markdown_table_str = "\n".join(md_lines); logger.info(f"Generated Markdown for table '{caption_id}'.")
            else: logger.info(f"Not enough structured MD rows for table '{caption_id}'.")
        else: logger.info(f"Not enough columns (num_cols={num_cols}) for MD for table '{caption_id}'.")
    return raw_table_text, markdown_table_str

# --- extract_visual_elements_from_page ---
def extract_visual_elements_from_page(
    doc_fitz: fitz.Document, page_num: int, pdf_id: str,
    vlm_tracker: VLMUsageTracker, processed_captions_for_this_pdf: Set[str]
) -> List[Document]:
    page = doc_fitz.load_page(page_num); visual_docs: List[Document] = []
    pdf_image_save_dir = get_pdf_extracted_images_dir(pdf_id); os.makedirs(pdf_image_save_dir, exist_ok=True)
    primitive_visuals = []
    for img_idx, img_info_fitz in enumerate(page.get_images(full=True)):
        try:
            bbox = page.get_image_bbox(img_info_fitz, transform=False); img_bytes_check = doc_fitz.extract_image(img_info_fitz[0])["image"]
            if not img_bytes_check: continue
            pil_img_check = Image.open(io.BytesIO(img_bytes_check))
            if pil_img_check.width > settings.min_visual_width_pymupdf / 2 and pil_img_check.height > settings.min_visual_height_pymupdf / 2:
                primitive_visuals.append({"bbox": bbox, "type": "raster", "id": f"p{page_num}_raster{img_idx}", "raw_info": img_info_fitz})
        except Exception as e_img: logger.debug(f"Skipping image due to error: {e_img}")
    for dc_idx, dc_bbox in enumerate(get_pymupdf_drawing_clusters(page)):
        if not any(r_info["type"] == "raster" and r_info["bbox"].contains(dc_bbox) for r_info in primitive_visuals):
            primitive_visuals.append({"bbox": dc_bbox, "type": "drawing_cluster", "id": f"p{page_num}_draw{dc_idx}"})
    captions_on_page = find_pymupdf_captions(page); caption_to_visual_parts_map = defaultdict(list)
    unassigned_primitives = list(primitive_visuals); processed_primitive_indices_this_page = set()
    for cap_info in captions_on_page:
        if cap_info["id"] in processed_captions_for_this_pdf: continue
        found_visual_for_caption = False; potential_visuals_for_caption = []
        cap_info_bbox_center_y = (cap_info["bbox"].y0 + cap_info["bbox"].y1) / 2
        for i, prim_vis in enumerate(unassigned_primitives):
            if i in processed_primitive_indices_this_page: continue
            prim_vis_bbox_center_y = (prim_vis["bbox"].y0 + prim_vis["bbox"].y1) / 2; v_dist = abs(prim_vis_bbox_center_y - cap_info_bbox_center_y)
            h_overlap = max(0, min(prim_vis["bbox"].x1, cap_info["bbox"].x1) - max(prim_vis["bbox"].x0, cap_info["bbox"].x0)); h_overlap_ratio = 0
            if prim_vis["bbox"].width > 0 and cap_info["bbox"].width > 0 : h_overlap_ratio_prim = h_overlap / prim_vis["bbox"].width; h_overlap_ratio_cap = h_overlap / cap_info["bbox"].width; h_overlap_ratio = max(h_overlap_ratio_prim, h_overlap_ratio_cap)
            is_correctly_positioned = False; max_v_dist_figure = 200; max_v_dist_table = 100
            if cap_info["type"] == "figure":
                if ((prim_vis_bbox_center_y < cap_info_bbox_center_y and v_dist < max_v_dist_figure) or (abs(v_dist) < 40)): is_correctly_positioned = True
            elif cap_info["type"] == "table":
                if ((prim_vis_bbox_center_y > cap_info_bbox_center_y and v_dist < max_v_dist_table) or (abs(v_dist) < 40)): is_correctly_positioned = True
            if h_overlap_ratio > 0.2 and is_correctly_positioned: potential_visuals_for_caption.append((prim_vis, i))
        if potential_visuals_for_caption:
            potential_visuals_for_caption.sort(key=lambda x: (0 if cap_info["type"] == "figure" and x[0]["type"] == "drawing_cluster" else 1, abs((x[0]["bbox"].y0 + x[0]["bbox"].y1)/2 - cap_info_bbox_center_y), -(x[0]["bbox"].width * x[0]["bbox"].height)))
            best_prim_vis, best_prim_idx = potential_visuals_for_caption[0]
            target_list = caption_to_visual_parts_map[cap_info["id"]];
            if not any(item.get("is_caption_meta") for item in target_list): target_list.append({"is_caption_meta": True, "caption_obj": cap_info})
            target_list.append(best_prim_vis); processed_primitive_indices_this_page.add(best_prim_idx); found_visual_for_caption = True
        if not found_visual_for_caption and cap_info["id"] not in processed_captions_for_this_pdf:
            doc_type = "text_table_content" if cap_info["type"] == "table" else "text_figure_description"
            page_content_for_doc = f"IDENTITY: {cap_info['id']}\nCAPTION: {cap_info['text']}"
            table_as_markdown_output = None
            if cap_info["type"] == "table":
                raw_table_body, table_as_markdown_output = find_table_content(page, cap_info)
                if raw_table_body: page_content_for_doc += f"\n\n--- Extracted Table Text ---\n{raw_table_body}"
                else: page_content_for_doc += "\n\n--- Extracted Table Text ---\n(No distinct body text found for this table)"
            doc_metadata = {"source_pdf_id": pdf_id, "source_doc_name": os.path.basename(doc_fitz.name), "page_number": page_num + 1, "type": doc_type, "original_caption": cap_info["text"], "caption_id": cap_info["id"], "element_subtype": "textual_content_with_caption"}
            if table_as_markdown_output: doc_metadata["table_markdown"] = table_as_markdown_output
            visual_docs.append(Document(page_content=page_content_for_doc, metadata=doc_metadata))
            processed_captions_for_this_pdf.add(cap_info["id"])
    for i, prim_vis in enumerate(unassigned_primitives):
        if i in processed_primitive_indices_this_page: continue
        if prim_vis["bbox"].width > settings.min_visual_width_pymupdf * 2 and prim_vis["bbox"].height > settings.min_visual_height_pymupdf * 2: caption_to_visual_parts_map[prim_vis["id"]].append(prim_vis)
    for semantic_id, group_elements in caption_to_visual_parts_map.items():
        if semantic_id in processed_captions_for_this_pdf and any(item.get("is_caption_meta") for item in group_elements): pass
        elif semantic_id in processed_captions_for_this_pdf: continue
        visual_parts_data_list = [item for item in group_elements if not item.get("is_caption_meta")]; caption_meta_item = next((item for item in group_elements if item.get("is_caption_meta")), None)
        if not visual_parts_data_list: continue
        composite_render_roi = fitz.Rect(); final_caption_text = f"Uncaptioned Visual ({semantic_id})"; final_semantic_type = "figure"
        if caption_meta_item: current_caption_info = caption_meta_item["caption_obj"]; final_caption_text = current_caption_info["text"]; final_semantic_type = current_caption_info["type"]; composite_render_roi.include_rect(current_caption_info["bbox"])
        for part_data in visual_parts_data_list: composite_render_roi.include_rect(part_data["bbox"])
        composite_render_roi.intersect(page.rect)
        if composite_render_roi.is_empty or composite_render_roi.width < settings.min_visual_width_pymupdf or composite_render_roi.height < settings.min_visual_height_pymupdf: continue
        is_table_visual = final_semantic_type == "table"; refined_roi = refine_roi_by_content_and_text(page, composite_render_roi, is_table_visual)
        if not refined_roi: continue
        try:
            pix = page.get_pixmap(clip=refined_roi, dpi=settings.render_dpi_pymupdf, alpha=False)
            filename_part = re.sub(r'[^\w.-]', '_', str(semantic_id)); base_filename = f"page{page_num+1}_SEMANTIC_{filename_part}.png"; image_save_path = os.path.join(pdf_image_save_dir, base_filename); counter = 0
            while os.path.exists(image_save_path): counter += 1; image_save_path = os.path.join(pdf_image_save_dir, f"page{page_num+1}_SEMANTIC_{filename_part}_{counter}.png")
            pix.save(image_save_path); pix = None
            vlm_description = generate_detailed_image_description(image_save_path, f"visual element ({final_semantic_type})", vlm_tracker)
            relative_image_path = os.path.relpath(image_save_path, get_pdf_specific_data_dir(pdf_id))
            visual_docs.append(Document(page_content=vlm_description, metadata={"source_pdf_id": pdf_id, "source_doc_name": os.path.basename(doc_fitz.name), "page_number": page_num + 1, "type": "image_description", "image_path_on_server": image_save_path, "image_path_relative_to_pdf_data": relative_image_path, "original_caption": final_caption_text, "element_subtype": f"pymupdf_semantic_{final_semantic_type}", "caption_id": semantic_id}))
            processed_captions_for_this_pdf.add(semantic_id)
        except Exception as e: logger.error(f"Error in extract_visual_elements_from_page rendering/VLM for '{semantic_id}': {e}", exc_info=True)
    return visual_docs

# --- process_single_pdf_custom ---
def process_single_pdf_custom(pdf_file_path: str, pdf_id: str) -> Tuple[List[Document], Dict[str, Any]]:
    logger.info(f"Processing PDF (ID: {pdf_id}): {os.path.basename(pdf_file_path)}")
    vlm_tracker = VLMUsageTracker(limit=settings.max_elements_for_vlm_description_per_pdf)
    processed_captions_globally_for_this_pdf = set()
    final_documents: List[Document] = []
    doc_metadata_summary = {"pdf_id": pdf_id, "original_filename": os.path.basename(pdf_file_path), "title": "N/A", "abstract": "N/A", "is_scanned": False, "page_count": 0}
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    try:
        doc_fitz = fitz.open(pdf_file_path)
        doc_metadata_summary["page_count"] = len(doc_fitz)
        if len(doc_fitz) > 0:
            first_page = doc_fitz.load_page(0); page_width = first_page.rect.width; page_height = first_page.rect.height
            blocks_with_font_info = first_page.get_text("dict", flags=fitz.TEXTFLAGS_DICT)["blocks"]
            title_block_candidates = []
            for block in blocks_with_font_info:
                if block['type'] == 0:
                    bbox = fitz.Rect(block['bbox'])
                    if bbox.y1 < page_height * 0.40:
                        block_text_lines_val = []; current_block_max_font = 0; current_block_min_font = 1000
                        for line_val in block['lines']:
                            line_font_max_in_line = 0
                            for span_val in line_val['spans']: line_font_max_in_line = max(line_font_max_in_line, span_val['size']); current_block_min_font = min(current_block_min_font, span_val['size']); block_text_lines_val.append(span_val['text'])
                            current_block_max_font = max(current_block_max_font, line_font_max_in_line)
                        full_block_text_val = clean_parsed_text(" ".join(block_text_lines_val))
                        if not full_block_text_val or len(full_block_text_val) < 5: continue
                        if re.match(r"(^\d{1,2}(\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[a-z]*\.?)?\s+\d{4}$)", full_block_text_val.strip(), re.I): continue
                        if re.match(r"^arXiv:[\d\.\w\/v-]+$", full_block_text_val.strip(), re.I): continue
                        word_tokens = full_block_text_val.split()
                        if len(word_tokens) < 10 and any(kw in full_block_text_val.lower() for kw in ["university", "institute", "department", "college", "conference", "workshop", "proceedings", "journal"]): continue
                        if len(word_tokens) < 5 and sum(1 for word in word_tokens if word.isupper() or (word[0].isupper() and not word.islower())) > len(word_tokens) / 2 : continue
                        block_center_x_val = (bbox.x0 + bbox.x1) / 2; page_center_x_val = page_width / 2
                        is_centered_enough_val = abs(block_center_x_val - page_center_x_val) < page_width * 0.35; font_consistent_enough_val = (current_block_max_font - current_block_min_font) < 2.5
                        if is_centered_enough_val and font_consistent_enough_val and current_block_max_font > 11: title_block_candidates.append( (current_block_max_font, bbox.y0, bbox.x0, full_block_text_val, bbox) )
            extracted_title_val = ""
            title_lines_collected_val = [] # Initialize here
            if title_block_candidates:
                title_block_candidates.sort(key=lambda x_val: (-x_val[0], x_val[1]))
                if title_block_candidates:
                    best_font_size_val = title_block_candidates[0][0]; last_y1_val = 0
                    for cand_font_val, cand_y0_val, cand_x0_val, cand_text_val, cand_bbox_val in title_block_candidates:
                        if abs(cand_font_val - best_font_size_val) < 2.0: 
                            if not title_lines_collected_val or (cand_y0_val > last_y1_val - 15 and cand_y0_val < last_y1_val + cand_bbox_val.height * 2.5) :
                                if not (len(cand_text_val.split()) < 3 and cand_text_val.count('@') > 0) and not (len(cand_text_val.split()) < 4 and any(kw in cand_text_val.lower() for kw in ["university", "institute", "department"])):
                                    title_lines_collected_val.append(cand_text_val); last_y1_val = cand_bbox_val.y1
                                else: logger.debug(f"Filtered potential title line: {cand_text_val}");
                        elif title_lines_collected_val : break 
            if title_lines_collected_val: extracted_title_val = " ".join(title_lines_collected_val).strip()
            if len(extracted_title_val.split()) > 25: extracted_title_val = " ".join(extracted_title_val.split()[:25]) + "..."
            if extracted_title_val: doc_metadata_summary["title"] = extracted_title_val; logger.info(f"Extracted Title for {pdf_id}: {extracted_title_val}"); final_documents.append(Document(page_content=f"DOCUMENT TITLE: {extracted_title_val}", metadata={"source_pdf_id": pdf_id, "source_doc_name": os.path.basename(pdf_file_path), "page_number": 1, "type": "title_summary", "importance": "critical"}))
            else: logger.warning(f"Title extraction failed for {pdf_id}.")
            extracted_abstract_val = ""; abstract_found_val = False
            for block_val in blocks_with_font_info:
                if block_val['type'] == 0:
                    block_full_text_val_abs = clean_parsed_text("\n".join("".join(s['text'] for s in l['spans']) for l in block_val['lines']))
                    if not abstract_found_val and re.match(r"^\s*(Abstract|Summary)\b", block_full_text_val_abs, re.I): abstract_found_val = True; abstract_text_candidate_val = re.sub(r"^\s*(Abstract|Summary)\s*[:.\-\s]*\s*", "", block_full_text_val_abs, count=1, flags=re.I); extracted_abstract_val += abstract_text_candidate_val.strip() + "\n\n"; continue
                    if abstract_found_val:
                        if re.match(r"^\s*(Keywords|Index Terms|CCS Concepts|Introduction|1\s*\.|I\s*\.|\d{1,2}\s+[A-Z][a-z]+)", block_full_text_val_abs, re.I): abstract_found_val = False; break
                        if fitz.Rect(block_val['bbox']).y0 > page_height * 0.8: abstract_found_val = False; break
                        extracted_abstract_val += block_full_text_val_abs.strip() + "\n\n"
            extracted_abstract_val = extracted_abstract_val.strip()
            if len(extracted_abstract_val) > 50:
                if len(extracted_abstract_val.split()) > 500 : extracted_abstract_val = " ".join(extracted_abstract_val.split()[:500]) + "..."
                doc_metadata_summary["abstract"] = extracted_abstract_val; logger.info(f"Extracted Abstract for {pdf_id} (len {len(extracted_abstract_val)}): {extracted_abstract_val[:100]}..."); final_documents.append(Document(page_content=f"DOCUMENT ABSTRACT: {extracted_abstract_val}", metadata={"source_pdf_id": pdf_id, "source_doc_name": os.path.basename(pdf_file_path), "page_number": 1, "type": "abstract_summary", "importance": "critical"}))
            else: logger.warning(f"Abstract extraction failed for {pdf_id}.")
        digitally_extracted_text_sample_val = "".join(doc_fitz.load_page(i_val).get_text("text", sort=True) for i_val in range(min(3, len(doc_fitz))))
        if len(clean_parsed_text(digitally_extracted_text_sample_val)) < settings.min_ocr_text_length_for_scanned_pdf * min(3, len(doc_fitz)): doc_metadata_summary["is_scanned"] = True; logger.info(f"PDF '{os.path.basename(pdf_file_path)}' (ID: {pdf_id}) appears scanned.")
        for page_num_idx_val in range(len(doc_fitz)):
            page_val = doc_fitz.load_page(page_num_idx_val); page_text_for_chunking_val = ""; parser_source_log_val = "pymupdf_digital"
            if doc_metadata_summary["is_scanned"]:
                parser_source_log_val = "pymupdf_ocr"; temp_ocr_image_dir_val = get_pdf_extracted_images_dir(pdf_id); ocr_img_path_val = os.path.join(temp_ocr_image_dir_val, f"page{page_num_idx_val+1}_ocr_temp.png")
                pix_val = page_val.get_pixmap(dpi=settings.ocr_dpi, alpha=False); pix_val.save(ocr_img_path_val); pix_val = None
                page_text_for_chunking_val = ocr_image_to_text_unstructured(ocr_img_path_val)
                if vlm_tracker.can_use_vlm():
                    page_img_desc_val = generate_detailed_image_description(ocr_img_path_val, "scanned page", vlm_tracker); relative_ocr_img_path_val = os.path.relpath(ocr_img_path_val, get_pdf_specific_data_dir(pdf_id))
                    final_documents.append(Document(page_content=page_img_desc_val, metadata={"source_pdf_id": pdf_id, "source_doc_name": os.path.basename(pdf_file_path), "page_number": page_num_idx_val + 1, "type": "image_description", "image_path_on_server": ocr_img_path_val, "image_path_relative_to_pdf_data": relative_ocr_img_path_val, "original_caption": f"Full Scanned Page {page_num_idx_val+1}", "element_subtype": "scanned_page_full_vlm", "caption_id": f"ScannedPage{page_num_idx_val+1}_FullVLM"}))
            else:
                page_text_for_chunking_val = page_val.get_text("text", sort=True)
                visual_docs_from_page_val = extract_visual_elements_from_page(doc_fitz, page_num_idx_val, pdf_id, vlm_tracker, processed_captions_globally_for_this_pdf)
                final_documents.extend(visual_docs_from_page_val)
            cleaned_page_text_val = clean_parsed_text(page_text_for_chunking_val)
            if cleaned_page_text_val:
                text_to_process_val = cleaned_page_text_val
                if page_num_idx_val == 0 and doc_metadata_summary["title"] != "N/A":
                    if not cleaned_page_text_val.strip().lower().startswith(doc_metadata_summary["title"].strip().lower()[:min(30, len(doc_metadata_summary["title"]))]): text_to_process_val = f"Document Title Context: {doc_metadata_summary['title']}\n\n{cleaned_page_text_val}"
                page_splits_val = text_splitter.create_documents([text_to_process_val], metadatas=[{"source_pdf_id": pdf_id, "source_doc_name": os.path.basename(pdf_file_path), "page_number": page_num_idx_val + 1, "type": "text_chunk", "parser_source": parser_source_log_val}])
                final_documents.extend(page_splits_val)
        doc_fitz.close()
    except Exception as e: logger.error(f"Error processing PDF {pdf_file_path} (ID: {pdf_id}): {e}", exc_info=True);
    if 'doc_fitz' in locals() and hasattr(doc_fitz, 'is_open') and doc_fitz.is_open: doc_fitz.close()
    final_deduped_documents_val: List[Document] = []; seen_keys_val = set()
    for doc_item_val in final_documents:
        key_parts_val = [doc_item_val.metadata.get("type"), doc_item_val.metadata.get("page_number"), doc_item_val.metadata.get("source_pdf_id")]
        content_key_part_val = doc_item_val.page_content[:150]
        if doc_item_val.metadata.get("type") == "image_description": content_key_part_val = doc_item_val.metadata.get("image_path_on_server", content_key_part_val)
        elif doc_item_val.metadata.get("type") in ["text_table_content", "text_figure_description"]: content_key_part_val = doc_item_val.metadata.get("caption_id", content_key_part_val)
        key_parts_val.append(content_key_part_val); key_val = tuple(key_parts_val)
        if key_val not in seen_keys_val: final_deduped_documents_val.append(doc_item_val); seen_keys_val.add(key_val)
    logger.info(f"Finished PDF (ID: {pdf_id}): Extracted {len(final_deduped_documents_val)} unique elements. Title: '{doc_metadata_summary['title']}', Abstract found: {'Yes' if doc_metadata_summary['abstract'] != 'N/A' else 'No'}")
    return final_deduped_documents_val, doc_metadata_summary