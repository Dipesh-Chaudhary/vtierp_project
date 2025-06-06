import logging
import fitz  # PyMuPDF
import os
import shutil
import uuid
import io
import re
import math
from PIL import Image
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any, Set

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import settings, get_pdf_extracted_images_dir, get_pdf_specific_data_dir
from app.dependencies_config.llm_config import get_aux_llm
from app.services.utils import clean_parsed_text, image_to_base64, ocr_image_to_text_unstructured

logger = logging.getLogger(__name__)

class VLMUsageTracker: # No change
    def __init__(self, limit: int): self.count = 0; self.limit = limit
    def increment(self): self.count += 1
    def can_use_vlm(self) -> bool: return self.count < self.limit

def generate_detailed_image_description(image_path: str, element_type: str, vlm_tracker: VLMUsageTracker) -> str: # No change
    # ... (same as before)
    aux_llm = get_aux_llm()
    if not aux_llm: logger.error(f"VLM (aux_llm) not initialized for {os.path.basename(image_path)}."); return f"VLM_NOT_INIT_{os.path.basename(image_path)}"
    if not vlm_tracker.can_use_vlm(): return f"VLM_LIMIT_REACHED_{os.path.basename(image_path)}"
    try:
        img_base64, mime_type = image_to_base64(image_path)
        if not img_base64 or not mime_type: logger.warning(f"Could not load/convert image {os.path.basename(image_path)} for VLM."); return f"VLM_LOAD_FAIL_{os.path.basename(image_path)}"
        prompt_text = (f"Expert document analyst: Analyze this {element_type} from a research paper. Describe its key visual components, structure, any text present within it, and its apparent purpose or the information it conveys. For Tables (if image): describe columns, rows, data types, and notable trends or relationships. For Diagrams/Charts: describe type, axes, trends, and flow. Provide a concise, comprehensive summary suitable for Q&A.")
        from langchain_core.messages import HumanMessage
        message = HumanMessage(content=[ {"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}}])
        response = aux_llm.invoke([message]); vlm_tracker.increment()
        return clean_parsed_text(response.content)
    except Exception as e: logger.error(f"Error in VLM for {os.path.basename(image_path)}: {e}", exc_info=True); return f"VLM_ERROR_{os.path.basename(image_path)}"

def find_pymupdf_captions(page: fitz.Page) -> List[Dict]: # No change
    # ... (same as before)
    captions = []; blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES & ~fitz.TEXT_PRESERVE_IMAGES)["blocks"]
    for block in blocks:
        if block['type'] == 0:
            bt = "".join(s['text'] for l_val in block['lines'] for s in l_val['spans']); ct = clean_parsed_text(bt)
            m = re.match(r"^(Figure|Fig\.?|Table)\s+([A-Za-z0-9]+\.?\d*)\s*[:\.]?\s*", ct, re.I)
            if m: c_type = "figure" if "fig" in m.group(1).lower() else "table"; c_id = f"{c_type.capitalize()}{m.group(2)}"; captions.append({"text": ct, "bbox": fitz.Rect(block['bbox']), "type": c_type, "page_num": page.number, "id": c_id})
    return captions

def get_pymupdf_drawing_clusters(page: fitz.Page) -> List[fitz.Rect]: # No change
    # ... (same as before)
    drawings = page.get_drawings()
    path_rect_tuples = [(d['rect'].x0, d['rect'].y0, d['rect'].x1, d['rect'].y1) for d in drawings if d['rect'].width > 1 and d['rect'].height > 1 and (d.get('type') != 'fill' or d.get('color') is not None)]
    if not path_rect_tuples: return []
    path_rects = sorted([fitz.Rect(r) for r in path_rect_tuples], key=lambda r_obj: (r_obj.y0, r_obj.x0))
    merged_rects: List[fitz.Rect] = []
    page_diagonal = math.sqrt(page.rect.width**2 + page.rect.height**2) if page.rect.width > 0 and page.rect.height > 0 else 1000.0
    max_merge_distance = page_diagonal * settings.drawing_cluster_max_dist_factor
    for current_rect in path_rects:
        if not merged_rects: merged_rects.append(current_rect)
        else:
            last_merged_rect = merged_rects[-1]
            expanded_last_rect = fitz.Rect(last_merged_rect) + (-max_merge_distance, -max_merge_distance, max_merge_distance, max_merge_distance)
            if current_rect.intersects(expanded_last_rect): merged_rects[-1] = last_merged_rect | current_rect
            else: merged_rects.append(current_rect)
    return [r_item for r_item in merged_rects if r_item.width > settings.min_visual_width_pymupdf / 2 and r_item.height > settings.min_visual_height_pymupdf / 2]

def refine_roi_by_content_and_text(page: fitz.Page, initial_roi: fitz.Rect, is_likely_table: bool) -> Optional[fitz.Rect]: # No change
    # ... (same as before)
    current_roi_base = fitz.Rect(initial_roi); page_rect = page.rect
    if not current_roi_base.intersects(page_rect) or current_roi_base.is_empty: return None
    content_bbox = fitz.Rect(); has_explicit_visual_content = False
    for path in page.get_drawings():
        path_rect = fitz.Rect(path['rect'])
        if path_rect.intersects(current_roi_base) and path_rect.width > 1 and path_rect.height > 1: content_bbox.include_rect(path_rect); has_explicit_visual_content = True
    for img_info in page.get_images(full=True):
        try: img_bbox_candidate = page.get_image_bbox(img_info, transform=False)
        except Exception: continue
        if img_bbox_candidate.intersects(current_roi_base): content_bbox.include_rect(img_bbox_candidate); has_explicit_visual_content = True
    if has_explicit_visual_content and not content_bbox.is_empty and content_bbox.width > 5 and content_bbox.height > 5:
        working_roi = content_bbox.intersect(current_roi_base)
        if working_roi.is_empty or working_roi.width < 5 or working_roi.height < 5: working_roi = fitz.Rect(current_roi_base)
    elif not is_likely_table and (current_roi_base.width < settings.min_visual_width_pymupdf or current_roi_base.height < settings.min_visual_height_pymupdf): return None
    else: working_roi = fitz.Rect(current_roi_base)
    if working_roi.is_empty or working_roi.width < settings.min_visual_width_pymupdf / 2 or working_roi.height < settings.min_visual_height_pymupdf / 2: return None
    final_trimmed_roi = fitz.Rect(working_roi); text_blocks = [fitz.Rect(b[:4]) for b in page.get_text("blocks", flags=0) if b[6] == 0] 
    lowest_text_above = None; text_above_candidates = [tb for tb in text_blocks if tb.y1 < final_trimmed_roi.y0 + 10 and tb.x1 > final_trimmed_roi.x0 - 5 and tb.x0 < final_trimmed_roi.x1 + 5]
    if text_above_candidates: lowest_text_above = max(text_above_candidates, key=lambda r_item: r_item.y1)
    if lowest_text_above and lowest_text_above.get_area() > settings.text_block_min_area_for_obstruction and lowest_text_above.y1 < final_trimmed_roi.y1 - 5: final_trimmed_roi.y0 = lowest_text_above.y1 + 2
    highest_text_below = None; text_below_candidates = [tb for tb in text_blocks if tb.y0 > final_trimmed_roi.y1 - 10 and tb.x1 > final_trimmed_roi.x0 - 5 and tb.x0 < final_trimmed_roi.x1 + 5]
    if text_below_candidates: highest_text_below = min(text_below_candidates, key=lambda r_item: r_item.y0)
    if highest_text_below and highest_text_below.get_area() > settings.text_block_min_area_for_obstruction and highest_text_below.y0 > final_trimmed_roi.y0 + 5: final_trimmed_roi.y1 = highest_text_below.y0 - 2
    final_trimmed_roi.intersect(page_rect)
    if final_trimmed_roi.is_empty or final_trimmed_roi.width < settings.min_visual_width_pymupdf or final_trimmed_roi.height < settings.min_visual_height_pymupdf: return None
    return final_trimmed_roi

# --- find_table_content (Simplified Markdown, focus on raw text and bbox) ---
def find_table_content(page: fitz.Page, table_caption_info: Dict) -> Tuple[str, Optional[str], Optional[fitz.Rect]]:
    caption_bbox_rect = fitz.Rect(table_caption_info["bbox"])
    caption_id = table_caption_info["id"]; caption_text_cleaned = clean_parsed_text(table_caption_info["text"])
    markdown_table_str: Optional[str] = None
    table_text_overall_bbox: Optional[fitz.Rect] = None
    logger.debug(f"TABLE EXTRACTION: Start for '{caption_id}' on page {page.number + 1}")

    search_y_start = caption_bbox_rect.y1 - 5 # Allow slight overlap with caption bottom
    min_y_of_next_major_element = page.rect.height
    for other_cap_info in find_pymupdf_captions(page):
        if other_cap_info["id"] != caption_id and fitz.Rect(other_cap_info["bbox"]).y0 > caption_bbox_rect.y1:
            min_y_of_next_major_element = min(min_y_of_next_major_element, fitz.Rect(other_cap_info["bbox"]).y0)
    for img_info in page.get_images(full=True):
        try: img_bbox = page.get_image_bbox(img_info)
        except Exception: continue
        if img_bbox.y0 > caption_bbox_rect.y1 and img_bbox.height > 30:
             min_y_of_next_major_element = min(min_y_of_next_major_element, img_bbox.y0)
    search_y_end_effective = min(page.rect.height - 10, min_y_of_next_major_element - 5)
    table_search_roi = fitz.Rect(page.rect.x0 + 5, search_y_start, page.rect.x1 - 5, search_y_end_effective)

    if table_search_roi.is_empty or table_search_roi.height < 5:
        logger.warning(f"TABLE EXTRACTION: Insufficient search area for '{caption_id}'."); return "", None, None
        
    # Get text blocks in reading order for better line reconstruction
    blocks_in_roi = page.get_text("blocks", clip=table_search_roi, sort=True)
    if not blocks_in_roi: logger.debug(f"TABLE EXTRACTION: No text blocks in ROI for '{caption_id}'."); return "", None, None

    reconstructed_lines_data: List[Dict[str, Any]] = []
    
    for block_idx, block_tuple in enumerate(blocks_in_roi):
        if block_tuple[6] != 0: continue # Not a text block
        block_text_content = block_tuple[4]
        current_block_rect = fitz.Rect(block_tuple[:4]) # Bbox of the current text block

        lines_in_block_from_pymupdf = block_text_content.strip().split('\n')
        
        for line_text_raw in lines_in_block_from_pymupdf:
            line_text_cleaned = clean_parsed_text(line_text_raw)
            if not line_text_cleaned: continue

            # Heuristic to get a bbox for this specific line within the block
            # This is approximate; for perfect line bboxes, word-level iteration is needed but complex.
            # For now, use the block's bbox for the line if it's a single-line block,
            # or approximate if multi-line (this part is hard to do accurately without words).
            # Let's simplify: we'll primarily use the raw text of lines.
            # The table_text_overall_bbox will be from accumulating these block bboxes.

            is_part_of_caption = (line_text_cleaned == caption_text_cleaned and 
                                 (current_block_rect.intersects(caption_bbox_rect) or 
                                  abs(current_block_rect.y0 - caption_bbox_rect.y0) < current_block_rect.height / 2))
            if is_part_of_caption: logger.debug(f"Skipping line as caption '{caption_id}': '{line_text_cleaned}'"); continue
            
            # If the block starts above the end of the caption, and it's not the caption itself, skip
            if current_block_rect.y0 < caption_bbox_rect.y1 - 5 and not is_part_of_caption: continue
            
            # If too far below previous content and very short, might be footerish
            if len(reconstructed_lines_data) > 2 and \
               current_block_rect.y0 > reconstructed_lines_data[-1]['bbox'].y1 + 40 and \
               len(line_text_cleaned.split()) < 3:
                logger.debug(f"Stopping for '{caption_id}' due to distant short line: '{line_text_cleaned}'"); break
            
            reconstructed_lines_data.append({'text': line_text_cleaned, 'bbox': current_block_rect}) # Store block_bbox per line
            if table_text_overall_bbox is None: table_text_overall_bbox = fitz.Rect(current_block_rect)
            else: table_text_overall_bbox.include_rect(current_block_rect)
        else: # Inner loop (lines_in_block_from_pymupdf) completed without break
            continue
        break # Outer loop (blocks_in_roi) broken due to inner break

    if not reconstructed_lines_data: 
        logger.warning(f"TABLE EXTRACTION: No distinct text lines for table body of '{caption_id}'.")
        return "", None, None
        
    raw_table_text = "\n".join([line_data['text'] for line_data in reconstructed_lines_data]).strip()

    # Simplified Markdown: only if multiple "words" (space separated segments) are common
    if len(reconstructed_lines_data) >= 1:
        # Count words per line for a simple column heuristic
        line_word_counts = [len(line['text'].split()) for line in reconstructed_lines_data if line['text'].strip()]
        if line_word_counts:
            avg_words_per_line = sum(line_word_counts) / len(line_word_counts)
            # Try to infer num_cols if avg_words > N (e.g., 2.5, tunable) and some consistency
            # This is a very rough heuristic for "does it look like it has columns?"
            num_cols_heuristic = 1
            if avg_words_per_line > 2.5 and len(line_word_counts) > 1 : # If lines average >2.5 words, guess it might have columns
                # Take the most frequent word count as num_cols if it's > 1
                count_freq = defaultdict(int)
                for wc in line_word_counts: count_freq[wc]+=1
                if count_freq:
                    sorted_counts = sorted(count_freq.items(), key=lambda x:x[1], reverse=True)
                    if sorted_counts[0][0] > 1:
                         num_cols_heuristic = sorted_counts[0][0]
            
            if num_cols_heuristic > 1 and num_cols_heuristic < 15: # Max 15 cols for this heuristic
                logger.info(f"TABLE '{caption_id}': Attempting simplified MD with {num_cols_heuristic} columns.")
                md_lines = []
                # Assume first line is header
                header_cells = reconstructed_lines_data[0]['text'].split(None, num_cols_heuristic - 1) # Split into N parts
                header_cells += [""] * (num_cols_heuristic - len(header_cells))
                md_lines.append("| " + " | ".join(h.strip().replace("|", "\\|") for h in header_cells) + " |")
                md_lines.append("|" + " :-- |" * num_cols_heuristic)
                for line_data in reconstructed_lines_data[1:]:
                    data_cells = line_data['text'].split(None, num_cols_heuristic - 1)
                    data_cells += [""] * (num_cols_heuristic - len(data_cells))
                    md_lines.append("| " + " | ".join(rc.strip().replace("|", "\\|") for rc in data_cells) + " |")
                if len(md_lines) > 2: markdown_table_str = "\n".join(md_lines)
                else: logger.info(f"Simplified MD for '{caption_id}' too few rows.")
            else: logger.info(f"Skipping simplified MD for '{caption_id}', num_cols_heuristic={num_cols_heuristic}.")
        else: logger.info(f"Not enough line data to guess columns for MD for '{caption_id}'.")

    if not raw_table_text: logger.warning(f"TABLE EXTRACTION: No text for table '{caption_id}'"); return "", None, None
    return raw_table_text, markdown_table_str, table_text_overall_bbox

# --- extract_visual_elements_from_page (Bug Fix and Refined Table Image Fallback) ---
def extract_visual_elements_from_page(
    doc_fitz: fitz.Document, page_num: int, pdf_id: str,
    vlm_tracker: VLMUsageTracker, processed_captions_for_this_pdf: Set[str],
    original_pdf_filename: str
) -> List[Document]:
    page = doc_fitz.load_page(page_num); visual_docs: List[Document] = []
    pdf_image_save_dir = get_pdf_extracted_images_dir(pdf_id); os.makedirs(pdf_image_save_dir, exist_ok=True)
    # ... (primitive visual identification - same) ...
    primitive_visuals = []
    for img_idx, img_info_fitz in enumerate(page.get_images(full=True)):
        try:
            bbox = page.get_image_bbox(img_info_fitz, transform=False); img_bytes_check = doc_fitz.extract_image(img_info_fitz[0])["image"]
            if not img_bytes_check: continue
            pil_img_check = Image.open(io.BytesIO(img_bytes_check))
            if pil_img_check.width > settings.min_visual_width_pymupdf / 2 and pil_img_check.height > settings.min_visual_height_pymupdf / 2:
                primitive_visuals.append({"bbox": bbox, "type": "raster", "id": f"p{page_num}_raster{img_idx}", "raw_info": img_info_fitz})
        except Exception as e_img: logger.debug(f"Skipping image on page {page_num+1} due to error: {e_img}")
    for dc_idx, dc_bbox in enumerate(get_pymupdf_drawing_clusters(page)):
        if not any(r_info["type"] == "raster" and r_info["bbox"].contains(dc_bbox) for r_info in primitive_visuals):
            primitive_visuals.append({"bbox": dc_bbox, "type": "drawing_cluster", "id": f"p{page_num}_draw{dc_idx}"})

    captions_on_page = find_pymupdf_captions(page); caption_to_visual_parts_map = defaultdict(list)
    unassigned_primitives = list(primitive_visuals); processed_primitive_indices_this_page = set()

    for cap_info in captions_on_page:
        current_caption_bbox_from_cap_info = fitz.Rect(cap_info['bbox'])
        # CRITICAL BUG FIX: Initialize potential_visuals_for_caption HERE
        potential_visuals_for_caption: List[Tuple[Dict, int]] = [] 

        if cap_info["type"] == "figure" and cap_info["id"] in processed_captions_for_this_pdf:
             logger.debug(f"Figure caption '{cap_info['id']}' already processed. Skipping visual association.")
             continue
        
        found_visual_for_caption = False
        cap_info_bbox_center_y = (current_caption_bbox_from_cap_info.y0 + current_caption_bbox_from_cap_info.y1) / 2
        for i, prim_vis in enumerate(unassigned_primitives):
            if i in processed_primitive_indices_this_page: continue
            prim_vis_bbox_center_y = (prim_vis["bbox"].y0 + prim_vis["bbox"].y1) / 2; v_dist = abs(prim_vis_bbox_center_y - cap_info_bbox_center_y)
            h_overlap = max(0, min(prim_vis["bbox"].x1, current_caption_bbox_from_cap_info.x1) - max(prim_vis["bbox"].x0, current_caption_bbox_from_cap_info.x0)); h_overlap_ratio = 0 
            if prim_vis["bbox"].width > 0 and current_caption_bbox_from_cap_info.width > 0 : h_overlap_ratio_prim = h_overlap / prim_vis["bbox"].width; h_overlap_ratio_cap = h_overlap / current_caption_bbox_from_cap_info.width; h_overlap_ratio = max(h_overlap_ratio_prim, h_overlap_ratio_cap)
            is_correctly_positioned = False; max_v_dist_figure = 200; max_v_dist_table = 100
            if cap_info["type"] == "figure":
                if ((prim_vis_bbox_center_y < cap_info_bbox_center_y and v_dist < max_v_dist_figure) or (abs(v_dist) < 40)): is_correctly_positioned = True
            elif cap_info["type"] == "table": 
                if ((prim_vis_bbox_center_y > cap_info_bbox_center_y and v_dist < max_v_dist_table) or (abs(v_dist) < 40)): is_correctly_positioned = True
            if h_overlap_ratio > 0.2 and is_correctly_positioned: potential_visuals_for_caption.append((prim_vis, i))
        
        if potential_visuals_for_caption: # This logic block is now safe
            potential_visuals_for_caption.sort(key=lambda x: (0 if cap_info["type"] == "figure" and x[0]["type"] == "drawing_cluster" else 1, abs((x[0]["bbox"].y0 + x[0]["bbox"].y1)/2 - cap_info_bbox_center_y), -(x[0]["bbox"].width * x[0]["bbox"].height)))
            best_prim_vis, best_prim_idx = potential_visuals_for_caption[0]
            target_list = caption_to_visual_parts_map[cap_info["id"]];
            if not any(item.get("is_caption_meta") for item in target_list): target_list.append({"is_caption_meta": True, "caption_obj": cap_info})
            target_list.append(best_prim_vis); processed_primitive_indices_this_page.add(best_prim_idx); found_visual_for_caption = True
            logger.info(f"Associated caption '{cap_info['id']}' with visual primitive '{best_prim_vis['id']}' on page {page_num+1}.")
        
        text_part_already_processed = cap_info["id"] in processed_captions_for_this_pdf and \
                                      not any(item.get("type") == "table_image_fallback" for item in caption_to_visual_parts_map.get(cap_info["id"],[]))

        if (cap_info["type"] == "table" or \
           (cap_info["type"] == "figure" and not found_visual_for_caption)) and \
           not text_part_already_processed :
            page_content_for_doc = f"IDENTITY: {cap_info['id']}\nCAPTION: {cap_info['text']}"
            table_as_markdown_output = None; table_text_content_bbox = None 
            element_subtype = "textual_content_with_caption"
            doc_type = "text_figure_description" if cap_info["type"] == "figure" else "text_table_content"

            if cap_info["type"] == "table":
                raw_table_body, table_as_markdown_output, table_text_content_bbox = find_table_content(page, cap_info)
                if raw_table_body: page_content_for_doc += f"\n\n--- Extracted Table Text ---\n{raw_table_body}"
                else: page_content_for_doc += "\n\n--- Extracted Table Text ---\n(No distinct body text found for this table)"
                
                # Fallback for tables: only if no direct visual AND (no good MD OR raw text is poor)
                # AND table_text_content_bbox (from find_table_content) is valid
                if not found_visual_for_caption and \
                   (not table_as_markdown_output or len(raw_table_body) < 30) and \
                   table_text_content_bbox and not table_text_content_bbox.is_empty:
                    logger.info(f"Table '{cap_info['id']}' trying image fallback using text bbox: {table_text_content_bbox}.")
                    table_image_initial_roi = fitz.Rect(table_text_content_bbox)
                    # Include caption if it's close and typically above the identified text
                    if current_caption_bbox_from_cap_info.y1 < table_image_initial_roi.y0 + 15: 
                        table_image_initial_roi.include_rect(current_caption_bbox_from_cap_info)
                    
                    refined_table_image_roi = refine_roi_by_content_and_text(page, table_image_initial_roi, is_likely_table=True)
                    if refined_table_image_roi:
                        table_image_part_id = f"{cap_info['id']}_image_fallback"
                        target_list_fallback = caption_to_visual_parts_map[cap_info["id"]]
                        if not any(item.get("is_caption_meta") for item in target_list_fallback): target_list_fallback.append({"is_caption_meta": True, "caption_obj": cap_info})
                        target_list_fallback.append({"bbox": refined_table_image_roi, "type": "table_image_fallback", "id": table_image_part_id})
                        logger.info(f"Scheduled image fallback for table '{cap_info['id']}' with ROI {refined_table_image_roi}")
                        element_subtype += "_plus_image_fallback"
                    else: logger.warning(f"Could not define ROI for image fallback of table '{cap_info['id']}'.")
                elif not (table_text_content_bbox and not table_text_content_bbox.is_empty):
                     logger.warning(f"Skipping image fallback for table '{cap_info['id']}', no table body text identified to define an ROI.")
            
            doc_metadata = {"source_pdf_id": pdf_id, "source_doc_name": original_pdf_filename, "page_number": page_num + 1, "type": doc_type, "original_caption": cap_info["text"], "caption_id": cap_info["id"], "element_subtype": element_subtype}
            if table_as_markdown_output: doc_metadata["generated_markdown_table"] = table_as_markdown_output
            visual_docs.append(Document(page_content=page_content_for_doc, metadata=doc_metadata))
            if "image_fallback" not in element_subtype: processed_captions_for_this_pdf.add(cap_info["id"])
    
    # ... (Unassigned primitives, and Semantic Visual Group processing - same as your last correct version which included NameError fix for `doc_check`) ...
    for i, prim_vis in enumerate(unassigned_primitives):
        if i in processed_primitive_indices_this_page: continue
        if prim_vis["bbox"].width > settings.min_visual_width_pymupdf * 2 and prim_vis["bbox"].height > settings.min_visual_height_pymupdf * 2:
            caption_to_visual_parts_map[prim_vis["id"]].append(prim_vis) 
            logger.info(f"Identified large uncaptioned visual '{prim_vis['id']}' for processing on page {page_num+1}.")

    for semantic_id, group_elements in caption_to_visual_parts_map.items():
        already_rendered_and_described_this_page_run = any(
            doc_item.metadata.get("caption_id") == semantic_id and doc_item.metadata.get("type") == "image_description"
            for doc_item in visual_docs 
        )
        if semantic_id in processed_captions_for_this_pdf and already_rendered_and_described_this_page_run :
            logger.debug(f"Visual for '{semantic_id}' already processed this page run/globally. Skipping re-rendering.")
            continue
        visual_parts_data_list = [item for item in group_elements if not item.get("is_caption_meta")]
        caption_meta_item = next((item for item in group_elements if item.get("is_caption_meta")), None)
        if not visual_parts_data_list: continue 
        composite_render_roi = fitz.Rect(); final_caption_text = f"Uncaptioned Visual ({semantic_id})"; final_semantic_type = "figure"
        current_element_subtype = "pymupdf_semantic_figure"
        if caption_meta_item:
            current_caption_info_obj = caption_meta_item["caption_obj"] 
            final_caption_text = current_caption_info_obj["text"]; final_semantic_type = current_caption_info_obj["type"]
            composite_render_roi.include_rect(fitz.Rect(current_caption_info_obj["bbox"])) 
            current_element_subtype = f"pymupdf_semantic_{final_semantic_type}"
        is_table_img_fallback = any(part.get("type") == "table_image_fallback" for part in visual_parts_data_list)
        if is_table_img_fallback:
            final_semantic_type = "table"; current_element_subtype = "table_image_vlm_description"
        for part_data in visual_parts_data_list: composite_render_roi.include_rect(part_data["bbox"])
        composite_render_roi.intersect(page.rect)
        if composite_render_roi.is_empty or composite_render_roi.width < settings.min_visual_width_pymupdf or composite_render_roi.height < settings.min_visual_height_pymupdf: continue
        is_likely_table_for_refine = (final_semantic_type == "table")
        refined_roi = refine_roi_by_content_and_text(page, composite_render_roi, is_likely_table_for_refine)
        if not refined_roi: logger.warning(f"ROI refinement failed for visual '{semantic_id}' on page {page_num+1}. Skipping."); continue
        try:
            pix = page.get_pixmap(clip=refined_roi, dpi=settings.render_dpi_pymupdf, alpha=False)
            if pix.width == 0 or pix.height == 0: logger.warning(f"Pixmap for '{semantic_id}' is empty. ROI: {refined_roi}. Skipping save."); continue
            filename_part = re.sub(r'[^\w.-]', '_', str(semantic_id)); base_filename = f"page{page_num+1}_SEMANTIC_{filename_part}.png"; image_save_path = os.path.join(pdf_image_save_dir, base_filename); counter = 0
            while os.path.exists(image_save_path): counter += 1; image_save_path = os.path.join(pdf_image_save_dir, f"page{page_num+1}_SEMANTIC_{filename_part}_{counter}.png")
            pix.save(image_save_path); pix = None
            vlm_element_type_for_prompt = f"image of a table (Caption: '{final_caption_text[:50]}...')" if is_table_img_fallback else f"visual element ({final_semantic_type}, Caption: '{final_caption_text[:50]}...')"
            vlm_description = generate_detailed_image_description(image_save_path, vlm_element_type_for_prompt, vlm_tracker)
            relative_image_path = os.path.relpath(image_save_path, get_pdf_specific_data_dir(pdf_id))
            visual_docs.append(Document(page_content=vlm_description, metadata={"source_pdf_id": pdf_id, "source_doc_name": original_pdf_filename, "page_number": page_num + 1, "type": "image_description", "image_path_on_server": image_save_path, "image_path_relative_to_pdf_data": relative_image_path, "original_caption": final_caption_text, "element_subtype": current_element_subtype, "caption_id": semantic_id}))
            processed_captions_for_this_pdf.add(semantic_id) 
            logger.info(f"Rendered and VLM-described visual '{semantic_id}' (Type: {current_element_subtype}) on page {page_num+1}.")
        except Exception as e: logger.error(f"Error in rendering/VLM for visual '{semantic_id}': {e}", exc_info=True)
    return visual_docs


# --- process_single_pdf_custom (Main Orchestrator - with corrected HTML extraction) ---
def process_single_pdf_custom(pdf_file_path: str, pdf_id: str, original_pdf_filename_for_metadata: str) -> Tuple[List[Document], Dict[str, Any]]:
    logger.info(f"Starting processing for PDF (ID: {pdf_id}): {original_pdf_filename_for_metadata}")
    vlm_tracker = VLMUsageTracker(limit=settings.max_elements_for_vlm_description_per_pdf)
    processed_captions_globally_for_this_pdf = set() 
    final_documents: List[Document] = []
    doc_metadata_summary = {"pdf_id": pdf_id, "original_filename": original_pdf_filename_for_metadata, "title": original_pdf_filename_for_metadata, "abstract": "N/A", "is_scanned": False, "page_count": 0}
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    final_deduped_documents_val: List[Document] = []
    try:
        doc_fitz = fitz.open(pdf_file_path)
        doc_metadata_summary["page_count"] = len(doc_fitz)
        if len(doc_fitz) > 0: 
            # --- Title/Abstract Extraction (same as your last correct version) ---
            # ... (This logic was good, keep it)
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
            title_lines_collected_val = []
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
            if extracted_title_val: doc_metadata_summary["title"] = extracted_title_val; logger.info(f"Extracted Title for {pdf_id}: {extracted_title_val}"); final_documents.append(Document(page_content=f"DOCUMENT TITLE: {extracted_title_val}", metadata={"source_pdf_id": pdf_id, "source_doc_name": original_pdf_filename_for_metadata, "page_number": 1, "type": "title_summary", "importance": "critical"}))
            else: logger.warning(f"Title extraction failed for {pdf_id}. Defaulting to filename."); 
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
                doc_metadata_summary["abstract"] = extracted_abstract_val; logger.info(f"Extracted Abstract for {pdf_id} (len {len(extracted_abstract_val)}): {extracted_abstract_val[:100]}..."); final_documents.append(Document(page_content=f"DOCUMENT ABSTRACT: {extracted_abstract_val}", metadata={"source_pdf_id": pdf_id, "source_doc_name": original_pdf_filename_for_metadata, "page_number": 1, "type": "abstract_summary", "importance": "critical"}))
            else: logger.warning(f"Abstract extraction failed or abstract too short for {pdf_id}.")
            # --- End Title/Abstract Extraction ---
        digitally_extracted_text_sample_val = "".join(doc_fitz.load_page(i_val).get_text("text", sort=True) for i_val in range(min(3, len(doc_fitz))))
        if len(clean_parsed_text(digitally_extracted_text_sample_val)) < settings.min_ocr_text_length_for_scanned_pdf * min(3, len(doc_fitz)): doc_metadata_summary["is_scanned"] = True; logger.info(f"PDF '{original_pdf_filename_for_metadata}' (ID: {pdf_id}) appears scanned.")
        
        for page_num_idx_val in range(len(doc_fitz)):
            page_val = doc_fitz.load_page(page_num_idx_val)
            page_text_for_chunking_val = ""
            parser_source_log_val = "pymupdf_digital_text"

            if doc_metadata_summary["is_scanned"]:
                parser_source_log_val = "pymupdf_ocr"; temp_ocr_image_dir_val = get_pdf_extracted_images_dir(pdf_id); ocr_img_path_val = os.path.join(temp_ocr_image_dir_val, f"page{page_num_idx_val+1}_ocr_temp.png")
                pix_val = page_val.get_pixmap(dpi=settings.ocr_dpi, alpha=False); pix_val.save(ocr_img_path_val); pix_val = None
                page_text_for_chunking_val = ocr_image_to_text_unstructured(ocr_img_path_val)
                if vlm_tracker.can_use_vlm():
                    page_img_desc_val = generate_detailed_image_description(ocr_img_path_val, "scanned page", vlm_tracker); relative_ocr_img_path_val = os.path.relpath(ocr_img_path_val, get_pdf_specific_data_dir(pdf_id))
                    final_documents.append(Document(page_content=page_img_desc_val, metadata={"source_pdf_id": pdf_id, "source_doc_name": original_pdf_filename_for_metadata, "page_number": page_num_idx_val + 1, "type": "image_description", "image_path_on_server": ocr_img_path_val, "image_path_relative_to_pdf_data": relative_ocr_img_path_val, "original_caption": f"Full Scanned Page {page_num_idx_val+1}", "element_subtype": "scanned_page_full_vlm", "caption_id": f"ScannedPage{page_num_idx_val+1}_FullVLM"}))
            else: 
                try: # Corrected HTML extraction call - removed invalid kwargs
                    page_html_content = page_val.get_text("html") # Basic HTML
                    text_from_html = re.sub(r'<style.*?</style>', '', page_html_content, flags=re.DOTALL | re.IGNORECASE) 
                    text_from_html = re.sub(r'<script.*?</script>', '', text_from_html, flags=re.DOTALL | re.IGNORECASE) 
                    text_from_html = re.sub(r'<sup>(.*?)</sup>', r'^{\1}', text_from_html) # Convert sup
                    text_from_html = re.sub(r'<sub>(.*?)</sub>', r'_{\1}', text_from_html) # Convert sub
                    text_from_html = re.sub(r'<[^>]+>', ' ', text_from_html) 
                    text_from_html = text_from_html.replace("<", "<").replace(">", ">").replace("&", "&").replace(" ", " ") 
                    page_text_for_chunking_val = clean_parsed_text(text_from_html)
                    parser_source_log_val = "pymupdf_html_cleaned"
                    if not page_text_for_chunking_val.strip(): 
                        logger.warning(f"HTML parsing yielded empty text for page {page_num_idx_val+1} of {pdf_id}. Falling back.")
                        page_text_for_chunking_val = page_val.get_text("text", sort=True)
                        parser_source_log_val = "pymupdf_digital_text_fallback"
                except Exception as e_html:
                    logger.warning(f"Error during HTML text extraction for page {page_num_idx_val+1} of {pdf_id}: {e_html}. Falling back.")
                    page_text_for_chunking_val = page_val.get_text("text", sort=True)
                    parser_source_log_val = "pymupdf_digital_text_error_fallback"
            
            if not doc_metadata_summary["is_scanned"]: 
                visual_docs_from_page_val = extract_visual_elements_from_page(doc_fitz, page_num_idx_val, pdf_id, vlm_tracker, processed_captions_globally_for_this_pdf, original_pdf_filename_for_metadata)
                final_documents.extend(visual_docs_from_page_val)

            cleaned_page_text_for_chunking = clean_parsed_text(page_text_for_chunking_val) 
            if cleaned_page_text_for_chunking:
                text_to_process_val = cleaned_page_text_for_chunking
                if page_num_idx_val == 0 and doc_metadata_summary["title"] != "N/A" and doc_metadata_summary["title"] != original_pdf_filename_for_metadata :
                    if not cleaned_page_text_for_chunking.strip().lower().startswith(doc_metadata_summary["title"].strip().lower()[:min(30, len(doc_metadata_summary["title"]))]):
                        text_to_process_val = f"Document Title Context: {doc_metadata_summary['title']}\n\n{cleaned_page_text_for_chunking}"
                page_splits_val = text_splitter.create_documents([text_to_process_val], metadatas=[{"source_pdf_id": pdf_id, "source_doc_name": original_pdf_filename_for_metadata, "page_number": page_num_idx_val + 1, "type": "text_chunk", "parser_source": parser_source_log_val}])
                final_documents.extend(page_splits_val)
        doc_fitz.close()
    except Exception as e: 
        logger.error(f"Critical error during PDF processing for {pdf_id} ({original_pdf_filename_for_metadata}): {e}", exc_info=True)
        if 'doc_fitz' in locals() and hasattr(doc_fitz, 'is_open') and doc_fitz.is_open:
            doc_fitz.close()
    
    seen_keys_val = set() 
    for doc_item_val in final_documents: 
        key_tuple_parts = [doc_item_val.metadata.get("type", "unknown_type"), doc_item_val.metadata.get("source_pdf_id", pdf_id), doc_item_val.metadata.get("page_number", 0) ]
        if doc_item_val.metadata.get("type") == "image_description": key_tuple_parts.append(doc_item_val.metadata.get("image_path_on_server", doc_item_val.page_content[:50]))
        elif doc_item_val.metadata.get("type") in ["text_table_content", "text_figure_description"]: key_tuple_parts.append(doc_item_val.metadata.get("caption_id", doc_item_val.page_content[:50]))
        else: key_tuple_parts.append(doc_item_val.page_content[:100]) 
        key_val = tuple(key_tuple_parts)
        if key_val not in seen_keys_val: final_deduped_documents_val.append(doc_item_val); seen_keys_val.add(key_val)
    
    logger.info(f"Finished PDF (ID: {pdf_id}, File: {original_pdf_filename_for_metadata}): Extracted {len(final_deduped_documents_val)} unique elements. Title: '{doc_metadata_summary['title']}', Abstract found: {'Yes' if doc_metadata_summary['abstract'] != 'N/A' else 'No'}")
    return final_deduped_documents_val, doc_metadata_summary
