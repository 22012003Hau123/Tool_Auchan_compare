"""
Mode 2: Kiá»ƒm tra popup annotations Ä‘Ă£ Ä‘Æ°á»£c thá»±c thi trong PDF final báº±ng OpenAI Vision.
Äá»™c láº­p vá»›i step6/step6_gemini. DĂ¹ng template/prompt riĂªng.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mode2_annotations")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.INFO)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(_h)


@contextlib.contextmanager
def _suppress_mupdf_stderr():
    """Táº¡m áº©n stderr khi MuPDF in cáº£nh bĂ¡o (vd: 'No common ancestor in structure tree')."""
    if getattr(sys.stderr, "fileno", None) is None:
        yield
        return
    try:
        import io
        _bak = sys.stderr
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stderr = _bak

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from dotenv import load_dotenv
    from pathlib import Path
    _project_root = Path(__file__).resolve().parent
    _env_path = _project_root / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=True)
except Exception:
    pass

GPT_MODEL = os.getenv("MODE2_GPT_MODEL") or os.getenv("GPT_MODEL", "gpt-5.2")


def _smart_preprocess(ref_path: str, final_path: str) -> tuple[str, dict]:
    """
    Preprocess tá»‘i giáº£n (no-op). Tráº£ vá» path vĂ  metadata.
    CĂ³ thá»ƒ má»Ÿ rá»™ng sau náº¿u cáº§n pdf_optimizer.
    """
    return ref_path, {"preprocessed": False, "reason": "minimal"}


def get_openai_client(api_key: Optional[str] = None) -> Optional[Any]:
    """Khá»Ÿi táº¡o OpenAI client tá»« api_key (Æ°u tiĂªn) hoáº·c env."""
    if OpenAI is None:
        return None
    key = api_key or os.getenv("MODE2_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    if not key or key == "your-api-key-here":
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None


def extract_popup_annotations(pdf_path: str) -> List[Dict]:
    """
    TrĂ­ch xuáº¥t táº¥t cáº£ annotations cĂ³ ná»™i dung tá»« PDF reference.
    KhĂ´ng lÆ°u object annot (khĂ´ng serializable), chá»‰ page, content, rect, type.
    """
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is required. Install with: pip install PyMuPDF")
    with _suppress_mupdf_stderr():
        doc = fitz.open(pdf_path)
    annotations: List[Dict] = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        annots = page.annots()
        if not annots:
            continue

        for annot in annots:
            try:
                content = ""
                content = annot.info.get("content", "") or annot.info.get("title", "")
                if not content and hasattr(annot, "popup"):
                    popup = getattr(annot, "popup", None)
                    if popup:
                        content = popup.info.get("content", "")
                if not content:
                    annot_type = annot.type[0]
                    if annot_type == fitz.PDF_ANNOT_FREE_TEXT:
                        content = annot.info.get("subject", "")
                if not content:
                    try:
                        text_in_rect = page.get_text("text", clip=annot.rect).strip()
                        if text_in_rect:
                            content = text_in_rect
                    except Exception:
                        pass

                if content and content.strip():
                    r = annot.rect
                    annotations.append({
                        "page": page_num,
                        "content": content.strip(),
                        "rect": {"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1},
                        "type": annot.type[0],
                    })
            except Exception as e:
                logger.warning("Could not extract annotation on page %s: %s", page_num + 1, e)

    doc.close()
    return annotations


def _rect_from_dict(d: dict) -> "fitz.Rect":
    return fitz.Rect(d["x0"], d["y0"], d["x1"], d["y1"])


def convert_full_page_to_image(page: "fitz.Page", zoom: float = 1.5) -> bytes:
    """Convert toĂ n bá»™ page thĂ nh PNG bytes."""
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    pix = None
    return img_bytes


def check_annotation_with_gpt(
    client: Optional[Any],
    annotation_content: str,
    ref_page_image_bytes: bytes,
    final_page_image_bytes: bytes,
    model: str = GPT_MODEL,
) -> Dict:
    """Gá»i OpenAI Vision Ä‘á»ƒ Ä‘Ă¡nh giĂ¡ annotation Ä‘Ă£ thá»±c hiá»‡n hay chÆ°a."""
    if client is None:
        return {
            "implemented": False,
            "confidence": 0.0,
            "reasoning": "OpenAI client not available",
            "evidence": "",
            "status": "unclear",
        }

    ref_base64 = base64.b64encode(ref_page_image_bytes).decode("utf-8")
    final_base64 = base64.b64encode(final_page_image_bytes).decode("utf-8")

    prompt = f"""Comparez IMAGE 1 (RĂ©fĂ©rence) et IMAGE 2 (Final).

DEMANDE: {annotation_content}

RĂ©pondez UNIQUEMENT en JSON:
- status: "implemented" / "not_implemented" / "partial" / "unclear"
- implemented: true/false
- confidence: 0-1

AUCUNE explication. RĂ©ponse JSON UNIQUEMENT.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Expert PDF. JSON ONLY. NO explanation."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{ref_base64}", "detail": "high"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{final_base64}", "detail": "high"},
                        },
                    ],
                },
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "implemented": bool(result.get("implemented", False)),
            "confidence": float(result.get("confidence", 0.0)),
            "reasoning": result.get("reasoning", ""),
            "evidence": "",
            "status": result.get("status", "unclear"),
        }
    except Exception as e:
        return {
            "implemented": False,
            "confidence": 0.0,
            "reasoning": f"Error: {e}",
            "evidence": "",
            "status": "unclear",
        }


def _annotate_status(final_page: "fitz.Page", rect: "fitz.Rect", result: Dict) -> None:
    """TĂ´ mĂ u rect theo status, khĂ´ng thĂªm popup text."""
    status = result.get("status", "unclear")
    if status == "implemented":
        color = (0, 1, 0)
    elif status == "not_implemented":
        color = (1, 0, 0)
    elif status == "partial":
        color = (1, 1, 0)
    else:
        color = (0.5, 0.5, 0.5)

    annot = final_page.add_rect_annot(rect)
    annot.set_colors(stroke=color)
    annot.set_border(width=2.0)
    annot.set_opacity(0.5)
    annot.update()


def compare_mode2(
    ref_pdf_path: str,
    final_pdf_path: str,
    output_path: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_pages: int = 500,
    max_annotations: int = 1000,
    save_images_dir: Optional[str] = None,
) -> Dict:
    """
    Mode 2 â€“ Äá»c popup annotations tá»« ref_pdf, kiá»ƒm tra báº±ng OpenAI Vision, annotate vĂ o final_pdf.
    """
    if fitz is None:
        raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")

    logger.info("[Mode2] Báº¯t Ä‘áº§u so sĂ¡nh: ref=%s, final=%s", ref_pdf_path, final_pdf_path)

    ref_pdf_path, preprocess_metadata = _smart_preprocess(ref_pdf_path, final_pdf_path)

    model_name = model or GPT_MODEL
    if output_path is None:
        base = os.path.splitext(final_pdf_path)[0]
        output_path = f"{base}_mode2_diff.pdf"

    annotations = extract_popup_annotations(ref_pdf_path)
    logger.info("[Mode2] TrĂ­ch xuáº¥t %d annotations tá»« PDF reference", len(annotations))
    if len(annotations) > max_annotations:
        annotations = annotations[:max_annotations]
        logger.warning("[Mode2] Giá»›i háº¡n %d annotations (max=%d)", max_annotations, max_annotations)

    with _suppress_mupdf_stderr():
        ref_doc = fitz.open(ref_pdf_path)
        final_doc = fitz.open(final_pdf_path)

    if ref_doc.page_count > max_pages or final_doc.page_count > max_pages:
        ref_doc.close()
        final_doc.close()
        raise ValueError(f"PDF exceeds max_pages={max_pages}")

    client = get_openai_client(api_key=api_key)

    annotations_by_page: Dict[int, List[Dict]] = {}
    for ann in annotations:
        annotations_by_page.setdefault(ann["page"], []).append(ann)

    num_pages = min(ref_doc.page_count, final_doc.page_count)
    logger.info("[Mode2] Ref %d trang, Final %d trang â†’ xá»­ lĂ½ %d trang", ref_doc.page_count, final_doc.page_count, num_pages)

    results: List[Dict] = []
    ref_page_images: Dict[int, bytes] = {}
    final_page_images: Dict[int, bytes] = {}

    for i in range(num_pages):
        if i not in annotations_by_page:
            continue

        ref_page = ref_doc.load_page(i)
        final_page = final_doc.load_page(i)

        if i not in ref_page_images:
            ref_page_images[i] = convert_full_page_to_image(ref_page, zoom=1.5)
        if i not in final_page_images:
            final_page_images[i] = convert_full_page_to_image(final_page, zoom=1.5)

        if save_images_dir:
            img_dir = os.path.join(save_images_dir, "gpt_images")
            os.makedirs(img_dir, exist_ok=True)
            page_label = i + 1
            ref_img_path = os.path.join(img_dir, f"ref_page_{page_label}.png")
            final_img_path = os.path.join(img_dir, f"final_page_{page_label}.png")
            try:
                with open(ref_img_path, "wb") as f:
                    f.write(ref_page_images[i])
                with open(final_img_path, "wb") as f:
                    f.write(final_page_images[i])
                logger.info("[Mode2] Da luu anh trang %d -> %s", page_label, img_dir)
            except Exception as e:
                logger.warning("[Mode2] Khong luu duoc anh: %s", e)
        ann_list = annotations_by_page[i]
        logger.info("[Mode2] Trang %d: kiá»ƒm tra %d annotation(s)", i + 1, len(ann_list))

        for idx, ann_data in enumerate(ann_list):
            annotation_content = ann_data["content"]
            rect = _rect_from_dict(ann_data["rect"])

            logger.info("[Mode2]   [%d/%d] GPT Vision: %s...", idx + 1, len(ann_list), (annotation_content[:50] + "â€¦") if len(annotation_content) > 50 else annotation_content)

            check_result = check_annotation_with_gpt(
                client=client,
                annotation_content=annotation_content,
                ref_page_image_bytes=ref_page_images[i],
                final_page_image_bytes=final_page_images[i],
                model=model_name,
            )

            status_val = check_result.get("status", "unclear")
            logger.info("[Mode2]     â†’ status=%s, confidence=%.2f", status_val, check_result.get("confidence", 0))

            result_entry = {
                "page": i + 1,
                "status": status_val,
                "implemented": check_result.get("implemented"),
                "reasoning": check_result.get("reasoning", ""),
                "evidence": check_result.get("evidence", ""),
                "confidence": check_result.get("confidence", 0.0),
                "annotation": annotation_content,
            }
            results.append(result_entry)

            try:
                _annotate_status(final_page, rect, check_result)
            except Exception:
                pass

    with _suppress_mupdf_stderr():
        final_doc.save(output_path, garbage=4, deflate=True)
    ref_doc.close()
    final_doc.close()

    summary = {
        "total_annotations": len(results),
        "implemented": sum(1 for r in results if r["status"] == "implemented"),
        "not_implemented": sum(1 for r in results if r["status"] == "not_implemented"),
        "partial": sum(1 for r in results if r["status"] == "partial"),
        "unclear": sum(1 for r in results if r["status"] == "unclear"),
    }

    logger.info("[Mode2] HoĂ n thĂ nh: output=%s | implemented=%d, not_implemented=%d, partial=%d, unclear=%d",
                output_path, summary["implemented"], summary["not_implemented"], summary["partial"], summary["unclear"])

    return {
        "output_pdf": output_path,
        "results": results,
        "summary": summary,
        "preprocessing": preprocess_metadata,
    }


__all__ = [
    "compare_mode2",
    "extract_popup_annotations",
    "check_annotation_with_gpt",
    "get_openai_client",
]



