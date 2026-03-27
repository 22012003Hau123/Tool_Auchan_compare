"""
Mode 3: Charte Compliance Check
Segmentation / YOLO detect zones + deterministic code checks + GPT Vision crop checks + annotated PDF output.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image
import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception:
    A = None
    ToTensorV2 = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent / ".env"
    if _env.exists():
        load_dotenv(dotenv_path=_env, override=True)
except Exception:
    pass

logger = logging.getLogger("mode3_charte")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [mode3] %(levelname)s: %(message)s"))
    logger.addHandler(_h)

DPI = 300
PX_TO_MM = 25.4 / DPI

SEG_IMG_SIZE = 512
SEG_CLASS_NAMES = {
    0: "background",
    1: "brand_logo",
    2: "price_main",
    3: "product_image",
    4: "product_name",
    5: "promo_badge",
}

# ---------------------------------------------------------------------------
#  Model discovery
# ---------------------------------------------------------------------------

def _find_yolo_model() -> Optional[Path]:
    base = Path(__file__).resolve().parent
    for candidate in [
        base / "models" / "bbox_id.pt",
        base.parent / "models" / "bbox_id.pt",
    ]:
        if candidate.exists():
            return candidate
    return None


def _find_seg_model() -> Optional[Path]:
    base = Path(__file__).resolve().parent
    for candidate in [
        base / "models" / "best_model.pth",
        base.parent / "models" / "best_model.pth",
    ]:
        if candidate.exists():
            return candidate
    return None


def _render_page_to_pil(page: fitz.Page) -> Image.Image:
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


# ---------------------------------------------------------------------------
#  Segmentation model (DeepLabV3+ best_model.pth)
# ---------------------------------------------------------------------------

_seg_model_cache: Dict[str, Any] = {}


def _load_seg_model(model_path: Path) -> Optional[Any]:
    if torch is None or smp is None:
        return None

    key = str(model_path)
    if key in _seg_model_cache:
        return _seg_model_cache[key]

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)

        num_classes = checkpoint.get("num_classes", 6)
        encoder = checkpoint.get("encoder", "resnet34")

        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        _seg_model_cache[key] = {"model": model, "device": device}
        logger.info("Loaded segmentation model: %s (encoder=%s, classes=%d)", model_path, encoder, num_classes)
        return _seg_model_cache[key]
    except Exception as e:
        logger.warning("Cannot load segmentation model: %s", e)
        return None


def _seg_predict_crop(model_info: Dict, crop_img: Image.Image) -> np.ndarray:
    """Run segmentation on a single YOLO crop, return class mask at crop size."""
    if A is None or ToTensorV2 is None:
        raise RuntimeError("albumentations required for segmentation")

    import cv2
    model = model_info["model"]
    device = model_info["device"]

    img_np = np.array(crop_img.convert("RGB"))
    orig_h, orig_w = img_np.shape[:2]

    transform = A.Compose([
        A.Resize(SEG_IMG_SIZE, SEG_IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    augmented = transform(image=img_np)
    tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        mask_512 = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    mask_full = cv2.resize(mask_512.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask_full


def _extract_subzones_from_mask(
    seg_mask_crop: np.ndarray,
    yolo_bbox: List[float],
    page_idx: int,
    page_img: Image.Image,
    yolo_conf: float,
) -> List[Dict[str, Any]]:
    """Extract ALL classes found in a segmentation mask of one YOLO crop.
    Each class with enough pixels becomes a sub-zone with its own bbox and mask."""
    import cv2

    ix1, iy1 = int(yolo_bbox[0]), int(yolo_bbox[1])
    crop_h, crop_w = seg_mask_crop.shape[:2]
    total_px = crop_h * crop_w
    min_area = max(200, total_px * 0.005)

    subzones: List[Dict[str, Any]] = []

    for cls_id in range(1, len(SEG_CLASS_NAMES)):
        cls_name = SEG_CLASS_NAMES[cls_id]
        binary = (seg_mask_crop == cls_id).astype(np.uint8)
        px_count = int(binary.sum())
        if px_count < min_area:
            continue

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            cx, cy, cw, ch = cv2.boundingRect(cnt)

            abs_x1 = float(ix1 + cx)
            abs_y1 = float(iy1 + cy)
            abs_x2 = float(ix1 + cx + cw)
            abs_y2 = float(iy1 + cy + ch)

            component_mask_crop = np.zeros_like(binary)
            cv2.drawContours(component_mask_crop, [cnt], -1, 1, cv2.FILLED)

            full_mask = np.zeros((page_img.height, page_img.width), dtype=np.uint8)
            full_mask[iy1:iy1 + crop_h, ix1:ix1 + crop_w] = component_mask_crop

            ratio = area / total_px

            subzones.append({
                "class_name": cls_name,
                "confidence": yolo_conf * max(ratio, 0.3),
                "seg_ratio": round(ratio, 3),
                "page_idx": page_idx,
                "bbox_px": [abs_x1, abs_y1, abs_x2, abs_y2],
                "bbox_mm": [
                    abs_x1 * PX_TO_MM, abs_y1 * PX_TO_MM,
                    abs_x2 * PX_TO_MM, abs_y2 * PX_TO_MM,
                ],
                "width_mm": cw * PX_TO_MM,
                "height_mm": ch * PX_TO_MM,
                "pil_image": page_img,
                "seg_mask": full_mask,
                "seg_mask_crop": component_mask_crop,
                "yolo_bbox_px": list(yolo_bbox),
            })

    return subzones


# ---------------------------------------------------------------------------
#  YOLO Detection
# ---------------------------------------------------------------------------

def _run_yolo(final_doc: fitz.Document) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    """Run YOLO on all pages. Returns raw detections and page images."""
    model_path = _find_yolo_model()
    if model_path is None:
        logger.warning("YOLO model not found")
        return [], []

    try:
        from services.detector import BBoxIDDetector
        detector = BBoxIDDetector(model_path, conf_threshold=0.3)
    except Exception as e:
        logger.warning("Cannot load YOLO detector: %s", e)
        return [], []

    all_dets: List[Dict[str, Any]] = []
    page_imgs: List[Image.Image] = []

    for page_idx in range(final_doc.page_count):
        page = final_doc.load_page(page_idx)
        img = _render_page_to_pil(page)
        page_imgs.append(img)

        detections = detector.predict(img, conf=0.3)
        for det in detections:
            x1, y1, x2, y2 = [float(v) for v in det.bbox[:4]]
            all_dets.append({
                "confidence": det.confidence,
                "page_idx": page_idx,
                "bbox_px": [x1, y1, x2, y2],
                "pil_image": img,
            })

    return all_dets, page_imgs


# ---------------------------------------------------------------------------
#  Pipeline: YOLO bbox → crop → Segmentation → extract ALL classes per bbox
# ---------------------------------------------------------------------------

def _detect_zones(final_doc: fitz.Document) -> Tuple[List[Dict[str, Any]], str]:
    """
    Step 1: YOLO detects bounding boxes (generic 'bbox') on each page.
    Step 2: For each YOLO bbox, crop the region.
    Step 3: Run Segmentation on the crop → pixel mask with multiple classes.
    Step 4: Extract ALL classes found → each becomes a sub-zone.
    Returns (sub_zones, method_used).
    """

    yolo_dets, page_imgs = _run_yolo(final_doc)
    if not yolo_dets:
        return [], "none"

    seg_path = _find_seg_model()
    seg_model = None
    if seg_path and torch is not None and smp is not None and A is not None:
        seg_model = _load_seg_model(seg_path)

    if not seg_model:
        logger.warning("Segmentation model not available — cannot classify YOLO boxes")
        zones = []
        for det in yolo_dets:
            x1, y1, x2, y2 = det["bbox_px"]
            zones.append({
                "class_name": "bbox",
                "confidence": det["confidence"],
                "page_idx": det["page_idx"],
                "bbox_px": det["bbox_px"],
                "bbox_mm": [x1 * PX_TO_MM, y1 * PX_TO_MM, x2 * PX_TO_MM, y2 * PX_TO_MM],
                "width_mm": (x2 - x1) * PX_TO_MM,
                "height_mm": (y2 - y1) * PX_TO_MM,
                "pil_image": det["pil_image"],
            })
        return zones, "yolo"

    method = "yolo+segmentation"
    logger.info("Pipeline: YOLO found %d boxes, running segmentation on each crop...", len(yolo_dets))

    all_subzones: List[Dict[str, Any]] = []
    yolo_boxes_info: List[Dict] = []

    for det in yolo_dets:
        x1, y1, x2, y2 = det["bbox_px"]
        img = det["pil_image"]
        page_idx = det["page_idx"]

        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(img.width, int(x2)), min(img.height, int(y2))
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        crop = img.crop((ix1, iy1, ix2, iy2))
        seg_mask_crop = _seg_predict_crop(seg_model, crop)

        subzones = _extract_subzones_from_mask(
            seg_mask_crop, [x1, y1, x2, y2], page_idx, img, det["confidence"],
        )

        classes_found = [sz["class_name"] for sz in subzones]
        yolo_boxes_info.append({
            "bbox_px": det["bbox_px"],
            "page_idx": page_idx,
            "seg_mask_crop": seg_mask_crop,
            "classes_found": classes_found,
        })

        all_subzones.extend(subzones)

    class_counts = {}
    for sz in all_subzones:
        class_counts[sz["class_name"]] = class_counts.get(sz["class_name"], 0) + 1
    logger.info("Extracted %d sub-zones from %d YOLO boxes: %s",
                len(all_subzones), len(yolo_dets), class_counts)

    for sz in all_subzones:
        sz["_yolo_boxes_info"] = yolo_boxes_info

    return all_subzones, method


# ---------------------------------------------------------------------------
#  Hard Rules Loading
# ---------------------------------------------------------------------------

def _load_hard_rules(hard_rules_path: str) -> Dict[str, Any]:
    p = Path(hard_rules_path)
    if not p.exists():
        logger.error("Hard rules file not found: %s", hard_rules_path)
        return {"metadata": {"missing": True}, "rules": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"metadata": {"invalid_json": True}, "rules": []}


def _hex_to_rgb(hex_color: str) -> Optional[Tuple[int, int, int]]:
    v = (hex_color or "").strip().lstrip("#")
    if len(v) != 6:
        return None
    try:
        return (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Phase 1: Deterministic Code Checks
# ---------------------------------------------------------------------------

def _sample_color_in_zone(zone: Dict, target_rgb: Tuple[int, int, int], tol: int = 20) -> bool:
    """Sample color within a detected zone, using pixel mask if available (segmentation)."""
    img = zone["pil_image"]
    bbox_px = zone["bbox_px"]
    x1, y1, x2, y2 = [int(v) for v in bbox_px]
    w, h = img.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return False

    crop = np.array(img.crop((x1, y1, x2, y2)))
    tr, tg, tb = target_rgb

    seg_mask = zone.get("seg_mask")
    if seg_mask is not None:
        mask_crop = seg_mask[y1:y2, x1:x2]
        if mask_crop.shape[:2] != crop.shape[:2]:
            return False
        pixels = crop[mask_crop > 0]
        if len(pixels) == 0:
            return False
        match = (
            (np.abs(pixels[:, 0].astype(int) - tr) <= tol) &
            (np.abs(pixels[:, 1].astype(int) - tg) <= tol) &
            (np.abs(pixels[:, 2].astype(int) - tb) <= tol)
        )
        return bool(np.any(match))

    color_mask = (
        (np.abs(crop[:, :, 0].astype(int) - tr) <= tol) &
        (np.abs(crop[:, :, 1].astype(int) - tg) <= tol) &
        (np.abs(crop[:, :, 2].astype(int) - tb) <= tol)
    )
    return bool(np.any(color_mask))


def _has_color_fullpage(doc: fitz.Document, target_rgb: Tuple[int, int, int], tol: int = 20) -> bool:
    tr, tg, tb = target_rgb
    for i in range(min(doc.page_count, 10)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5), alpha=False)
        data = pix.samples
        for idx in range(0, len(data), 3):
            if (abs(data[idx] - tr) <= tol and
                abs(data[idx + 1] - tg) <= tol and
                abs(data[idx + 2] - tb) <= tol):
                return True
    return False


def _extract_text_from_rect(page: fitz.Page, bbox_px: List[float], img_size: Tuple[int, int]) -> str:
    """Extract text from a PDF rect corresponding to a pixel bbox."""
    page_rect = page.rect
    iw, ih = img_size
    sx = page_rect.width / iw
    sy = page_rect.height / ih
    x1, y1, x2, y2 = bbox_px
    clip = fitz.Rect(x1 * sx, y1 * sy, x2 * sx, y2 * sy)
    return page.get_text("text", clip=clip).strip()


def _zones_by_applicable(zones: List[Dict], applicable_to: Optional[List[str]]) -> Dict[str, List[Dict]]:
    """Filter zones by applicable_to; return dict of class_name -> zones."""
    if not applicable_to:
        all_zones = {c: [z for z in zones if z["class_name"] == c] for c in SEG_CLASS_NAMES.values() if c != "background"}
        return {k: v for k, v in all_zones.items() if v}
    out: Dict[str, List[Dict]] = {}
    for c in applicable_to:
        lst = [z for z in zones if z["class_name"] == c]
        if lst:
            out[c] = lst
    return out


def _check_color(
    final_doc: fitz.Document,
    rule: Dict,
    zones_by_class: Dict[str, List[Dict]],
    lang: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Check color rule: sample pixels in seg_mask, tolerance 20."""
    constraint = rule.get("constraint", {}) or {}
    rgb = _hex_to_rgb(constraint.get("hex", ""))
    if rgb is None:
        return _make_result(rule, "unclear", 0.2, "Missing/invalid HEX value", lang)

    relevant_zones = []
    for k in ("promo_badge", "price_main", "brand_logo", "product_name"):
        relevant_zones.extend(zones_by_class.get(k, []))

    found = False
    evidence_bbox = None
    for z in relevant_zones:
        if _sample_color_in_zone(z, rgb, tol=20):
            found = True
            evidence_bbox = z
            break

    if not found:
        found = _has_color_fullpage(final_doc, rgb, tol=20)

    status = "pass" if found else "fail"
    reason = f"Color {constraint.get('hex')}: {'found' if found else 'not found'}"
    r = _make_result(rule, status, 0.7, reason, lang)
    if evidence_bbox:
        r["evidence_page"] = evidence_bbox["page_idx"]
        r["evidence_bbox_px"] = evidence_bbox["bbox_px"]
    return r


def _check_dimension(
    rule: Dict,
    zones_by_class: Dict[str, List[Dict]],
    lang: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Check dimension rule: measure bbox in mm."""
    constraint = rule.get("constraint", {}) or {}
    max_w = constraint.get("max_width_mm")
    max_h = constraint.get("max_height_mm")
    min_w = constraint.get("min_width_mm")
    min_h = constraint.get("min_height_mm")

    applicable = rule.get("applicable_to") or ["price_main"]
    cand_zones = []
    for c in applicable:
        cand_zones.extend(zones_by_class.get(c, []))

    if not cand_zones:
        return _make_result(rule, "unclear", 0.3, "No applicable zones detected", lang)

    if max_w is not None:
        violations = [z for z in cand_zones if z.get("width_mm", 0) > max_w]
        if violations:
            r = _make_result(rule, "fail", 0.8,
                f"width {violations[0]['width_mm']:.1f}mm > max {max_w}mm", lang)
            r["evidence_page"] = violations[0]["page_idx"]
            r["evidence_bbox_px"] = violations[0]["bbox_px"]
            return r
        widths = [z.get("width_mm", 0) for z in cand_zones]
        return _make_result(rule, "pass", 0.8, f"All widths within {max_w}mm (max: {max(widths):.1f}mm)", lang)

    if max_h is not None:
        violations = [z for z in cand_zones if z.get("height_mm", 0) > max_h]
        if violations:
            r = _make_result(rule, "fail", 0.8,
                f"height {violations[0]['height_mm']:.1f}mm > max {max_h}mm", lang)
            r["evidence_page"] = violations[0]["page_idx"]
            r["evidence_bbox_px"] = violations[0]["bbox_px"]
            return r
        return _make_result(rule, "pass", 0.8, f"All heights within {max_h}mm", lang)

    if min_w is not None or min_h is not None:
        return _make_result(rule, "unclear", 0.3, "min_width/min_height not yet implemented", lang)

    return _make_result(rule, "unclear", 0.2, "No dimension constraint in rule", lang)


def _check_spacing(
    rule: Dict,
    zones_by_class: Dict[str, List[Dict]],
    lang: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Check spacing rule: distance between sub-zones in same bbox."""
    constraint = rule.get("constraint", {}) or {}
    dist_mm = constraint.get("distance_to_price_block_mm") or constraint.get("distance_mm")
    if dist_mm is None:
        return _make_result(rule, "unclear", 0.2, "No distance constraint", lang)

    price_zones = zones_by_class.get("price_main", [])
    promo_zones = zones_by_class.get("promo_badge", [])
    name_zones = zones_by_class.get("product_name", [])

    if not price_zones:
        return _make_result(rule, "unclear", 0.3, "No price_main zone detected", lang)

    for pz in price_zones:
        pz_yolo = pz.get("yolo_bbox_px")
        neighbors = [z for z in promo_zones + name_zones
                     if z["page_idx"] == pz["page_idx"]
                     and z.get("yolo_bbox_px") == pz_yolo]
        if not neighbors:
            neighbors = [z for z in promo_zones + name_zones if z["page_idx"] == pz["page_idx"]]

        for other in neighbors:
            gap_px = abs(other["bbox_px"][1] - pz["bbox_px"][3])
            gap_mm = gap_px * PX_TO_MM
            tolerance = max(1.0, dist_mm * 0.5)
            status = "pass" if abs(gap_mm - dist_mm) <= tolerance else "fail"
            r = _make_result(rule, status, 0.6, f"Measured gap: {gap_mm:.1f}mm (expected: {dist_mm}mm)", lang)
            r["evidence_page"] = pz["page_idx"]
            r["evidence_bbox_px"] = pz["bbox_px"]
            return r

    return _make_result(rule, "unclear", 0.3, "Could not find adjacent elements to measure spacing", lang)


def _evaluate_hard_rules_code(
    final_doc: fitz.Document,
    rules: List[Dict],
    zones: List[Dict],
    lang: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Phase 1: deterministic checks — router by check_method and type."""

    code_rules = [r for r in rules if r.get("check_method") == "code"]
    if not code_rules:
        return []

    results: List[Dict[str, Any]] = []

    for rule in code_rules:
        rtype = rule.get("type", "unknown")
        constraint = rule.get("constraint", {}) if isinstance(rule.get("constraint"), dict) else {}
        applicable_to = rule.get("applicable_to")

        zones_by_class = _zones_by_applicable(zones, applicable_to)

        if rtype == "color":
            if constraint.get("check_method") == "visual":
                continue  # defer to GPT
            r = _check_color(final_doc, rule, zones_by_class, lang)
            if r:
                results.append(r)

        elif rtype == "dimension":
            r = _check_dimension(rule, zones_by_class, lang)
            if r:
                results.append(r)

        elif rtype == "spacing":
            r = _check_spacing(rule, zones_by_class, lang)
            if r:
                results.append(r)

        # content removed — Mode 1 handles content
        # typography, forbidden_effect, layout → GPT phase

    return results


def _get_rule_title(rule: Dict, lang: Optional[str] = None) -> str:
    """Get rule title by lang (fr/vi/en). Falls back to title_fr, then title."""
    if lang and lang in ("vi", "en"):
        key = f"title_{lang}"
        val = rule.get(key)
        if val:
            return val
    return rule.get("title_fr") or rule.get("title", "")


def _make_result(rule: Dict, status: str, confidence: float, reason: str, lang: Optional[str] = None) -> Dict[str, Any]:
    return {
        "rule_id": rule.get("rule_id", "unknown"),
        "title": _get_rule_title(rule, lang),
        "type": rule.get("type", "unknown"),
        "severity": rule.get("severity", "major"),
        "status": status,
        "confidence": confidence,
        "reason": reason,
        "source": rule.get("source", {}),
    }


# ---------------------------------------------------------------------------
#  Phase 2: GPT Vision Crop Checks
# ---------------------------------------------------------------------------

def _openai_client() -> Optional[Any]:
    if OpenAI is None:
        return None
    key = os.getenv("MODE3_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None


def _pil_to_b64(img: Image.Image, max_side: int = 1024) -> str:
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _crop_zone(img: Image.Image, bbox_px: List[float], padding: int = 20) -> Image.Image:
    w, h = img.size
    x1 = max(0, int(bbox_px[0]) - padding)
    y1 = max(0, int(bbox_px[1]) - padding)
    x2 = min(w, int(bbox_px[2]) + padding)
    y2 = min(h, int(bbox_px[3]) + padding)
    return img.crop((x1, y1, x2, y2))


def _pick_zone_for_rule(zones: List[Dict], rule: Dict) -> Optional[Dict]:
    """Pick first zone matching rule's applicable_to, or fallback by type."""
    applicable = rule.get("applicable_to") or []
    rtype = rule.get("type", "")
    for cls in applicable:
        cand = [z for z in zones if z["class_name"] == cls]
        if cand:
            return cand[0]
    if rtype == "typography":
        for cls in ("price_main", "promo_badge", "product_name"):
            cand = [z for z in zones if z["class_name"] == cls]
            if cand:
                return cand[0]
    elif rtype == "forbidden_effect":
        cand = [z for z in zones if z["class_name"] == "product_image"]
        if cand:
            return cand[0]
    return zones[0] if zones else None


def _evaluate_hard_rules_gpt(
    rules: List[Dict],
    zones: List[Dict],
    page_images: Optional[List[Image.Image]] = None,
    model: Optional[str] = None,
    lang: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Phase 2: GPT Vision for typography, forbidden_effect, layout, visual checks."""
    client = _openai_client()
    if client is None:
        return [_make_result(r, "unclear", 0.1, "OpenAI unavailable", lang) for r in rules]

    model_name = model or os.getenv("MODE3_GPT_MODEL", "gpt-5.2")

    gpt_rules = [r for r in rules if r.get("type") in ("typography", "forbidden_effect", "layout")
                 or r.get("check_method") == "visual"
                 or (r.get("constraint") or {}).get("check_method") == "visual"]

    if not gpt_rules:
        return []

    results: List[Dict[str, Any]] = []

    typography_rules = [r for r in gpt_rules if r.get("type") == "typography"]
    typo_result_cache: Optional[Dict] = None

    for rule in gpt_rules:
        rtype = rule.get("type", "")
        constraint = rule.get("constraint", {}) or {}
        rid = rule.get("rule_id", "unknown")
        rule_title = _get_rule_title(rule, lang)

        zone = None
        crop = None
        prompt = ""

        if rtype == "layout":
            img = None
            if page_images and len(page_images) > 0:
                img = page_images[0]
            elif zones:
                img = zones[0]["pil_image"]
            if img:
                crop = img
                prompt = (
                    f"Analyze this full page from a retail flyer.\n"
                    f"Rule: {rule_title}\n"
                    f"Constraint: {json.dumps(constraint, ensure_ascii=False)}\n"
                    f"Source: {rule.get('source', {}).get('quote', '')}\n\n"
                    f"Check layout (max cases per page, flaps, bombes placement, prix/origine position).\n"
                    f"Return JSON: {{\"status\": \"pass\"|\"fail\"|\"unclear\", "
                    f"\"confidence\": 0.0-1.0, \"reason\": \"...\"}}"
                )
            else:
                results.append(_make_result(rule, "unclear", 0.1, "No page image for layout check", lang))
                continue

        elif rtype == "typography":
            if typo_result_cache is not None:
                r = _make_result(rule, typo_result_cache["status"],
                    typo_result_cache["confidence"], typo_result_cache["reason"], lang)
                if typo_result_cache.get("evidence_page") is not None:
                    r["evidence_page"] = typo_result_cache["evidence_page"]
                if typo_result_cache.get("evidence_bbox_px") is not None:
                    r["evidence_bbox_px"] = typo_result_cache["evidence_bbox_px"]
                results.append(r)
                continue
            zone = _pick_zone_for_rule(zones, rule)
            if zone:
                crop = _crop_zone(zone["pil_image"], zone["bbox_px"])
                prompt = (
                    "Analyze this cropped image from a retail flyer.\n"
                    "The only check: Is the main TITLE or SECTION HEADING text displayed in BOLD?\n"
                    "Ignore font name, font size - only verify if the title/heading appears bold.\n"
                    "Return JSON: {\"status\": \"pass\"|\"fail\"|\"unclear\", "
                    "\"confidence\": 0.0-1.0, \"reason\": \"...\"}"
                )
            else:
                results.append(_make_result(rule, "unclear", 0.2, "No zone for typography check", lang))
                continue

        elif rtype == "forbidden_effect":
            zone = _pick_zone_for_rule(zones, rule)
            if zone:
                crop = _crop_zone(zone["pil_image"], zone["bbox_px"])
                prompt = (
                    f"Analyze this cropped packshot image from a retail flyer.\n"
                    f"Rule: No drop shadows, diffuse shadows, or outer glow effects allowed.\n"
                    f"Forbidden: {constraint.get('forbidden', [])}\n"
                    f"Exception: {constraint.get('exception', 'none')}\n\n"
                    f"Does this packshot have any shadow, glow, or visual effects?\n"
                    f"Return JSON: {{\"status\": \"pass\"|\"fail\"|\"unclear\", "
                    f"\"confidence\": 0.0-1.0, \"reason\": \"...\"}}"
                )
            else:
                results.append(_make_result(rule, "unclear", 0.2, "No product_image zone for effect check", lang))
                continue

        elif rule.get("check_method") == "visual" or constraint.get("check_method") == "visual":
            zone = _pick_zone_for_rule(zones, rule)
            if zone:
                crop = _crop_zone(zone["pil_image"], zone["bbox_px"])
                prompt = (
                    f"Analyze this image area from a retail flyer.\n"
                    f"Rule: {rule_title}\n"
                    f"Constraint: {json.dumps(constraint, ensure_ascii=False)}\n"
                    f"Source: {rule.get('source', {}).get('quote', '')}\n\n"
                    f"Does this comply with the rule?\n"
                    f"Return JSON: {{\"status\": \"pass\"|\"fail\"|\"unclear\", "
                    f"\"confidence\": 0.0-1.0, \"reason\": \"...\"}}"
                )
            else:
                results.append(_make_result(rule, "unclear", 0.1, "No zones to crop for visual check", lang))
                continue
        else:
            results.append(_make_result(rule, "unclear", 0.2, f"No zone available for {rtype} check", lang))
            continue

        if crop is None or not prompt:
            continue

        try:
            b64 = _pil_to_b64(crop)
            resp = client.chat.completions.create(
                model=model_name,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_completion_tokens=300,
                messages=[
                    {"role": "system", "content": "You are a print design compliance auditor. Respond with JSON only."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ]},
                ],
            )
            raw = resp.choices[0].message.content if resp.choices else None
            if not raw or not isinstance(raw, (str, bytes, bytearray)):
                if rtype == "typography" and typo_result_cache is None:
                    typo_result_cache = {"status": "unclear", "confidence": 0.1, "reason": "GPT returned empty response",
                        "evidence_page": None, "evidence_bbox_px": None}
                results.append(_make_result(rule, "unclear", 0.1, "GPT returned empty response", lang))
                continue
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, TypeError) as je:
                logger.warning("GPT JSON parse failed for %s: %s", rid, je)
                if rtype == "typography" and typo_result_cache is None:
                    typo_result_cache = {"status": "unclear", "confidence": 0.1, "reason": str(je),
                        "evidence_page": None, "evidence_bbox_px": None}
                results.append(_make_result(rule, "unclear", 0.1, f"GPT response invalid: {je}", lang))
                continue
            r = _make_result(rule,
                payload.get("status", "unclear"),
                float(payload.get("confidence", 0.5)),
                payload.get("reason", "GPT response"),
                lang)
            if zone and "page_idx" in zone:
                r["evidence_page"] = zone["page_idx"]
                r["evidence_bbox_px"] = zone["bbox_px"]
            elif page_images and len(page_images) > 0:
                r["evidence_page"] = 0
            if rtype == "typography" and typo_result_cache is None:
                typo_result_cache = {
                    "status": r["status"],
                    "confidence": r["confidence"],
                    "reason": r["reason"],
                    "evidence_page": r.get("evidence_page"),
                    "evidence_bbox_px": r.get("evidence_bbox_px"),
                }
            results.append(r)
        except Exception as e:
            logger.warning("GPT check failed for %s: %s", rid, e)
            err_r = _make_result(rule, "unclear", 0.1, f"GPT call failed: {e}", lang)
            if rtype == "typography" and typo_result_cache is None:
                typo_result_cache = {"status": "unclear", "confidence": 0.1, "reason": str(e),
                    "evidence_page": None, "evidence_bbox_px": None}
            results.append(err_r)

    return results


# ---------------------------------------------------------------------------
#  Phase 3: Soft Rules (GPT Vision full-page)
# ---------------------------------------------------------------------------

def _evaluate_soft_rules(
    charte_doc: fitz.Document,
    final_doc: fitz.Document,
    hard_rules: List[Dict],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    client = _openai_client()
    if client is None:
        return {"status": "skipped", "score": 0.0, "reason": "OpenAI unavailable", "issues": []}

    model_name = model or os.getenv("MODE3_GPT_MODEL", "gpt-5.2")

    charte_pages = set()
    for r in hard_rules:
        src = r.get("source", {})
        if "page" in src:
            charte_pages.add(src["page"] - 1)
    charte_pages = sorted(charte_pages)[:3]
    if not charte_pages:
        charte_pages = [0]

    charte_context = []
    for pi in charte_pages:
        if pi < charte_doc.page_count:
            txt = charte_doc.load_page(pi).get_text("text")[:500]
            if txt.strip():
                charte_context.append(f"[Charte p.{pi+1}]: {txt}")

    images = []
    for pi in range(min(final_doc.page_count, 4)):
        img = _render_page_to_pil(final_doc.load_page(pi))
        images.append(_pil_to_b64(img, max_side=800))

    context_text = "\n".join(charte_context[:3]) if charte_context else "No charte text available."

    prompt = (
        "You are evaluating a Produit Fini (retail flyer) against the Auchan Charte design guidelines.\n\n"
        f"Charte context:\n{context_text}\n\n"
        "Evaluate the following images of the Produit Fini for:\n"
        "1. Grid layout consistency\n"
        "2. Price hierarchy (main price prominent, secondary prices smaller)\n"
        "3. Visual balance and spacing\n"
        "4. Color palette consistency with Auchan branding\n"
        "5. Typography consistency\n"
        "6. Overall professional quality\n\n"
        "Return JSON: {\"status\": \"pass\"|\"partial\"|\"fail\", "
        "\"score\": 0.0-1.0, \"issues\": [\"issue1\", ...], \"reasoning\": \"...\"}"
    )

    content: list = [{"type": "text", "text": prompt}]
    for b64 in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    try:
        resp = client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            temperature=0.1,
            max_completion_tokens=500,
            messages=[
                {"role": "system", "content": "You are a design compliance auditor for Auchan retail. JSON only."},
                {"role": "user", "content": content},
            ],
        )
        payload = json.loads(resp.choices[0].message.content)
        return {
            "status": payload.get("status", "ok"),
            "score": float(payload.get("score", 0.0)),
            "reason": payload.get("reasoning", ""),
            "issues": payload.get("issues", []),
        }
    except Exception as e:
        logger.error("Soft rules evaluation failed: %s", e)
        return {"status": "error", "score": 0.0, "reason": str(e), "issues": []}


# ---------------------------------------------------------------------------
#  Annotate PDF
# ---------------------------------------------------------------------------

def _annotate_results(
    final_doc: fitz.Document,
    hard_results: List[Dict],
    page_images: Dict[int, Tuple[int, int]],
) -> None:
    """Add colored rect annotations to the PDF based on rule results."""
    color_map = {
        "pass": (0, 0.7, 0),
        "fail": (1, 0, 0),
        "unclear": (1, 0.8, 0),
    }

    for r in hard_results:
        status = r.get("status", "unclear")
        color = color_map.get(status, (0.5, 0.5, 0.5))
        page_idx = r.get("evidence_page")
        bbox_px = r.get("evidence_bbox_px")

        if page_idx is not None and page_idx < final_doc.page_count:
            page = final_doc.load_page(page_idx)
            page_rect = page.rect
            img_size = page_images.get(page_idx)

            if bbox_px and img_size:
                iw, ih = img_size
                sx = page_rect.width / iw
                sy = page_rect.height / ih
                rect = fitz.Rect(
                    bbox_px[0] * sx, bbox_px[1] * sy,
                    bbox_px[2] * sx, bbox_px[3] * sy,
                )
            else:
                margin = 5
                rect = fitz.Rect(
                    page_rect.x0 + margin, page_rect.y0 + margin,
                    page_rect.x1 - margin, page_rect.y1 - margin,
                )

            try:
                annot = page.add_rect_annot(rect)
                annot.set_colors(stroke=color)
                annot.set_border(width=1.5 if status == "pass" else 2.5)
                annot.set_opacity(0.5)

                label = f"{r.get('rule_id', '?')} [{status}]"
                annot.set_info(title="Mode3", content=label)
                annot.update()
            except Exception as e:
                logger.warning("Failed to annotate %s: %s", r.get("rule_id"), e)


# ---------------------------------------------------------------------------
#  Debug Visualization
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    "product_image": (0, 200, 0),
    "price_main": (255, 0, 0),
    "promo_badge": (255, 165, 0),
    "brand_logo": (0, 100, 255),
    "product_name": (200, 0, 200),
    "bbox": (128, 128, 128),
}


def _save_debug_images(
    zones: List[Dict[str, Any]],
    detect_method: str,
    gpt_rules: List[Dict],
    output_dir: Path,
) -> Dict[str, Any]:
    """Save debug visualizations: YOLO boxes + seg sub-zones overlay, crops, seg masks."""
    import cv2

    debug_dir = output_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    page_groups: Dict[int, List[Dict]] = {}
    for z in zones:
        page_groups.setdefault(z["page_idx"], []).append(z)

    overlay_files = []
    crop_files = []
    yolo_crop_files = []

    for page_idx, page_zones in sorted(page_groups.items()):
        if not page_zones:
            continue
        img = page_zones[0]["pil_image"].copy()
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        for i, z in enumerate(page_zones):
            cls = z["class_name"]
            seg_ratio = z.get("seg_ratio")
            color = CLASS_COLORS.get(cls, (128, 128, 128))
            bgr = (color[2], color[1], color[0])
            x1, y1, x2, y2 = [int(v) for v in z["bbox_px"]]

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), bgr, 2)

            pct = f" {seg_ratio:.0%}" if seg_ratio is not None else ""
            label = f"{cls}{pct}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img_cv, (x1, y1 - th - 6), (x1 + tw + 4, y1), bgr, -1)
            cv2.putText(img_cv, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            crop_pil = img.crop((max(0, x1 - 5), max(0, y1 - 5),
                                 min(img.width, x2 + 5), min(img.height, y2 + 5)))
            crop_name = f"crop_p{page_idx}_{i}_{cls}.jpg"
            crop_pil.save(str(debug_dir / crop_name), quality=85)

            seg_mask_crop = z.get("seg_mask_crop")
            mask_name = None
            if seg_mask_crop is not None:
                mask_vis = np.zeros((*seg_mask_crop.shape, 3), dtype=np.uint8)
                c = CLASS_COLORS.get(cls, (128, 128, 128))
                mask_vis[seg_mask_crop > 0] = c
                mask_name = f"segmask_p{page_idx}_{i}_{cls}.jpg"
                cv2.imwrite(str(debug_dir / mask_name),
                            cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])

            crop_files.append({
                "file": crop_name,
                "mask_file": mask_name,
                "class": cls,
                "seg_ratio": round(seg_ratio, 3) if seg_ratio is not None else None,
                "page": page_idx,
                "width_mm": round(z["width_mm"], 1),
                "height_mm": round(z["height_mm"], 1),
            })

        drawn_yolo = set()
        for z in page_zones:
            yolo_box = z.get("yolo_bbox_px")
            if yolo_box:
                key = tuple(int(v) for v in yolo_box)
                if key not in drawn_yolo:
                    drawn_yolo.add(key)
                    yx1, yy1, yx2, yy2 = key
                    cv2.rectangle(img_cv, (yx1, yy1), (yx2, yy2), (0, 255, 255), 4, cv2.LINE_AA)
                    cv2.putText(img_cv, "YOLO", (yx1, yy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        yolo_info = page_zones[0].get("_yolo_boxes_info", [])
        for yi, yb in enumerate(yb for yb in yolo_info if yb["page_idx"] == page_idx):
            seg_mask_full = yb.get("seg_mask_crop")
            if seg_mask_full is not None:
                mask_vis = np.zeros((*seg_mask_full.shape, 3), dtype=np.uint8)
                for cid, cname in SEG_CLASS_NAMES.items():
                    if cid == 0:
                        continue
                    c = CLASS_COLORS.get(cname, (128, 128, 128))
                    mask_vis[seg_mask_full == cid] = c
                yolo_mask_name = f"yolo_seg_p{page_idx}_{yi}.jpg"
                cv2.imwrite(str(debug_dir / yolo_mask_name),
                            cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])

                ybx = yb["bbox_px"]
                ix1, iy1, ix2, iy2 = int(ybx[0]), int(ybx[1]), int(ybx[2]), int(ybx[3])
                yolo_crop_pil = img.crop((max(0, ix1), max(0, iy1), min(img.width, ix2), min(img.height, iy2)))
                yolo_crop_name = f"yolo_crop_p{page_idx}_{yi}.jpg"
                yolo_crop_pil.save(str(debug_dir / yolo_crop_name), quality=85)

                yolo_crop_files.append({
                    "crop_file": yolo_crop_name,
                    "mask_file": yolo_mask_name,
                    "classes": yb.get("classes_found", []),
                    "page": page_idx,
                })

        overlay_name = f"detect_page_{page_idx}.jpg"
        cv2.imwrite(str(debug_dir / overlay_name), img_cv, [cv2.IMWRITE_JPEG_QUALITY, 85])
        overlay_files.append({"file": overlay_name, "page": page_idx, "zone_count": len(page_zones)})

    gpt_crop_files = []
    price_zones = [z for z in zones if z["class_name"] == "price_main"]
    product_zones = [z for z in zones if z["class_name"] == "product_image"]

    for rule in gpt_rules:
        rtype = rule.get("type", "")
        rid = rule.get("rule_id", "?")
        constraint = rule.get("constraint", {}) or {}

        if rtype == "layout" and zones:
            z = zones[0]
            crop = z["pil_image"]
        elif rtype == "typography" and price_zones:
            z = price_zones[0]
            crop = _crop_zone(z["pil_image"], z["bbox_px"], padding=20)
        elif rtype == "forbidden_effect" and product_zones:
            z = product_zones[0]
            crop = _crop_zone(z["pil_image"], z["bbox_px"], padding=20)
        elif (rule.get("check_method") == "visual" or constraint.get("check_method") == "visual") and (product_zones or zones):
            z = product_zones[0] if product_zones else zones[0]
            crop = _crop_zone(z["pil_image"], z["bbox_px"], padding=20)
        else:
            continue
        crop_name = f"gpt_crop_{rid}.jpg"
        crop.save(str(debug_dir / crop_name), quality=90)
        gpt_crop_files.append({
            "file": crop_name,
            "rule_id": rid,
            "rule_type": rtype,
            "rule_title": rule.get("title_fr") or rule.get("title", ""),
        })

    return {
        "overlays": overlay_files,
        "yolo_crops": yolo_crop_files,
        "crops": crop_files,
        "gpt_crops": gpt_crop_files,
    }


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def compare_mode3(
    charte_pdf_path: str,
    final_pdf_path: str,
    output_dir: str,
    hard_rules_path: Optional[str] = None,
    model: Optional[str] = None,
    lang: Optional[str] = None,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pdf = out_dir / "output_mode3_review.pdf"
    report_file = out_dir / "mode3_report.json"
    hard_path = hard_rules_path or str(
        Path(__file__).resolve().parent / "rules" / "hard_rule" / "hard_rules_auchan_vclients.json"
    )

    logger.info("Loading PDFs: charte=%s, final=%s", charte_pdf_path, final_pdf_path)
    charte_doc = fitz.open(charte_pdf_path)
    final_doc = fitz.open(final_pdf_path)

    # Load rules
    hard_rules_data = _load_hard_rules(hard_path)
    rules = hard_rules_data.get("rules", [])
    logger.info("Loaded %d hard rules from %s", len(rules), hard_path)

    # Detect zones (segmentation preferred, YOLO fallback)
    logger.info("Running zone detection...")
    zones, detect_method = _detect_zones(final_doc)
    logger.info("Detected %d zones via %s across %d pages", len(zones), detect_method, final_doc.page_count)

    # Collect page image sizes for annotation + full page images for layout GPT
    page_size_map: Dict[int, Tuple[int, int]] = {}
    page_pil_by_idx: Dict[int, Image.Image] = {}
    for z in zones:
        pi = z["page_idx"]
        if pi not in page_size_map:
            page_size_map[pi] = z["pil_image"].size
            page_pil_by_idx[pi] = z["pil_image"]
    page_pil_list = [page_pil_by_idx[i] for i in sorted(page_pil_by_idx.keys())]

    # Phase 1: Code checks
    logger.info("Phase 1: Deterministic code checks...")
    code_results = _evaluate_hard_rules_code(final_doc, rules, zones, lang=lang)
    logger.info("Code checks: %d results", len(code_results))

    # Phase 2: GPT Vision checks
    logger.info("Phase 2: GPT Vision crop checks...")
    gpt_rules = [r for r in rules if r.get("type") in ("typography", "forbidden_effect", "layout")
                 or r.get("check_method") == "visual"
                 or (r.get("constraint") or {}).get("check_method") == "visual"]
    gpt_results = _evaluate_hard_rules_gpt(
        gpt_rules, zones, page_images=page_pil_list, model=model, lang=lang
    )
    logger.info("GPT checks: %d results", len(gpt_results))

    # Merge hard results
    all_hard_results = code_results + gpt_results
    checked_ids = {r["rule_id"] for r in all_hard_results}
    for rule in rules:
        if rule.get("rule_id") not in checked_ids:
            all_hard_results.append(_make_result(rule, "unclear", 0.1, "No checker available", lang))

    hard_summary = {
        "total": len(all_hard_results),
        "pass": sum(1 for r in all_hard_results if r["status"] == "pass"),
        "fail": sum(1 for r in all_hard_results if r["status"] == "fail"),
        "unclear": sum(1 for r in all_hard_results if r["status"] == "unclear"),
    }
    logger.info("Hard rules summary: %s", hard_summary)

    # Phase 3: Soft rules
    logger.info("Phase 3: Soft rules evaluation...")
    soft_eval = _evaluate_soft_rules(charte_doc, final_doc, rules, model=model)
    logger.info("Soft rules: status=%s, score=%.2f", soft_eval.get("status"), soft_eval.get("score", 0))

    charte_doc.close()

    # Debug images
    logger.info("Saving debug images...")
    debug_info = _save_debug_images(zones, detect_method, gpt_rules, out_dir)
    logger.info("Debug: %d overlays, %d crops, %d gpt_crops",
                len(debug_info["overlays"]), len(debug_info["crops"]), len(debug_info["gpt_crops"]))

    # Annotate PDF
    logger.info("Annotating PDF...")
    _annotate_results(final_doc, all_hard_results, page_size_map)
    final_doc.save(str(out_pdf))
    final_doc.close()

    # Clean up non-serializable references
    for r in all_hard_results:
        r.pop("pil_image", None)
        r.pop("seg_mask", None)
        r.pop("seg_mask_crop", None)
        r.pop("_yolo_boxes_info", None)
        r.pop("yolo_bbox_px", None)

    # Build detection summary
    zone_summary = {}
    for z in zones:
        cn = z["class_name"]
        zone_summary[cn] = zone_summary.get(cn, 0) + 1

    payload = {
        "output_pdf": str(out_pdf),
        "detect_method": detect_method,
        "zones_detected": zone_summary,
        "hard_rules": {"results": all_hard_results, "summary": hard_summary},
        "soft_rules": soft_eval,
        "hard_rules_source": hard_path,
        "debug": debug_info,
    }
    report_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    return {
        "output_pdf": str(out_pdf),
        "report_file": str(report_file),
        "detect_method": detect_method,
        "hard_summary": hard_summary,
        "hard_results": all_hard_results,
        "soft_summary": {"status": soft_eval.get("status"), "score": soft_eval.get("score")},
        "soft_details": soft_eval,
        "zones_detected": zone_summary,
        "debug": debug_info,
    }


__all__ = ["compare_mode3"]
