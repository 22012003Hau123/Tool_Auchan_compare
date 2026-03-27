"""
Step 5 (DINOv2): Match produits bang visual similarity thay vi ID.
Pass 1 — DINOv2 full-image matching  (B duyet A, threshold ~0.75)
Pass 2 — Segment product_image (class 3) + DINOv2 re-match (cho B con sot)
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ─────────────────────────────────────────────
#  DINOv2 helpers
# ─────────────────────────────────────────────

_dino_cache: dict = {}


def _load_dino(model_name: str = "dinov2_vits14"):
    """Load DINOv2 (cached)."""
    if model_name in _dino_cache:
        return _dino_cache[model_name]

    print(f"[step5_dino] Loading DINOv2 model: {model_name}...")
    t0 = time.perf_counter()
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    _dino_cache[model_name] = (model, preprocess)
    print(f"[step5_dino] Model loaded in {time.perf_counter() - t0:.1f}s")
    return model, preprocess


def _encode_images(
    files: list[Path],
    model,
    preprocess,
) -> torch.Tensor:
    """Encode list of images -> (N, D) normalised embeddings."""
    embs = []
    for f in files:
        img = preprocess(Image.open(f).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            emb = model(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        embs.append(emb)
    return torch.cat(embs, dim=0) if embs else torch.empty(0)


def _encode_pil_images(
    images: list[Image.Image],
    model,
    preprocess,
) -> torch.Tensor:
    """Encode list of PIL images -> (N, D) normalised embeddings."""
    embs = []
    for img in images:
        t = preprocess(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            emb = model(t)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        embs.append(emb)
    return torch.cat(embs, dim=0) if embs else torch.empty(0)


# ─────────────────────────────────────────────
#  Segmentation helpers (reuse DeepLabV3+)
# ─────────────────────────────────────────────

_seg_cache: dict = {}
PRODUCT_IMAGE_CLASS = 3


def _load_seg_model(model_path: Path):
    """Load DeepLabV3+ from best_model.pth (cached)."""
    key = str(model_path)
    if key in _seg_cache:
        return _seg_cache[key]

    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        print("[step5_dino] segmentation_models_pytorch not installed")
        return None

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

        _seg_cache[key] = {"model": model, "device": device}
        print(f"[step5_dino] Loaded seg model: {model_path.name} (encoder={encoder}, classes={num_classes})")
        return _seg_cache[key]
    except Exception as e:
        print(f"[step5_dino] Cannot load seg model: {e}")
        return None


def _segment_product_image(img: Image.Image, seg_info: dict) -> Image.Image | None:
    """
    Segment anh, lay vung product_image (class 3), crop bbox cua vung do.
    Tra ve PIL Image cua vung product_image, hoac None neu khong tim thay.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import cv2
    except ImportError:
        return None

    model = seg_info["model"]
    device = seg_info["device"]
    img_np = np.array(img.convert("RGB"))
    orig_h, orig_w = img_np.shape[:2]

    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    augmented = transform(image=img_np)
    tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        mask_512 = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Resize mask ve kich thuoc goc
    mask_full = cv2.resize(mask_512.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Lay vung product_image (class 3)
    binary = (mask_full == PRODUCT_IMAGE_CLASS).astype(np.uint8)
    px_count = int(binary.sum())
    min_area = max(200, orig_h * orig_w * 0.01)  # it nhat 1% dien tich

    if px_count < min_area:
        return None

    # Tim bbox cua vung product_image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Lay contour lon nhat
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    crop = img.crop((x, y, x + w, y + h))
    return crop


# ─────────────────────────────────────────────
#  Sequential 1-1 matching (B duyet A)
# ─────────────────────────────────────────────

def _sequential_match(
    emb_source: torch.Tensor,
    emb_pool: torch.Tensor,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """
    Dùng Hungarian algorithm (optimal assignment) thay vì greedy sequential.
    Tìm phép ghép 1-1 TỐI ƯU TOÀN CỤC (maximize tổng similarity).
    Return: list of (source_idx, pool_idx, similarity)
    """
    if emb_source.numel() == 0 or emb_pool.numel() == 0:
        return []

    sim = (emb_source @ emb_pool.T).cpu().numpy()  # (Nb, Na)

    try:
        from scipy.optimize import linear_sum_assignment
        # Hungarian minimizes cost → dùng -sim để maximize similarity
        cost = -sim
        row_ind, col_ind = linear_sum_assignment(cost)

        matches = []
        for r, c in zip(row_ind, col_ind):
            s = sim[r, c]
            if s >= threshold:
                matches.append((int(r), int(c), float(s)))
        return matches
    except ImportError:
        print("[step5_dino] WARNING: scipy not installed, fallback to greedy matching")
        # Fallback greedy nếu không có scipy
        available_pool = set(range(emb_pool.shape[0]))
        matches = []

        for i in range(emb_source.shape[0]):
            if not available_pool:
                break

            best_j = -1
            best_sim = -1.0
            for j in available_pool:
                s = sim[i, j]
                if s > best_sim:
                    best_sim = s
                    best_j = j

            if best_sim >= threshold and best_j >= 0:
                available_pool.remove(best_j)
                matches.append((i, best_j, float(best_sim)))

        return matches


# ─────────────────────────────────────────────
#  OCR-based product ID extraction
# ─────────────────────────────────────────────

_ocr_reader = None


def _get_ocr_reader():
    """Load EasyOCR reader (cached)."""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            _ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
            print("[step5_dino] EasyOCR loaded")
        except ImportError:
            print("[step5_dino] EasyOCR not available")
            return None
    return _ocr_reader


def _extract_id_from_filename(filepath: Path) -> str | None:
    """Extract numeric product ID from filename like 'page2_10018.png' -> '10018'."""
    import re
    stem = filepath.stem  # e.g., 'page2_10018'
    # Match product IDs: 4-5 digit numbers after underscore
    m = re.search(r'_(\d{4,5})$', stem)
    if m:
        return m.group(1)
    return None


def _extract_id_from_image(filepath: Path) -> str | None:
    """
    OCR the green badge area to extract product ID.
    Returns the numeric ID string or None if not found.
    """
    import re
    import cv2
    reader = _get_ocr_reader()
    if reader is None:
        return None

    try:
        img = Image.open(filepath).convert("RGB")
        w, h = img.size
        # Green badge is typically in the top-left corner
        # Make crop generous: up to 35% width, 15% height
        crop_w = min(int(w * 0.35), 450)
        crop_h = min(int(h * 0.15), 150)
        crop = img.crop((0, 0, crop_w, crop_h))
        crop_np = np.array(crop)

        # Convert to grayscale and apply simple thresholding to boost contrast
        gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
        
        # EasyOCR works best with high contrast
        results = reader.readtext(gray, detail=0)
        text = " ".join(results).strip()

        # Find 4-5 digit product ID
        m = re.search(r'(\d{4,5})', text)
        if m:
            return m.group(1)
            
        # Try again with original color image if gray fails
        results2 = reader.readtext(crop_np, detail=0)
        text2 = " ".join(results2).strip()
        m2 = re.search(r'(\d{4,5})', text2)
        if m2:
            return m2.group(1)
            
    except Exception as e:
        print(f"[step5_dino] OCR error on {filepath.name}: {e}")

    return None


def _verify_and_correct_matches(
    all_matches: list[tuple[int, int, float]],
    files_a: list[Path],
    files_b: list[Path],
    pass_labels: list[str],
) -> list[tuple[int, int, float]]:
    """
    Verify DINOv2 matches using OCR-extracted product IDs.
    If A filename has ID '10018' but B image's OCR'd badge says '10026',
    try to find and swap with the correct pair even if the other B image failed OCR.
    """
    print()
    print("=" * 60)
    print("VERIFY: OCR-based product ID verification")
    print("=" * 60)

    # Build OCR ID cache for B images
    b_ocr_ids: dict[int, str | None] = {}
    for b_idx, _, _ in all_matches:
        if b_idx not in b_ocr_ids:
            ocr_id = _extract_id_from_image(files_b[b_idx])
            b_ocr_ids[b_idx] = ocr_id
            print(f"  [OCR] B={files_b[b_idx].stem} -> ID={ocr_id}")

    # Build A filename ID cache
    a_file_ids: dict[int, str | None] = {}
    for _, a_idx, _ in all_matches:
        if a_idx not in a_file_ids:
            a_file_ids[a_idx] = _extract_id_from_filename(files_a[a_idx])

    # Check for mismatches and build correction map
    mismatches = []
    for i, (b_idx, a_idx, sim) in enumerate(all_matches):
        a_id = a_file_ids.get(a_idx)
        b_id = b_ocr_ids.get(b_idx)
        if a_id and b_id and a_id != b_id:
            mismatches.append((i, a_id, b_id, b_idx, a_idx))
            print(f"  ⚠️  MISMATCH: match{i:02d} A_file={a_id} vs B_ocr={b_id}  (A={files_a[a_idx].stem}, B={files_b[b_idx].stem})")

    if not mismatches:
        print("  ✅ All matches verified OK (or OCR not available)")
        return all_matches

    # Try to correct mismatches by swapping
    corrected = list(all_matches)
    corrected_count = 0

    for idx_i, a_id_i, b_id_i, b_idx_i, a_idx_i in mismatches:
        # We know A_i matched with B_i, but B_i actually belongs to a_id == b_id_i!
        # Find the match that has A's ID == this B's OCR'd ID
        for idx_j, (b_idx_j, a_idx_j, sim_j) in enumerate(corrected):
            if idx_j == idx_i:
                continue
            a_id_j = a_file_ids.get(a_idx_j)
            if a_id_j == b_id_i:
                # Found the A that should have gotten B_i!
                # Even if B_j's OCR failed, we swap them because B_i is a guaranteed 100% verified match for A_j.
                b_id_j = b_ocr_ids.get(b_idx_j)
                
                # Check if it's already swapped or logically sound
                old_i = corrected[idx_i]
                old_j = corrected[idx_j]
                
                # Swap A indices:
                corrected[idx_i] = (b_idx_i, a_idx_j, sim_j)
                corrected[idx_j] = (b_idx_j, a_idx_i, corrected[idx_j][2])
                corrected_count += 1
                
                status_str = f"A_{a_id_j} <- B_{b_id_i}"
                if b_id_j == a_id_i:
                    status_str += f", A_{a_id_i} <- B_{b_id_j} (Perfect 2-way swap)"
                else:
                    status_str += f", A_{a_id_i} <- B_{b_id_j or 'Unknown'} (Forced 1-way swap)"
                    
                print(f"  🔄 SWAPPED: match{idx_i:02d} <-> match{idx_j:02d} | {status_str}")
                break

    print(f"  Corrected {corrected_count} mismatched pairs")
    return corrected


# ─────────────────────────────────────────────
#  Main: step5_match_with_dino
# ─────────────────────────────────────────────

def step5_match_with_dino(
    bbox_dir_a: Path,
    bbox_dir_b: Path,
    output_dir: Path,
    *,
    threshold_pass1: float = 0.75,
    threshold_pass2: float = 0.60,
    seg_model_path: Optional[Path] = None,
    dino_model_name: str = "dinov2_vits14",
) -> dict:
    """
    Match san pham giua PDF A (brief) va PDF B (produ fini) bang DINOv2.

    Pass 1: Full-image DINOv2 matching (B duyet A)
    Pass 2: Segment product_image tu anh chua match, re-match bang DINOv2

    Output: copy cac cap matched vao output_dir/pdf_a/ va output_dir/pdf_b/
            voi ten file page{N}_match{XX}.png de Step 6 ghep cap.
    """
    print("=" * 60)
    print("STEP 5 (DINOv2): VISUAL MATCHING")
    print("=" * 60)
    print(f"BBox A (brief): {bbox_dir_a}")
    print(f"BBox B (produ fini): {bbox_dir_b}")
    print(f"Output: {output_dir}")
    print(f"Threshold Pass1={threshold_pass1}, Pass2={threshold_pass2}")
    print()

    # Validate dirs
    if not bbox_dir_a.exists():
        print(f"ERROR: BBox A directory not found: {bbox_dir_a}")
        return {"total_a": 0, "total_b": 0, "matched": 0, "matched_ids": []}

    if not bbox_dir_b.exists():
        print(f"ERROR: BBox B directory not found: {bbox_dir_b}")
        return {"total_a": 0, "total_b": 0, "matched": 0, "matched_ids": []}

    # Clean output
    if output_dir.exists():
        shutil.rmtree(output_dir)
    out_a = output_dir / "pdf_a"
    out_b = output_dir / "pdf_b"
    out_a.mkdir(parents=True, exist_ok=True)
    out_b.mkdir(parents=True, exist_ok=True)

    # List images
    exts = (".png", ".jpg", ".jpeg", ".webp")
    files_a = sorted(f for f in bbox_dir_a.iterdir() if f.suffix.lower() in exts)
    files_b = sorted(f for f in bbox_dir_b.iterdir() if f.suffix.lower() in exts)

    print(f"PDF A (brief): {len(files_a)} images")
    print(f"PDF B (produ fini): {len(files_b)} images")
    print()

    if not files_a or not files_b:
        print("ERROR: One of the directories is empty")
        return {"total_a": len(files_a), "total_b": len(files_b), "matched": 0, "matched_ids": []}

    # Load DINOv2
    model, preprocess = _load_dino(dino_model_name)

    # ════════════════════════════════════════════
    #  PASS 1: Full-image DINOv2 matching
    # ════════════════════════════════════════════
    print("=" * 60)
    print(f"PASS 1: Full-Image DINOv2 Matching (threshold={threshold_pass1})")
    print("=" * 60)

    t0 = time.perf_counter()
    emb_a = _encode_images(files_a, model, preprocess)
    emb_b = _encode_images(files_b, model, preprocess)
    print(f"[TIME] Encoding {len(files_a)}+{len(files_b)} images: {time.perf_counter() - t0:.2f}s")

    # B duyet A
    pass1_matches = _sequential_match(emb_b, emb_a, threshold_pass1)

    print(f"\nPass 1 matched: {len(pass1_matches)} pairs")
    for b_idx, a_idx, sim_val in pass1_matches:
        print(f"  {files_b[b_idx].stem:>25} -> {files_a[a_idx].stem:<25}  sim={sim_val:.4f}")

    # Track matched indices
    matched_b_idx = {b_idx for b_idx, _, _ in pass1_matches}
    matched_a_idx = {a_idx for _, a_idx, _ in pass1_matches}

    # ════════════════════════════════════════════
    #  PASS 2: Segment product_image + re-match
    # ════════════════════════════════════════════
    unmatched_b = [i for i in range(len(files_b)) if i not in matched_b_idx]
    unmatched_a = [i for i in range(len(files_a)) if i not in matched_a_idx]
    pass2_matches: list[tuple[int, int, float]] = []

    if unmatched_b and unmatched_a and seg_model_path and seg_model_path.exists():
        print()
        print("=" * 60)
        print(f"PASS 2: Segment product_image + Re-match (threshold={threshold_pass2})")
        print(f"  Unmatched B: {len(unmatched_b)}, Unmatched A: {len(unmatched_a)}")
        print("=" * 60)

        seg_info = _load_seg_model(seg_model_path)
        if seg_info:
            # Segment toan bo unmatched B images
            crops_b: list[tuple[int, Image.Image]] = []
            for bi in unmatched_b:
                img_b = Image.open(files_b[bi]).convert("RGB")
                crop = _segment_product_image(img_b, seg_info)
                if crop:
                    crops_b.append((bi, crop))
                    print(f"  [SEG] {files_b[bi].stem}: product_image found ({crop.size[0]}x{crop.size[1]})")
                else:
                    print(f"  [SEG] {files_b[bi].stem}: no product_image found")

            # Segment toan bo unmatched A images
            crops_a: list[tuple[int, Image.Image]] = []
            for ai in unmatched_a:
                img_a = Image.open(files_a[ai]).convert("RGB")
                crop = _segment_product_image(img_a, seg_info)
                if crop:
                    crops_a.append((ai, crop))
                    print(f"  [SEG] {files_a[ai].stem}: product_image found ({crop.size[0]}x{crop.size[1]})")
                else:
                    print(f"  [SEG] {files_a[ai].stem}: no product_image found")

            if crops_b and crops_a:
                # Encode crops
                emb_crops_b = _encode_pil_images([c for _, c in crops_b], model, preprocess)
                emb_crops_a = _encode_pil_images([c for _, c in crops_a], model, preprocess)

                # Match (B crops duyet A crops)
                raw_matches = _sequential_match(emb_crops_b, emb_crops_a, threshold_pass2)

                for cb_idx, ca_idx, sim_val in raw_matches:
                    real_b_idx = crops_b[cb_idx][0]
                    real_a_idx = crops_a[ca_idx][0]
                    pass2_matches.append((real_b_idx, real_a_idx, sim_val))
                    matched_b_idx.add(real_b_idx)
                    matched_a_idx.add(real_a_idx)
                    print(f"  [MATCH] {files_b[real_b_idx].stem} -> {files_a[real_a_idx].stem}  sim={sim_val:.4f}")

                print(f"\nPass 2 matched: {len(pass2_matches)} pairs")
            else:
                print("  No valid crops to match in Pass 2")
        else:
            print("  [WARN] Seg model not available, skipping Pass 2")
    elif unmatched_b:
        print(f"\n[INFO] {len(unmatched_b)} unmatched B images (no seg model for Pass 2)")

    # ════════════════════════════════════════════
    #  PASS 3 (Force): Match tat ca B con lai voi best A con lai
    #  Khong can threshold — moi B PHAI duoc match
    # ════════════════════════════════════════════
    still_unmatched_b = [i for i in range(len(files_b)) if i not in matched_b_idx]
    still_unmatched_a = [i for i in range(len(files_a)) if i not in matched_a_idx]
    force_matches: list[tuple[int, int, float]] = []

    if still_unmatched_b and still_unmatched_a:
        print()
        print("=" * 60)
        print(f"PASS 3 (FORCE): Match {len(still_unmatched_b)} B con lai (no threshold)")
        print("=" * 60)

        # Dùng Hungarian algorithm cho cả pass 3 (threshold=0 = no threshold)
        sub_emb_b = emb_b[still_unmatched_b]    # (Nb_sub, D)
        sub_emb_a = emb_a[still_unmatched_a]    # (Na_sub, D)
        raw_force = _sequential_match(sub_emb_b, sub_emb_a, threshold=0.0)

        for sub_bi, sub_ai, sim_val in raw_force:
            real_bi = still_unmatched_b[sub_bi]
            real_ai = still_unmatched_a[sub_ai]
            force_matches.append((real_bi, real_ai, sim_val))
            matched_b_idx.add(real_bi)
            matched_a_idx.add(real_ai)
            print(f"  [FORCE] {files_b[real_bi].stem} -> {files_a[real_ai].stem}  sim={sim_val:.4f}")

        print(f"\nPass 3 force-matched: {len(force_matches)} pairs")

    # ════════════════════════════════════════════
    #  Copy matched pairs to output
    # ════════════════════════════════════════════
    all_matches = pass1_matches + pass2_matches + force_matches

    # ════════════════════════════════════════════
    #  OCR Verification: detect & fix swapped pairs
    # ════════════════════════════════════════════
    pass_labels = (["P1"] * len(pass1_matches) +
                   ["P2"] * len(pass2_matches) +
                   ["P3"] * len(force_matches))
    all_matches = _verify_and_correct_matches(all_matches, files_a, files_b, pass_labels)

    matched_ids = []
    mapping = {}

    print()
    print("=" * 60)
    print(f"COPYING {len(all_matches)} MATCHED PAIRS TO OUTPUT")
    print("=" * 60)

    import json as _json

    for match_idx, (b_idx, a_idx, sim_val) in enumerate(all_matches, 1):
        match_id = f"match{match_idx:02d}"
        src_pass = "P1" if match_idx <= len(pass1_matches) else ("P2" if match_idx <= len(pass1_matches) + len(pass2_matches) else "P3")

        # Extract page number from original filename (page1_xxx -> 1)
        import re
        page_match_b = re.search(r"page(\d+)", files_b[b_idx].stem)
        page_match_a = re.search(r"page(\d+)", files_a[a_idx].stem)
        page_b = page_match_b.group(1) if page_match_b else "1"
        page_a = page_match_a.group(1) if page_match_a else "1"

        # Output filenames: page{N}_{match_id}.png
        out_name_a = f"page{page_a}_{match_id}.png"
        out_name_b = f"page{page_b}_{match_id}.png"

        shutil.copy2(files_a[a_idx], out_a / out_name_a)
        shutil.copy2(files_b[b_idx], out_b / out_name_b)

        print(f"  [{src_pass}] {files_b[b_idx].stem} <-> {files_a[a_idx].stem}"
              f"  sim={sim_val:.4f}  -> {match_id}")

        matched_ids.append(match_id)

        # Build mapping entry
        mapping[match_id] = {
            "file_a_original": files_a[a_idx].name,
            "file_b_original": files_b[b_idx].name,
            "file_a_renamed": out_name_a,
            "file_b_renamed": out_name_b,
            "page_a": int(page_a),
            "page_b": int(page_b),
            "similarity": round(sim_val, 4),
            "pass": src_pass,
        }

    # Save match_mapping.json
    mapping_file = output_dir / "match_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        _json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"\n  [SAVED] match_mapping.json -> {mapping_file}")

    # ════════════════════════════════════════════
    #  Summary
    # ════════════════════════════════════════════
    final_unmatched_b = [files_b[i].stem for i in range(len(files_b)) if i not in matched_b_idx]
    final_unmatched_a = [files_a[i].stem for i in range(len(files_a)) if i not in matched_a_idx]

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total A (brief):       {len(files_a)}")
    print(f"  Total B (produ fini):  {len(files_b)}")
    print(f"  Pass 1 matched:        {len(pass1_matches)}")
    print(f"  Pass 2 matched:        {len(pass2_matches)}")
    print(f"  Pass 3 forced:         {len(force_matches)}")
    print(f"  Total matched:         {len(all_matches)}")
    print(f"  Unmatched A:           {len(final_unmatched_a)}  {final_unmatched_a if final_unmatched_a else ''}")
    print(f"  Unmatched B:           {len(final_unmatched_b)}  {final_unmatched_b if final_unmatched_b else ''}")
    print(f"\n  Output: {output_dir}")
    print("  Done!")

    return {
        "total_a": len(files_a),
        "total_b": len(files_b),
        "matched": len(all_matches),
        "pass1_matched": len(pass1_matches),
        "pass2_matched": len(pass2_matches),
        "matched_ids": matched_ids,
        "unmatched_a": final_unmatched_a,
        "unmatched_b": final_unmatched_b,
        "copied_a": len(all_matches),
        "copied_b": len(all_matches),
    }


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Step 5 (DINOv2): Visual Matching")
    parser.add_argument("--bbox-a", type=str, required=True, help="Thu muc bbox PDF A (brief)")
    parser.add_argument("--bbox-b", type=str, required=True, help="Thu muc bbox PDF B (produ fini)")
    parser.add_argument("--output", type=str, required=True, help="Thu muc output compare_temp")
    parser.add_argument("--threshold-p1", type=float, default=0.75, help="Threshold Pass 1 (default=0.75)")
    parser.add_argument("--threshold-p2", type=float, default=0.60, help="Threshold Pass 2 (default=0.60)")
    parser.add_argument("--seg-model", type=str, default=None, help="Path to best_model.pth")
    parser.add_argument("--dino-model", type=str, default="dinov2_vits14", help="DINOv2 model name")
    args = parser.parse_args()

    seg_path = Path(args.seg_model) if args.seg_model else Path(__file__).resolve().parent / "models" / "best_model.pth"

    stats = step5_match_with_dino(
        bbox_dir_a=Path(args.bbox_a),
        bbox_dir_b=Path(args.bbox_b),
        output_dir=Path(args.output),
        threshold_pass1=args.threshold_p1,
        threshold_pass2=args.threshold_p2,
        seg_model_path=seg_path,
        dino_model_name=args.dino_model,
    )

    return 0 if stats["matched"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
