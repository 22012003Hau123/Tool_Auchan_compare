"""
Bước 6 (Gemini version): Chatbot với LangChain + Google Gemini để so sánh ảnh bbox.
"""

from __future__ import annotations

import sys
import os
import base64
import json
import unicodedata
import io
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor

# Load .env để lấy GOOGLE_API_KEY hoặc GEMINI_API_KEY
try:
    from dotenv import load_dotenv
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
except ImportError:
    pass

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    print("ERROR: langchain-google-genai not installed. Install with: pip install langchain-google-genai")
    sys.exit(1)

import re
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

NUMBERED_ITEM_PATTERN = re.compile(r"\b\d{1,2}\s*[-–]\s*")


def normalize_text_for_alignment(text: str) -> str:
    """Normalize text for alignment comparison.
    
    GIỮ NGUYÊN dấu accent (è, é, à, ÈME...) vì đó là sai biệt thật trong tiếng Pháp.
    Chỉ bỏ ký tự đặc biệt (không phải chữ/số) và lowercase.
    """
    if not text:
        return ""
    # NFC normalize (gộp combining chars thành 1 ký tự, giữ nguyên accent)
    normalized = unicodedata.normalize("NFC", str(text))
    # Bỏ ký tự không phải chữ/số, lowercase
    normalized = re.sub(r"[^\w]+", "", normalized).lower()
    return normalized


def strip_code_fence(raw: str) -> str:
    """Bỏ ```json ... ``` để dễ json.loads."""
    if not raw:
        return ""
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def ensure_list_of_strings(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            if item is None:
                result.append("")
            else:
                result.append(str(item))
        return result
    return [str(value)]


def split_numbered_segment(text: str) -> list[str]:
    """Tách chuỗi chứa nhiều mục '1 -', '2 -'... thành danh sách."""
    if not text:
        return [""]

    segment = text.strip()
    if not segment:
        return [""]

    matches = list(NUMBERED_ITEM_PATTERN.finditer(segment))
    if len(matches) <= 1:
        return [segment]

    parts: list[str] = []
    first_start = matches[0].start()
    leading = segment[:first_start].strip()
    if leading:
        parts.append(leading)

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(segment)
        chunk = segment[start:end].strip()
        if chunk:
            parts.append(chunk)
    return parts or [segment]


def split_text_snippets(text: str) -> list[str]:
    """Tách text theo newline và numbered list để đảm bảo giống image1."""
    if text is None:
        return [""]

    raw = str(text)
    if not raw.strip():
        return [""]

    lines = [line.strip() for line in re.split(r"[\r\n]+", raw) if line.strip()]
    if not lines:
        return [""]

    snippets: list[str] = []
    for line in lines:
        snippets.extend(split_numbered_segment(line))

    return snippets or [raw.strip()]


def normalize_simple_diff_arrays(diff_section: dict) -> bool:
    """Đảm bảo differences.image1/image2 luôn được tách giống nhau."""
    lower_keys = {str(k).strip().lower() for k in diff_section.keys()}
    if not lower_keys or not lower_keys.issubset({"image1", "image 1", "image2", "image 2"}):
        return False

    key_img1 = next((k for k in diff_section.keys() if str(k).strip().lower() in {"image1", "image 1"}), None)
    key_img2 = next((k for k in diff_section.keys() if str(k).strip().lower() in {"image2", "image 2"}), None)
    if key_img1 is None or key_img2 is None:
        return False

    image1_values = ensure_list_of_strings(diff_section.get(key_img1))
    image2_values = ensure_list_of_strings(diff_section.get(key_img2))

    normalized_image1: list[str] = []
    for value in image1_values:
        normalized_image1.extend(split_text_snippets(value))

    normalized_image2: list[str] = []
    for value in image2_values:
        normalized_image2.extend(split_text_snippets(value))

    max_len = max(len(normalized_image1), len(normalized_image2))
    if max_len == 0:
        return False

    if len(normalized_image1) < max_len:
        normalized_image1.extend([""] * (max_len - len(normalized_image1)))
    if len(normalized_image2) < max_len:
        normalized_image2.extend([""] * (max_len - len(normalized_image2)))

    for idx in range(max_len):
        img1_value = normalized_image1[idx]
        img2_value = normalized_image2[idx]
        if not img1_value or not img2_value:
            continue
        if img2_value.strip().lower() == "null":
            continue
        if normalize_text_for_alignment(img1_value) == normalize_text_for_alignment(img2_value):
            normalized_image2[idx] = ""

    changed = (
        normalized_image1 != image1_values
        or normalized_image2 != image2_values
    )

    if changed:
        diff_section[key_img1] = normalized_image1
        diff_section[key_img2] = normalized_image2

    return changed


def postprocess_comparison_output(raw: str) -> str:
    """Chuẩn hoá output GPT để image2 luôn tách từng phần giống image1."""
    if not raw:
        return raw

    cleaned = strip_code_fence(raw)
    if not cleaned:
        return raw

    try:
        data = json.loads(cleaned)
    except Exception:
        return raw

    changed = False
    differences = data.get("differences")
    if isinstance(differences, dict):
        if normalize_simple_diff_arrays(differences):
            changed = True
        key_img1 = next((k for k in differences.keys() if str(k).strip().lower() in {"image1", "image 1"}), None)
        key_img2 = next((k for k in differences.keys() if str(k).strip().lower() in {"image2", "image 2"}), None)
        if key_img1 is not None and key_img2 is not None:
            image1_values = ensure_list_of_strings(differences.get(key_img1))
            image2_values = ensure_list_of_strings(differences.get(key_img2))
            max_len = max(len(image1_values), len(image2_values))
    if not changed:
        return raw

    return json.dumps(data, ensure_ascii=False, indent=2)


def extract_id_from_filename(filename: str) -> str | None:
    """
    Trích xuất ID từ tên file.

    Args:
        filename: Tên file (ví dụ: page1_15599.png)

    Returns:
        ID (ví dụ: "15599") hoặc None nếu không tìm thấy
    """
    # Pattern: page{number}_{ID}.png
    match = re.match(r"page\d+_(.+)\.png", filename)
    if match:
        return match.group(1)
    return None


def encode_image_to_base64(image_path: Path) -> str:
    """
    Encode ảnh sang base64.

    Args:
        image_path: Đường dẫn đến ảnh

    Returns:
        Base64 string
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_image_to_base64_with_resize(
    image_path: Path,
    *,
    max_side: int = 1024,
) -> tuple[str, int, int, list[int]]:
    """
    Resize ảnh trước khi gửi cho model để coordinate ổn định hơn.

    Trả về base64 ảnh PNG đã resize + kích thước (w, h) theo pixel.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        scale = min(1.0, max_side / max(orig_w, orig_h))
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        if (new_w, new_h) != (orig_w, orig_h):
            img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        # Draw ruler overlay so GPT can output coordinates more reliably.
        draw = ImageDraw.Draw(img)
        ox = min(120, max(30, int(new_w * 0.08)))
        oy = min(120, max(30, int(new_h * 0.08)))
        ruler_color = (255, 70, 70)
        grid_color = (255, 220, 220)
        tick_color = (255, 120, 120)
        step = 100

        draw.line([(ox, oy), (new_w - 1, oy)], fill=ruler_color, width=3)
        draw.line([(ox, oy), (ox, new_h - 1)], fill=ruler_color, width=3)
        cross = 8
        draw.line([(ox - cross, oy), (ox + cross, oy)], fill=ruler_color, width=3)
        draw.line([(ox, oy - cross), (ox, oy + cross)], fill=ruler_color, width=3)

        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

        x = ox
        while x <= new_w - 1:
            dx = x - ox
            draw.line([(x, oy), (x, new_h - 1)], fill=grid_color, width=1)
            draw.line([(x, oy - 6), (x, oy + 6)], fill=tick_color, width=2)
            if dx % (step * 2) == 0:
                draw.text((x + 4, oy - 28), f"{dx}", fill=ruler_color, font=font)
            x += step

        y = oy
        while y <= new_h - 1:
            dy = y - oy
            draw.line([(ox, y), (new_w - 1, y)], fill=grid_color, width=1)
            draw.line([(ox - 6, y), (ox + 6, y)], fill=tick_color, width=2)
            if dy % (step * 2) == 0:
                draw.text((ox - 70, y + 2), f"{dy}", fill=ruler_color, font=font)
            y += step

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64, new_w, new_h, [int(ox), int(oy)]


def compare_two_images(
    image_a_path: Path,
    image_b_path: Path,
    *,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
) -> tuple[str, list[int], list[int]]:
    """
    So sánh 2 ảnh bbox có cùng ID và trả về JSON CỐ ĐỊNH (sử dụng Google Gemini).

    Cấu trúc JSON BẮT BUỘC:
    {
      "differ_product": [
        {
          "aspect": "<visual difference category>",
          "image1": "<description>",
          "image2": "<description>"
        }
      ],
      "differences": {
        "field_name": {
          "image1": "<value in image1>",
          "image2": "<value in image2>"
        }
      }
    }
    """
    # Lấy API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables")
    
    # Khởi tạo ChatGoogleGenerativeAI
    chat = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )

    # Encode 2 ảnh
    image_a_b64 = encode_image_to_base64(image_a_path)
    image_b_b64, image_b_model_w, image_b_model_h, image_b_ruler_origin_px = encode_image_to_base64_with_resize(
        image_b_path,
        max_side=1024,
    )

    # Tạo prompt cho Gemini
    messages = [
        SystemMessage(
            content="""You are an advanced vision and text comparison assistant for advertising or catalog images.



Your task:

Compare two advertising images and return a JSON output with:

1. "differ_product" — product comparison only

2. "differences" — text comparison only

3. "background_differences" — background color comparison only

4. "missing_elements" — check if any elements are missing in one image compared to the other



========================================================

🖼️ 1️⃣ DIFFER_PRODUCT — PRODUCT COMPARISON

========================================================

Compare only the product identity based on:

- packaging shape,

- dominant colors,

- visual layout and pattern,

- brand appearance,

- product-line design.



Two products must be considered the SAME even if:

- zoomed in/out, cropped, rotated, repositioned

- size, scale, visibility, count differs

- printed text differs or unreadable



Only report differences when the product TYPE is truly different.



If same:

"differ_product": []



========================================================

🏷️ 2️⃣ DIFFERENCES — TEXT COMPARISON

========================================================

Compare only text outside the product packaging:

- price

- promotion text

- product name / variant name

- reference number

- short marketing bullets ("Existe aussi :", "Dès 3 ans", etc.)



Ignore:

- text printed ON the product

- layout/spacing/visibility differences

- line breaks / text being split onto different lines (same wording, just different layout)

- repeated spaces, hyphenation, soft line-breaks, capitalization variants when the words are the same

- colored highlights/background rectangles behind the same text



Output format (STRICT):

```

"differences": {

  "image1": ["<FULL context from Image 1, always complete>", "..."],

  "image2": ["<ONLY different text from Image 2, use '' if same>", "..."],

  "image2_bboxes_px": [
    [x0, y0, x1, y1] or null
  ]

}

```



Coordinate rules (STRICT):
- `image2_bboxes_px` MUST be an array aligned by index with `differences.image2` (same length).
- Use `null` when `image2[i]` is '' (same text) or when `image2[i]` is the literal string "null" (missing text).
- For each non-empty `image2[i]`, return the tight bounding box of EXACTLY that text snippet in the INPUT `Image 2` (the cropped PNG you send to the model).
- Coordinate system: pixel coordinates on the provided Image 2 crop.
-  origin (0,0) is the intersection of the drawn "ruler" axes on Image 2 (NOT the top-left corner).
-  x increases to the right, y increases downward
- Return numbers as integers (or floats are accepted).
- bbox format MUST be `[x_min, y_min, x_max, y_max]` (NOT `[x, y, width, height]`).
- Never return markdown; output must be valid JSON only.

CRITICAL LOGIC:

- **image1** (brief): ALWAYS provide FULL context from brief (Image 1), split into separate elements

- **image2** (produit fini): ONLY provide text that is DIFFERENT from brief (Image 1):

  - If the content is the SAME (after normalization, ignoring punctuation/capitalization) → use empty string ""

  - If the content is DIFFERENT → provide the actual text from Image 2

  - If Image 1 contains a snippet that is completely MISSING from Image 2 → use the literal string "null" (all lowercase) for that element

  - Only report differences in actual CONTENT, not just formatting/ordering changes

- NEVER use "null" if Image 2 still contains the snippet in any form. Format variations (prefixes like "1 -", capitalization, punctuation, accents, line-breaks, added numbering, etc.) must be returned as the actual Image 2 text instead of "null".

- Whenever you output "null" for any index, you MUST also add a corresponding entry in `missing_elements` that clearly states the Image 1 text and that Image 2 lacks it. Missing text counts as a missing element.



CRITICAL RULES FOR TEXT SPLITTING AND ORDERING:

- Use JSON arrays of strings. Each array element must be a SEPARATE, DISTINCT text snippet.

- When Image 2 contains multiple different text segments (e.g., "DÈS LA NAISSANCE", "0€", "1 - OURS ASSIS..."), you MUST split them into separate array elements:

  ✅ CORRECT: "image2": ["DÈS LA NAISSANCE", "0€", "1 - OURS ASSIS 1 mètre. Réf 619855. 2 - OURS MAGNÉTIQUE..."]

  ❌ WRONG: "image2": ["DÈS LA NAISSANCE 0€ 1 - OURS ASSIS 1 mètre..."] (all in one string)



ORDERING RULES (CRITICAL):

- Arrays must follow a CONSISTENT ORDER for both image1 (brief) and image2 (produit fini):

  1. Age labels FIRST: "DÈS 3 ANS", "DÈS 5 ANS", "DÈS LA NAISSANCE" → index 0 (use "" if missing in brief/image1)

  2. Price badges SECOND: "0€", "25€99", "10€ DE REMISE" → index 1 (use "" if missing in brief/image1)

  3. Product names/variants THIRD:

     - For image1 (brief): Split into separate elements: ["Product name", "Existe en : variant1", "variant2", ...]

     - For image2 (produit fini): Only include if different from brief/image1 (use "" if same)

  4. Product lists LAST: Numbered items (1 -, 2 -, 3 -, etc.) → start after product names/variants

     - For image1 (brief): Always include all items

     - For image2 (produit fini): Only include if different from brief/image1 (use "" if same)



GROUPING RULES FOR image1 (brief):

- For image1 (brief): Split product names and variants into separate elements for clarity:

  ✅ CORRECT image1 (brief): ["MARIE 40 cm Réf. 760674", "Existe en : Simba, 40 cm Réf. 634233", "Bambi, 40 cm Réf. 680652"]

  - This allows better comparison with image2 (produit fini)

- For image2 (produit fini): Only include elements that differ from brief/image1 (use "" if same)
  ✅ CORRECT image2: ["1 - MARIE 40 cm Réf. 760674", "Existe en : 2 - Simba, 40 cm Réf. 634233", "3 - Bambi, 40 cm Réf. 680652"]



SPLITTING RULES (APPLY TO BOTH image1 AND image2):

- Split by:

  - Age labels: "DÈS 3 ANS", "DÈS 5 ANS", "DÈS LA NAISSANCE" → separate element

  - Price badges: "0€", "25€99", "10€ DE REMISE" → separate element

  - Product lists: MUST split numbered items into separate elements FOR BOTH IMAGES:

    ✅ CORRECT image1: ["1 - SAC ANIMAL 25 cm. Réf 476922", "2 - Peluche kawaii 25 cm : 7€99 Réf. 585899", "3 - Panda ou Panda roux 28 cm : 8€99 Réf. 476922", "4 - Chien assis 28 cm : 7€99 Réf. 522125"]

    ✅ CORRECT image2: ["1 - PANDA 25 cm. Réf 462854.", "2 - PELUCHE KAWAII 25 cm : 7€99 Réf. 585899.", "3 - PANDA OU PANDA ROUX 28 cm : 8€99 Réf. 476922.", "4 - CHIEN ASSIS 28 cm : 7€99 Réf. 522125*"]

    ❌ WRONG: ["1 - SAC ANIMAL 25 cm. Réf 476922 2 - Peluche kawaii 25 cm : 7€99 Réf. 585899 3 - Panda ou Panda roux 28 cm : 8€99 Réf. 476922 4 - Chien assis 28 cm : 7€99 Réf. 522125"] (all in one string)

    - Each numbered item (starting with "1 -", "2 -", "3 -", "1-", "2-", etc.) must be a separate array element

    - Split at the pattern: number followed by " -" or "-" or " - "

    - Apply this splitting rule to BOTH image1 and image2 arrays

  - Promotional text: "Existe aussi :", "Exclusivité" → separate element

- The strings must be raw text snippets copied from the corresponding image (no explanations, no "Image 1 contains …" prefixes).



ALIGNMENT RULES (CRITICAL):

- Arrays must be aligned by position: index 0 of image1 corresponds to index 0 of image2, etc.

- Follow the ORDERING RULES above: Age labels → Price badges → Product names/variants → Product lists

- Example alignment (NEW LOGIC):

  ✅ CORRECT:

  image1: ["", "0€", "MARIE 40 cm Réf. 760674", "Existe en : Simba, 40 cm Réf. 634233", "Bambi, 40 cm Réf. 680652"]

  image2: ["", "", "1 - MARIE 40 cm Réf. 760674. ", "Existe en : 2 - SIMBA 40 cm Réf. 63423. ", "3 - BAMBI 40 cm Réf. 680652"]

  Explanation:

  - Index 0: Both missing age label → image2 = ""

  - Index 1: image1 has "0€", image2 doesn't have it → image2 = "" (missing, not different)

  - Index 2: image1 = "MARIE 40 cm Réf. 760674", image2 = "1 - MARIE 40 cm Réf. 760674. " (different format: has "1 -" prefix) → image2 = "1 - MARIE 40 cm Réf. 760674. "

  - Index 3: image1 = "Existe en : Simba, 40 cm Réf. 634233", image2 = "Existe en : 2 - SIMBA 40 cm Réf. 63423. " (different format: has "2 -" prefix) → image2 = "Existe en : 2 - SIMBA 40 cm Réf. 63423. "

  - Index 4: image1 = "Bambi, 40 cm Réf. 680652", image2 = "3 - BAMBI 40 cm Réf. 680652" (different format: has "3 -" prefix) → image2 = "3 - BAMBI 40 cm Réf. 680652"



  ✅ CORRECT:

  image1: ["", "0€", "1 - SAC ANIMAL 25 cm. Réf 476922", "2 - Peluche kawaii 25 cm : 7€99 Réf. 585899", "3 - Panda ou Panda roux 28 cm : 8€99 Réf. 476922", "4 - Chien assis 28 cm : 7€99 Réf. 522125"]

  image2: ["DÈS LA NAISSANCE", "", "1 - PANDA 25 cm. Réf 462854.", "2 - PELUCHE KAWAII 25 cm : 7€99 Réf. 585899.", "", "4 - CHIEN ASSIS 28 cm : 7€99 Réf. 522125*"]

  Explanation:

  - Index 0: image1 missing, image2 has "DÈS LA NAISSANCE" → image2 = "DÈS LA NAISSANCE"

  - Index 1: image1 has "0€", image2 doesn't → image2 = ""

  - Index 2: Different product name (SAC ANIMAL vs PANDA) and ref (476922 vs 462854) → image2 = "1 - PANDA 25 cm. Réf 462854."

  - Index 3: Same content (only capitalization differs, which is ignored) → image2 = "" (same after normalization)

  - Index 4: Same content (only capitalization differs) → image2 = "" (same after normalization)

  - Index 5: Different (has "*" at end) → image2 = "4 - CHIEN ASSIS 28 cm : 7€99 Réf. 522125*"



  ✅ CORRECT (MISSING TEXT IN IMAGE 2):

  image1: ["MARIE 40 cm Réf. 760674", "Existe en : Simba, 40 cm Réf. 634233", "Bambi, 40 cm Réf. 680652"]

  image2: ["null", "Existe en : 2 - Simba, 40 cm Réf. 634233", "3 - Bambi, 40 cm Réf. 680652"]

  Explanation:

  - Index 0: Image 1 has the base line but Image 2 lacks it entirely → set image2 = "null"

  - Index 1-2: Provide only the changed wording for Image 2

  ❌ WRONG:

  image1: ["Exclusivité", "0€", "1 - STITCH...", "2 - ANGEL...", "..."]

  image2: ["null", "null", "null", "null", ...]

  This is invalid because Image 2 clearly contains those lines. Use the actual Image 2 text (with numbering/prefix differences) or "" if identical—reserve "null" strictly for content that is fully absent.



COMPARISON RULES:

- Normalize whitespace before comparing (treat multiple spaces/newlines as a single space)

- Ignore ONLY these minor formatting differences: extra spaces, line breaks, soft hyphens → these should NOT be considered different

- REPORT AS DIFFERENT (these are REAL content differences, NOT formatting):
  • Accent differences: "E" vs "È", "e" vs "é", "a" vs "à", "EME" vs "ÈME" — accents change meaning in French
  • Missing or extra letters: "variétés" vs "variété" (missing "s"), "Off re" vs "Offre" — even 1 character difference matters
  • Spelling errors or typos: any word spelled differently
  • Different punctuation that changes meaning: "3€99" vs "3,99€"

- Capitalization alone (same letters, same accents, just upper/lower case) may be ignored

- If content is the same after normalization → image2 element = ""

- If every snippet is identical (after normalization) → `"image1": [], "image2": []`



========================================================

🎨 3️⃣ BACKGROUND_DIFFERENCES — BRAND LOGO BACKDROP ONLY

========================================================

Focus ONLY on the background immediately behind **brand logos** (One Two Fun, Smoby, VTech, Disney, etc.).

Ignore promotional banners, price badges, CTA ribbons, or large panels like "Soit 0€33 … Auchan".

You are **not** comparing the overall page background or banner backgrounds.



Rules:

- Inspect the area directly behind each logo.

- Determine if the logo sits on a backdrop (white rectangle, colored blob, etc.) or if it is transparent (no backdrop).

- If both logos have no backdrop → return {}.

- If both logos have the same backdrop color/shape → return {}.

- Only report when Image 2 has a backdrop that Image 1 does **not** have, or when the backdrop colors/shapes are clearly different (e.g., Image 2 adds a white rectangle while Image 1 is transparent).

- The backdrop color should be different from the surrounding page background to be considered a difference.

- Describe succinctly (e.g., "transparent", "white rectangle", "yellow blob").

- Do NOT report differences for:

  • price banners, promo strips, reduction tags

  • Auchan bird inside a red bar, CTA blocks, or any rectangular panel spanning the header

  • text badges where the entire banner changed color

- If the logo simply sits on the same color as the product card background (e.g., whole card is white/pink) → treat it as "transparent" (no backdrop difference).

- Only report a difference when there is a clearly visible additional shape/patch directly behind the logo.



Output:

"background_differences": {

  "image1": "<logo backdrop description>",

  "image2": "<logo backdrop description>"

}



Example (highlight):

- Image 1: logo is transparent (no backdrop)

- Image 2: logo sits on a yellow rectangle



"background_differences": {

  "image1": "transparent around logo",

  "image2": "yellow rectangle behind logo"

}



========================================================

🔍 4️⃣ MISSING_ELEMENTS — VISUAL ELEMENT COMPLETENESS CHECK

========================================================

Check if Image 2 (produit fini) is MISSING any VISUAL elements (images, icons, graphics) that exist in Image 1 (brief).

IMPORTANT: This section is ONLY for VISUAL elements (images, photos, icons, graphics). Text differences are already handled in the "differences" section above. DO NOT report missing text here.



CRITICAL RULES:

- ONLY report if Image 2 (produit fini) truly has LESS VISUAL elements than Image 1 (brief).

- Include ONLY visual components:
  - Product photos
  - Variant thumbnails
  - Pictograms
  - Icons
  - Brand logos
  - CTA badges (visual)
  - Promo banners (visual)
  - "Existe aussi :" visual blocks
  - Any other graphics or images

- DO NOT include textual snippets here - text differences are handled in the "differences" section.

- DO NOT report if Image 2 (produit fini) has MORE visual elements than Image 1 (brief) (that's additional, not missing)

- DO NOT report if Image 1 (brief) is missing visual elements (only check Image 2 - produit fini)

- Ignore size differences, position differences, or layout changes



Examples of CORRECT usage (VISUAL ELEMENTS ONLY):

✅ Image 1 (brief) has: product image + "Existe aussi :" visual block (graphic/image)

   Image 2 (produit fini) has: only product image

   → Image 2 is MISSING "Existe aussi :" visual block → report missing_elements



✅ Image 1 (brief) has: brand logo (image/graphic)

   Image 2 (produit fini) has: no logo

   → Image 2 is MISSING logo → report missing_elements



✅ Image 1 (brief) has: variant thumbnail images

   Image 2 (produit fini) has: no variant thumbnails

   → Image 2 is MISSING variant thumbnails → report missing_elements



Examples of INCORRECT usage (DO NOT report):

❌ Image 1 (brief) has text "MARIE 40 cm Réf. 760674", Image 2 (produit fini) lacks that text

   → This is a TEXT difference, NOT a visual element → Handle in "differences" section, NOT in missing_elements

❌ Image 2 (produit fini) contains the text but with different numbering, casing, punctuation, or layout

   → This is a TEXT difference → Handle in "differences" section, NOT in missing_elements

❌ Image 1 (brief) has: only product image

   Image 2 (produit fini) has: product image + "Existe aussi :" visual section

   → Image 2 has MORE visual elements, not missing → DO NOT report missing_elements



IMPORTANT:

- Extra visual elements (images, graphics, icons) that appear ONLY in Image 2 (produit fini) must be reported under `differences` (if they affect text) or ignored (if purely visual additions).

- Extra text that appears ONLY in Image 2 (produit fini) must be reported under `differences`, not `missing_elements`.

- Use `missing_elements` ONLY when Image 2 (produit fini) lacks a VISUAL element (image, graphic, icon) that Image 1 (brief) already has.

- Text differences are handled in the "differences" section - DO NOT duplicate them in missing_elements.



Output format:

- If Image 2 has all elements (even if different size/position) → return {} (empty object)

- If Image 2 is missing visual elements → return:



"missing_elements": {

  "image1": "<visual element that exists in image1>",

  "image2": "<how image2 lacks that element>",

  "missing": true

}



========================================================

📤 OUTPUT FORMAT (STRICT JSON)

========================================================

{

  "differ_product": [...],
  "differences": {...},
  "background_differences": {...},
  "missing_elements": {...}
}



Rules:

- If products match → differ_product = []

- If text matches → differences = {}

- If background colors match → background_differences = {}

- If Image 2 has all elements → missing_elements = {}

- NEVER add explanations, reasoning, or extra descriptions anywhere in the JSON. Return only the raw factual values.

- If you output "null" anywhere in `differences.image2`, double-check the snippet is truly missing AND add a `missing_elements` entry describing it.

- Never use "null" to represent formatting/casing/numbering changes—return Image 2's actual wording or "" if identical after normalization.



Return ONLY valid JSON. No markdown. No explanations.



"""
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Compare Image 1 (brief - reference) and Image 2 (produit fini - final product).\n\nReturn JSON with:\n- differ_product: product visual differences (empty array if products match)\n- differences: text differences outside product (empty object if all text matches)\n- background_differences: background color differences (empty object if colors match)\n- missing_elements: check if Image 2 (produit fini) is missing any elements from Image 1 (brief) (empty object if no missing elements)\n\nIn the JSON output, use:\n- \"image1\" to represent \"brief\" (reference image)\n- \"image2\" to represent \"produit fini\" (final product image)\n\nImage 1 (brief):",
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{image_a_b64}",
                },
                {
                    "type": "text",
                    "text": "\n\nImage 2 (produit fini):",
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{image_b_b64}",
                },
            ]
        ),
    ]

    # Gọi API
    response = chat.invoke(messages)

    content = response.content
    # Đảm bảo luôn trả về string JSON
    if isinstance(content, list):
        content = "".join(str(c) for c in content)
    elif not isinstance(content, str):
        content = str(content)

    return (
        content,
        [int(image_b_model_w), int(image_b_model_h)],
        [int(image_b_ruler_origin_px[0]), int(image_b_ruler_origin_px[1])],
    )


def step6_compare_with_gemini(
    compare_dir: Path,
    *,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    max_comparisons: int = None,
) -> dict:
    """
    So sánh các cặp bbox có ID trùng bằng Google Gemini.

    Args:
        compare_dir: Thư mục chứa compare_temp/pdf_a và compare_temp/pdf_b
        model: Gemini model (gemini-2.5-flash, gemini-1.5-flash, gemini-pro, etc.)
        temperature: Temperature
        max_comparisons: Số lượng so sánh tối đa (None = tất cả)

    Returns:
        Kết quả so sánh
    """
    print("=" * 60)
    print("BƯỚC 6: SO SÁNH BBOX BẰNG GEMINI")
    print("=" * 60)
    print(f"Compare dir: {compare_dir}")
    print()

    pdf_a_dir = compare_dir / "pdf_a"
    pdf_b_dir = compare_dir / "pdf_b"

    if not pdf_a_dir.exists() or not pdf_b_dir.exists():
        print("ERROR: Compare directories not found. Run step5 first.")
        return {"comparisons": 0, "results": []}

    # Lấy danh sách file và tạo map theo ID
    files_a_list = list(pdf_a_dir.glob("*.png"))
    files_b_list = list(pdf_b_dir.glob("*.png"))

    id_map_a: dict[str, Path] = {}
    id_map_b: dict[str, Path] = {}

    for file_a in files_a_list:
        id_text = extract_id_from_filename(file_a.name)
        if id_text:
            id_map_a[id_text] = file_a

    for file_b in files_b_list:
        id_text = extract_id_from_filename(file_b.name)
        if id_text:
            id_map_b[id_text] = file_b

    # Tìm ID trùng
    common_ids = set(id_map_a.keys()) & set(id_map_b.keys())

    if not common_ids:
        print("No common IDs found between pdf_a and pdf_b.")
        return {"comparisons": 0, "results": []}

    ids_to_compare = sorted(common_ids)
    if max_comparisons is not None:
        ids_to_compare = ids_to_compare[:max_comparisons]

    print(f"Total common IDs: {len(common_ids)}")
    print(f"IDs to compare: {len(ids_to_compare)}")
    print()

    results: List[dict] = []

    def process_id(idx_id_tuple):
        idx, id_text = idx_id_tuple
        print(f"\n[{idx}/{len(ids_to_compare)}] Comparing ID: {id_text}")
        
        image_a = id_map_a[id_text]
        image_b = id_map_b[id_text]

        try:
            comparison, image_b_model_input_size_px, image_b_ruler_origin_px = compare_two_images(
                image_a_path=image_a,
                image_b_path=image_b,
                model=model,
                temperature=temperature,
            )
            comparison = postprocess_comparison_output(comparison)
            
            return {
                "id": id_text,
                "filename_a": image_a.name,
                "filename_b": image_b.name,
                "image_a": str(image_a),
                "image_b": str(image_b),
                "image_b_model_input_size_px": image_b_model_input_size_px,
                "image_b_ruler_origin_px": image_b_ruler_origin_px,
                "comparison": comparison,
            }
        except Exception as e:
            print(f"  ERROR while comparing ID {id_text}: {e}")
            return {
                "id": id_text,
                "filename_a": image_a.name,
                "filename_b": image_b.name,
                "image_a": str(image_a),
                "image_b": str(image_b),
                "comparison": "",
                "error": str(e),
            }

    # Parallelize Comparisons
    max_workers = min(15, len(ids_to_compare))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_id, enumerate(ids_to_compare, 1)))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total comparisons: {len(results)}")

    # Lưu kết quả vào file JSON
    output_file = compare_dir / "step6_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {"comparisons": len(results), "results": results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Results saved to: {output_file}")
    print("\nDone!")

    return {"comparisons": len(results), "results": results}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Bước 6: So sánh bbox bằng Google Gemini")
    parser.add_argument(
        "--compare-dir",
        type=str,
        default=None,
        help="Thư mục compare_temp (mặc định: app_v2/compare_temp)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model (mặc định: gemini-2.5-flash). Các model khác: gemini-1.5-flash, gemini-pro",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature (mặc định: 0.0)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Số lượng so sánh tối đa",
    )

    args = parser.parse_args()

    # Thư mục compare
    if args.compare_dir:
        compare_dir = Path(args.compare_dir)
    else:
        compare_dir = Path(__file__).resolve().parent / "compare_temp"

    # Chạy step 6 với Gemini
    results = step6_compare_with_gemini(
        compare_dir=compare_dir,
        model=args.model,
        temperature=args.temperature,
        max_comparisons=args.max,
    )

    return 0 if results["comparisons"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

