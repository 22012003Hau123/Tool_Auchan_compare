"""
Xử lý bbox và ID detection.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from PIL import Image
import pytesseract

from .detector import BBoxIDDetector, Detection

# Load .env để lấy TESSERACT_CMD
try:
    from dotenv import load_dotenv
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
except ImportError:
    pass

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def crop_image(image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
    """Cắt ảnh theo bbox [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return image.crop((x1, y1, x2, y2))


def ocr_id(image: Image.Image, lang: str = "fra") -> str:
    """
    Dùng Tesseract OCR để nhận diện ID từ ảnh.
    
    Args:
        image: Ảnh chứa ID
        lang: Ngôn ngữ OCR (mặc định: fra)
        
    Returns:
        Text ID đã nhận diện (đã trim)
    """
    try:
        text = pytesseract.image_to_string(image, lang=lang)
        # Lấy dòng đầu tiên không rỗng
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return text.strip()
    except Exception:
        return ""


def process_bbox_with_id(
    bbox_image: Image.Image,
    detector: BBoxIDDetector,
    *,
    bbox_conf: float = 0.25,
    id_conf: float = 0.1,
    ocr_lang: str = "fra",
) -> Optional[str]:
    """
    Xử lý ảnh bbox để tìm ID và OCR.
    
    Args:
        bbox_image: Ảnh bbox đã cắt
        detector: YOLO detector
        bbox_conf: Confidence threshold cho bbox detection
        id_conf: Confidence threshold cho ID detection
        ocr_lang: Ngôn ngữ OCR
        
    Returns:
        ID text nếu tìm thấy, None nếu không
    """
    # Detect ID trong bbox image
    detections = detector.predict(bbox_image, conf=id_conf)
    
    # Lọc chỉ lấy class "id"
    id_detections = [
        d for d in detections 
        if d.class_name == "id" and d.confidence >= id_conf
    ]
    
    if not id_detections:
        return None
    
    # Lấy ID detection có confidence cao nhất
    best_id = max(id_detections, key=lambda d: d.confidence)
    
    # Cắt ảnh ID
    id_image = crop_image(bbox_image, best_id.bbox)
    
    # OCR ID
    id_text = ocr_id(id_image, lang=ocr_lang)
    
    return id_text if id_text else None


def save_bbox_image(
    bbox_image: Image.Image,
    output_dir: Path,
    identifier: Optional[str] = None,
    prefix: str = "",
) -> Path:
    """
    Lưu ảnh bbox với tên file.
    
    Args:
        bbox_image: Ảnh bbox
        output_dir: Thư mục output
        identifier: ID text (nếu có, sẽ dùng làm tên file)
        prefix: Tiền tố cho tên file
        
    Returns:
        Đường dẫn file đã lưu
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tạo tên file
    if identifier:
        # Dùng ID làm tên file (sanitize)
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in identifier)
        filename = f"{prefix}{safe_id}.png"
    else:
        # Dùng timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}{timestamp}.png"
    
    filepath = output_dir / filename
    bbox_image.save(filepath, "PNG")
    return filepath

