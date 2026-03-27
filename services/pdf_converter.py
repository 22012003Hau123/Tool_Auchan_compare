"""
Convert PDF sang ảnh.
"""

from __future__ import annotations

import os
from pathlib import Path
from pdf2image import convert_from_bytes
from PIL import Image

# Load .env để lấy POPPLER_PATH
try:
    from dotenv import load_dotenv
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
except ImportError:
    pass

POPPLER_PATH = os.getenv("POPPLER_PATH")


def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> list[Image.Image]:
    """
    Convert PDF bytes sang danh sách ảnh.
    
    Args:
        pdf_bytes: Nội dung PDF (bytes)
        dpi: Độ phân giải (mặc định: 300)
        
    Returns:
        Danh sách PIL Image
    """
    convert_kwargs = {"dpi": dpi}
    if POPPLER_PATH:
        convert_kwargs["poppler_path"] = POPPLER_PATH
    
    images = convert_from_bytes(pdf_bytes, **convert_kwargs)
    return images

