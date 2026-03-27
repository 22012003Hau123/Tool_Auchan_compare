"""
Bước 3: Dùng EasyOCR để trích xuất ID từ thư mục id, 
sau đó đổi tên file bbox tương ứng thành page{number}_{ID}.png
"""

from __future__ import annotations

import sys
import os
import re
import base64
from pathlib import Path
from PIL import Image

# Load .env để lấy OPENAI_API_KEY
try:
    from dotenv import load_dotenv
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
except ImportError:
    pass

# Global EasyOCR reader (sẽ được khởi tạo khi cần)
_easyocr_reader = None


def get_easyocr_reader(lang: str = 'en', gpu: bool = False):
    """
    Lấy EasyOCR reader (khởi tạo 1 lần).
    
    Args:
        lang: Ngôn ngữ OCR
        gpu: Dùng GPU hay không
        
    Returns:
        EasyOCR reader
    """
    global _easyocr_reader
    
    if _easyocr_reader is None:
        import easyocr
        print(f"Initializing EasyOCR (lang={lang}, gpu={gpu})...")
        _easyocr_reader = easyocr.Reader([lang], gpu=gpu)
        print("EasyOCR ready")
    
    return _easyocr_reader


def encode_image_to_base64(image_path: Path) -> str:
    """
    Encode ảnh sang base64.
    
    Args:
        image_path: Đường dẫn đến ảnh
        
    Returns:
        Base64 string
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def ocr_id_with_openai(image_path: Path, model: str = "gpt-4o") -> str:
    """
    OCR ID từ ảnh dùng OpenAI khi EasyOCR không trích xuất được.
    
    Args:
        image_path: Đường dẫn đến ảnh ID
        model: OpenAI model (mặc định: gpt-4o-mini)
        
    Returns:
        ID text (chỉ số) hoặc "" nếu không trích xuất được
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(f"    ERROR: OPENAI_API_KEY not found in .env")
            return ""
        
        # Encode ảnh
        image_b64 = encode_image_to_base64(image_path)
        
        # Khởi tạo ChatOpenAI
        chat = ChatOpenAI(
            model=model,
            temperature=0.0,
            api_key=api_key,
        )
        
        # Prompt đơn giản: chỉ yêu cầu trả về số ID
        messages = [
            SystemMessage(content="You extract only the ID number from the image. Respond with ONLY the number, ID will be top left of the image, no explanation."),
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": "Extract the ID number from this image. Respond with ONLY the number, nothing else."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                        "detail": "high"
                    }
                }
            ])
        ]
        
        response = chat.invoke(messages)
        id_text = response.content.strip()
        
        # Chỉ giữ số
        number_sequences = re.findall(r'\d+', id_text)
        if not number_sequences:
            return ""
        
        # Lấy chuỗi số dài nhất
        longest_number = max(number_sequences, key=len)
        return longest_number
        
    except Exception as e:
        print(f"    ERROR OpenAI OCR: {e}")
        return ""


def ocr_id(image_path: Path, lang: str = "en", gpu: bool = False, use_openai_fallback: bool = True) -> str:
    """
    OCR ID từ ảnh dùng EasyOCR.
    
    Args:
        image_path: Đường dẫn đến ảnh ID
        lang: Ngôn ngữ OCR (mặc định: en cho số)
        gpu: Dùng GPU hay không
        
    Returns:
        ID text (chỉ giữ số)
    """
    try:
        reader = get_easyocr_reader(lang=lang, gpu=gpu)
        
        # Load và convert ảnh sang grayscale để tránh lỗi với EasyOCR
        from PIL import Image
        import numpy as np
        
        image = Image.open(image_path)
        
        # Convert sang grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert sang numpy array
        img_array = np.array(image)
        
        # OCR với numpy array grayscale - lấy detail để biết vị trí
        result_detail = reader.readtext(img_array, detail=1, paragraph=False)
        
        # result_detail là list of tuples: [(bbox, text, confidence), ...]
        if not result_detail:
            # EasyOCR không tìm thấy text, thử OpenAI nếu có
            if use_openai_fallback:
                print(f"  EasyOCR: No text detected, trying OpenAI OCR...")
                openai_id = ocr_id_with_openai(image_path)
                if openai_id:
                    return openai_id
            return ""
        
        # Xử lý từng detection riêng biệt, không gộp lại
        # Ưu tiên text ở vùng trên của ảnh (ID thường ở đó)
        img_height = img_array.shape[0]
        candidates = []
        
        for detection in result_detail:
            bbox, text, confidence = detection
            # Lấy tọa độ y trung bình của bbox (vị trí dọc)
            y_coords = [point[1] for point in bbox]
            y_center = sum(y_coords) / len(y_coords)
            # Tính vị trí tương đối từ trên xuống (0 = trên cùng, 1 = dưới cùng)
            y_ratio = y_center / img_height
            
            # Tìm tất cả chuỗi số trong text này
            number_sequences = re.findall(r'\d+', text)
            for num_str in number_sequences:
                # Ưu tiên số ở vùng trên (y_ratio < 0.4) và có độ dài hợp lý (3-6 chữ số)
                priority = 0
                if y_ratio < 0.4:  # Vùng trên 40% của ảnh
                    priority += 10
                if 3 <= len(num_str) <= 6:  # Độ dài hợp lý cho ID
                    priority += 5
                priority += len(num_str)  # Ưu tiên số dài hơn
                priority += confidence  # Ưu tiên confidence cao hơn
                
                candidates.append((priority, num_str, y_ratio, confidence))
        
        if not candidates:
            # EasyOCR tìm thấy text nhưng không có số, thử OpenAI nếu có
            if use_openai_fallback:
                print(f"  EasyOCR: No numbers found, trying OpenAI OCR...")
                openai_id = ocr_id_with_openai(image_path)
                if openai_id:
                    return openai_id
            return ""
        
        # Sắp xếp theo priority và lấy số tốt nhất
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_number = candidates[0][1]
        
        return best_number
    except Exception as e:
        print(f"    ERROR OCR: {e}")
        # Nếu EasyOCR lỗi, thử OpenAI nếu có
        if use_openai_fallback:
            print(f"  Trying OpenAI OCR as fallback...")
            openai_id = ocr_id_with_openai(image_path)
            if openai_id:
                return openai_id
        return ""


def step3_ocr_and_rename(
    id_dir: Path,
    bbox_dir: Path,
    *,
    ocr_lang: str = "en",
    use_gpu: bool = False,
    dry_run: bool = False,
    use_openai_fallback: bool = True,
    openai_model: str = "gpt-4o-mini",
) -> dict:
    """
    Bước 3: OCR ID và đổi tên bbox.
    
    Args:
        id_dir: Thư mục chứa ảnh ID
        bbox_dir: Thư mục chứa ảnh bbox
        ocr_lang: Ngôn ngữ OCR (en cho số)
        use_gpu: Dùng GPU cho EasyOCR
        dry_run: Chỉ hiển thị không đổi tên thực sự
        
    Returns:
        Thống kê
    """
    print(f"ID directory: {id_dir}")
    print(f"BBox directory: {bbox_dir}")
    print(f"OCR: EasyOCR (lang={ocr_lang}, gpu={use_gpu})")
    print(f"Dry run: {dry_run}")
    print()
    
    # Import locally to avoid circular import
    from step2 import ocr_id_from_bbox_with_openai
    
    if not id_dir.exists():
        print(f"ERROR: ID directory not found: {id_dir}")
        return {"total": 0, "success": 0, "failed": 0}
    
    if not bbox_dir.exists():
        print(f"ERROR: BBox directory not found: {bbox_dir}")
        return {"total": 0, "success": 0, "failed": 0}
    
    # Lấy tất cả file ID
    id_files = sorted(id_dir.glob("*.png"))
    
    if not id_files:
        print("ERROR: No ID images found")
        return {"total": 0, "success": 0, "failed": 0}
    
    print(f"Found {len(id_files)} ID images")
    print()
    
    stats = {"total": len(id_files), "success": 0, "failed": 0, "no_text": 0}
    processed_bbox_files: set[str] = set()
    
    for idx, id_file in enumerate(id_files, 1):
        print(f"[{idx}/{len(id_files)}] {id_file.name}")
        
        # OCR để lấy ID text (chỉ số)
        print(f"  OCR (EasyOCR)...")
        id_text = ocr_id(id_file, lang=ocr_lang, gpu=use_gpu, use_openai_fallback=use_openai_fallback)
        
        bbox_file = bbox_dir / id_file.name
        if not id_text and bbox_file.exists():
            print("  ⚠️  OCR ID không thành công → thử GPT trực tiếp trên ảnh bbox...")
            # Sử dụng hàm có prompt chuyên dụng thay vì hàm generic của step 3
            gpt_id = ocr_id_from_bbox_with_openai(bbox_file, model=openai_model)
            if gpt_id:
                id_text = gpt_id
                print(f"  🤖 GPT đọc ID: {id_text}")

        if not id_text:
            print(f"  WARNING: No text detected (tried EasyOCR" + (f" + OpenAI {openai_model}" if use_openai_fallback else "") + " + GPT bbox fallback)")
            stats["no_text"] += 1
            continue
        
        # Clean ID text để dùng làm tên file (bỏ ký tự đặc biệt)
        id_text_clean = re.sub(r'[^\w\s-]', '', id_text)
        id_text_clean = re.sub(r'\s+', '_', id_text_clean)
        id_text_clean = id_text_clean[:100]  # Giới hạn độ dài
        
        print(f"  ID: '{id_text}' -> '{id_text_clean}'")
        
        # Tìm file bbox tương ứng (cùng tên)
        # id_file: page1_bbox0_20251116_xxx.png
        # bbox_file: page1_bbox0_20251116_xxx.png (cùng tên)
        if not bbox_file.exists():
            print(f"  WARNING: BBox file not found: {bbox_file.name}")
            stats["failed"] += 1
            continue
        processed_bbox_files.add(bbox_file.name)
        
        # Extract page number từ tên file
        page_match = re.match(r'page(\d+)', id_file.name)
        if not page_match:
            print(f"  WARNING: Cannot extract page number")
            stats["failed"] += 1
            continue
        
        page_num = page_match.group(1)
        
        # Tên file mới: page{number}_{ID}.png
        new_filename = f"page{page_num}_{id_text_clean}.png"
        new_filepath = bbox_dir / new_filename
        
        # Kiểm tra nếu file mới đã tồn tại
        if new_filepath.exists() and new_filepath != bbox_file:
            print(f"  WARNING: Target file already exists: {new_filename}")
            # Thêm suffix để tránh trùng
            counter = 1
            while new_filepath.exists():
                new_filename = f"page{page_num}_{id_text_clean}_{counter}.png"
                new_filepath = bbox_dir / new_filename
                counter += 1
            print(f"  Using: {new_filename}")
        
        # Đổi tên
        if dry_run:
            print(f"  DRY RUN: {bbox_file.name} -> {new_filename}")
        else:
            try:
                bbox_file.rename(new_filepath)
                print(f"  Renamed: {new_filename}")
                stats["success"] += 1
            except Exception as e:
                print(f"  ERROR: Failed to rename - {e}")
                stats["failed"] += 1
    
    # GPT fallback trực tiếp trên các bbox chưa xử lý
    remaining_bbox_files = sorted(
        f for f in bbox_dir.glob("*.png")
        if f.name not in processed_bbox_files and "_bbox" in f.name
    )
    if remaining_bbox_files:
        print("\n🤖 GPT FALLBACK: Đang đọc ID trực tiếp từ các bbox còn lại...")
    for bbox_file in remaining_bbox_files:
        print(f"  GPT → {bbox_file.name}")
        # Dùng hàm có prompt tìm banner xanh lá chuyên dụng
        gpt_id = ocr_id_from_bbox_with_openai(bbox_file, model=openai_model)
        if not gpt_id:
            print("    ❌ Không đọc được ID")
            stats["failed"] += 1
            continue

        page_match = re.match(r"page(\d+)", bbox_file.name)
        page_num = page_match.group(1) if page_match else ""
        id_text_clean = re.sub(r"[^\w\s-]", "", gpt_id)
        id_text_clean = re.sub(r"\s+", "_", id_text_clean)[:100]
        new_filename = f"page{page_num}_{id_text_clean}.png" if page_num else f"{id_text_clean}.png"
        new_filepath = bbox_dir / new_filename
        counter = 1
        while new_filepath.exists():
            new_filename = f"page{page_num}_{id_text_clean}_{counter}.png" if page_num else f"{id_text_clean}_{counter}.png"
            new_filepath = bbox_dir / new_filename
            counter += 1
        try:
            bbox_file.rename(new_filepath)
            print(f"    ✅ Renamed: {new_filename}")
            stats["success"] += 1
        except Exception as e:
            print(f"    ⚠️  Rename failed: {e}")
            stats["failed"] += 1

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total ID files: {stats['total']}")
    print(f"Successfully renamed: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"No text detected: {stats['no_text']}")
    print("\nDone!")

    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Bước 3: OCR ID và đổi tên bbox (dùng EasyOCR)")
    parser.add_argument("--id-dir", type=str, default=None, help="Thư mục chứa ảnh ID (mặc định: app_v2/id)")
    parser.add_argument("--bbox-dir", type=str, default=None, help="Thư mục chứa ảnh bbox (mặc định: app_v2/bbox)")
    parser.add_argument("--lang", type=str, default="en", help="Ngôn ngữ OCR (mặc định: en cho số)")
    parser.add_argument("--gpu", action="store_true", help="Dùng GPU cho EasyOCR")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ hiển thị, không đổi tên thực sự")
    parser.add_argument("--no-openai-fallback", action="store_true", help="Không dùng OpenAI khi EasyOCR thất bại")
    parser.add_argument("--openai-model", type=str, default="gpt-4o", help="OpenAI model cho fallback (mặc định: gpt-4o-mini)")
    
    args = parser.parse_args()
    
    # Thư mục ID
    if args.id_dir:
        id_dir = Path(args.id_dir)
    else:
        id_dir = Path(__file__).resolve().parent / "id"
    
    # Thư mục bbox
    if args.bbox_dir:
        bbox_dir = Path(args.bbox_dir)
    else:
        bbox_dir = Path(__file__).resolve().parent / "bbox"
    
    # Gọi hàm bước 3
    stats = step3_ocr_and_rename(
        id_dir=id_dir,
        bbox_dir=bbox_dir,
        ocr_lang=args.lang,
        use_gpu=args.gpu,
        dry_run=args.dry_run,
        use_openai_fallback=not args.no_openai_fallback,
        openai_model=args.openai_model,
    )
    
    return 0 if stats["success"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

