"""
Bước 2: Detect ID trong các ảnh bbox, cắt ID và lưu vào thư mục id.
"""

from __future__ import annotations

import sys
import os
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import numpy as np

from step3 import ocr_id_with_openai, encode_image_to_base64

# Load .env
try:
    from dotenv import load_dotenv
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
except ImportError:
    pass


def draw_ids_on_bbox(bbox_image: Image.Image, id_bbox: tuple) -> Image.Image:
    """
    Vẽ ID bbox lên ảnh bbox để tạo ảnh overview.
    
    Args:
        bbox_image: Ảnh bbox gốc
        id_bbox: (x1, y1, x2, y2) của ID
        
    Returns:
        Ảnh đã vẽ ID bbox
    """
    img_with_id = bbox_image.copy()
    draw = ImageDraw.Draw(img_with_id)
    
    x1, y1, x2, y2 = [int(v) for v in id_bbox]
    
    # Vẽ rectangle màu đỏ
    draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=3)
    
    # Vẽ label "ID"
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    label = "ID"
    bbox_text = draw.textbbox((0, 0), label, font=font)
    text_width = bbox_text[2] - bbox_text[0]
    text_height = bbox_text[3] - bbox_text[1]
    
    # Background cho text
    text_bg = [(x1, y1 - text_height - 4), (x1 + text_width + 8, y1)]
    draw.rectangle(text_bg, fill=(255, 0, 0))
    
    # Text
    draw.text((x1 + 4, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)
    
    return img_with_id


def ocr_id_from_bbox_with_openai(bbox_image: Image.Image | Path, model: str = "gpt-4o") -> str:
    """
    OCR ID từ bbox image dùng OpenAI với prompt tối ưu cho trường hợp đọc từ bbox.
    ID là số trên banner màu xanh lá, thường ở vùng trên của ảnh.
    
    Args:
        bbox_image: PIL Image hoặc Path đến file ảnh bbox
        model: OpenAI model (mặc định: gpt-4o-mini)
        
    Returns:
        ID text (chỉ số) hoặc "" nếu không trích xuất được
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        import base64
        import io
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(f"    ERROR: OPENAI_API_KEY not found in .env")
            return ""
        
        # Encode ảnh
        if isinstance(bbox_image, Path):
            image_b64 = encode_image_to_base64(bbox_image)
        else:
            # Convert PIL Image sang base64
            buffer = io.BytesIO()
            bbox_image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Khởi tạo ChatOpenAI
        chat = ChatOpenAI(
            model=model,
            temperature=0.0,
            api_key=api_key,
        )
        
        # Prompt tối ưu cho việc đọc ID từ bbox image
        # ID là số trên banner màu xanh lá, thường ở vùng trên của ảnh
        system_prompt = (
            "You are an OCR expert specialized in extracting a single ID number. "
            "Always return exactly one sequence of digits (3 to 6 digits long). "
            "The ID is printed in white on a solid green rectangular banner, usually near the top edge of the image. "
            "Ignore any other numbers, barcodes, captions, or price tags that are outside the green banner. "
            "If multiple digit groups exist, pick the one fully inside the green banner and closest to the top-left. "
            "Never concatenate multiple numbers. Respond with ONLY the digits, no spaces, no text, no punctuation."
        )
        user_prompt = (
            "Read the ID number from this cropped product image. "
            "Instructions:\n"
            "1. Find the bright green rectangular banner (white text on green background) near the top of the image.\n"
            "2. Extract ONLY the digits printed inside that banner (usually 3-6 digits).\n"
            "3. Ignore all other text or numbers elsewhere in the image.\n"
            "4. Output exactly that digit sequence with nothing else."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": user_prompt
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
        print(f"    ERROR OpenAI OCR from bbox: {e}")
        return ""


def step2_detect_id_and_save(
    bbox_dir: Path,
    id_dir: Path,
    model_path: Path,
    *,
    id_conf: float = 0.1,
    min_confidence: float = 0.0,
    show_progress: bool = True,
    save_low_conf: bool = False,
    save_report: bool = True,
    save_overview: bool = False,
) -> dict[str, int]:
    """
    Bước 2: Detect ID trong các ảnh bbox, cắt và lưu vào thư mục id.
    
    Args:
        bbox_dir: Thư mục chứa các ảnh bbox
        id_dir: Thư mục output để lưu ảnh ID đã cắt
        model_path: Đường dẫn đến model YOLO
        id_conf: Confidence threshold cho ID detection
        min_confidence: Confidence tối thiểu để lưu (lọc bỏ confidence quá thấp)
        show_progress: Hiển thị progress chi tiết
        save_low_conf: Lưu ID có confidence thấp vào thư mục riêng
        save_report: Lưu báo cáo chi tiết vào file text
        save_overview: Lưu ảnh bbox với ID bbox đã vẽ lên vào thư mục
        
    Returns:
        Dict với thống kê chi tiết
    """
    print(f"📁 Thư mục bbox: {bbox_dir}")
    print(f"📁 Thư mục id: {id_dir}")
    print(f"🤖 Model: {model_path}")
    print(f"⚙️  ID confidence threshold: {id_conf}")
    if min_confidence > 0:
        print(f"⚙️  Min confidence để lưu: {min_confidence}")
    print()
    
    # Xác định tên file PDF từ bbox_dir
    # Nếu bbox_dir là thư mục con (ví dụ: bbox/file_name/), lấy tên thư mục con
    # Nếu bbox_dir là thư mục gốc (bbox/), lấy tên từ thư mục cha
    pdf_name = bbox_dir.name if bbox_dir.parent.name == "bbox" else bbox_dir.stem
    
    # Tạo thư mục id với tên file PDF
    pdf_id_dir = id_dir / pdf_name
    pdf_id_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Thư mục id cho PDF '{pdf_name}': {pdf_id_dir}")
    print()
    
    # Load YOLO model
    print("⏳ Đang load YOLO model...")
    model = YOLO(str(model_path))
    print("✅ Đã load model")
    
    # Tạo thư mục cho ID confidence thấp nếu cần
    low_conf_dir = None
    if save_low_conf and min_confidence > 0:
        low_conf_dir = pdf_id_dir.parent / f"{pdf_id_dir.name}_low_conf"
        low_conf_dir.mkdir(parents=True, exist_ok=True)
    
    # Tạo thư mục overview nếu cần
    overview_dir = None
    if save_overview:
        overview_dir = pdf_id_dir.parent / f"{pdf_id_dir.name}_bbox_with_id_overview"
        overview_dir.mkdir(parents=True, exist_ok=True)
    
    # Lấy tất cả file ảnh trong thư mục bbox
    bbox_files = sorted(bbox_dir.glob("*.png"))
    
    if not bbox_files:
        print("❌ Không tìm thấy file ảnh bbox nào trong thư mục")
        return {
            "total": 0, 
            "with_id": 0, 
            "saved": 0, 
            "skipped_low_conf": 0,
            "no_id": 0,
            "confidence_stats": {"min": 0, "max": 0, "avg": 0}
        }
    
    print(f"📊 Tìm thấy {len(bbox_files)} file ảnh bbox")
    print()
    
    stats = {
        "total": len(bbox_files),
        "with_id": 0,
        "saved": 0,
        "skipped_low_conf": 0,
        "no_id": 0,
        "gpt_renamed": 0,
        "gpt_failed": 0,
        "confidence_stats": {"min": 1.0, "max": 0.0, "sum": 0.0, "count": 0},
        "no_id_files": [],
        "low_conf_files": [],
        "gpt_files": [],
        "gpt_failed_files": [],
        "page_stats": {}
    }

    def normalize_id_text_for_filename(id_text: str) -> str:
        if not id_text:
            return ""
        number_sequences = re.findall(r"\d+", id_text)
        if number_sequences:
            clean = max(number_sequences, key=len)
        else:
            clean = re.sub(r"[^\w\s-]", "", id_text)
            clean = re.sub(r"\s+", "_", clean)
        return clean[:80]

    def update_page_stats(page_num: str, *, has_id: bool) -> None:
        if not page_num:
            return
        if page_num not in stats["page_stats"]:
            stats["page_stats"][page_num] = {"total": 0, "with_id": 0, "no_id": 0}
        stats["page_stats"][page_num]["total"] += 1
        if has_id:
            stats["page_stats"][page_num]["with_id"] += 1
        else:
            stats["page_stats"][page_num]["no_id"] += 1

    def try_gpt_rename(
        bbox_file: Path,
        bbox_image: Image.Image,
    ) -> bool:
        """Fallback: dùng GPT OCR trực tiếp bbox để lấy ID và rename."""
        print("  🤖 GPT fallback: Đang đọc ID từ bbox...")
        gpt_id = ocr_id_from_bbox_with_openai(bbox_image)
        if not gpt_id:
            stats["gpt_failed"] += 1
            stats["gpt_failed_files"].append(bbox_file.name)
            print("  ❌ GPT fallback: Không đọc được ID")
            return False

        id_clean = normalize_id_text_for_filename(gpt_id)
        if not id_clean:
            stats["gpt_failed"] += 1
            stats["gpt_failed_files"].append(bbox_file.name)
            print(f"  ❌ GPT fallback: ID không hợp lệ ({gpt_id})")
            return False

        page_match = re.match(r"page(\d+)", bbox_file.name)
        page_num = page_match.group(1) if page_match else ""
        new_filename = f"page{page_num}_{id_clean}.png" if page_num else f"{id_clean}.png"
        new_bbox_path = bbox_file.with_name(new_filename)
        suffix = 1
        while new_bbox_path.exists():
            new_filename = (
                f"page{page_num}_{id_clean}_{suffix}.png"
                if page_num
                else f"{id_clean}_{suffix}.png"
            )
            new_bbox_path = bbox_file.with_name(new_filename)
            suffix += 1

        # Chỉ đổi tên file bbox trong thư mục bbox, không lưu vào thư mục id
        # vì không có ID đã cắt (YOLO không detect được)
        try:
            bbox_file.rename(new_bbox_path)
        except Exception as e:
            print(f"  ⚠️  GPT fallback: Đổi tên bbox thất bại ({e})")
            return False

        stats["gpt_renamed"] += 1
        stats["with_id"] += 1
        stats["saved"] += 1
        stats["gpt_files"].append(new_filename)
        if page_num:
            update_page_stats(page_num, has_id=True)

        print(f"  ✅ GPT fallback: Đổi tên bbox thành {new_filename}")
        return True
    
    # Xử lý từng file
    for idx, bbox_file in enumerate(bbox_files, 1):
        if show_progress:
            progress = f"[{idx}/{len(bbox_files)}]"
            percent = (idx / len(bbox_files)) * 100
            print(f"{progress} ({percent:.1f}%) {bbox_file.name}")
        else:
            if idx % 10 == 0 or idx == len(bbox_files):
                print(f"[{idx}/{len(bbox_files)}] Đang xử lý...")
        
        # Đọc ảnh
        bbox_image = Image.open(bbox_file)
        
        # Convert sang numpy array cho YOLO
        np_image = np.array(bbox_image.convert("RGB"))
        
        # Detect ID trong bbox image
        results = model.predict(np_image, conf=id_conf, verbose=False)
        
        if not results or not results[0].boxes:
            if try_gpt_rename(bbox_file, bbox_image):
                continue
            stats["no_id"] += 1
            stats["no_id_files"].append(bbox_file.name)
            page_match = re.match(r"page(\d+)", bbox_file.name)
            if page_match:
                update_page_stats(page_match.group(1), has_id=False)
            if show_progress:
                print(f"  ⚠️  Không tìm thấy ID")
            continue
        
        # Lọc chỉ lấy class "id"
        boxes = results[0].boxes
        id_detections = []
        
        for box in boxes:
            cls_id = int(box.cls.item())
            class_name = model.names.get(cls_id, str(cls_id))
            
            if class_name == "id":
                confidence = float(box.conf.item()) if box.conf is not None else 0.0
                xyxy = box.xyxy.tolist()[0]
                id_detections.append((confidence, xyxy))
        
        if not id_detections:
            if try_gpt_rename(bbox_file, bbox_image):
                continue
            stats["no_id"] += 1
            stats["no_id_files"].append(bbox_file.name)
            page_match = re.match(r"page(\d+)", bbox_file.name)
            if page_match:
                update_page_stats(page_match.group(1), has_id=False)
            if show_progress:
                print(f"  ⚠️  Không tìm thấy class 'id'")
            continue
        
        # Lấy ID detection có confidence cao nhất
        best_id = max(id_detections, key=lambda x: x[0])
        confidence, xyxy = best_id
        
        # Kiểm tra min confidence
        if confidence < min_confidence:
            stats["skipped_low_conf"] += 1
            stats["low_conf_files"].append((bbox_file.name, confidence))
            
            # Lưu vào thư mục low_conf nếu được yêu cầu
            if save_low_conf and low_conf_dir:
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                id_image = bbox_image.crop((x1, y1, x2, y2))
                low_conf_filename = bbox_file.name
                low_conf_filepath = low_conf_dir / low_conf_filename
                id_image.save(low_conf_filepath, "PNG")
            
            if show_progress:
                print(f"  ⚠️  Confidence quá thấp ({confidence:.2f} < {min_confidence:.2f}), bỏ qua")
            continue
        
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        
        # Cắt ảnh ID
        id_image = bbox_image.crop((x1, y1, x2, y2))
        
        stats["with_id"] += 1
        
        # Thống kê theo trang
        page_match = re.match(r"page(\d+)", bbox_file.name)
        if page_match:
            update_page_stats(page_match.group(1), has_id=True)
        
        # Cập nhật thống kê confidence
        conf_stats = stats["confidence_stats"]
        conf_stats["min"] = min(conf_stats["min"], confidence)
        conf_stats["max"] = max(conf_stats["max"], confidence)
        conf_stats["sum"] += confidence
        conf_stats["count"] += 1
        
        # Lưu ảnh ID với tên file giống bbox (chỉ thay thư mục)
        id_filename = bbox_file.name
        id_filepath = pdf_id_dir / id_filename
        
        id_image.save(id_filepath, "PNG")
        stats["saved"] += 1
        
        # Lưu ảnh overview nếu được yêu cầu
        if save_overview and overview_dir:
            overview_image = draw_ids_on_bbox(bbox_image, xyxy)
            overview_filename = bbox_file.name
            overview_filepath = overview_dir / overview_filename
            overview_image.save(overview_filepath, "PNG")
        
        if show_progress:
            print(f"  ✅ ID detected (conf={confidence:.2f}) → {id_filename}")
    
    # Tính average confidence
    if stats["confidence_stats"]["count"] > 0:
        stats["confidence_stats"]["avg"] = stats["confidence_stats"]["sum"] / stats["confidence_stats"]["count"]
    else:
        stats["confidence_stats"]["min"] = 0
    
    # Lưu báo cáo
    if save_report:
        report_path = pdf_id_dir.parent / f"step2_report_{pdf_name}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write("📊 BÁO CÁO BƯỚC 2 - DETECT ID\n")
            f.write("="*60 + "\n\n")
            f.write(f"Tổng số file: {stats['total']}\n")
            f.write(f"File có ID: {stats['with_id']}\n")
            f.write(f"File đã lưu: {stats['saved']}\n")
            f.write(f"File bỏ qua (confidence thấp): {stats['skipped_low_conf']}\n")
            f.write(f"File không có ID: {stats['no_id']}\n\n")
            
            if stats["confidence_stats"]["count"] > 0:
                conf_stats = stats["confidence_stats"]
                f.write("📈 Thống kê confidence:\n")
                f.write(f"  Min: {conf_stats['min']:.2f}\n")
                f.write(f"  Max: {conf_stats['max']:.2f}\n")
                f.write(f"  Avg: {conf_stats['avg']:.2f}\n\n")

            if stats["gpt_renamed"] > 0:
                f.write(f"🤖 GPT fallback renamed: {stats['gpt_renamed']}\n")
            if stats["gpt_failed"] > 0:
                f.write(f"🤖 GPT fallback failed: {stats['gpt_failed']}\n")
            f.write("\n")
            
            # Thống kê theo trang
            if stats["page_stats"]:
                f.write("📄 Thống kê theo trang:\n")
                for page_num in sorted(stats["page_stats"].keys(), key=int):
                    page_stat = stats["page_stats"][page_num]
                    f.write(f"  Trang {page_num}: {page_stat['with_id']}/{page_stat['total']} có ID "
                           f"({page_stat['with_id']/page_stat['total']*100:.1f}%)\n")
                f.write("\n")
            
            # Danh sách file không có ID
            if stats["no_id_files"]:
                f.write(f"📋 Danh sách {len(stats['no_id_files'])} file không có ID:\n")
                for filename in stats["no_id_files"][:50]:  # Chỉ hiển thị 50 file đầu
                    f.write(f"  - {filename}\n")
                if len(stats["no_id_files"]) > 50:
                    f.write(f"  ... và {len(stats['no_id_files']) - 50} file khác\n")
                f.write("\n")

            if stats["gpt_files"]:
                f.write(f"🤖 Danh sách {len(stats['gpt_files'])} file đổi tên bằng GPT:\n")
                for filename in stats["gpt_files"]:
                    f.write(f"  - {filename}\n")
                f.write("\n")
            
            # Danh sách file confidence thấp
            if stats["low_conf_files"]:
                f.write(f"📋 Danh sách {len(stats['low_conf_files'])} file có confidence thấp:\n")
                for filename, conf in stats["low_conf_files"]:
                    f.write(f"  - {filename} (conf={conf:.2f})\n")
                f.write("\n")
            
            f.write(f"📁 Thư mục output: {pdf_id_dir}\n")
            if low_conf_dir:
                f.write(f"📁 Thư mục low confidence: {low_conf_dir}\n")
    
    print("\n" + "="*60)
    print("📊 KẾT QUẢ BƯỚC 2")
    print("="*60)
    print(f"Tổng số file: {stats['total']}")
    print(f"File có ID: {stats['with_id']}")
    print(f"File đã lưu: {stats['saved']}")
    print(f"File bỏ qua (confidence thấp): {stats['skipped_low_conf']}")
    print(f"File không có ID: {stats['no_id']}")
    if stats["gpt_renamed"] > 0:
        print(f"🤖 GPT fallback renamed: {stats['gpt_renamed']}")
    if stats["gpt_failed"] > 0:
        print(f"🤖 GPT fallback failed: {stats['gpt_failed']}")
    
    if stats["confidence_stats"]["count"] > 0:
        conf_stats = stats["confidence_stats"]
        print(f"\n📈 Thống kê confidence:")
        print(f"  Min: {conf_stats['min']:.2f}")
        print(f"  Max: {conf_stats['max']:.2f}")
        print(f"  Avg: {conf_stats['avg']:.2f}")
    
    # Thống kê theo trang
    if stats["page_stats"]:
        print(f"\n📄 Thống kê theo trang:")
        for page_num in sorted(stats["page_stats"].keys(), key=int)[:10]:  # Hiển thị 10 trang đầu
            page_stat = stats["page_stats"][page_num]
            print(f"  Trang {page_num}: {page_stat['with_id']}/{page_stat['total']} có ID "
                  f"({page_stat['with_id']/page_stat['total']*100:.1f}%)")
        if len(stats["page_stats"]) > 10:
            print(f"  ... và {len(stats['page_stats']) - 10} trang khác")
    
    print(f"\n📁 Thư mục output: {pdf_id_dir}")
    if low_conf_dir:
        print(f"📁 Thư mục low confidence: {low_conf_dir}")
    if overview_dir:
        print(f"📁 Thư mục overview: {overview_dir}")
        # In ra danh sách file overview
        overview_files = sorted(overview_dir.glob("*.png"))
        if overview_files:
            print(f"\n📸 Ảnh overview ({len(overview_files)} files):")
            for ovf in overview_files[:5]:  # Hiển thị 5 file đầu
                print(f"  - {ovf.name}")
            if len(overview_files) > 5:
                print(f"  ... và {len(overview_files) - 5} file khác")
    if save_report:
        report_path = pdf_id_dir.parent / f"step2_report_{pdf_name}.txt"
        print(f"📄 Báo cáo chi tiết: {report_path}")
    print("\n✅ Hoàn thành bước 2!")
    
    return stats


def main():
    """Chạy bước 2: Detect ID trong bbox và lưu vào thư mục id."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bước 2: Detect ID trong bbox và lưu vào thư mục id")
    parser.add_argument("--bbox-dir", type=str, default=None, help="Thư mục chứa ảnh bbox (mặc định: app_v2/bbox)")
    parser.add_argument("--id-dir", type=str, default=None, help="Thư mục output cho ảnh ID (mặc định: app_v2/id)")
    parser.add_argument("--model", type=str, default=None, help="Đường dẫn đến model YOLO")
    parser.add_argument("--id-conf", type=float, default=0.1, help="Confidence threshold cho ID detection")
    parser.add_argument("--min-conf", type=float, default=0.0, help="Confidence tối thiểu để lưu (lọc bỏ confidence thấp)")
    parser.add_argument("--quiet", action="store_true", help="Không hiển thị progress chi tiết")
    parser.add_argument("--save-low-conf", action="store_true", help="Lưu ID có confidence thấp vào thư mục riêng")
    parser.add_argument("--no-report", action="store_true", help="Không lưu báo cáo chi tiết")
    parser.add_argument("--overview", action="store_true", help="Lưu ảnh bbox với ID bbox đã vẽ lên vào thư mục")
    
    args = parser.parse_args()
    
    # Tìm model path
    if args.model:
        model_path = Path(args.model)
    else:
        # Ưu tiên tìm id_plus.pt trong app_v2/models
        model_path = Path(__file__).resolve().parent / "models" / "id_plus.pt"
        if not model_path.exists():
            # Fallback về bbox_id.pt
            model_path = Path(__file__).resolve().parent.parent / "models" / "bbox_id.pt"
            if not model_path.exists():
                model_path = Path("/home/hault/pdf_compare_app/models/bbox_id.pt")
    
    if not model_path.exists():
        print(f"❌ Không tìm thấy model tại: {model_path}")
        return 1
    
    # Thư mục bbox
    if args.bbox_dir:
        bbox_dir = Path(args.bbox_dir)
    else:
        bbox_dir = Path(__file__).resolve().parent / "bbox"
    
    if not bbox_dir.exists():
        print(f"❌ Không tìm thấy thư mục bbox tại: {bbox_dir}")
        return 1
    
    # Thư mục id
    if args.id_dir:
        id_dir = Path(args.id_dir)
    else:
        id_dir = Path(__file__).resolve().parent / "id"
    
    # Gọi hàm bước 2
    stats = step2_detect_id_and_save(
        bbox_dir=bbox_dir,
        id_dir=id_dir,
        model_path=model_path,
        id_conf=args.id_conf,
        min_confidence=args.min_conf,
        show_progress=not args.quiet,
        save_low_conf=args.save_low_conf,
        save_report=not args.no_report,
        save_overview=args.overview,
    )
    
    return 0 if stats["saved"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

