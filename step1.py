"""
Bước 1: Convert PDF sang ảnh và detect bbox, lưu vào thư mục bbox.
Đơn giản hóa để giống lệnh YOLO CLI trên Colab.
"""

from __future__ import annotations

import sys
import os
import math
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_bytes
from ultralytics import YOLO

from bbox_splitter import split_bbox_if_needed  # Import module cắt bbox

# Load .env để lấy POPPLER_PATH
try:
    from dotenv import load_dotenv
    PROJECT_ROOT = Path(__file__).resolve().parent
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
except ImportError:
    pass

POPPLER_PATH = os.getenv("POPPLER_PATH")

# Tự động detect poppler path cho Linux
if sys.platform.startswith('linux'):
    # Nếu POPPLER_PATH từ .env là Windows path (không tồn tại trên Linux), override nó
    if POPPLER_PATH and not os.path.exists(POPPLER_PATH):
        print(f"⚠️  POPPLER_PATH từ .env không tồn tại: {POPPLER_PATH}")
        POPPLER_PATH = None
    
    # Auto-detect nếu chưa có hoặc path không hợp lệ
    if not POPPLER_PATH:
        # Trong Docker Linux, poppler thường ở /usr/bin
        if os.path.exists('/usr/bin/pdftoppm') and os.path.exists('/usr/bin/pdfinfo'):
            POPPLER_PATH = '/usr/bin'
            print(f"✅ Auto-detected poppler at: {POPPLER_PATH}")


def calculate_iou(bbox1: tuple, bbox2: tuple) -> float:
    """
    Tính IoU (Intersection over Union) giữa 2 bbox.
    
    Args:
        bbox1: (x1, y1, x2, y2, conf)
        bbox2: (x1, y1, x2, y2, conf)
        
    Returns:
        IoU value (0.0 - 1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
    x1_2, y1_2, x2_2, y2_2 = bbox2[:4]
    
    # Tính intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Tính union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def merge_overlapping_bboxes(bbox_list: list, iou_threshold: float = 0.5, verbose: bool = False) -> list:
    """
    Merge các bbox overlap dựa trên IoU threshold.
    Sử dụng phương pháp đệ quy để merge tất cả bbox trong cùng một nhóm overlap.
    
    Args:
        bbox_list: List of (x1, y1, x2, y2, confidence)
        iou_threshold: Ngưỡng IoU để merge (mặc định: 0.5)
        verbose: Hiển thị thông tin debug
        
    Returns:
        List bbox đã merge
    """
    if not bbox_list:
        return []
    
    if len(bbox_list) == 1:
        return bbox_list
    
    # Sắp xếp theo confidence giảm dần
    bbox_list = sorted(bbox_list, key=lambda b: b[4], reverse=True)
    
    merged = []
    used = [False] * len(bbox_list)
    
    for i in range(len(bbox_list)):
        if used[i]:
            continue
        
        # Bắt đầu với bbox i
        group = [i]
        used[i] = True
        
        # Tìm tất cả bbox overlap với nhóm hiện tại (đệ quy)
        changed = True
        while changed:
            changed = False
            for idx in group:
                current = bbox_list[idx]
                for j in range(len(bbox_list)):
                    if used[j] or j in group:
                        continue
                    
                    iou = calculate_iou(current, bbox_list[j])
                    if verbose and iou > 0.1:  # Chỉ hiển thị nếu IoU > 0.1
                        print(f"    IoU between bbox {idx} and {j}: {iou:.3f}")
                    
                    if iou >= iou_threshold:
                        group.append(j)
                        used[j] = True
                        changed = True
        
        # Merge tất cả bbox trong nhóm
        if len(group) > 1:
            x1_min = min(bbox_list[idx][0] for idx in group)
            y1_min = min(bbox_list[idx][1] for idx in group)
            x2_max = max(bbox_list[idx][2] for idx in group)
            y2_max = max(bbox_list[idx][3] for idx in group)
            conf_max = max(bbox_list[idx][4] for idx in group)
            
            if verbose:
                print(f"    Merged {len(group)} bboxes into one")
            
            merged.append((x1_min, y1_min, x2_max, y2_max, conf_max))
        else:
            # Chỉ có 1 bbox, giữ nguyên
            merged.append(bbox_list[group[0]])
    
    return merged


def draw_bboxes_on_image(image: Image.Image, bbox_list: list) -> Image.Image:
    """
    Vẽ các bbox lên ảnh để tạo ảnh tổng thể.
    
    Args:
        image: Ảnh gốc
        bbox_list: List of (x1, y1, x2, y2, confidence)
        
    Returns:
        Ảnh đã vẽ bbox
    """
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Màu sắc
    colors = [
        (255, 0, 0),    # Đỏ
        (0, 255, 0),    # Xanh lá
        (0, 0, 255),    # Xanh dương
        (255, 255, 0),  # Vàng
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    
    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
        except:
            font = ImageFont.load_default()
    
    for idx, (x1, y1, x2, y2, confidence) in enumerate(bbox_list):
        color = colors[idx % len(colors)]
        
        # Vẽ rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        
        # Vẽ label
        label = f"#{idx+1} ({confidence:.2f})"
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # Background cho text
        text_bg = [(x1, y1 - text_height - 4), (x1 + text_width + 8, y1)]
        draw.rectangle(text_bg, fill=color)
        
        # Text
        draw.text((x1 + 4, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)
    
    return img_with_boxes


def step1_detect_and_save_bbox(
    pdf_path: Path,
    model_path: Path,
    output_dir: Path,
    *,
    conf: float = 0.8,
    dpi: int = 300,
    imgsz: int = 640,
    device: str | None = None,
    max_det: int = 300,
    save_overview: bool = False,
    merge_iou: float = 0.45,
) -> int:
    """
    Bước 1: Convert PDF sang ảnh, detect bbox và lưu vào thư mục bbox.
    Đơn giản - giống lệnh: yolo predict model=... source=... save=True
    
    Args:
        pdf_path: Đường dẫn đến file PDF
        model_path: Đường dẫn đến model YOLO
        output_dir: Thư mục output để lưu bbox
        conf: Confidence threshold (mặc định: 0.25)
        dpi: DPI khi convert PDF (mặc định: 300)
        imgsz: Kích thước ảnh input cho YOLO (mặc định: 640)
        device: Thiết bị chạy YOLO (None = auto, hoặc 'cuda', 'cpu')
        max_det: Số lượng detection tối đa (mặc định: 300)
        save_overview: Lưu ảnh tổng thể với các bbox đã vẽ
        
    Returns:
        Tổng số bbox đã lưu
    """
    print(f"PDF: {pdf_path.name}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Config: conf={conf}, imgsz={imgsz}, max_det={max_det}, device={device or 'auto'}")
    print()
    
    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(str(model_path))
    print("Model loaded")
    
    # Đọc PDF
    print("Reading PDF...")
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    # Convert PDF sang ảnh
    print("Converting PDF to images...")
    convert_kwargs = {"dpi": dpi}
    if POPPLER_PATH:
        convert_kwargs["poppler_path"] = POPPLER_PATH
    
    images = convert_from_bytes(pdf_bytes, **convert_kwargs)
    print(f"Converted {len(images)} pages")
    
    # Tạo thư mục output với tên file PDF (bỏ extension)
    pdf_name = pdf_path.stem  # Tên file không có extension
    pdf_output_dir = output_dir / pdf_name
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {pdf_output_dir}")
    
    # Xử lý từng trang
    total_bbox = 0
    
    for page_idx, page_image in enumerate(images, 1):
        print(f"\n[Page {page_idx}/{len(images)}]")
        
        # Predict - giống lệnh: yolo predict model=... source=...
        print(f"  Detecting bboxes...")
        
        # Tạo kwargs cho predict
        predict_kwargs = {
            "conf": conf,
            "iou": 0.45,  # NMS IOU threshold (mặc định YOLO)
            "verbose": False,
            "save": False,
            "max_det": max_det,
            "imgsz": imgsz,
        }
        
        if device:
            predict_kwargs["device"] = device
        
        # Predict trực tiếp với PIL Image
        results = model.predict(page_image, **predict_kwargs)
        
        if not results or len(results) == 0:
            print(f"  WARNING: No results from YOLO")
            continue
        
        result = results[0]
        if not result.boxes or len(result.boxes) == 0:
            print(f"  WARNING: No boxes detected")
            continue
        
        # Lấy class names
        class_names = model.names
        
        # Lọc chỉ lấy class "bbox"
        bbox_list = []
        stats = {"total": len(result.boxes), "bbox": 0, "id": 0, "other": 0}
        
        for box in result.boxes:
            cls_id = int(box.cls.item())
            class_name = class_names.get(cls_id, str(cls_id))
            confidence = float(box.conf.item()) if box.conf is not None else 0.0
            
            # Đếm theo class
            if class_name == "bbox":
                stats["bbox"] += 1
            elif class_name == "id":
                stats["id"] += 1
            else:
                stats["other"] += 1
            
            # Chỉ xử lý class "bbox"
            if class_name != "bbox":
                continue
            
            # Lấy tọa độ từ YOLO (absolute pixel coordinates: x1, y1, x2, y2)
            xyxy = box.xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]
            x1_raw, y1_raw, x2_raw, y2_raw = [float(v) for v in xyxy]
            
            # Convert sang int để cắt theo pixel (giống code mẫu)
            img_width, img_height = page_image.size
            x_min = int(x1_raw)
            y_min = int(y1_raw)
            x_max = int(x2_raw)
            y_max = int(y2_raw)
            
            # Ensure coordinates are within image bounds (giống code mẫu)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)
            
            # Kiểm tra hợp lệ
            if x_max > x_min and y_max > y_min:
                bbox_list.append((x_min, y_min, x_max, y_max, confidence))
        
        # Hiển thị thống kê
        print(f"  YOLO detected: {stats['total']} total (bbox={stats['bbox']}, id={stats['id']}, other={stats['other']})")
        
        if not bbox_list:
            print(f"  WARNING: No bbox class found")
            continue
        
        # Hiển thị confidence range
        confidences = [c for _, _, _, _, c in bbox_list]
        print(f"  Confidence: min={min(confidences):.3f}, max={max(confidences):.3f}, avg={sum(confidences)/len(confidences):.3f}")
        
        # Merge bbox overlap
        if merge_iou > 0:
            bbox_before_merge = len(bbox_list)
            print(f"  Before merge: {bbox_before_merge} bboxes")
            bbox_list = merge_overlapping_bboxes(bbox_list, merge_iou, verbose=True)
            bbox_after_merge = len(bbox_list)
            if bbox_before_merge != bbox_after_merge:
                print(f"  ✅ Merged: {bbox_before_merge} -> {bbox_after_merge} bboxes (IoU threshold: {merge_iou})")
            else:
                print(f"  ⚠️  No merge occurred (IoU threshold: {merge_iou} may be too high)")
        
        # Lưu ảnh tổng thể nếu được yêu cầu
        if save_overview:
            overview_image = draw_bboxes_on_image(page_image, bbox_list)
            overview_filename = f"page{page_idx}_overview.png"
            overview_filepath = pdf_output_dir / overview_filename
            overview_image.save(overview_filepath, "PNG")
            print(f"  Saved overview: {overview_filename}")
        
        extra_top_padding = 30  # pixel padding thêm phía trên để tránh cắt thiếu
        
        # Lưu từng bbox
        for bbox_idx, (x1, y1, x2, y2, confidence) in enumerate(bbox_list):
            # Cắt ảnh bbox theo đúng tọa độ từ YOLO (đã được clip khi thêm vào list)
            crop_y1 = max(0, y1 - extra_top_padding)
            bbox_image = page_image.crop((x1, crop_y1, x2, y2))
            
            # --- LUỒNG MỚI: Kiểm tra và cắt ảnh dính bằng GPT-4o-mini ---
            split_images = split_bbox_if_needed(bbox_image)
            
            # Lưu các ảnh (1 ảnh nếu ko cắt, 2 ảnh nếu đã cắt thành công)
            for part_idx, part_image in enumerate(split_images):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                
                if len(split_images) == 1:
                    filename = f"page{page_idx}_bbox{bbox_idx}_{timestamp}.png"
                else:
                    filename = f"page{page_idx}_bbox{bbox_idx}_part{part_idx+1}_{timestamp}.png"
                    
                filepath = pdf_output_dir / filename
                part_image.save(filepath, "PNG")
                
                aw, ah = part_image.size
                print(f"  [{bbox_idx+1}{'.'+str(part_idx+1) if len(split_images)>1 else ''}] conf={confidence:.2f}, size={aw}x{ah} -> {filename}")
                total_bbox += 1
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total bboxes: {total_bbox}")
    print(f"Output directory: {pdf_output_dir}")
    print("\nDone!")
    
    return total_bbox


def main():
    """Chạy bước 1: Detect và lưu bbox."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Detect bbox from PDF (giống lệnh: yolo predict model=... source=...)"
    )
    parser.add_argument("pdf_path", type=str, help="Đường dẫn đến file PDF")
    parser.add_argument("--model", type=str, default=None, help="Đường dẫn đến model YOLO")
    parser.add_argument("--conf", type=float, default=0.8, help="Confidence threshold (mặc định: 0.25)")
    parser.add_argument("--output", type=str, default=None, help="Thư mục output (mặc định: app_v2/bbox)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI khi convert PDF (mặc định: 300)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh input cho YOLO (mặc định: 640)")
    parser.add_argument("--device", type=str, default=None, help="Thiết bị: cuda, cuda:0, cpu (mặc định: auto)")
    parser.add_argument("--max-det", type=int, default=300, help="Số lượng detection tối đa (mặc định: 300)")
    parser.add_argument("--overview", action="store_true", help="Lưu ảnh tổng thể với bbox đã vẽ")
    parser.add_argument("--merge-iou", type=float, default=0.5, help="IoU threshold để merge bbox overlap (0.0 = tắt, mặc định: 0.5)")
    
    args = parser.parse_args()
    
    # Tìm model path
    if args.model:
        model_path = Path(args.model)
    else:
        # Tìm trong app_v2/models trước
        model_path = Path(__file__).resolve().parent / "models" / "bbox_id.pt"
        if not model_path.exists():
                model_path = Path("/home/hault/pdf_compare_app/models/bbox_id.pt")
    
    if not model_path.exists():
        print(f"ERROR: Model not found at: {model_path}")
        return 1
    
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found at: {pdf_path}")
        return 1
    
    # Thư mục output
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).resolve().parent / "bbox"
    
    # Gọi hàm bước 1
    total_bbox = step1_detect_and_save_bbox(
        pdf_path=pdf_path,
        model_path=model_path,
        output_dir=output_dir,
        conf=args.conf,
        dpi=args.dpi,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        save_overview=args.overview,
        merge_iou=args.merge_iou,
    )
    
    return 0 if total_bbox > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
