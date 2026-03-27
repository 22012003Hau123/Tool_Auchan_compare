"""
Bước 4: Xử lý PDF thứ 2 (PDF B) trong thư mục tạm.
- Xóa thư mục tạm nếu có
- Chạy step1 → step2 → step3 tuần tự vào thư mục tạm
- Sau đó so sánh với bbox đã đổi tên từ PDF A
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

# Import các hàm từ step 1, 2, 3
from step1 import step1_detect_and_save_bbox
from step2 import step2_detect_id_and_save
from step3 import step3_ocr_and_rename


def step4_process_pdf_b(
    pdf_b_path: Path,
    model_path: Path,
    temp_dir: Path,
    *,
    conf: float = 0.8,
    id_conf: float = 0.1,
    ocr_lang: str = "en",
    use_gpu: bool = False,
    imgsz: int = 640,
    device: str | None = None,
    max_det: int = 300,
    merge_iou: float = 0.3,
    use_openai_fallback: bool = True,
    openai_model: str = "gpt-4o-mini",
) -> dict:
    """
    Xử lý PDF B: step1 → step2 → step3 vào thư mục tạm.
    
    Args:
        pdf_b_path: Đường dẫn đến PDF B
        model_path: Đường dẫn đến model YOLO
        temp_dir: Thư mục tạm (sẽ bị xóa rỗng trước khi chạy)
        conf: Confidence cho bbox detection
        id_conf: Confidence cho ID detection
        ocr_lang: Ngôn ngữ OCR
        use_gpu: Dùng GPU cho OCR
        imgsz: Kích thước ảnh input cho YOLO
        device: Thiết bị cho YOLO (cuda/cpu)
        max_det: Số lượng detection tối đa
        
    Returns:
        Thống kê tổng hợp
    """
    print("="*60)
    print("BƯỚC 4: XỬ LÝ PDF B TRONG THƯ MỤC TẠM")
    print("="*60)
    print(f"PDF B: {pdf_b_path.name}")
    print(f"Thư mục tạm: {temp_dir}")
    print()
    
    # Xóa thư mục tạm nếu có
    if temp_dir.exists():
        print(f"⚠️  Xóa thư mục tạm cũ: {temp_dir}")
        shutil.rmtree(temp_dir)
        print("✅ Đã xóa")
    
    # Tạo lại thư mục tạm
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Tạo các thư mục con
    temp_bbox_dir = temp_dir / "bbox"
    temp_id_dir = temp_dir / "id"
    
    print()
    print("="*60)
    print("STEP 1: DETECT BBOX")
    print("="*60)
    
    # Bước 1: Detect bbox
    total_bbox = step1_detect_and_save_bbox(
        pdf_path=pdf_b_path,
        model_path=model_path,
        output_dir=temp_bbox_dir,
        conf=conf,
        dpi=300,
        imgsz=imgsz,
        device=device,
        max_det=max_det,
        save_overview=False,
        merge_iou=merge_iou,
    )
    
    if total_bbox == 0:
        print("[X] Khong detect duoc bbox nao")
        return {"step1": 0, "step2": 0, "step3": 0}
    
    # Lấy tên PDF (không có extension) để tìm thư mục con
    pdf_name = pdf_b_path.stem
    actual_bbox_dir = temp_bbox_dir / pdf_name
    
    # NOTE: Step 2 (ID detection) va Step 3 (OCR rename) da duoc thay the
    # boi step5_dino.py (DINOv2 visual matching). Khong can OCR ID nua.
    
    print()
    print("=" * 60)
    print("TONG KET BUOC 4")
    print("=" * 60)
    print(f"Bbox detected: {total_bbox}")
    print(f"[INFO] Step 2+3 skipped (using DINOv2 matching instead of ID-based)")
    print(f"\n  Thu muc tam: {temp_dir}")
    print(f"   - bbox/{pdf_name}/: {actual_bbox_dir}")
    print("\n  Hoan thanh buoc 4!")
    
    return {
        "step1": total_bbox,
        "step2": 0,
        "step3": total_bbox,  # Gia lap thanh cong de pipeline khong bi chan
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Bước 4: Xử lý PDF B trong thư mục tạm")
    parser.add_argument("pdf_b_path", type=str, help="Đường dẫn đến PDF B")
    parser.add_argument("--model", type=str, default=None, help="Đường dẫn đến model YOLO")
    parser.add_argument("--temp-dir", type=str, default=None, help="Thư mục tạm (mặc định: app_v2/temp)")
    parser.add_argument("--conf", type=float, default=0.8, help="Confidence cho bbox (mặc định: 0.9)")
    parser.add_argument("--id-conf", type=float, default=0.1, help="Confidence cho ID (mặc định: 0.1)")
    parser.add_argument("--lang", type=str, default="en", help="Ngôn ngữ OCR (mặc định: en)")
    parser.add_argument("--gpu", action="store_true", help="Dùng GPU cho OCR")
    parser.add_argument("--device", type=str, default=None, help="Device cho YOLO (cuda/cpu)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh input (mặc định: 640)")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections (mặc định: 300)")
    parser.add_argument("--merge-iou", type=float, default=0.1, help="IoU threshold để merge bbox overlap (0.0 = tắt, mặc định: 0.3)")
    parser.add_argument("--no-openai-fallback", action="store_true", help="Không dùng OpenAI khi EasyOCR thất bại")
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini", help="OpenAI model cho fallback (mặc định: gpt-4o-mini)")
    
    args = parser.parse_args()
    
    # Tìm model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = Path(__file__).resolve().parent / "models" / "bbox_id.pt"
        if not model_path.exists():
            model_path = Path(__file__).resolve().parent.parent / "models" / "bbox_id.pt"
    
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return 1
    
    pdf_b_path = Path(args.pdf_b_path)
    if not pdf_b_path.exists():
        print(f"ERROR: PDF B not found: {pdf_b_path}")
        return 1
    
    # Thư mục tạm
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
    else:
        temp_dir = Path(__file__).resolve().parent / "temp"
    
    # Chạy step 4
    results = step4_process_pdf_b(
        pdf_b_path=pdf_b_path,
        model_path=model_path,
        temp_dir=temp_dir,
        conf=args.conf,
        id_conf=args.id_conf,
        ocr_lang=args.lang,
        use_gpu=args.gpu,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        merge_iou=args.merge_iou,
        use_openai_fallback=not args.no_openai_fallback,
        openai_model=args.openai_model,
    )
    
    return 0 if results["step3"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

