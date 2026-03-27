"""
Bước 5: Tìm ID trùng giữa bbox (PDF A) và temp/bbox (PDF B).
Lưu các cặp bbox có ID trùng vào thư mục compare_temp để so sánh.
"""

from __future__ import annotations

import sys
import shutil
import re
from pathlib import Path


def extract_id_from_filename(filename: str) -> str | None:
    """
    Trích xuất ID từ tên file.
    
    Args:
        filename: Tên file (ví dụ: page1_15599.png)
        
    Returns:
        ID (ví dụ: "15599") hoặc None nếu không tìm thấy
    """
    # Pattern: page{number}_{ID}.png
    match = re.match(r'page\d+_(.+)\.png', filename)
    if match:
        return match.group(1)
    return None


def step5_find_matching_ids(
    bbox_dir_a: Path,
    bbox_dir_b: Path,
    output_dir: Path,
) -> dict:
    """
    Tìm ID trùng giữa 2 thư mục bbox và lưu vào thư mục so sánh.
    
    Args:
        bbox_dir_a: Thư mục bbox của PDF A (bbox/)
        bbox_dir_b: Thư mục bbox của PDF B (temp/bbox/)
        output_dir: Thư mục output để lưu cặp bbox trùng (compare_temp/)
        
    Returns:
        Thống kê
    """
    print("="*60)
    print("BƯỚC 5: TÌM ID TRÙNG VÀ LƯU VÀO COMPARE_TEMP")
    print("="*60)
    print(f"BBox A (PDF A): {bbox_dir_a}")
    print(f"BBox B (PDF B): {bbox_dir_b}")
    print(f"Output: {output_dir}")
    print()
    
    # Kiểm tra thư mục
    if not bbox_dir_a.exists():
        print(f"ERROR: BBox A directory not found: {bbox_dir_a}")
        return {"total_a": 0, "total_b": 0, "matched": 0}
    
    if not bbox_dir_b.exists():
        print(f"ERROR: BBox B directory not found: {bbox_dir_b}")
        return {"total_a": 0, "total_b": 0, "matched": 0}
    
    # Xóa thư mục output nếu có
    if output_dir.exists():
        print(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)
        print("Cleaned")
    
    # Tạo lại thư mục output với 2 thư mục con
    output_a_dir = output_dir / "pdf_a"
    output_b_dir = output_dir / "pdf_b"
    output_a_dir.mkdir(parents=True, exist_ok=True)
    output_b_dir.mkdir(parents=True, exist_ok=True)
    
    # Lấy danh sách file từ cả 2 thư mục
    files_a = list(bbox_dir_a.glob("*.png"))
    files_b = list(bbox_dir_b.glob("*.png"))
    
    print(f"PDF A: {len(files_a)} bbox files")
    print(f"PDF B: {len(files_b)} bbox files")
    print()
    
    # Tạo dict: ID -> file path cho cả 2 thư mục
    id_map_a = {}
    id_map_b = {}
    
    for file_a in files_a:
        id_text = extract_id_from_filename(file_a.name)
        if id_text:
            if id_text not in id_map_a:
                id_map_a[id_text] = []
            id_map_a[id_text].append(file_a)
    
    for file_b in files_b:
        id_text = extract_id_from_filename(file_b.name)
        if id_text:
            if id_text not in id_map_b:
                id_map_b[id_text] = []
            id_map_b[id_text].append(file_b)
    
    print(f"PDF A: {len(id_map_a)} unique IDs")
    print(f"PDF B: {len(id_map_b)} unique IDs")
    print()
    
    # Tìm ID trùng
    matched_ids = set(id_map_a.keys()) & set(id_map_b.keys())
    
    print(f"Found {len(matched_ids)} matching IDs")
    print()
    
    if not matched_ids:
        print("No matching IDs found")
        return {
            "total_a": len(files_a),
            "total_b": len(files_b),
            "unique_a": len(id_map_a),
            "unique_b": len(id_map_b),
            "matched": 0,
            "matched_ids": []
        }
    
    # Copy các file có ID trùng vào thư mục output
    stats = {
        "total_a": len(files_a),
        "total_b": len(files_b),
        "unique_a": len(id_map_a),
        "unique_b": len(id_map_b),
        "matched": len(matched_ids),
        "matched_ids": [],
        "copied_a": 0,
        "copied_b": 0,
    }
    
    for idx, id_text in enumerate(sorted(matched_ids), 1):
        print(f"[{idx}/{len(matched_ids)}] ID: {id_text}")
        
        # Copy files từ PDF A
        files_from_a = id_map_a[id_text]
        for file_a in files_from_a:
            dest = output_a_dir / file_a.name
            shutil.copy2(file_a, dest)
            print(f"  A: {file_a.name} -> pdf_a/")
            stats["copied_a"] += 1
        
        # Copy files từ PDF B
        files_from_b = id_map_b[id_text]
        for file_b in files_from_b:
            dest = output_b_dir / file_b.name
            shutil.copy2(file_b, dest)
            print(f"  B: {file_b.name} -> pdf_b/")
            stats["copied_b"] += 1
        
        stats["matched_ids"].append(id_text)
    
    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total bbox A: {stats['total_a']} ({stats['unique_a']} unique IDs)")
    print(f"Total bbox B: {stats['total_b']} ({stats['unique_b']} unique IDs)")
    print(f"Matching IDs: {stats['matched']}")
    print(f"Copied from A: {stats['copied_a']} files")
    print(f"Copied from B: {stats['copied_b']} files")
    print(f"\nOutput directories:")
    print(f"  PDF A: {output_a_dir}")
    print(f"  PDF B: {output_b_dir}")
    print("\nDone!")
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Bước 5: Tìm ID trùng và lưu vào compare_temp")
    parser.add_argument("--bbox-a", type=str, default=None, help="Thư mục bbox PDF A (mặc định: app_v2/bbox)")
    parser.add_argument("--bbox-b", type=str, default=None, help="Thư mục bbox PDF B (mặc định: app_v2/temp/bbox)")
    parser.add_argument("--output", type=str, default=None, help="Thư mục output (mặc định: app_v2/compare_temp)")
    
    args = parser.parse_args()
    
    # Thư mục bbox A
    if args.bbox_a:
        bbox_dir_a = Path(args.bbox_a)
    else:
        bbox_dir_a = Path(__file__).resolve().parent / "bbox"
    
    # Thư mục bbox B
    if args.bbox_b:
        bbox_dir_b = Path(args.bbox_b)
    else:
        bbox_dir_b = Path(__file__).resolve().parent / "temp" / "bbox"
    
    # Thư mục output
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).resolve().parent / "compare_temp"
    
    # Chạy step 5
    stats = step5_find_matching_ids(
        bbox_dir_a=bbox_dir_a,
        bbox_dir_b=bbox_dir_b,
        output_dir=output_dir,
    )
    
    return 0 if stats["matched"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

