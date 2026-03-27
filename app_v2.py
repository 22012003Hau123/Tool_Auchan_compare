"""
Streamlit App: So sánh PDF với workflow tự động.
- Chọn file PDF A (mới hoặc từ danh sách có sẵn)
- Chọn file PDF B
- Chạy step1 → step4 → step5 → step6 → step7
- Xuất PDF đã highlight
"""

from __future__ import annotations
import streamlit as st
import sys
from pathlib import Path
import shutil
import io
import base64
from datetime import datetime
import uuid

# Import các hàm từ các step
from step1 import step1_detect_and_save_bbox
from step2 import step2_detect_id_and_save
from step3 import step3_ocr_and_rename, ocr_id_with_openai
from step4 import step4_process_pdf_b
from step5 import step5_find_matching_ids, extract_id_from_filename
from step6 import step6_compare_with_gpt
from step6_gemini import step6_compare_with_gemini
from step7_pymupdf import step7_highlight_pdf_b
import re
import os

# Cấu hình ngôn ngữ
LANGUAGE_OPTIONS = {
    "vi": "Tiếng Việt",
    "en": "English",
    "fr": "Français",
}
DEFAULT_LANGUAGE = "vi"


def get_current_language() -> str:
    return st.session_state.get("language", DEFAULT_LANGUAGE)


def tr(vi: str, en: str, fr: str, **kwargs) -> str:
    texts = {"vi": vi, "en": en, "fr": fr}
    lang = get_current_language()
    text = texts.get(lang, texts[DEFAULT_LANGUAGE])
    if kwargs:
        text = text.format(**kwargs)
    return text


# Cấu hình trang
st.set_page_config(
    page_title="PDF Comparison Tool",
    page_icon="📄",
    layout="wide"
)

# Thư mục gốc
APP_DIR = Path(__file__).resolve().parent
BBOX_DIR = APP_DIR / "bbox"
ID_DIR = APP_DIR / "id"
OUTPUT_DIR = APP_DIR / "output"
MODELS_DIR = APP_DIR / "models"
SESSIONS_DIR = APP_DIR / "sessions"

SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def find_model():
    """Tìm model YOLO."""
    model_path = MODELS_DIR / "bbox_id.pt"
    if not model_path.exists():
        model_path = APP_DIR.parent / "models" / "bbox_id.pt"
    return model_path if model_path.exists() else None


def get_processed_pdfs():
    """Lấy danh sách PDF đã xử lý từ thư mục bbox và id."""
    processed = set()
    
    # Lấy từ bbox
    if BBOX_DIR.exists():
        for subdir in BBOX_DIR.iterdir():
            if subdir.is_dir():
                processed.add(subdir.name)
    
    # Lấy từ id
    if ID_DIR.exists():
        for subdir in ID_DIR.iterdir():
            if subdir.is_dir():
                processed.add(subdir.name)
    
    return sorted(list(processed))


def clear_output_directory():
    """Xóa toàn bộ file trong thư mục output."""
    if not OUTPUT_DIR.exists():
        return 0
    
    deleted_count = 0
    try:
        for file_path in OUTPUT_DIR.iterdir():
            if file_path.is_file():
                file_path.unlink()
                deleted_count += 1
            elif file_path.is_dir():
                shutil.rmtree(file_path)
                deleted_count += 1
    except Exception as e:
        print(f"Warning: Error clearing output directory: {e}")
    
    return deleted_count


def init_session_workspace() -> dict:
    """
    Tạo (hoặc trả về) workspace tạm cho session hiện tại.
    Workspace gồm root + các thư mục con: temp/, compare_temp/, temp_pdf_upload/, pdf_a_temp/, output/.
    """
    workspace = st.session_state.get("session_workspace")
    if workspace:
        return workspace

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:6]
    session_id = f"{timestamp}_{unique_suffix}"

    root_dir = SESSIONS_DIR / session_id
    temp_dir = root_dir / "temp"
    compare_dir = root_dir / "compare_temp"
    uploads_dir = root_dir / "temp_pdf_upload"
    pdf_a_temp_dir = root_dir / "pdf_a_temp"
    session_output_dir = root_dir / "output"

    for path in (temp_dir, compare_dir, uploads_dir, pdf_a_temp_dir, session_output_dir):
        path.mkdir(parents=True, exist_ok=True)

    workspace = {
        "id": session_id,
        "root": root_dir,
        "temp": temp_dir,
        "compare": compare_dir,
        "uploads": uploads_dir,
        "pdf_a_temp": pdf_a_temp_dir,
        "output": session_output_dir,
    }
    st.session_state["session_workspace"] = workspace
    return workspace


def cleanup_session_workspace():
    """Xóa workspace tạm của session hiện tại (nếu có)."""
    workspace = st.session_state.get("session_workspace")
    if not workspace:
        return

    root_dir = workspace.get("root")
    if root_dir and root_dir.exists():
        try:
            shutil.rmtree(root_dir)
            print(f"✅ Đã xóa session workspace: {root_dir}")
        except Exception as e:
            print(f"Warning: Cannot remove session workspace {root_dir}: {e}")

    st.session_state.pop("session_workspace", None)


def cleanup_old_sessions(max_age_hours: int = 168, min_file_age_hours: int = 24):
    """
    Xóa các session cũ không còn active (orphaned sessions).
    Chỉ xóa session nếu:
    1. Session cũ hơn max_age_hours (mặc định: 7 ngày = 168 giờ)
    2. VÀ file mới nhất trong session cũ hơn min_file_age_hours (mặc định: 24 giờ)
    
    Điều này đảm bảo không xóa session đang được sử dụng bởi người dùng khác.
    
    Args:
        max_age_hours: Xóa session cũ hơn số giờ này (mặc định: 168 giờ = 7 ngày)
        min_file_age_hours: File mới nhất trong session phải cũ hơn số giờ này (mặc định: 24 giờ)
    """
    if not SESSIONS_DIR.exists():
        return
    
    current_time = datetime.now()
    deleted_count = 0
    current_workspace = st.session_state.get("session_workspace")
    current_session_id = current_workspace.get("id") if current_workspace else None
    
    for session_dir in SESSIONS_DIR.iterdir():
        if not session_dir.is_dir():
            continue
        
        # Bỏ qua session hiện tại của user này
        if current_session_id and session_dir.name == current_session_id:
            continue
        
        # Parse timestamp từ tên thư mục: YYYYMMDD_HHMMSS_suffix
        try:
            timestamp_str = session_dir.name.split("_")[:2]  # ["YYYYMMDD", "HHMMSS"]
            if len(timestamp_str) == 2:
                session_time = datetime.strptime("_".join(timestamp_str), "%Y%m%d_%H%M%S")
                age_hours = (current_time - session_time).total_seconds() / 3600
                
                # Chỉ xóa nếu session cũ hơn max_age_hours
                if age_hours > max_age_hours:
                    # Kiểm tra file mới nhất trong session để đảm bảo không có activity gần đây
                    latest_file_time = None
                    try:
                        for file_path in session_dir.rglob("*"):
                            if file_path.is_file():
                                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                                if latest_file_time is None or file_mtime > latest_file_time:
                                    latest_file_time = file_mtime
                    except Exception:
                        # Nếu không đọc được file, bỏ qua session này
                        continue
                    
                    # Chỉ xóa nếu file mới nhất cũ hơn min_file_age_hours
                    if latest_file_time:
                        file_age_hours = (current_time - latest_file_time).total_seconds() / 3600
                        if file_age_hours < min_file_age_hours:
                            # Session có activity gần đây, không xóa (có thể đang được dùng)
                            continue
                    
                    # Xóa session cũ và không có activity gần đây
                    try:
                        shutil.rmtree(session_dir)
                        deleted_count += 1
                        print(f"🗑️ Đã xóa session cũ: {session_dir.name} (tuổi: {age_hours:.1f} giờ)")
                    except Exception as e:
                        print(f"Warning: Cannot remove old session {session_dir}: {e}")
        except (ValueError, IndexError):
            # Nếu không parse được timestamp, bỏ qua
            continue
    
    if deleted_count > 0:
        print(f"✅ Đã xóa {deleted_count} session cũ")
    
    return deleted_count


def fix_bbox_ids_with_openai(bbox_b_dir: Path, bbox_a_dir: Path, model: str = "gpt-4o-mini", matched_ids: set = None) -> dict:
    """
    Dùng OpenAI OCR để đọc ID từ ảnh bbox trong temp/bbox và đổi tên nếu tìm thấy ID tương ứng từ PDF A.
    
    Args:
        bbox_b_dir: Thư mục bbox của PDF B (temp/bbox/...)
        bbox_a_dir: Thư mục bbox của PDF A (bbox/...)
        model: OpenAI model để OCR
        matched_ids: Set các ID đã được match (để bỏ qua các file đã match)
        
    Returns:
        Dict với thống kê: {"ocr_count": int, "renamed_count": int, "matched_ids": list}
    """
    if not bbox_b_dir.exists():
        return {"ocr_count": 0, "renamed_count": 0, "matched_ids": []}
    
    # Lấy danh sách ID từ PDF A
    pdf_a_ids = set()
    if bbox_a_dir.exists():
        for file_a in bbox_a_dir.glob("*.png"):
            id_text = extract_id_from_filename(file_a.name)
            if id_text:
                pdf_a_ids.add(id_text)
    
    if not pdf_a_ids:
        return {"ocr_count": 0, "renamed_count": 0, "matched_ids": []}
    
    # Lấy danh sách ID đã match (nếu có)
    if matched_ids is None:
        matched_ids = set()
    
    # OCR các ảnh bbox trong PDF B (chỉ các file chưa match)
    bbox_files = list(bbox_b_dir.glob("*.png"))
    ocr_count = 0
    renamed_count = 0
    new_matched_ids = []
    
    for bbox_file in bbox_files:
        # Kiểm tra xem file này đã được match chưa (dựa vào tên file)
        file_id = extract_id_from_filename(bbox_file.name)
        if file_id and file_id in matched_ids:
            # File này đã được match, bỏ qua
            continue
        
        try:
            # OCR bằng OpenAI
            ocr_result = ocr_id_with_openai(bbox_file, model=model)
            ocr_count += 1
            
            if not ocr_result:
                continue
            
            # Tìm ID trong kết quả OCR (tìm số có 4-5 chữ số)
            id_match = re.search(r'\b(\d{4,5})\b', ocr_result)
            if not id_match:
                # Thử tìm số ngắn hơn (3-6 chữ số)
                id_match = re.search(r'\b(\d{3,6})\b', ocr_result)
            
            if id_match:
                found_id = id_match.group(1)
                
                # Kiểm tra xem ID này có trong PDF A không
                if found_id in pdf_a_ids:
                    # Đổi tên file theo format: page{page}_bbox{idx}_{timestamp}.png -> page{page}_{id}.png
                    # Giữ nguyên page number từ tên file cũ
                    page_match = re.search(r'page(\d+)_', bbox_file.name)
                    if page_match:
                        page_num = page_match.group(1)
                        new_name = f"page{page_num}_{found_id}.png"
                        new_path = bbox_file.parent / new_name
                        
                        # Chỉ đổi tên nếu tên file mới khác tên file cũ
                        if bbox_file.name != new_name:
                            # Kiểm tra xem file mới đã tồn tại chưa
                            if new_path.exists():
                                # Nếu đã tồn tại, xóa file cũ (có thể là duplicate)
                                bbox_file.unlink()
                            else:
                                bbox_file.rename(new_path)
                                renamed_count += 1
                                new_matched_ids.append(found_id)
                                print(f"  ✅ Renamed: {bbox_file.name} -> {new_name} (ID: {found_id})")
        except Exception as e:
            print(f"  ⚠️  Error processing {bbox_file.name}: {e}")
            continue
    
    return {
        "ocr_count": ocr_count,
        "renamed_count": renamed_count,
        "matched_ids": new_matched_ids
    }

# Main app
def main():
    # Cleanup các session cũ khi khởi động (chỉ chạy 1 lần mỗi session)
    # Chỉ xóa session cũ hơn 7 ngày VÀ không có file activity trong 24 giờ gần đây
    # Điều này đảm bảo không xóa session đang được sử dụng bởi người dùng khác
    if "sessions_cleaned" not in st.session_state:
        cleanup_old_sessions(max_age_hours=168, min_file_age_hours=24)  # 7 ngày, file activity > 24h
        st.session_state.sessions_cleaned = True
    
    if "language" not in st.session_state:
        st.session_state.language = DEFAULT_LANGUAGE

    lang_select = st.sidebar.selectbox(
        tr("Language / Ngôn ngữ / Langue", "Language / Ngôn ngữ / Langue", "Language / Ngôn ngữ / Langue"),
        options=list(LANGUAGE_OPTIONS.keys()),
        format_func=lambda code: LANGUAGE_OPTIONS[code],
        index=list(LANGUAGE_OPTIONS.keys()).index(st.session_state.language),
    )
    if lang_select != st.session_state.language:
        st.session_state.language = lang_select
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()

    st.title(tr("📄 Công cụ so sánh PDF", "📄 PDF Comparison Tool", "📄 Outil de comparaison PDF"))
    st.markdown("---")
    
    # Tìm model
    model_path = find_model()
    if not model_path:
        st.error(tr("❌ Không tìm thấy model YOLO. Vui lòng đặt model vào thư mục models/",
                    "❌ YOLO model not found. Please place the model inside models/",
                    "❌ Modèle YOLO introuvable. Merci de le placer dans le dossier models/"))
        return
    
    # Session state
    if "pdf_a_name" not in st.session_state:
        st.session_state.pdf_a_name = None
    if "pdf_b_path" not in st.session_state:
        st.session_state.pdf_b_path = None
    if "output_pdf_path" not in st.session_state:
        st.session_state.output_pdf_path = None
    if "output_cleared_for" not in st.session_state:
        st.session_state.output_cleared_for = None
    if "data_cleared_for" not in st.session_state:
        st.session_state.data_cleared_for = None
    if "pdf_a_exists_flag" not in st.session_state:
        st.session_state.pdf_a_exists_flag = False
    # Session state cho việc hiển thị PDF
    if "show_pdf_states" not in st.session_state:
        st.session_state.show_pdf_states = {}
    
    # BƯỚC 1: Chọn PDF A
    st.header(tr("📋 Bước 1: Chọn PDF A", "📋 Step 1: Select PDF A", "📋 Étape 1 : Sélectionner PDF A"))
    
    # Tùy chọn: Upload mới hoặc chọn từ danh sách
    option = st.radio(
        tr("Chọn phương thức:", "Choose a method:", "Choisir une méthode :"),
        options=["upload", "existing"],
        format_func=lambda value: tr(
            "📤 Upload file PDF mới" if value == "upload" else "📁 Chọn từ danh sách có sẵn",
            "📤 Upload new PDF" if value == "upload" else "📁 Choose from existing list",
            "📤 Importer un nouveau PDF" if value == "upload" else "📁 Choisir dans la liste existante",
        ),
        horizontal=True
    )
    
    # Lưu option hiện tại để detect thay đổi
    if "last_option" not in st.session_state:
        st.session_state.last_option = option
    elif st.session_state.last_option != option:
        # User chuyển giữa upload và existing, cleanup workspace cũ
        cleanup_session_workspace()
        st.session_state.last_option = option
    
    pdf_a_path = None
    pdf_a_name = None
    
    if option == "upload":
        uploaded_file = st.file_uploader(tr("Chọn file PDF A", "Choose PDF A", "Choisir le PDF A"), type=["pdf"])
        if not uploaded_file:
            st.session_state.pdf_a_exists_flag = False
            # Cleanup session workspace khi xóa file upload
            current_pdf_a_name = st.session_state.get("pdf_a_name")
            if current_pdf_a_name:
                # Reset PDF A name và cleanup workspace
                st.session_state.pdf_a_name = None
                cleanup_session_workspace()
        if uploaded_file:
            pdf_a_candidate = Path(uploaded_file.name)
            pdf_a_name_candidate = pdf_a_candidate.stem
            
            # Cleanup session workspace cũ nếu PDF A name thay đổi
            current_pdf_a_name = st.session_state.get("pdf_a_name")
            if current_pdf_a_name and current_pdf_a_name != pdf_a_name_candidate:
                cleanup_session_workspace()
                # Reset các flags liên quan
                st.session_state.output_cleared_for = None
                st.session_state.data_cleared_for = None
            
            # Khởi tạo session workspace để lưu file PDF A tạm
            workspace = init_session_workspace()
            
            bbox_a_dir = BBOX_DIR / pdf_a_name_candidate
            id_a_dir = ID_DIR / pdf_a_name_candidate
            # Lưu vào session workspace thay vì pdf_a_library
            temp_pdf_path = workspace["pdf_a_temp"] / uploaded_file.name

            if bbox_a_dir.exists() and id_a_dir.exists():
                st.session_state.pdf_a_exists_flag = True
                pdf_a_name = pdf_a_name_candidate
                st.session_state.pdf_a_name = pdf_a_name
                # Không cần file PDF khi đã có bbox + ID
                pdf_a_path = None
                # Không lưu file upload vào session khi đã có dữ liệu
                st.info(
                    tr(
                        "📁 PDF này đã có dữ liệu bbox + ID. Vui lòng dùng mục 'Chọn từ danh sách có sẵn' hoặc tiếp tục với dữ liệu hiện có.",
                        "📁 This PDF already has bbox + ID data. Please use 'Choose from existing list' or continue with the existing data.",
                        "📁 Ce PDF possède déjà des bbox + ID. Utilisez « Choisir dans la liste existante » ou continuez avec les données disponibles.",
                    )
                )
            else:
                st.session_state.pdf_a_exists_flag = False
                # Lưu file vào session workspace (sẽ tự xóa khi session kết thúc)
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                pdf_a_path = temp_pdf_path
                pdf_a_name = temp_pdf_path.stem
                
                # Xóa output nếu chưa xóa cho file này
                if st.session_state.output_cleared_for != pdf_a_name:
                    deleted_count = clear_output_directory()
                    if deleted_count > 0:
                        st.info(tr("🗑️ Đã xóa {count} file trong thư mục output",
                                   "🗑️ Deleted {count} file(s) in output directory",
                                   "🗑️ {count} fichier(s) supprimé(s) du dossier output",
                                   count=deleted_count))
                    st.session_state.output_cleared_for = pdf_a_name
                
                # Chỉ xóa dữ liệu cũ một lần duy nhất khi upload file mới
                # Tránh xóa lại khi Streamlit rerun sau khi đã chạy step1
                if st.session_state.data_cleared_for != pdf_a_name:
                    if bbox_a_dir.exists() or id_a_dir.exists():
                        # Tự động xóa dữ liệu cũ
                        deleted_items = []
                        try:
                            if bbox_a_dir.exists():
                                shutil.rmtree(bbox_a_dir)
                                deleted_items.append("bbox")
                            if id_a_dir.exists():
                                shutil.rmtree(id_a_dir)
                                deleted_items.append("ID")
                            
                            if deleted_items:
                                st.warning(tr("⚠️ Đã xóa dữ liệu cũ: {items}",
                                              "⚠️ Removed old data: {items}",
                                              "⚠️ Anciennes données supprimées : {items}",
                                              items=", ".join(deleted_items)))
                                st.info(tr("📄 File mới sẽ được xử lý từ đầu: {filename}",
                                           "📄 The new file will be processed from scratch: {filename}",
                                           "📄 Le nouveau fichier sera traité depuis le début : {filename}",
                                           filename=uploaded_file.name))
                            st.session_state.data_cleared_for = pdf_a_name
                        except Exception as e:
                            st.error(f"❌ Lỗi khi xóa dữ liệu cũ: {e}")
                    else:
                        st.success(tr("✅ Đã tải lên: {filename}",
                                      "✅ Uploaded: {filename}",
                                      "✅ Téléversé : {filename}",
                                      filename=uploaded_file.name))
                        st.session_state.data_cleared_for = pdf_a_name
    
    else:
        st.session_state.pdf_a_exists_flag = False
        # Chọn từ danh sách có sẵn
        processed_pdfs = get_processed_pdfs()
        if not processed_pdfs:
            st.warning(tr("⚠️ Chưa có PDF nào đã xử lý. Vui lòng upload file mới.",
                          "⚠️ No processed PDFs yet. Please upload a new file.",
                          "⚠️ Aucun PDF traité. Veuillez importer un nouveau fichier."))
        else:
            selected_pdf = st.selectbox(
                tr("Chọn PDF đã xử lý:", "Select a processed PDF:", "Choisir un PDF traité :"),
                processed_pdfs,
                index=0 if processed_pdfs else None
            )
            if selected_pdf:
                pdf_a_name = selected_pdf
                
                # Cleanup session workspace cũ nếu PDF A name thay đổi
                current_pdf_a_name = st.session_state.get("pdf_a_name")
                if current_pdf_a_name and current_pdf_a_name != pdf_a_name:
                    cleanup_session_workspace()
                    # Reset các flags liên quan
                    st.session_state.output_cleared_for = None
                    st.session_state.data_cleared_for = None
                
                st.session_state.pdf_a_name = pdf_a_name  # Lưu vào session state để dùng cho các bước tiếp theo
                
                # Xóa output nếu chưa xóa cho file này
                if st.session_state.output_cleared_for != pdf_a_name:
                    deleted_count = clear_output_directory()
                    if deleted_count > 0:
                        st.info(tr("🗑️ Đã xóa {count} file trong thư mục output",
                                   "🗑️ Deleted {count} file(s) in output directory",
                                   "🗑️ {count} fichier(s) supprimé(s) du dossier output",
                                   count=deleted_count))
                    st.session_state.output_cleared_for = pdf_a_name
                
                st.success(tr("✅ Đã chọn: {name}", "✅ Selected: {name}", "✅ Sélectionné : {name}", name=selected_pdf))
                
                # Khi chọn từ danh sách có sẵn, không cần file PDF gốc vì đã có bbox + ID
                # Chỉ cần pdf_a_name để xử lý các bước tiếp theo
                pdf_a_path = None
    
    # Nếu có file mới, chạy step1
    if pdf_a_path and option == "upload" and not st.session_state.get("pdf_a_exists_flag"):
        if st.button(tr("🚀 Chạy Step 1 (Detect Bbox)", "🚀 Run Step 1 (Detect Bbox)", "🚀 Lancer l'étape 1 (Detect Bbox)"), type="primary"):
            with st.spinner(tr("Đang xử lý Step 1...", "Processing Step 1...", "Traitement de l'étape 1...")):
                try:
                    bbox_output_dir = BBOX_DIR / pdf_a_name
                    total_bbox = step1_detect_and_save_bbox(
                        pdf_path=pdf_a_path,
                        model_path=model_path,
                        output_dir=BBOX_DIR,  # step1 sẽ tự tạo thư mục con
                        conf=0.4,
                        dpi=300,
                        imgsz=640,
                        device=None,
                        max_det=300,
                        save_overview=False,
                        merge_iou=0.1,
                    )
                    
                    if total_bbox > 0:
                        st.success(tr("✅ Step 1 hoàn thành! Đã detect {count} bbox",
                                      "✅ Step 1 completed! Detected {count} bbox",
                                      "✅ Étape 1 terminée ! {count} bbox détectés",
                                      count=total_bbox))
                        st.session_state.pdf_a_name = pdf_a_name
                    else:
                        st.error(tr("❌ Step 1 thất bại: Không detect được bbox nào",
                                    "❌ Step 1 failed: No bbox detected",
                                    "❌ Étape 1 échouée : aucun bbox détecté"))
                except Exception as e:
                    st.error(tr("❌ Lỗi Step 1: {error}",
                                "❌ Step 1 error: {error}",
                                "❌ Erreur Étape 1 : {error}",
                                error=e))
                    st.exception(e)
    
    # Nếu đã có PDF A (từ danh sách hoặc đã chạy step1)
    if pdf_a_name:
        st.session_state.pdf_a_name = pdf_a_name
        st.info(tr("📄 PDF A: **{name}**", "📄 PDF A: **{name}**", "📄 PDF A : **{name}**", name=pdf_a_name))
        
        # Kiểm tra xem đã có bbox và id chưa
        bbox_a_dir = BBOX_DIR / pdf_a_name
        id_a_dir = ID_DIR / pdf_a_name
        
        bbox_count = 0
        if bbox_a_dir.exists():
            bbox_count = len(list(bbox_a_dir.glob("*.png")))
            st.success(tr("✅ Bbox: {count} files", "✅ Bbox: {count} files", "✅ Bbox : {count} fichiers", count=bbox_count))
        else:
            st.warning(tr("⚠️ Chưa có bbox. Vui lòng chạy Step 1.",
                          "⚠️ No bbox yet. Please run Step 1.",
                          "⚠️ Aucun bbox. Veuillez exécuter l'étape 1."))
        
        if id_a_dir.exists():
            id_count = len(list(id_a_dir.glob("*.png")))
            st.success(tr("✅ ID: {count} files", "✅ ID: {count} files", "✅ ID : {count} fichiers", count=id_count))
        else:
            st.warning(tr("⚠️ Chưa có ID. Cần chạy Step 2 và Step 3.",
                          "⚠️ No IDs yet. Run Step 2 & 3.",
                          "⚠️ Aucun ID. Exécutez les étapes 2 et 3."))
        
        # Nút xóa file (thư mục bbox và id)
        if (bbox_a_dir.exists() or id_a_dir.exists()):
            st.markdown("---")
            
            # Session state cho xác nhận xóa
            confirm_key = f"show_confirm_delete_{pdf_a_name}"
            if confirm_key not in st.session_state:
                st.session_state[confirm_key] = False
            
            # Nút xóa
            if st.button(tr("🗑️ Xóa file", "🗑️ Delete files", "🗑️ Supprimer les fichiers"), type="secondary", key="delete_pdf_a"):
                st.session_state[confirm_key] = True
                st.rerun()
            
            # Hiển thị cảnh báo và nút xác nhận nếu đã nhấn xóa
            if st.session_state[confirm_key]:
                st.warning(tr("⚠️ **CẢNH BÁO:** Xóa file sẽ xóa toàn bộ dữ liệu đã xử lý của PDF A này:",
                              "⚠️ **WARNING:** Deleting will remove all processed data for this PDF A:",
                              "⚠️ **AVERTISSEMENT :** La suppression retirera toutes les données traitées de ce PDF A :"))
                
                delete_info = []
                if bbox_a_dir.exists():
                    bbox_count = len(list(bbox_a_dir.glob("*.png")))
                    delete_info.append(tr("  • Bbox: {count} files",
                                          "  • Bbox: {count} files",
                                          "  • Bbox : {count} fichiers",
                                          count=bbox_count))
                if id_a_dir.exists():
                    id_count = len(list(id_a_dir.glob("*.png")))
                    delete_info.append(tr("  • ID: {count} files",
                                          "  • ID: {count} files",
                                          "  • ID : {count} fichiers",
                                          count=id_count))
                
                for info in delete_info:
                    st.text(info)
                
                st.error(tr("**Hành động này không thể hoàn tác!**",
                            "**This action cannot be undone!**",
                            "**Cette action est irréversible !**"))
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(tr(" Xác nhận xóa", " Confirm deletion", " Confirmer la suppression"), type="primary", use_container_width=True, key="confirm_delete_btn"):
                        try:
                            deleted_items = []
                            if bbox_a_dir.exists():
                                shutil.rmtree(bbox_a_dir)
                                deleted_items.append("bbox")
                            if id_a_dir.exists():
                                shutil.rmtree(id_a_dir)
                                deleted_items.append("ID")
                            
                            st.session_state[confirm_key] = False
                            if deleted_items:
                                st.success(tr("✅ Đã xóa: {items}",
                                              "✅ Deleted: {items}",
                                              "✅ Supprimé : {items}",
                                              items=", ".join(deleted_items)))
                                st.rerun()
                            else:
                                st.info(tr("ℹ️ Không có gì để xóa",
                                           "ℹ️ Nothing to delete",
                                           "ℹ️ Rien à supprimer"))
                                st.rerun()
                        except Exception as e:
                            st.error(tr("❌ Lỗi khi xóa: {error}",
                                        "❌ Error deleting: {error}",
                                        "❌ Erreur lors de la suppression : {error}",
                                        error=e))
                            st.session_state[confirm_key] = False
                
                with col2:
                    if st.button(tr("Hủy", "Cancel", "Annuler"), use_container_width=True, key="cancel_delete_btn"):
                        st.session_state[confirm_key] = False
                        st.rerun()
        
        # Tự động chạy step2 và step3 nếu có bbox
        if bbox_a_dir.exists() and bbox_count > 0:
            # Chỉ hiển thị nút Step 2 & 3 nếu chưa có ID nào được tạo
            if id_a_dir.exists():
                id_count = len(list(id_a_dir.glob("*.png")))
            else:
                id_count = 0

            if id_count == 0 and st.button(tr("🚀 Chạy Step 2 & 3 (Detect ID và OCR)",
                                             "🚀 Run Step 2 & 3 (Detect ID & OCR)",
                                             "🚀 Lancer Étape 2 & 3 (Detect ID & OCR)"), type="primary"):
                with st.spinner(tr("Đang xử lý Step 2 & 3...", "Processing Step 2 & 3...", "Traitement des étapes 2 & 3...")):
                    try:
                        # Tìm model ID
                        id_model_path = MODELS_DIR / "id_plus.pt"
                        if not id_model_path.exists():
                            id_model_path = model_path
                        
                        # Step 2: Detect ID
                        step2_results = step2_detect_id_and_save(
                            bbox_dir=bbox_a_dir,
                            id_dir=ID_DIR,
                            model_path=id_model_path,
                            id_conf=0.1,
                            min_confidence=0.0,
                            show_progress=False,
                            save_low_conf=False,
                            save_report=False,
                            save_overview=False,
                        )
                        
                        if step2_results["saved"] > 0:
                            st.success(tr("✅ Step 2: Đã detect {count} ID",
                                          "✅ Step 2: Detected {count} IDs",
                                          "✅ Étape 2 : {count} ID détectés",
                                          count=step2_results["saved"]))
                            
                            # Step 3: OCR và rename
                            step3_results = step3_ocr_and_rename(
                                id_dir=id_a_dir,
                                bbox_dir=bbox_a_dir,
                                ocr_lang="en",
                                use_gpu=False,
                                dry_run=False,
                                use_openai_fallback=True,
                                openai_model="gpt-4.1",
                            )
                            
                            if step3_results["success"] > 0:
                                st.success(tr("✅ Step 3: Đã OCR và rename {count} file",
                                              "✅ Step 3: OCR + renamed {count} files",
                                              "✅ Étape 3 : OCR + renommé {count} fichiers",
                                              count=step3_results["success"]))
                                st.rerun()  # Reload để cập nhật UI
                            else:
                                st.warning(tr("⚠️ Step 3: Không OCR được file nào",
                                              "⚠️ Step 3: No files OCR'd",
                                              "⚠️ Étape 3 : Aucun fichier OCR"))
                        else:
                            st.warning(tr("⚠️ Step 2: Không detect được ID nào",
                                          "⚠️ Step 2: No IDs detected",
                                          "⚠️ Étape 2 : Aucun ID détecté"))
                    except Exception as e:
                        st.error(tr("❌ Lỗi Step 2 & 3: {error}",
                                    "❌ Step 2 & 3 error: {error}",
                                    "❌ Erreur Étapes 2 & 3 : {error}",
                                    error=e))
                        st.exception(e)
    
    st.markdown("---")
    
    # BƯỚC 2: Chọn PDF B và xử lý
    if st.session_state.pdf_a_name:
        st.header(tr("📋 Bước 2: Chọn PDF B và Xử lý",
                     "📋 Step 2: Select PDF B and Process",
                     "📋 Étape 2 : Sélectionner PDF B et Traiter"))
        
        uploaded_files_b = st.file_uploader(
            tr("Chọn file PDF B (có thể chọn nhiều file)",
               "Select PDF B files (multiple allowed)",
               "Sélectionner les PDF B (multiples autorisés)"), 
            type=["pdf"], 
            key="pdf_b",
            accept_multiple_files=True
        )
        
        if uploaded_files_b:
            st.success(tr("✅ Đã chọn {count} file PDF B:",
                          "✅ Selected {count} PDF B file(s):",
                          "✅ {count} fichier(s) PDF B sélectionné(s) :",
                          count=len(uploaded_files_b)))
            for i, uploaded_file_b in enumerate(uploaded_files_b, 1):
                st.text(f"  {i}. {uploaded_file_b.name}")
            
            # Chọn AI model cho Step 6
            st.subheader(tr("🤖 Chọn AI Model cho So sánh ",
                            "🤖 Choose AI Model for Comparison ",
                            "🤖 Choisir le modèle IA pour la comparaison "))
            ai_model = st.radio(
                tr("Chọn model AI:", "Choose AI model:", "Choisir le modèle IA :"),
                ["OpenAI GPT", "Google Gemini"],
                horizontal=True,
                help=tr("OpenAI GPT: Sử dụng GPT-4.1\nGoogle Gemini: Sử dụng Gemini 2.5 Flash (nhanh hơn, miễn phí hơn)",
                        "OpenAI GPT: Uses GPT-4.1\nGoogle Gemini: Uses Gemini 2.5 Flash (faster, cheaper)",
                        "OpenAI GPT : utilise GPT-4.1\nGoogle Gemini : utilise Gemini 2.5 Flash (plus rapide, moins cher)")
            )
            
            # Nút chạy toàn bộ workflow cho tất cả file
            if st.button(tr("🚀 Chạy Toàn Bộ Workflow cho {count} file PDF B (Step 4 → 5 → 6 → 7)",
                            "🚀 Run full workflow for {count} PDF B file(s) (Step 4 → 5 → 6 → 7)",
                            "🚀 Lancer tout le workflow pour {count} PDF B (Étapes 4 → 5 → 6 → 7)",
                            count=len(uploaded_files_b)), type="primary"):
                # Luôn bắt đầu với workspace mới để tránh xung đột với session khác
                cleanup_session_workspace()
                workspace = init_session_workspace()
                uploads_dir = workspace["uploads"]
                temp_dir = workspace["temp"]
                compare_root = workspace["compare"]

                pdf_b_paths = []
                for uploaded_file_b in uploaded_files_b:
                    temp_pdf_b_path = uploads_dir / uploaded_file_b.name
                    with open(temp_pdf_b_path, "wb") as f:
                        f.write(uploaded_file_b.getbuffer())
                    pdf_b_paths.append(temp_pdf_b_path)

                all_results = []
                bbox_a_dir = BBOX_DIR / st.session_state.pdf_a_name
                
                try:
                    # Lặp qua từng file PDF B
                    for file_idx, pdf_b_path in enumerate(pdf_b_paths, 1):
                        pdf_b_name = pdf_b_path.stem
                        st.markdown("---")
                        st.subheader(tr("📄 Xử lý file {idx}/{total}: {name}",
                                        "📄 Processing file {idx}/{total}: {name}",
                                        "📄 Traitement du fichier {idx}/{total} : {name}",
                                        idx=file_idx, total=len(pdf_b_paths), name=pdf_b_path.name))
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            # Step 4: Xử lý PDF B
                            status_text.text(tr("🔄 [{idx}/{total}] Đang chạy Step 4: Xử lý PDF B...",
                                                "🔄 [{idx}/{total}] Running Step 4: Processing PDF B...",
                                                "🔄 [{idx}/{total}] Étape 4 : Traitement du PDF B...",
                                                idx=file_idx, total=len(pdf_b_paths)))
                            progress_bar.progress(10)
                            
                            step4_results = step4_process_pdf_b(
                                pdf_b_path=pdf_b_path,
                                model_path=model_path,
                                temp_dir=temp_dir,
                                conf=0.4,
                                id_conf=0.1,
                                ocr_lang="en",
                                use_gpu=False,
                                imgsz=640,
                                device=None,
                                max_det=300,
                                merge_iou=0.1,
                                use_openai_fallback=True,
                                openai_model="gpt-4.1",
                            )
                            
                            if step4_results["step3"] == 0:
                                st.error(tr("❌ [{idx}/{total}] Step 4 thất bại: Không xử lý được PDF B",
                                            "❌ [{idx}/{total}] Step 4 failed: Cannot process PDF B",
                                            "❌ [{idx}/{total}] Étape 4 échouée : impossible de traiter le PDF B",
                                            idx=file_idx, total=len(pdf_b_paths)))
                                all_results.append({
                                    "file_name": pdf_b_path.name,
                                    "status": "failed",
                                    "error": "Step 4 failed"
                                })
                                continue
                            
                            st.success(tr("✅ [{idx}/{total}] Step 4: {bbox} bbox, {ids} ID, {renamed} renamed",
                                          "✅ [{idx}/{total}] Step 4: {bbox} bbox, {ids} ID, {renamed} renamed",
                                          "✅ [{idx}/{total}] Étape 4 : {bbox} bbox, {ids} ID, {renamed} renommés",
                                          idx=file_idx, total=len(pdf_b_paths),
                                          bbox=step4_results["step1"],
                                          ids=step4_results["step2"],
                                          renamed=step4_results["step3"]))
                            progress_bar.progress(30)
                            
                            # Step 5: Tìm ID trùng
                            status_text.text(tr("🔄 [{idx}/{total}] Đang chạy Step 5: Tìm ID trùng...",
                                                "🔄 [{idx}/{total}] Running Step 5: Matching IDs...",
                                                "🔄 [{idx}/{total}] Étape 5 : Recherche des ID correspondants...",
                                                idx=file_idx, total=len(pdf_b_paths)))
                            progress_bar.progress(40)
                            
                            bbox_b_dir = temp_dir / "bbox" / pdf_b_name
                            current_compare_dir = compare_root / pdf_b_name
                            
                            step5_results = step5_find_matching_ids(
                                bbox_dir_a=bbox_a_dir,
                                bbox_dir_b=bbox_b_dir,
                                output_dir=current_compare_dir,
                            )
                            
                            # Kiểm tra xem có bbox nào chưa được match không
                            total_bbox_b = len(list(bbox_b_dir.glob("*.png"))) if bbox_b_dir.exists() else 0
                            unmatched_count = total_bbox_b - step5_results.get("copied_b", 0)
                            
                            # Lấy danh sách ID đã match để bỏ qua khi OCR
                            matched_ids_set = set(step5_results.get("matched_ids", []))
                            
                            # Nếu không có ID trùng hoặc có bbox chưa được match, thử sửa ID
                            if step5_results["matched"] == 0 or unmatched_count > 0:
                                if step5_results["matched"] == 0:
                                    st.warning(tr("⚠️ [{idx}/{total}] Step 5: Không tìm thấy ID trùng nào",
                                                  "⚠️ [{idx}/{total}] Step 5: No matching IDs found",
                                                  "⚠️ [{idx}/{total}] Étape 5 : Aucun ID correspondant",
                                                  idx=file_idx, total=len(pdf_b_paths)))
                                else:
                                    st.info(tr("ℹ️ [{idx}/{total}] Step 5: Tìm thấy {matched} ID trùng, nhưng còn {unmatched} bbox chưa match",
                                               "ℹ️ [{idx}/{total}] Step 5: Found {matched} matches, {unmatched} unmatched bbox",
                                               "ℹ️ [{idx}/{total}] Étape 5 : {matched} correspondances, {unmatched} bbox sans correspondance",
                                               idx=file_idx, total=len(pdf_b_paths),
                                               matched=step5_results["matched"], unmatched=unmatched_count))
                                
                                # Thử sửa ID bằng OpenAI OCR cho các bbox chưa match
                                status_text.text(tr("🔄 [{idx}/{total}] Đang sửa ID bằng OpenAI OCR cho {count} bbox chưa match...",
                                                    "🔄 [{idx}/{total}] Fixing IDs via OpenAI OCR for {count} unmatched bbox...",
                                                    "🔄 [{idx}/{total}] Correction des ID via OpenAI OCR pour {count} bbox sans correspondance...",
                                                    idx=file_idx, total=len(pdf_b_paths), count=unmatched_count))
                                fix_results = fix_bbox_ids_with_openai(
                                    bbox_b_dir=bbox_b_dir,
                                    bbox_a_dir=bbox_a_dir,
                                    model="gpt-4o-mini",
                                    matched_ids=matched_ids_set
                                )
                                
                                if fix_results["renamed_count"] > 0:
                                    st.info(tr("ℹ️ [{idx}/{total}] Đã sửa {count} file bbox bằng OpenAI OCR",
                                               "ℹ️ [{idx}/{total}] Fixed {count} bbox files via OpenAI OCR",
                                               "ℹ️ [{idx}/{total}] {count} bbox corrigés via OpenAI OCR",
                                               idx=file_idx, total=len(pdf_b_paths), count=fix_results["renamed_count"]))
                                    st.info(tr("   Tìm thấy {count} ID: {ids}",
                                               "   Found {count} IDs: {ids}",
                                               "   {count} ID trouvés : {ids}",
                                               count=len(fix_results["matched_ids"]),
                                               ids=", ".join(fix_results["matched_ids"][:10]) + ("..." if len(fix_results["matched_ids"]) > 10 else "")))
                                    
                                    # Chạy lại Step 5 sau khi sửa ID
                                    status_text.text(tr("🔄 [{idx}/{total}] Đang chạy lại Step 5 sau khi sửa ID...",
                                                        "🔄 [{idx}/{total}] Rerunning Step 5 after fixing IDs...",
                                                        "🔄 [{idx}/{total}] Relance de l'étape 5 après correction des ID...",
                                                        idx=file_idx, total=len(pdf_b_paths)))
                                    step5_results = step5_find_matching_ids(
                                        bbox_dir_a=bbox_a_dir,
                                        bbox_dir_b=bbox_b_dir,
                                        output_dir=current_compare_dir,
                                    )
                                    
                                    if step5_results["matched"] == 0:
                                        st.warning(tr("⚠️ [{idx}/{total}] Step 5: Vẫn không tìm thấy ID trùng sau khi sửa",
                                                      "⚠️ [{idx}/{total}] Step 5: Still no matches after fixing",
                                                      "⚠️ [{idx}/{total}] Étape 5 : toujours aucune correspondance après correction",
                                                      idx=file_idx, total=len(pdf_b_paths)))
                                        all_results.append({
                                            "file_name": pdf_b_path.name,
                                            "status": "no_match",
                                            "matched": 0,
                                            "fix_attempt": fix_results
                                        })
                                        continue
                                    else:
                                        st.success(tr("✅ [{idx}/{total}] Step 5 (sau khi sửa): Tìm thấy {matched} ID trùng",
                                                      "✅ [{idx}/{total}] Step 5 (after fix): Found {matched} matching IDs",
                                                      "✅ [{idx}/{total}] Étape 5 (après correction) : {matched} ID correspondants",
                                                      idx=file_idx, total=len(pdf_b_paths), matched=step5_results["matched"]))
                                else:
                                    if step5_results["matched"] == 0:
                                        st.warning(tr("⚠️ [{idx}/{total}] Không sửa được ID nào bằng OpenAI OCR",
                                                      "⚠️ [{idx}/{total}] Could not fix any ID with OpenAI OCR",
                                                      "⚠️ [{idx}/{total}] Impossible de corriger les ID via OpenAI OCR",
                                                      idx=file_idx, total=len(pdf_b_paths)))
                                        all_results.append({
                                            "file_name": pdf_b_path.name,
                                            "status": "no_match",
                                            "matched": 0,
                                            "fix_attempt": fix_results
                                        })
                                        continue
                                    else:
                                        st.info(tr("ℹ️ [{idx}/{total}] Không sửa được thêm ID nào, nhưng vẫn có {matched} ID trùng",
                                                   "ℹ️ [{idx}/{total}] No additional IDs fixed, but {matched} matches found",
                                                   "ℹ️ [{idx}/{total}] Pas d'ID supplémentaire corrigé, mais {matched} correspondances trouvées",
                                                   idx=file_idx, total=len(pdf_b_paths), matched=step5_results["matched"]))
                            
                            st.success(tr("✅ [{idx}/{total}] Step 5: Tìm thấy {matched} ID trùng",
                                          "✅ [{idx}/{total}] Step 5: Found {matched} matching IDs",
                                          "✅ [{idx}/{total}] Étape 5 : {matched} ID correspondants",
                                          idx=file_idx, total=len(pdf_b_paths), matched=step5_results["matched"]))
                            progress_bar.progress(60)
                            
                            # Step 6: So sánh bằng AI (GPT hoặc Gemini)
                            if ai_model == "OpenAI GPT":
                                status_text.text(tr("🔄 [{idx}/{total}] Đang chạy Step 6: So sánh bằng OpenAI GPT...",
                                                    "🔄 [{idx}/{total}] Running Step 6: Comparing with OpenAI GPT...",
                                                    "🔄 [{idx}/{total}] Étape 6 : Comparaison via OpenAI GPT...",
                                                    idx=file_idx, total=len(pdf_b_paths)))
                                progress_bar.progress(70)
                                
                                step6_results = step6_compare_with_gpt(
                                    compare_dir=current_compare_dir,
                                    model="gpt-4.1",
                                    temperature=0.0,
                                    max_comparisons=None,
                                )
                            else:  # Google Gemini
                                status_text.text(tr("🔄 [{idx}/{total}] Đang chạy Step 6: So sánh bằng Google Gemini...",
                                                    "🔄 [{idx}/{total}] Running Step 6: Comparing with Google Gemini...",
                                                    "🔄 [{idx}/{total}] Étape 6 : Comparaison via Google Gemini...",
                                                    idx=file_idx, total=len(pdf_b_paths)))
                                progress_bar.progress(70)
                                
                                step6_results = step6_compare_with_gemini(
                                    compare_dir=current_compare_dir,
                                    model="gemini-2.5-flash",
                                    temperature=0.0,
                                    max_comparisons=None,
                                )
                            
                            if step6_results["comparisons"] == 0:
                                st.warning(tr("⚠️ [{idx}/{total}] Step 6: Không có kết quả so sánh",
                                              "⚠️ [{idx}/{total}] Step 6: No comparison results",
                                              "⚠️ [{idx}/{total}] Étape 6 : Aucun résultat de comparaison",
                                              idx=file_idx, total=len(pdf_b_paths)))
                                all_results.append({
                                    "file_name": pdf_b_path.name,
                                    "status": "no_comparison",
                                    "matched": step5_results["matched"]
                                })
                                continue
                            
                            st.success(tr("✅ [{idx}/{total}] Step 6: Đã so sánh {count} cặp bằng {model}",
                                          "✅ [{idx}/{total}] Step 6: Compared {count} pairs using {model}",
                                          "✅ [{idx}/{total}] Étape 6 : {count} comparaisons avec {model}",
                                          idx=file_idx, total=len(pdf_b_paths),
                                          count=step6_results["comparisons"],
                                          model=ai_model))
                            progress_bar.progress(85)
                            
                            # Step 7: Highlight PDF B
                            status_text.text(tr("🔄 [{idx}/{total}] Đang chạy Step 7: Highlight PDF B...",
                                                "🔄 [{idx}/{total}] Running Step 7: Highlighting PDF B...",
                                                "🔄 [{idx}/{total}] Étape 7 : Mise en évidence du PDF B...",
                                                idx=file_idx, total=len(pdf_b_paths)))
                            progress_bar.progress(90)
                            
                            output_pdf_name = f"{st.session_state.pdf_a_name}_vs_{pdf_b_name}_highlighted.pdf"
                            output_pdf_path = workspace["output"] / output_pdf_name
                            workspace["output"].mkdir(parents=True, exist_ok=True)
                            
                            step6_results_file = current_compare_dir / "step6_results.json"
                            
                            step7_results = step7_highlight_pdf_b(
                                pdf_b_path=pdf_b_path,
                                step6_results_file=step6_results_file,
                                output_path=output_pdf_path,
                                markup_color="Green",
                            )
                            
                            if step7_results["highlighted"] == 0:
                                st.warning(tr("⚠️ [{idx}/{total}] Step 7: Không highlight được text nào",
                                              "⚠️ [{idx}/{total}] Step 7: No highlights applied",
                                              "⚠️ [{idx}/{total}] Étape 7 : aucun surlignage",
                                              idx=file_idx, total=len(pdf_b_paths)))
                            else:
                                st.success(tr("✅ [{idx}/{total}] Step 7: Đã highlight {count} phần",
                                              "✅ [{idx}/{total}] Step 7: Highlighted {count} sections",
                                              "✅ [{idx}/{total}] Étape 7 : {count} zones surlignées",
                                              idx=file_idx, total=len(pdf_b_paths),
                                              count=step7_results["highlighted"]))
                            
                            progress_bar.progress(100)
                            status_text.text(tr("✅ [{idx}/{total}] Hoàn thành!",
                                                "✅ [{idx}/{total}] Done!",
                                                "✅ [{idx}/{total}] Terminé !",
                                                idx=file_idx, total=len(pdf_b_paths)))
                            
                            # Lưu kết quả
                            result = {
                                "file_name": pdf_b_path.name,
                                "status": "success",
                                "output_path": output_pdf_path,
                                "output_name": output_pdf_name,
                                "step4": step4_results,
                                "step5": step5_results,
                                "step6": step6_results,
                                "step7": step7_results,
                            }
                            all_results.append(result)
                            
                            # Hiển thị kết quả ngay sau khi xử lý xong file này
                            st.markdown("---")
                            st.subheader(tr("📄 Kết quả: {name}",
                                            "📄 Result: {name}",
                                            "📄 Résultat : {name}",
                                            name=pdf_b_path.name))
                            
                            st.success(tr("✅ Đã xử lý thành công",
                                          "✅ Processed successfully",
                                          "✅ Traitement réussi"))
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(tr("Step 4 - Bbox", "Step 4 - Bbox", "Étape 4 - Bbox"), step4_results['step1'])
                                st.caption(tr("ID: {ids}, Renamed: {renamed}",
                                              "ID: {ids}, Renamed: {renamed}",
                                              "ID : {ids}, Renommés : {renamed}",
                                              ids=step4_results['step2'], renamed=step4_results['step3']))
                            with col2:
                                st.metric(tr("Step 5 - Matched", "Step 5 - Matched", "Étape 5 - Correspondances"),
                                          step5_results['matched'])
                            with col3:
                                st.metric(tr("Step 6 - Comparisons", "Step 6 - Comparisons", "Étape 6 - Comparaisons"),
                                          step6_results['comparisons'])
                            with col4:
                                st.metric(tr("Step 7 - Highlighted", "Step 7 - Highlighted", "Étape 7 - Surlignés"),
                                          step7_results['highlighted'])
                            
                            # Tự động hiển thị PDF ngay lập tức
                            if output_pdf_path.exists():
                                with open(output_pdf_path, "rb") as f:
                                    pdf_bytes = f.read()
                                
                                st.markdown("---")
                                st.subheader(tr("📄 Xem trước PDF",
                                                "📄 Preview PDF",
                                                "📄 Aperçu du PDF"))
                                base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
                                st.markdown(pdf_display, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(tr("❌ [{idx}/{total}] Lỗi: {error}",
                                        "❌ [{idx}/{total}] Error: {error}",
                                        "❌ [{idx}/{total}] Erreur : {error}",
                                        idx=file_idx, total=len(pdf_b_paths), error=e))
                            st.exception(e)
                            result = {
                                "file_name": pdf_b_path.name,
                                "status": "error",
                                "error": str(e)
                            }
                            all_results.append(result)
                            progress_bar.progress(0)
                            status_text.text(tr("❌ [{idx}/{total}] Đã xảy ra lỗi",
                                                "❌ [{idx}/{total}] An error occurred",
                                                "❌ [{idx}/{total}] Une erreur est survenue",
                                                idx=file_idx, total=len(pdf_b_paths)))
                finally:
                    cleanup_session_workspace()
                
                # Hiển thị tổng kết tất cả kết quả
                if len(pdf_b_paths) > 1:
                    st.markdown("---")
                    st.header(tr("📊 Tổng Kết Tất Cả File",
                                 "📊 Summary of All Files",
                                 "📊 Récapitulatif de tous les fichiers"))
                    
                    success_count = sum(1 for r in all_results if r["status"] == "success")
                    st.info(tr("✅ Hoàn thành: {success}/{total} file",
                               "✅ Completed: {success}/{total} files",
                               "✅ Terminés : {success}/{total} fichiers",
                               success=success_count, total=len(all_results)))
                    
                    # Hiển thị bảng tổng kết
                    summary_data = []
                    for result in all_results:
                        if result["status"] == "success":
                            summary_data.append({
                                tr("File", "File", "Fichier"): result['file_name'],
                                tr("Status", "Status", "Statut"): "✅ " + tr("Thành công", "Success", "Succès"),
                                "Bbox": result['step4']['step1'],
                                tr("Matched", "Matched", "Correspondances"): result['step5']['matched'],
                                tr("Comparisons", "Comparisons", "Comparaisons"): result['step6']['comparisons'],
                                tr("Highlighted", "Highlighted", "Surlignés"): result['step7']['highlighted']
                            })
                        elif result["status"] == "no_match":
                            summary_data.append({
                                tr("File", "File", "Fichier"): result['file_name'],
                                tr("Status", "Status", "Statut"): "⚠️ " + tr("Không trùng", "No Match", "Pas de correspondance"),
                                "Bbox": "-",
                                tr("Matched", "Matched", "Correspondances"): "0",
                                tr("Comparisons", "Comparisons", "Comparaisons"): "-",
                                tr("Highlighted", "Highlighted", "Surlignés"): "-"
                            })
                        else:
                            summary_data.append({
                                tr("File", "File", "Fichier"): result['file_name'],
                                tr("Status", "Status", "Statut"): f"❌ {result.get('error', 'Error')[:30]}...",
                                "Bbox": "-",
                                tr("Matched", "Matched", "Correspondances"): "-",
                                tr("Comparisons", "Comparisons", "Comparaisons"): "-",
                                tr("Highlighted", "Highlighted", "Surlignés"): "-"
                            })
                    
                    if summary_data:
                        st.dataframe(summary_data, use_container_width=True)
    else:
        st.info(tr("ℹ️ Vui lòng chọn PDF A trước",
                   "ℹ️ Please select PDF A first",
                   "ℹ️ Veuillez d'abord sélectionner le PDF A"))
    
    # Footer
    st.markdown("---")
    st.markdown("### 📁 Folders")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text(f"Bbox: {BBOX_DIR}")
    with col2:
        st.text(f"ID: {ID_DIR}")
    with col3:
        st.text(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

