"""
FastAPI Backend: PDF Comparison Tool
Replaces Streamlit app with REST API + SSE for real-time progress.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Load .env
try:
    from dotenv import load_dotenv
    PROJECT_ROOT = Path(__file__).resolve().parent
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
except ImportError:
    pass

# Import pipeline steps
from step1 import step1_detect_and_save_bbox
from step2 import step2_detect_id_and_save
from step3 import step3_ocr_and_rename, ocr_id_with_openai
from step4 import step4_process_pdf_b
from step5_dino import step5_match_with_dino
from step6 import step6_compare_with_gpt
from step6_gemini import step6_compare_with_gemini
from step7_pymupdf import step7_highlight_pdf_b
from mode2_annotations import compare_mode2, get_openai_client as mode2_get_client
from mode3_charte import compare_mode3

# === App setup ===
app = FastAPI(title="PDF Comparison Tool", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
APP_DIR = Path(__file__).resolve().parent
BBOX_DIR = APP_DIR / "bbox"
ID_DIR = APP_DIR / "id"
OUTPUT_DIR = APP_DIR / "output"
MODELS_DIR = APP_DIR / "models"
SESSIONS_DIR = APP_DIR / "sessions"
STATIC_DIR = APP_DIR / "static"
MODE3_FIXED_CHARTE = Path(
    os.getenv("MODE3_CHARTE_PATH", str(APP_DIR / "Auchan - Charte tract - vClients .pdf"))
)
PDFS_A_DIR = APP_DIR / "pdfs_a"
PDFS_A_DIR.mkdir(exist_ok=True)

SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# === Progress tracking ===
# In-memory dict: session_id -> list of messages
progress_store: dict[str, list[dict]] = {}


def push_progress(session_id: str, step: str, status: str, detail: str = "",
                  progress: int = 0, data: dict | None = None):
    """Push a progress event to the store."""
    if session_id not in progress_store:
        progress_store[session_id] = []
    event = {
        "step": step,
        "status": status,
        "detail": detail,
        "progress": progress,
        "data": data or {},
        "timestamp": datetime.now().isoformat(),
    }
    progress_store[session_id].append(event)


# === Helpers ===
def find_model() -> Optional[Path]:
    model_path = MODELS_DIR / "bbox_id.pt"
    if not model_path.exists():
        model_path = APP_DIR.parent / "models" / "bbox_id.pt"
    return model_path if model_path.exists() else None


def find_id_model() -> Optional[Path]:
    id_model_path = MODELS_DIR / "id_plus.pt"
    if id_model_path.exists():
        return id_model_path
    return find_model()


VERSIONED_STEM_RE = re.compile(r"^(?P<base>.+?)_(?P<ts>\d{8}_\d{6})(?:_(?P<seq>\d+))?$")


def split_versioned_stem(stem: str) -> tuple[str, str | None]:
    match = VERSIONED_STEM_RE.match(stem)
    if not match:
        return stem, None
    return match.group("base"), match.group("ts")


def format_timestamp_label(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        return dt.strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        return None


def find_pdf_a_file_by_stem(stem: str) -> Optional[Path]:
    direct = PDFS_A_DIR / f"{stem}.pdf"
    if direct.exists():
        return direct
    for path in PDFS_A_DIR.glob("*.pdf"):
        if path.stem == stem:
            return path
    return None


def build_pdf_display_info(stem: str, pdf_file: Optional[Path] = None) -> dict:
    base_stem, ts = split_versioned_stem(stem)
    display_name = f"{base_stem}.pdf"
    stored_at_display = format_timestamp_label(ts)
    if stored_at_display:
        return {
            "display_name": display_name,
            "stored_at_display": stored_at_display,
        }

    if pdf_file and pdf_file.exists():
        try:
            mtime = datetime.fromtimestamp(pdf_file.stat().st_mtime)
            return {
                "display_name": display_name,
                "stored_at_display": mtime.strftime("%d/%m/%Y %H:%M:%S"),
            }
        except Exception:
            pass

    return {
        "display_name": display_name,
        "stored_at_display": None,
    }


def get_processed_pdfs() -> list[str]:
    processed = set()
    if BBOX_DIR.exists():
        for subdir in BBOX_DIR.iterdir():
            if subdir.is_dir():
                processed.add(subdir.name)
    if ID_DIR.exists():
        for subdir in ID_DIR.iterdir():
            if subdir.is_dir():
                processed.add(subdir.name)
    return sorted(list(processed))


def fix_bbox_ids_with_openai(bbox_b_dir: Path, bbox_a_dir: Path,
                              model: str = "gpt-5.4",
                              matched_ids: set = None) -> dict:
    """Use OpenAI OCR to read and fix IDs from bbox images."""
    if not bbox_b_dir.exists():
        return {"ocr_count": 0, "renamed_count": 0, "matched_ids": []}

    pdf_a_ids = set()
    if bbox_a_dir.exists():
        for file_a in bbox_a_dir.glob("*.png"):
            id_text = extract_id_from_filename(file_a.name)
            if id_text:
                pdf_a_ids.add(id_text)

    if not pdf_a_ids:
        return {"ocr_count": 0, "renamed_count": 0, "matched_ids": []}

    if matched_ids is None:
        matched_ids = set()

    bbox_files = list(bbox_b_dir.glob("*.png"))
    ocr_count = 0
    renamed_count = 0
    new_matched_ids = []

    for bbox_file in bbox_files:
        file_id = extract_id_from_filename(bbox_file.name)
        if file_id and file_id in matched_ids:
            continue

        try:
            ocr_result = ocr_id_with_openai(bbox_file, model=model)
            ocr_count += 1
            if not ocr_result:
                continue

            id_match = re.search(r'\b(\d{4,5})\b', ocr_result)
            if not id_match:
                id_match = re.search(r'\b(\d{3,6})\b', ocr_result)

            if id_match:
                found_id = id_match.group(1)
                if found_id in pdf_a_ids:
                    page_match = re.search(r'page(\d+)_', bbox_file.name)
                    if page_match:
                        page_num = page_match.group(1)
                        new_name = f"page{page_num}_{found_id}.png"
                        new_path = bbox_file.parent / new_name
                        if bbox_file.name != new_name:
                            if new_path.exists():
                                bbox_file.unlink()
                            else:
                                bbox_file.rename(new_path)
                                renamed_count += 1
                                new_matched_ids.append(found_id)
        except Exception as e:
            print(f"  â ï¸  Error processing {bbox_file.name}: {e}")
            continue

    return {
        "ocr_count": ocr_count,
        "renamed_count": renamed_count,
        "matched_ids": new_matched_ids,
    }


def create_session_workspace(session_id: str) -> dict:
    """Create workspace directories for a session."""
    root_dir = SESSIONS_DIR / session_id
    temp_dir = root_dir / "temp"
    compare_dir = root_dir / "compare_temp"
    uploads_dir = root_dir / "temp_pdf_upload"
    pdf_a_temp_dir = root_dir / "pdf_a_temp"
    session_output_dir = root_dir / "output"

    for path in (temp_dir, compare_dir, uploads_dir, pdf_a_temp_dir, session_output_dir):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "id": session_id,
        "root": root_dir,
        "temp": temp_dir,
        "compare": compare_dir,
        "uploads": uploads_dir,
        "pdf_a_temp": pdf_a_temp_dir,
        "output": session_output_dir,
    }


def cleanup_session(session_id: str):
    """Remove a session workspace."""
    root_dir = SESSIONS_DIR / session_id
    if root_dir.exists():
        shutil.rmtree(root_dir, ignore_errors=True)
    progress_store.pop(session_id, None)


# === API Routes ===

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "Frontend not found")
    return index_path.read_text(encoding="utf-8")


@app.get("/api/health")
async def health():
    model = find_model()
    return {
        "status": "ok",
        "model_found": model is not None,
        "model_path": str(model) if model else None,
    }


@app.get("/api/processed-pdfs")
async def list_processed_pdfs():
    pdfs = get_processed_pdfs()
    result = []
    for name in pdfs:
        bbox_dir = BBOX_DIR / name
        id_dir = ID_DIR / name
        bbox_count = len(list(bbox_dir.glob("*.png"))) if bbox_dir.exists() else 0
        id_count = len(list(id_dir.glob("*.png"))) if id_dir.exists() else 0
        pdf_file = find_pdf_a_file_by_stem(name)
        display_info = build_pdf_display_info(name, pdf_file)
        result.append({
            "name": name,
            "display_name": display_info["display_name"],
            "stored_at_display": display_info["stored_at_display"],
            "bbox_count": bbox_count,
            "id_count": id_count,
            "has_bbox": bbox_dir.exists(),
            "has_id": id_dir.exists(),
        })
    return {"pdfs": result}


@app.post("/api/upload-pdf-a")
async def upload_pdf_a(file: UploadFile = File(...)):
    """Upload PDF A and create a session."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    workspace = create_session_workspace(session_id)

    original_filename = file.filename
    suffix = Path(original_filename).suffix or ".pdf"
    original_stem = Path(original_filename).stem

    content = await file.read()

    stored_filename = original_filename
    perm_path = PDFS_A_DIR / stored_filename
    if perm_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stored_filename = f"{original_stem}_{ts}{suffix}"
        perm_path = PDFS_A_DIR / stored_filename
        seq = 1
        while perm_path.exists():
            seq += 1
            stored_filename = f"{original_stem}_{ts}_{seq}{suffix}"
            perm_path = PDFS_A_DIR / stored_filename

    perm_path.write_bytes(content)

    pdf_name = Path(stored_filename).stem
    bbox_a_dir = BBOX_DIR / pdf_name
    id_a_dir = ID_DIR / pdf_name

    # Save uploaded file for this session
    temp_path = workspace["pdf_a_temp"] / stored_filename
    temp_path.write_bytes(content)

    already_processed = bbox_a_dir.exists() and id_a_dir.exists()
    bbox_count = len(list(bbox_a_dir.glob("*.png"))) if bbox_a_dir.exists() else 0
    id_count = len(list(id_a_dir.glob("*.png"))) if id_a_dir.exists() else 0

    return {
        "session_id": session_id,
        "pdf_name": pdf_name,
        "display_name": original_filename,
        "stored_name": stored_filename,
        "stored_at_display": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "file_path": str(temp_path),
        "already_processed": already_processed,
        "bbox_count": bbox_count,
        "id_count": id_count,
    }


@app.post("/api/select-pdf-a")
async def select_existing_pdf_a(name: str = Form(...)):
    """Select an existing processed PDF A."""
    bbox_a_dir = BBOX_DIR / name
    id_a_dir = ID_DIR / name

    if not bbox_a_dir.exists() and not id_a_dir.exists():
        raise HTTPException(404, f"No data found for '{name}'")

    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    create_session_workspace(session_id)

    bbox_count = len(list(bbox_a_dir.glob("*.png"))) if bbox_a_dir.exists() else 0
    id_count = len(list(id_a_dir.glob("*.png"))) if id_a_dir.exists() else 0

    pdf_file = find_pdf_a_file_by_stem(name)
    display_info = build_pdf_display_info(name, pdf_file)

    return {
        "session_id": session_id,
        "pdf_name": name,
        "display_name": display_info["display_name"],
        "stored_at_display": display_info["stored_at_display"],
        "already_processed": True,
        "bbox_count": bbox_count,
        "id_count": id_count,
    }


@app.post("/api/run-step1")
async def run_step1(session_id: str = Form(...), pdf_name: str = Form(...)):
    """Run Step 1: Detect bbox from PDF A."""
    model_path = find_model()
    if not model_path:
        raise HTTPException(500, "YOLO model not found")

    workspace = create_session_workspace(session_id)
    pdf_path = workspace["pdf_a_temp"] / f"{pdf_name}.pdf"

    # Try to find the PDF file
    if not pdf_path.exists():
        # Search in pdf_a_temp for any PDF
        pdf_files = list(workspace["pdf_a_temp"].glob("*.pdf"))
        if pdf_files:
            pdf_path = pdf_files[0]
        else:
            raise HTTPException(404, "PDF file not found in session")

    # Clear old data
    bbox_a_dir = BBOX_DIR / pdf_name
    id_a_dir = ID_DIR / pdf_name
    if bbox_a_dir.exists():
        shutil.rmtree(bbox_a_dir)
    if id_a_dir.exists():
        shutil.rmtree(id_a_dir)

    try:
        total_bbox = step1_detect_and_save_bbox(
            pdf_path=pdf_path,
            model_path=model_path,
            output_dir=BBOX_DIR,
            conf=0.4,
            dpi=300,
            imgsz=640,
            device=None,
            max_det=300,
            save_overview=False,
            merge_iou=0.1,
        )
        return {"total_bbox": total_bbox, "pdf_name": pdf_name}
    except Exception as e:
        raise HTTPException(500, f"Step 1 error: {str(e)}")


@app.post("/api/run-step23")
async def run_step23(session_id: str = Form(...), pdf_name: str = Form(...)):
    """Run Step 2 & 3: Detect ID + OCR rename."""
    model_path = find_model()
    id_model_path = find_id_model()

    bbox_a_dir = BBOX_DIR / pdf_name
    id_a_dir = ID_DIR / pdf_name

    if not bbox_a_dir.exists():
        raise HTTPException(404, "Bbox directory not found. Run Step 1 first.")

    try:
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

        step3_results = {"success": 0}
        if step2_results["saved"] > 0:
            step3_results = step3_ocr_and_rename(
                id_dir=id_a_dir,
                bbox_dir=bbox_a_dir,
                ocr_lang="en",
                use_gpu=False,
                dry_run=False,
                use_openai_fallback=True,
                openai_model="gpt-4o",
            )

        return {
            "step2_saved": step2_results["saved"],
            "step3_renamed": step3_results["success"],
        }
    except Exception as e:
        raise HTTPException(500, f"Step 2&3 error: {str(e)}")


@app.post("/api/upload-pdf-b")
async def upload_pdf_b(
    session_id: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """Upload one or more PDF B files."""
    workspace_root = SESSIONS_DIR / session_id
    if not workspace_root.exists():
        raise HTTPException(404, "Session not found")

    uploads_dir = workspace_root / "temp_pdf_upload"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        dest = uploads_dir / f.filename
        content = await f.read()
        dest.write_bytes(content)
        saved_files.append({"name": f.filename, "path": str(dest)})

    return {"uploaded": len(saved_files), "files": saved_files}


@app.post("/api/run-pipeline")
async def run_pipeline(
    session_id: str = Form(...),
    pdf_a_name: str = Form(...),
    ai_model: str = Form("OpenAI GPT"),
):
    """Run the full pipeline (Step 4â†’5â†’6â†’7) for all uploaded PDF B files.
    Progress is pushed to SSE store."""
    model_path = find_model()
    if not model_path:
        raise HTTPException(500, "YOLO model not found")

    workspace_root = SESSIONS_DIR / session_id
    if not workspace_root.exists():
        raise HTTPException(404, "Session not found")

    uploads_dir = workspace_root / "temp_pdf_upload"
    temp_dir = workspace_root / "temp"
    compare_root = workspace_root / "compare_temp"
    output_dir = workspace_root / "output"

    pdf_b_files = list(uploads_dir.glob("*.pdf"))
    if not pdf_b_files:
        raise HTTPException(404, "No PDF B files found")

    bbox_a_dir = BBOX_DIR / pdf_a_name
    if not bbox_a_dir.exists():
        raise HTTPException(404, f"Bbox directory for '{pdf_a_name}' not found")

    # Clear progress
    progress_store[session_id] = []

    all_results = []
    total_files = len(pdf_b_files)

    async def process_one_pdf_b(file_idx, pdf_b_path):
        pdf_b_name = pdf_b_path.stem
        push_progress(session_id, "step4", "running",
                      f"[{file_idx}/{total_files}] Processing {pdf_b_path.name}...",
                      progress=int(10 + (file_idx - 1) / total_files * 80))

        try:
            # Step 4 (CPU bound mostly, but run in thread)
            step4_results = await asyncio.to_thread(
                step4_process_pdf_b,
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

            push_progress(session_id, "step4", "done",
                          f"[{file_idx}/{total_files}] Step 4: {step4_results['step1']} bbox, {step4_results['step2']} ID",
                          progress=int(10 + (file_idx - 0.7) / total_files * 80))

            if step4_results["step3"] == 0:
                push_progress(session_id, "step4", "error",
                              f"[{file_idx}/{total_files}] Step 4 failed for {pdf_b_path.name}")
                return {"file_name": pdf_b_path.name, "status": "failed"}

            # Step 5 (DINOv2 visual matching)
            push_progress(session_id, "step5", "running",
                          f"[{file_idx}/{total_files}] DINOv2 visual matching...",
                          progress=int(10 + (file_idx - 0.6) / total_files * 80))

            bbox_b_dir = temp_dir / "bbox" / pdf_b_name
            current_compare_dir = compare_root / pdf_b_name

            # Find segmentation model for Pass 2
            seg_model_path = MODELS_DIR / "best_model.pth"

            step5_results = await asyncio.to_thread(
                step5_match_with_dino,
                bbox_dir_a=bbox_a_dir,
                bbox_dir_b=bbox_b_dir,
                output_dir=current_compare_dir,
                threshold_pass1=0.75,
                threshold_pass2=0.60,
                seg_model_path=seg_model_path if seg_model_path.exists() else None,
            )

            if step5_results["matched"] == 0:
                push_progress(session_id, "step5", "warning",
                              f"[{file_idx}/{total_files}] No visual matches found for {pdf_b_path.name}")
                return {"file_name": pdf_b_path.name, "status": "no_match", "matched": 0}

            push_progress(session_id, "step5", "done",
                          f"[{file_idx}/{total_files}] Step 5: {step5_results['matched']} matches (P1={step5_results.get('pass1_matched', 0)}, P2={step5_results.get('pass2_matched', 0)})",
                          progress=int(10 + (file_idx - 0.4) / total_files * 80))

            # Step 6
            push_progress(session_id, "step6", "running",
                          f"[{file_idx}/{total_files}] Comparing with {ai_model}...",
                          progress=int(10 + (file_idx - 0.3) / total_files * 80))

            if ai_model == "OpenAI GPT":
                step6_results = await asyncio.to_thread(
                    step6_compare_with_gpt,
                    compare_dir=current_compare_dir,
                    model="gpt-4o",
                    temperature=0.0,
                    max_comparisons=None,
                )
            else:
                step6_results = await asyncio.to_thread(
                    step6_compare_with_gemini,
                    compare_dir=current_compare_dir,
                    model="gemini-2.5-flash",
                    temperature=0.0,
                    max_comparisons=None,
                )

            if step6_results["comparisons"] == 0:
                push_progress(session_id, "step6", "warning",
                              f"[{file_idx}/{total_files}] No comparison results")
                return {"file_name": pdf_b_path.name, "status": "no_comparison"}

            push_progress(session_id, "step6", "done",
                          f"[{file_idx}/{total_files}] Step 6: {step6_results['comparisons']} comparisons",
                          progress=int(10 + (file_idx - 0.15) / total_files * 80))

            # Step 7
            push_progress(session_id, "step7", "running",
                          f"[{file_idx}/{total_files}] Highlighting PDF B...",
                          progress=int(10 + (file_idx - 0.1) / total_files * 80))

            output_pdf_name = f"{pdf_a_name}_vs_{pdf_b_name}_highlighted.pdf"
            output_pdf_path = output_dir / output_pdf_name

            step6_results_file = current_compare_dir / "step6_results.json"

            step7_results = await asyncio.to_thread(
                step7_highlight_pdf_b,
                pdf_b_path=pdf_b_path,
                step6_results_file=step6_results_file,
                output_path=output_pdf_path,
                markup_color="Green",
            )

            push_progress(session_id, "step7", "done",
                          f"[{file_idx}/{total_files}] Step 7: {step7_results['highlighted']} highlights",
                          progress=int(10 + file_idx / total_files * 80))

            return {
                "file_name": pdf_b_path.name,
                "status": "success",
                "output_file": output_pdf_name,
                "step4": step4_results,
                "step5": {"matched": step5_results["matched"]},
                "step6": {"comparisons": step6_results["comparisons"]},
                "step7": {"highlighted": step7_results["highlighted"]},
            }

        except Exception as e:
            push_progress(session_id, "error", "error",
                          f"[{file_idx}/{total_files}] Error: {str(e)}")
            return {
                "file_name": pdf_b_path.name,
                "status": "error",
                "error": str(e),
            }

    # Run tasks concurrently
    tasks = [process_one_pdf_b(idx, path) for idx, path in enumerate(pdf_b_files, 1)]
    all_results = await asyncio.gather(*tasks)

    push_progress(session_id, "complete", "done",
                  f"Pipeline complete: {sum(1 for r in all_results if r['status'] == 'success')}/{total_files} successful",
                  progress=100, data={"results": all_results})

    return {"results": all_results}


@app.get("/api/stream-progress/{session_id}")
async def stream_progress(session_id: str):
    """SSE endpoint for streaming progress events."""
    async def event_generator():
        last_index = 0
        while True:
            events = progress_store.get(session_id, [])
            while last_index < len(events):
                event = events[last_index]
                yield f"data: {json.dumps(event)}\n\n"
                last_index += 1
                # If pipeline complete, stop
                if event.get("step") == "complete":
                    return
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/pdf-a/{name}")
async def serve_pdf_a(name: str):
    """Serve a PDF A from permanent storage."""
    # Try by stem first, then literal file name
    path = find_pdf_a_file_by_stem(name)
    if not path:
        literal = PDFS_A_DIR / name
        path = literal if literal.exists() else None
    
    if not path:
        raise HTTPException(404, "PDF A not found")
        
    return FileResponse(path, media_type="application/pdf", content_disposition_type="inline")


@app.get("/api/view/{session_id}/{filename}")
async def view_file(session_id: str, filename: str):
    """View a highlighted PDF output inline."""
    file_path = SESSIONS_DIR / session_id / "output" / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        content_disposition_type="inline",
    )


@app.get("/api/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    """Download a highlighted PDF output as attachment."""
    file_path = SESSIONS_DIR / session_id / "output" / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/pdf",
    )


@app.get("/api/pdf-info/{pdf_name}")
async def pdf_info(pdf_name: str):
    """Get info about a processed PDF A."""
    bbox_dir = BBOX_DIR / pdf_name
    id_dir = ID_DIR / pdf_name
    bbox_count = len(list(bbox_dir.glob("*.png"))) if bbox_dir.exists() else 0
    id_count = len(list(id_dir.glob("*.png"))) if id_dir.exists() else 0

    # Get bbox filenames with IDs
    bbox_files = []
    if bbox_dir.exists():
        for f in sorted(bbox_dir.glob("*.png")):
            file_id = extract_id_from_filename(f.name)
            bbox_files.append({"name": f.name, "id": file_id})

    pdf_file = find_pdf_a_file_by_stem(pdf_name)
    display_info = build_pdf_display_info(pdf_name, pdf_file)

    return {
        "name": pdf_name,
        "display_name": display_info["display_name"],
        "stored_at_display": display_info["stored_at_display"],
        "bbox_count": bbox_count,
        "id_count": id_count,
        "has_bbox": bbox_dir.exists(),
        "has_id": id_dir.exists(),
        "bbox_files": bbox_files,
    }


@app.delete("/api/delete-pdf-a/{name}")
async def delete_pdf_a(name: str):
    """Delete processed data for a PDF A."""
    bbox_dir = BBOX_DIR / name
    id_dir = ID_DIR / name
    deleted = []

    if bbox_dir.exists():
        shutil.rmtree(bbox_dir)
        deleted.append("bbox")
    if id_dir.exists():
        shutil.rmtree(id_dir)
        deleted.append("id")

    if not deleted:
        raise HTTPException(404, "No data to delete")

    return {"deleted": deleted, "name": name}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Cleanup a session workspace."""
    cleanup_session(session_id)
    return {"status": "cleaned", "session_id": session_id}



# === Mode 2 & 3: SPA - serve main index, use hash routing ===
@app.get("/mode2", response_class=HTMLResponse)
@app.get("/mode3", response_class=HTMLResponse)
async def serve_spa_modes():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "Frontend not found")
    return index_path.read_text(encoding="utf-8")


@app.post("/api/mode2/compare")
async def api_mode2_compare(
    ref_pdf: UploadFile = File(...),
    final_pdf: UploadFile = File(...),
    model: Optional[str] = Form(None),
):
    if not ref_pdf.filename or not ref_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "ref_pdf must be a PDF file")
    if not final_pdf.filename or not final_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "final_pdf must be a PDF file")

    client = mode2_get_client()
    if client is None:
        raise HTTPException(503, "OpenAI API key not configured for mode2")

    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    mode2_dir = SESSIONS_DIR / session_id / "mode2"
    mode2_dir.mkdir(parents=True, exist_ok=True)

    ref_path = mode2_dir / "ref.pdf"
    final_path = mode2_dir / "final.pdf"
    ref_path.write_bytes(await ref_pdf.read())
    final_path.write_bytes(await final_pdf.read())

    output_path = mode2_dir / "output_mode2_diff.pdf"

    try:
        result = await asyncio.to_thread(
            compare_mode2,
            ref_pdf_path=str(ref_path),
            final_pdf_path=str(final_path),
            output_path=str(output_path),
            model=model,
            save_images_dir=str(mode2_dir),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Mode 2 compare failed: {e}")

    output_filename = output_path.name

    gpt_images_dir = mode2_dir / "gpt_images"
    gpt_images = []
    if gpt_images_dir.exists():
        for f in sorted(gpt_images_dir.glob("*.png")):
            gpt_images.append({
                "name": f.name,
                "url": f"/api/mode2/images/{session_id}/{f.name}",
            })

    return {
        "session_id": session_id,
        "output_file": output_filename,
        "summary": result["summary"],
        "results": result["results"],
        "preprocessing": result.get("preprocessing", {}),
        "view_url": f"/api/mode2/result/{session_id}/{output_filename}",
        "download_url": f"/api/mode2/download/{session_id}/{output_filename}",
        "gpt_images": gpt_images,
    }

@app.get("/api/mode2/result/{session_id}/{filename}")
async def api_mode2_view_result(session_id: str, filename: str):
    file_path = SESSIONS_DIR / session_id / "mode2" / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path=str(file_path), media_type="application/pdf", content_disposition_type="inline")


@app.get("/api/mode2/download/{session_id}/{filename}")
async def api_mode2_download_result(session_id: str, filename: str):
    file_path = SESSIONS_DIR / session_id / "mode2" / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path=str(file_path), filename=filename, media_type="application/pdf")


@app.get("/api/mode2/images/{session_id}/{filename}")
async def api_mode2_view_image(session_id: str, filename: str):
    from fastapi.responses import Response
    file_path = SESSIONS_DIR / session_id / "mode2" / "gpt_images" / filename
    if not file_path.exists():
        raise HTTPException(404, "Image not found")
    return Response(content=file_path.read_bytes(), media_type="image/png")
@app.post("/api/mode3/check")
async def api_mode3_check(
    final_pdf: UploadFile = File(...),
    model: Optional[str] = Form(None),
    lang: Optional[str] = Form("fr"),
):
    if not final_pdf.filename or not final_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "final_pdf must be a PDF file")
    if not MODE3_FIXED_CHARTE.exists():
        raise HTTPException(500, f"Fixed charte not found: {MODE3_FIXED_CHARTE}")

    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    mode3_dir = SESSIONS_DIR / session_id / "mode3"
    mode3_dir.mkdir(parents=True, exist_ok=True)

    charte_path = mode3_dir / "charte.pdf"
    final_path = mode3_dir / "final.pdf"
    shutil.copy2(MODE3_FIXED_CHARTE, charte_path)
    final_path.write_bytes(await final_pdf.read())

    hard_rules_path = str(APP_DIR / "rules" / "hard_rule" / "hard_rules_auchan_vclients.json")

    try:
        result = await asyncio.to_thread(
            compare_mode3,
            charte_pdf_path=str(charte_path),
            final_pdf_path=str(final_path),
            output_dir=str(mode3_dir),
            hard_rules_path=hard_rules_path,
            model=model,
            lang=lang or "fr",
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Mode 3 check failed: {e}")

    output_file = Path(result["output_pdf"]).name
    return {
        "session_id": session_id,
        "output_file": output_file,
        "hard_summary": result.get("hard_summary", {}),
        "hard_results": result.get("hard_results", []),
        "soft_summary": result.get("soft_summary", {}),
        "soft_details": result.get("soft_details", {}),
        "zones_detected": result.get("zones_detected", {}),
        "detect_method": result.get("detect_method", "unknown"),
        "debug": result.get("debug", {}),
        "debug_base_url": f"/api/mode3/debug/{session_id}",
        "view_url": f"/api/mode3/result/{session_id}/{output_file}",
        "download_url": f"/api/mode3/download/{session_id}/{output_file}",
        "report_url": f"/api/mode3/report/{session_id}",
        "charte_source": str(MODE3_FIXED_CHARTE),
    }

@app.get("/api/mode3/debug/{session_id}/{filename}")
async def api_mode3_debug_image(session_id: str, filename: str):
    file_path = SESSIONS_DIR / session_id / "mode3" / "debug" / filename
    if not file_path.exists():
        raise HTTPException(404, "Debug image not found")
    return FileResponse(path=str(file_path), media_type="image/jpeg")

@app.get("/api/mode3/result/{session_id}/{filename}")
async def api_mode3_view_result(session_id: str, filename: str):
    file_path = SESSIONS_DIR / session_id / "mode3" / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path=str(file_path), media_type="application/pdf", content_disposition_type="inline")


@app.get("/api/mode3/download/{session_id}/{filename}")
async def api_mode3_download_result(session_id: str, filename: str):
    file_path = SESSIONS_DIR / session_id / "mode3" / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path=str(file_path), filename=filename, media_type="application/pdf")


@app.get("/api/mode3/report/{session_id}")
async def api_mode3_report(session_id: str):
    file_path = SESSIONS_DIR / session_id / "mode3" / "mode3_report.json"
    if not file_path.exists():
        raise HTTPException(404, "Report not found")
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {"error": "invalid_report_json"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)















