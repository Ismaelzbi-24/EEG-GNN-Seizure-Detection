"""
EEG-GNN web API.

  GET  /health          -> model status
  POST /jobs            -> upload an .edf (multipart field "file"); returns {"id": ...}
  GET  /jobs/{id}       -> {"status": queued|running|done|error, "progress": 0-1, "stage": str,
                            "result": {...} when done, "error": str when failed}

Uploaded files are written to a temp file, analysed, and deleted immediately.
Nothing is stored. Results are kept in memory for JOB_TTL_SEC, then dropped.

Environment variables (all optional):
  MODEL_PATH      default ../models/best_model_v4.pt
  SCALER_PATH     default ../models/scaler.pkl
  ALLOWED_ORIGINS comma-separated list for CORS, default "*"
  MAX_UPLOAD_MB   default 500
  MAX_WINDOWS     max 60-s windows analysed per file (evenly spread), default 40
  WORKERS         files analysed at the same time, default 1
"""
import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import pipeline

HERE = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(HERE.parent / "models" / "best_model_v4.pt"))
SCALER_PATH = os.environ.get("SCALER_PATH", str(HERE.parent / "models" / "scaler.pkl"))
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))
MAX_WINDOWS = int(os.environ.get("MAX_WINDOWS", "40"))
WORKERS = int(os.environ.get("WORKERS", "1"))
JOB_TTL_SEC = 3600
SITE_DIR = HERE.parent / "site"

app = FastAPI(title="EEG-GNN API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                   allow_methods=["GET", "POST"], allow_headers=["*"])

MODEL = SCALER = None
MODEL_META, LOAD_ERROR = {}, None
try:
    MODEL, SCALER, MODEL_META = pipeline.load_model(MODEL_PATH, SCALER_PATH)
except Exception as e:  # keep the server up so /health can explain the problem
    LOAD_ERROR = f"{type(e).__name__}: {e}"

JOBS = {}
LOCK = threading.Lock()
POOL = ThreadPoolExecutor(max_workers=WORKERS)


def _set(job_id, **kw):
    with LOCK:
        JOBS[job_id].update(kw)


def _cleanup():
    now = time.time()
    with LOCK:
        for k in [k for k, j in JOBS.items() if now - j["created"] > JOB_TTL_SEC]:
            del JOBS[k]


def _work(job_id, path):
    try:
        _set(job_id, status="running")

        def progress(done, total, stage):
            _set(job_id, progress=round(done / max(total, 1), 3), stage=stage)

        result = pipeline.run(path, MODEL, SCALER, max_windows=MAX_WINDOWS, progress=progress)
        result["model"] = {"name": "GAT v4", "device": str(pipeline.DEVICE), **MODEL_META}
        _set(job_id, status="done", progress=1.0, stage="Done", result=result)
    except pipeline.PipelineError as e:
        _set(job_id, status="error", error=str(e))
    except Exception as e:
        _set(job_id, status="error", error=f"Unexpected error while analysing: {type(e).__name__}: {e}")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.get("/health")
def health():
    return {"ok": MODEL is not None, "device": str(pipeline.DEVICE),
            "error": LOAD_ERROR, "max_upload_mb": MAX_UPLOAD_MB, "max_windows": MAX_WINDOWS}


@app.post("/jobs")
async def create_job(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(503, f"Model not loaded: {LOAD_ERROR}")
    if not (file.filename or "").lower().endswith(".edf"):
        raise HTTPException(400, "Please upload an .edf file.")
    _cleanup()
    fd, path = tempfile.mkstemp(suffix=".edf")
    size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
                out.write(chunk)
    except HTTPException:
        os.remove(path)
        raise
    job_id = uuid.uuid4().hex
    with LOCK:
        JOBS[job_id] = {"status": "queued", "progress": 0.0, "stage": "Waiting in queue",
                        "created": time.time(), "filename": file.filename,
                        "size_mb": round(size / 1048576, 1)}
    POOL.submit(_work, job_id, path)
    return {"id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown or expired job.")
        return {k: v for k, v in job.items() if k != "created"}


# Serve the website from the same server when it's next to the API.
if SITE_DIR.exists():
    @app.get("/")
    def index():
        return FileResponse(SITE_DIR / "index.html")

    app.mount("/", StaticFiles(directory=SITE_DIR), name="site")
