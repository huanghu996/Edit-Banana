#!/usr/bin/env python3
"""
FastAPI Backend Server — web service entry for Edit Banana.

Provides upload and conversion API. Run with: python server_pa.py
Server runs at http://localhost:8081
"""

import os
import sys
import threading
import tempfile
import shutil
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Status capture (wrap stdout before other imports that may print) ──────────
_orig_stdout = sys.stdout
_status_lock = threading.Lock()
_status: dict = {"step": "idle", "message": "Ready"}
_capturing = False

_STEP_MAP = {
    "[0]": "预处理 Preprocess",
    "[1]": "文字识别 OCR",
    "[2]": "图形分割 SAM3",
    "[3]": "图形处理 Shape",
    "[4]": "生成XML片段",
    "[7]": "合并XML Merge",
    "Done.": "完成",
}


class _Tee:
    """Write to original stdout and update _status when a conversion is active."""
    def write(self, s):
        _orig_stdout.write(s)
        if _capturing:
            line = s.strip()
            if line and "HTTP/1." not in line:
                with _status_lock:
                    _status["message"] = line[:300]
                    for key, label in _STEP_MAP.items():
                        if key in line:
                            _status["step"] = label
                            break
    def flush(self):
        _orig_stdout.flush()
    def isatty(self):
        return getattr(_orig_stdout, "isatty", lambda: False)()


sys.stdout = _Tee()

# ── Imports ───────────────────────────────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(
    title="Edit Banana API",
    description="Image to editable DrawIO (XML) — upload a diagram image, get DrawIO XML.",
    version="1.0.0",
)

_static_dir = os.path.join(PROJECT_ROOT, "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# ── Pipeline singleton ────────────────────────────────────────────────────────
_pipeline = None
_pipeline_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)  # one conversion at a time


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from main import load_config, Pipeline
                _pipeline = Pipeline(load_config())
    return _pipeline


@app.on_event("startup")
async def _startup():
    # Instantiate Pipeline early so it is ready before the first request.
    # (Models are still lazy-loaded on first use, but the object is shared.)
    _get_pipeline()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def get_status():
    with _status_lock:
        return dict(_status)


@app.get("/")
def root():
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    global _capturing

    name = file.filename or ""
    ext = Path(name).suffix.lower()
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format. Use: {', '.join(sorted(allowed))}")

    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    if not os.path.exists(config_path):
        raise HTTPException(503, "Missing config/config.yaml")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        pipeline = _get_pipeline()

        from main import load_config
        config = load_config()
        output_dir = config.get("paths", {}).get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)

        with _status_lock:
            _status["step"] = "开始处理"
            _status["message"] = "Starting conversion..."
        _capturing = True

        import asyncio
        loop = asyncio.get_event_loop()

        result_path = await loop.run_in_executor(
            _executor,
            lambda: pipeline.process_image(
                tmp_path,
                output_dir=output_dir,
                with_refinement=False,
                with_text=True,
            ),
        )

        with _status_lock:
            _status["step"] = "完成"
            _status["message"] = "Conversion complete"

        if not result_path or not os.path.exists(result_path):
            raise HTTPException(500, "Conversion failed — no output file produced")

        return FileResponse(
            result_path,
            media_type="application/xml",
            filename=Path(name).stem + ".xml",
        )

    except HTTPException:
        with _status_lock:
            _status["step"] = "error"
        raise
    except Exception as e:
        with _status_lock:
            _status["step"] = "error"
            _status["message"] = str(e)
        raise HTTPException(500, str(e))
    finally:
        _capturing = False
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def main():
    uvicorn.run(app, host="0.0.0.0", port=8081)


if __name__ == "__main__":
    main()
