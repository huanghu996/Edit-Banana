#!/usr/bin/env python3
"""
FastAPI Backend Server — web service entry for Edit Banana.

Provides upload and conversion API. Run with: python server_pa.py
Server runs at http://localhost:8081
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    """Upload an image and get editable DrawIO XML. Supported: PNG, JPG, BMP, TIFF, WebP."""
    name = file.filename or ""
    ext = Path(name).suffix.lower()
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format. Use one of: {', '.join(sorted(allowed))}.")

    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    if not os.path.exists(config_path):
        raise HTTPException(503, "Server not configured (missing config/config.yaml)")

    try:
        from main import load_config, Pipeline
        import tempfile
        import shutil

        config = load_config()
        output_dir = config.get("paths", {}).get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            pipeline = Pipeline(config)
            result_path = pipeline.process_image(
                tmp_path,
                output_dir=output_dir,
                with_refinement=False,
                with_text=True,
            )
            if not result_path or not os.path.exists(result_path):
                raise HTTPException(500, "Conversion failed")
            out_name = Path(name).stem + ".xml"
            return FileResponse(
                result_path,
                media_type="application/xml",
                filename=out_name,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


def main():
    uvicorn.run(app, host="0.0.0.0", port=8081)


if __name__ == "__main__":
    main()
