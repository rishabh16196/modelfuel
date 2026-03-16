"""FastAPI application exposing the DistillStack extraction pipeline over HTTP."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from distillstack.config import Settings, get_settings
from distillstack.pipeline.processor import DocumentProcessor

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DistillStack",
    version="0.1.0",
    description="Cognitive ETL Pipeline — document extraction endpoint.",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract")
async def extract_pdf(
    file: UploadFile,
    settings: Annotated[Settings, Depends(get_settings)],
) -> JSONResponse:
    """Accept a PDF upload and return the extracted InternalDocument as JSON."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        processor = DocumentProcessor(settings)
        document = await processor.process(tmp_path)
        return JSONResponse(content=document.model_dump(mode="json"))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/extract/markdown")
async def extract_pdf_markdown(
    file: UploadFile,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PlainTextResponse:
    """Accept a PDF upload and return the extracted Markdown."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        processor = DocumentProcessor(settings)
        document = await processor.process(tmp_path)
        return PlainTextResponse(content=document.markdown)
    finally:
        tmp_path.unlink(missing_ok=True)
