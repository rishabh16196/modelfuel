"""FastAPI application exposing the DistillStack pipeline over HTTP."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from distillstack.config import Settings, get_settings
from distillstack.models.document import QualityVerdict, SynthesisRecord
from distillstack.pipeline.processor import DocumentProcessor
from distillstack.pipeline.quality import QualityScorer
from distillstack.pipeline.synthesis import SynthesisAgent

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DistillStack",
    version="0.1.0",
    description="Cognitive ETL Pipeline for SLM instruction-tuning dataset synthesis.",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process")
async def process_pdf(
    file: UploadFile,
    settings: Annotated[Settings, Depends(get_settings)],
) -> StreamingResponse:
    """Accept a PDF upload, run the full pipeline, and return scored JSONL."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        processor = DocumentProcessor(settings)
        document = await processor.process(tmp_path)

        agent = SynthesisAgent(settings)
        records = await agent.synthesize(document)

        scorer = QualityScorer(settings)
        chunks = _rebuild_chunk_map(document.markdown, records)
        verdicts = await scorer.score_batch(chunks)

        output = _build_jsonl(records, verdicts)
        return StreamingResponse(
            iter([output]),
            media_type="application/jsonl",
            headers={
                "Content-Disposition": f'attachment; filename="{Path(file.filename).stem}_output.jsonl"'
            },
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def _rebuild_chunk_map(
    markdown: str, records: list[SynthesisRecord]
) -> list[tuple[SynthesisRecord, str]]:
    """Re-associate each record with its source chunk via the stored hash."""
    from distillstack.pipeline.synthesis import _chunk_by_headers

    chunks = _chunk_by_headers(markdown)
    hash_to_chunk = {SynthesisRecord.hash_chunk(c): c for c in chunks}
    return [
        (rec, hash_to_chunk.get(rec.source_chunk_hash, ""))
        for rec in records
    ]


def _build_jsonl(
    records: list[SynthesisRecord], verdicts: list[QualityVerdict]
) -> str:
    """Serialize records + verdicts into JSONL, keeping only passing rows."""
    lines: list[str] = []
    for record, verdict in zip(records, verdicts, strict=True):
        if not verdict.passed:
            logger.info(
                "Filtered out record (faithfulness=%.2f, complexity=%.2f): %s",
                verdict.faithfulness_score,
                verdict.complexity_score,
                record.instruction[:80],
            )
            continue
        row = {
            "instruction": record.instruction,
            "thought": record.thought,
            "output": record.output,
            "source_file": record.source_file,
            "quality": {
                "faithfulness": verdict.faithfulness_score,
                "complexity": verdict.complexity_score,
            },
        }
        lines.append(json.dumps(row, ensure_ascii=False))
    return "\n".join(lines) + "\n" if lines else ""
