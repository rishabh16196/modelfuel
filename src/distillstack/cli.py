"""CLI entry point for the DistillStack pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from distillstack.config import get_settings
from distillstack.models.document import SynthesisRecord
from distillstack.pipeline.processor import DocumentProcessor
from distillstack.pipeline.quality import QualityScorer
from distillstack.pipeline.synthesis import SynthesisAgent, _chunk_by_headers

logger = logging.getLogger(__name__)


async def _run_pipeline(pdf_path: Path, output_dir: Path) -> None:
    """Execute the full ETL pipeline: process -> synthesize -> score -> write."""
    settings = get_settings()
    settings.configure_logging()

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing %s", pdf_path)
    processor = DocumentProcessor(settings)
    document = await processor.process(pdf_path)
    logger.info(
        "Extracted %d pages, %d content blocks",
        document.total_pages,
        len(document.content_blocks),
    )

    logger.info("Running synthesis agent...")
    agent = SynthesisAgent(settings)
    records = await agent.synthesize(document)
    logger.info("Generated %d synthesis records", len(records))

    logger.info("Running quality scorer...")
    scorer = QualityScorer(settings)
    chunks = _chunk_by_headers(document.markdown)
    hash_to_chunk = {SynthesisRecord.hash_chunk(c): c for c in chunks}
    pairs = [
        (rec, hash_to_chunk.get(rec.source_chunk_hash, "")) for rec in records
    ]
    verdicts = await scorer.score_batch(pairs)

    output_file = output_dir / f"{pdf_path.stem}_output.jsonl"
    kept = 0
    with output_file.open("w", encoding="utf-8") as fh:
        for record, verdict in zip(records, verdicts, strict=True):
            if not verdict.passed:
                logger.debug(
                    "Filtered: faithfulness=%.2f complexity=%.2f — %s",
                    verdict.faithfulness_score,
                    verdict.complexity_score,
                    record.instruction[:60],
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
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    logger.info(
        "Done. %d / %d records passed quality gate -> %s",
        kept,
        len(records),
        output_file,
    )


def main() -> None:
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        prog="distillstack",
        description="DistillStack: Cognitive ETL Pipeline for SLM dataset synthesis.",
    )
    sub = parser.add_subparsers(dest="command")

    proc = sub.add_parser("process", help="Process a PDF through the full pipeline.")
    proc.add_argument("pdf_path", type=Path, help="Path to the input PDF file.")
    proc.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output JSONL (defaults to DISTILL_OUTPUT_DIR or ./output).",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "process":
        output_dir = args.output_dir or get_settings().output_dir
        asyncio.run(_run_pipeline(args.pdf_path, output_dir))


if __name__ == "__main__":
    main()
