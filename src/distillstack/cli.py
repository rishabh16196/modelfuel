"""CLI entry point for the DistillStack pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from distillstack.config import get_settings
from distillstack.pipeline.processor import DocumentProcessor

logger = logging.getLogger(__name__)


async def _run_extract(pdf_path: Path, output_dir: Path) -> None:
    """Extract structured data from a PDF and write the result to disk."""
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

    md_file = output_dir / f"{pdf_path.stem}.md"
    md_file.write_text(document.markdown, encoding="utf-8")
    logger.info("Markdown written to %s", md_file)

    json_file = output_dir / f"{pdf_path.stem}.json"
    json_file.write_text(
        document.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("Structured JSON written to %s", json_file)


def main() -> None:
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        prog="distillstack",
        description="DistillStack: Cognitive ETL Pipeline for SLM dataset synthesis.",
    )
    sub = parser.add_subparsers(dest="command")

    proc = sub.add_parser("extract", help="Extract structured data from a PDF.")
    proc.add_argument("pdf_path", type=Path, help="Path to the input PDF file.")
    proc.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files (defaults to DISTILL_OUTPUT_DIR or ./output).",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "extract":
        output_dir = args.output_dir or get_settings().output_dir
        asyncio.run(_run_extract(args.pdf_path, output_dir))


if __name__ == "__main__":
    main()
