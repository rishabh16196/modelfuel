"""Multi-modal document processor: Docling primary extraction with VLM fallback."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from docling.document_converter import DocumentConverter

from distillstack.models.document import ContentBlock, InternalDocument, PageMeta

if TYPE_CHECKING:
    from docling.datamodel.document import DoclingDocument

    from distillstack.config import Settings

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    """Raised when document processing fails irrecoverably."""


class DocumentProcessor:
    """Ingest a PDF via Docling, routing low-confidence / scanned pages to a VLM."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._converter = DocumentConverter()

    async def process(self, pdf_path: Path) -> InternalDocument:
        """Run the full extraction pipeline on a single PDF.

        1. Convert via Docling (offloaded to a thread).
        2. Assess per-page confidence; route weak pages to VLM.
        3. Merge into a unified InternalDocument.
        """
        if not pdf_path.exists():
            raise DocumentProcessingError(f"PDF not found: {pdf_path}")

        result = await asyncio.to_thread(self._converter.convert, str(pdf_path))
        doc: DoclingDocument = result.document

        pages: list[PageMeta] = []
        content_blocks: list[ContentBlock] = []

        page_count = len(doc.pages) if doc.pages else 1

        for page_no in range(1, page_count + 1):
            confidence, is_scanned = self._assess_page(doc, page_no)

            needs_vlm = (
                is_scanned or confidence < self._settings.vlm_confidence_threshold
            )

            if needs_vlm:
                logger.info(
                    "Page %d routed to VLM (confidence=%.2f, scanned=%s)",
                    page_no,
                    confidence,
                    is_scanned,
                )
                vlm_blocks = await self._process_with_vlm(b"", page_no)
                content_blocks.extend(vlm_blocks)
                pages.append(
                    PageMeta(
                        page_number=page_no,
                        is_scanned=is_scanned,
                        confidence=confidence,
                        extraction_method="vlm",
                    )
                )
            else:
                page_blocks = self._extract_blocks_for_page(doc, page_no)
                content_blocks.extend(page_blocks)
                pages.append(
                    PageMeta(
                        page_number=page_no,
                        is_scanned=False,
                        confidence=confidence,
                        extraction_method="docling",
                    )
                )

        markdown = doc.export_to_markdown()

        return InternalDocument(
            source_path=str(pdf_path),
            total_pages=page_count,
            pages=pages,
            content_blocks=content_blocks,
            markdown=markdown,
        )

    async def _process_with_vlm(
        self, page_image: bytes, page_number: int
    ) -> list[ContentBlock]:
        """Placeholder for VLM enrichment (e.g. DeepSeek-OCR 2).

        In production this would encode ``page_image`` as base64, send it to
        ``litellm.acompletion`` with a vision-capable model, and parse the
        structured output into ContentBlocks.
        """
        logger.warning(
            "VLM processing not yet implemented; skipping page %d", page_number
        )
        return []

    @staticmethod
    def _assess_page(doc: DoclingDocument, page_no: int) -> tuple[float, bool]:
        """Derive a confidence score and scan-detection flag for a page.

        Returns (confidence, is_scanned).  When Docling metadata isn't
        available we default to high confidence / not scanned so the page
        is processed through the standard path.
        """
        try:
            page_key = page_no
            if doc.pages and page_key in doc.pages:
                page = doc.pages[page_key]
                size = getattr(page, "size", None)
                has_text_layer = size is not None
                return (0.95 if has_text_layer else 0.3, not has_text_layer)
        except (KeyError, AttributeError, TypeError):
            pass
        return 0.95, False

    @staticmethod
    def _extract_blocks_for_page(
        doc: DoclingDocument, page_no: int
    ) -> list[ContentBlock]:
        """Walk the Docling document tree and collect blocks belonging to *page_no*."""
        blocks: list[ContentBlock] = []

        if doc.texts:
            for item in doc.texts:
                prov = getattr(item, "prov", None)
                if prov:
                    item_pages = [p.page_no for p in prov if hasattr(p, "page_no")]
                    if item_pages and page_no not in item_pages:
                        continue

                label = getattr(item, "label", "paragraph")
                block_type = _map_label_to_block_type(str(label))
                text = getattr(item, "text", "") or ""
                if text.strip():
                    blocks.append(
                        ContentBlock(
                            block_type=block_type,
                            content=text,
                            page_number=page_no,
                        )
                    )

        if doc.tables:
            for table in doc.tables:
                prov = getattr(table, "prov", None)
                if prov:
                    item_pages = [p.page_no for p in prov if hasattr(p, "page_no")]
                    if item_pages and page_no not in item_pages:
                        continue

                table_md = ""
                export = getattr(table, "export_to_markdown", None)
                if callable(export):
                    table_md = export()
                elif hasattr(table, "text"):
                    table_md = table.text or ""

                if table_md.strip():
                    blocks.append(
                        ContentBlock(
                            block_type="table",
                            content=table_md,
                            page_number=page_no,
                        )
                    )

        return blocks


def _map_label_to_block_type(label: str) -> str:
    """Map a Docling element label to our ContentBlock.block_type enum."""
    label_lower = label.lower()
    if "head" in label_lower or "title" in label_lower:
        return "heading"
    if "list" in label_lower:
        return "list"
    if "code" in label_lower:
        return "code"
    if "table" in label_lower:
        return "table"
    if "image" in label_lower or "figure" in label_lower or "picture" in label_lower:
        return "image_desc"
    return "paragraph"
