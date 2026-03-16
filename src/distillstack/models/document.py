"""Domain models for document representation, synthesis records, and quality verdicts."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class PageMeta(BaseModel):
    """Metadata for a single page extracted from a document."""

    model_config = ConfigDict(strict=True)

    page_number: int
    is_scanned: bool
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_method: Literal["docling", "vlm"]


class ContentBlock(BaseModel):
    """A single structural block extracted from a page."""

    model_config = ConfigDict(strict=True)

    block_type: Literal["heading", "paragraph", "table", "list", "code", "image_desc"]
    content: str
    page_number: int
    metadata: dict[str, Any] = {}


class InternalDocument(BaseModel):
    """Unified representation of a parsed document, merging all extraction sources."""

    model_config = ConfigDict(strict=True)

    source_path: str
    total_pages: int
    pages: list[PageMeta]
    content_blocks: list[ContentBlock]
    markdown: str


class SynthesisRecord(BaseModel):
    """One instruction-tuning row compatible with HuggingFace SFTTrainer."""

    model_config = ConfigDict(strict=True)

    instruction: str
    thought: str
    output: str
    source_file: str
    source_chunk_hash: str

    @staticmethod
    def hash_chunk(chunk: str) -> str:
        """Produce a deterministic short hash for a source chunk."""
        return hashlib.sha256(chunk.encode()).hexdigest()[:16]


class QualityVerdict(BaseModel):
    """Result of the quality audit for a single synthesis record."""

    model_config = ConfigDict(strict=True)

    faithfulness_score: float = Field(ge=0.0, le=1.0)
    complexity_score: float = Field(ge=0.0, le=1.0)
    reasoning: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return self.faithfulness_score >= 0.7 and self.complexity_score >= 0.4
