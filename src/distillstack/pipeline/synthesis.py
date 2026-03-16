"""Grounded synthesis agent: generate instruction-tuning pairs from extracted documents."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING

import litellm

from distillstack.models.document import InternalDocument, SynthesisRecord

if TYPE_CHECKING:
    from distillstack.config import Settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a Teacher creating high-quality training data for a small language model.

Given ONLY the context below, generate one instruction-tuning example.
Respond with a JSON object containing exactly three keys:
- "instruction": a clear, specific question or task derived from the context.
- "thought": a step-by-step reasoning trace that cites which part of the context supports the answer.
- "output": a comprehensive, accurate answer grounded entirely in the context.

Rules:
1. NEVER use information outside the provided context.
2. The instruction must be challenging enough to be useful for model training.
3. The thought must explicitly reference sentences or phrases from the context.
4. Respond ONLY with valid JSON. No markdown fences, no extra text.\
"""


class SynthesisError(Exception):
    """Raised when synthesis generation fails."""


class SynthesisAgent:
    """Generate grounded Instruction -> Thought -> Response triples from a document."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def synthesize(
        self, document: InternalDocument
    ) -> list[SynthesisRecord]:
        """Chunk the document by headers and generate pairs concurrently."""
        chunks = _chunk_by_headers(document.markdown)
        if not chunks:
            logger.warning("No chunks produced for %s", document.source_path)
            return []

        tasks = [
            self._generate_pair(chunk, document.source_path) for chunk in chunks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        records: list[SynthesisRecord] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Chunk %d failed: %s", i, result)
            else:
                records.append(result)
        return records

    async def _generate_pair(
        self, chunk: str, source_file: str
    ) -> SynthesisRecord:
        """Call the teacher LLM to produce a single synthesis record."""
        response = await litellm.acompletion(
            model=self._settings.teacher_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n\n{chunk}"},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or ""
        try:
            data = json.loads(_strip_json_fences(raw))
        except json.JSONDecodeError as exc:
            raise SynthesisError(f"Teacher returned invalid JSON: {raw[:200]}") from exc

        return SynthesisRecord(
            instruction=data["instruction"],
            thought=data["thought"],
            output=data["output"],
            source_file=source_file,
            source_chunk_hash=SynthesisRecord.hash_chunk(chunk),
        )


def _chunk_by_headers(markdown: str) -> list[str]:
    """Split markdown into sections on ``## `` boundaries.

    Each chunk retains its heading line.  Content before the first ``## ``
    heading is included as its own chunk if non-trivial.
    """
    parts = re.split(r"(?=\n## )", markdown)
    chunks: list[str] = []
    for part in parts:
        stripped = part.strip()
        if len(stripped) > 30:
            chunks.append(stripped)
    return chunks


def _strip_json_fences(text: str) -> str:
    """Remove optional ```json ... ``` wrappers that LLMs sometimes add."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()
