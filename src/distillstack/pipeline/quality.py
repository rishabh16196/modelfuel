"""Quality scorer (Critic): audit synthesis records for faithfulness and complexity."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING

import litellm

from distillstack.models.document import QualityVerdict, SynthesisRecord

if TYPE_CHECKING:
    from distillstack.config import Settings

logger = logging.getLogger(__name__)

_CRITIC_PROMPT = """\
You are a strict QA auditor for machine-learning training data.

You will receive a SOURCE text and a GENERATED question-answer pair.
Evaluate the pair on two dimensions:

1. **Faithfulness** (0.0 – 1.0): Is the answer fully and accurately supported by the source text?
   - 1.0 = every claim in the answer is directly traceable to the source.
   - 0.0 = the answer fabricates information not present in the source.

2. **Complexity** (0.0 – 1.0): Is the instruction challenging enough to meaningfully train a small language model?
   - 1.0 = requires multi-step reasoning, synthesis, or domain knowledge present in the source.
   - 0.0 = trivially answerable (e.g. "What is the title?").

Respond ONLY with a JSON object containing exactly three keys:
- "faithfulness_score": float
- "complexity_score": float
- "reasoning": a brief explanation of your scores

No markdown fences, no extra text.\
"""

_MAX_CONCURRENT = 10


class QualityScoringError(Exception):
    """Raised when quality scoring fails."""


class QualityScorer:
    """Use a critic LLM to verify faithfulness and complexity of synthesis records."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

    async def score(
        self, record: SynthesisRecord, source_chunk: str
    ) -> QualityVerdict:
        """Score a single record against its source chunk."""
        user_content = (
            f"SOURCE:\n{source_chunk}\n\n"
            f"INSTRUCTION:\n{record.instruction}\n\n"
            f"ANSWER:\n{record.output}"
        )

        async with self._semaphore:
            response = await litellm.acompletion(
                model=self._settings.critic_model,
                messages=[
                    {"role": "system", "content": _CRITIC_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

        raw = response.choices[0].message.content or ""
        try:
            data = json.loads(_strip_json_fences(raw))
        except json.JSONDecodeError as exc:
            raise QualityScoringError(
                f"Critic returned invalid JSON: {raw[:200]}"
            ) from exc

        return QualityVerdict(
            faithfulness_score=float(data["faithfulness_score"]),
            complexity_score=float(data["complexity_score"]),
            reasoning=data.get("reasoning", ""),
        )

    async def score_batch(
        self, records: list[tuple[SynthesisRecord, str]]
    ) -> list[QualityVerdict]:
        """Score multiple records concurrently, respecting the rate-limit semaphore."""
        tasks = [self.score(rec, chunk) for rec, chunk in records]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        verdicts: list[QualityVerdict] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Scoring failed for record %d: %s", i, result)
                verdicts.append(
                    QualityVerdict(
                        faithfulness_score=0.0,
                        complexity_score=0.0,
                        reasoning=f"Scoring error: {result}",
                    )
                )
            else:
                verdicts.append(result)
        return verdicts


def _strip_json_fences(text: str) -> str:
    """Remove optional ```json ... ``` wrappers."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()
