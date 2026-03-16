"""Pipeline stages: document processing, synthesis, and quality scoring."""

from distillstack.pipeline.processor import DocumentProcessor
from distillstack.pipeline.quality import QualityScorer
from distillstack.pipeline.synthesis import SynthesisAgent

__all__ = [
    "DocumentProcessor",
    "QualityScorer",
    "SynthesisAgent",
]
