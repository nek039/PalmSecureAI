"""
Core package for PalmSecureAI.

Provides abstract interfaces and unified pipeline for palm recognition.
"""

from .interfaces import (
    RecognitionResult,
    Preprocessor,
    FeatureExtractor,
    Matcher,
    ROIExtractor,
)

from .pipeline import PalmRecognizerPipeline, PipelineConfig

__all__ = [
    "RecognitionResult",
    "Preprocessor",
    "FeatureExtractor",
    "Matcher",
    "ROIExtractor",
    "PalmRecognizerPipeline",
    "PipelineConfig",
]
