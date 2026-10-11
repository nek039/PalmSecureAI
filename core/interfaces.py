"""
Abstract interfaces for palm recognition components.

These Protocol classes define the contracts that each component must follow,
enabling pluggable implementations and easier testing.
"""

from typing import Protocol, Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class RecognitionResult:
    """
    Standardized recognition result returned by matchers.
    """
    matched: bool
    user_id: Optional[str]
    similarity: float
    confidence: str  # 'high' | 'medium' | 'low'
    details: List[Tuple[str, float]]  # [(user_id, similarity), ...]
    quality: float = 0.0
    message: str = ""


class Preprocessor(Protocol):
    """
    Protocol for image preprocessing.

    All preprocessors must implement:
    - preprocess(image): returns normalized image [0,1]
    - normalize_for_model(image): returns batch tensor (1, 3, H, W)
    """

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess palm image and return normalized image."""
        ...

    def normalize_for_model(self, image: np.ndarray) -> np.ndarray:
        """Convert preprocessed image to model input batch tensor."""
        ...


class FeatureExtractor(Protocol):
    """
    Protocol for feature extraction.

    All extractors must implement:
    - feature_dim property: output feature dimension
    - forward(x): extract features from batch tensor
    - extract(image): extract features from single image (optional convenience method)
    """

    @property
    def feature_dim(self) -> int:
        """Output feature dimension."""
        ...

    def forward(self, x) -> np.ndarray:
        """Extract features from batch tensor."""
        ...

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from a single preprocessed image.
        Optional convenience method - implementations may raise NotImplementedError.
        """
        raise NotImplementedError


class Matcher(Protocol):
    """
    Protocol for template matching.

    All matchers must implement:
    - threshold property: matching threshold
    - match(query_feature, templates): match query against template library
    """

    @property
    def threshold(self) -> float:
        """Matching threshold."""
        ...

    def match(
        self,
        query_feature: np.ndarray,
        templates: List[Dict[str, Any]],
    ) -> RecognitionResult:
        """
        Match query feature against template library.

        Args:
            query_feature: Feature vector from the query image
            templates: List of template dicts with 'user_id' and 'feature' keys

        Returns:
            RecognitionResult with match status and details
        """
        ...


class ROIExtractor(Protocol):
    """
    Protocol for Region of Interest extraction.

    All ROI extractors must implement:
    - extract_roi(image): extract palm ROI from image
    - extract_roi_with_quality_check(image): extract ROI with quality score (optional)
    """

    def extract_roi(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract palm ROI from image. Returns None if extraction fails."""
        ...

    def extract_roi_with_quality_check(
        self,
        image: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Extract ROI and return quality score.

        Returns:
            (roi, quality_score) where quality_score is in [0, 1]
        """
        raise NotImplementedError
