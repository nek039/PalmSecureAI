"""
Unified palm recognition pipeline.

Provides a standard pipeline that wires together ROI extraction,
preprocessing, feature extraction, and template matching.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch

from .interfaces import (
    RecognitionResult,
    Preprocessor,
    FeatureExtractor,
    Matcher,
    ROIExtractor,
)


@dataclass
class PipelineConfig:
    """Configuration for recognition pipeline."""
    match_threshold: float = 0.80
    min_quality_threshold: float = 0.10
    max_samples_per_user: int = 10
    enable_multi_template: bool = True


class PalmRecognizerPipeline:
    """
    Unified palm recognition pipeline.

    Wires together the standard palm recognition components:
    - ROI extraction
    - Image preprocessing
    - Feature extraction
    - Template matching

    Example:
        pipeline = PalmRecognizerPipeline(
            feature_extractor=extractor,
            preprocessor=preprocessor,
            matcher=matcher,
            roi_extractor=roi_extractor,
        )
        result = pipeline.recognize(image)
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        preprocessor: Optional[Preprocessor] = None,
        matcher: Optional[Matcher] = None,
        roi_extractor: Optional[ROIExtractor] = None,
        config: Optional[PipelineConfig] = None,
    ):
        self.feature_extractor = feature_extractor
        self.preprocessor = preprocessor
        self.matcher = matcher
        self.roi_extractor = roi_extractor
        self.config = config or PipelineConfig()
        self._templates: List[Dict[str, Any]] = []

    @property
    def feature_dim(self) -> int:
        return self.feature_extractor.feature_dim

    def extract_feature(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Extract feature vector from an image.

        Steps:
        1. ROI extraction (if available)
        2. Preprocessing (if available)
        3. Feature extraction

        Args:
            image: Input BGR/RGB image

        Returns:
            (feature, quality) tuple. feature is None if extraction fails.
        """
        # Step 1: ROI extraction
        if self.roi_extractor is not None:
            if hasattr(self.roi_extractor, 'extract_roi_with_quality_check'):
                roi, quality = self.roi_extractor.extract_roi_with_quality_check(image)
            else:
                roi = self.roi_extractor.extract_roi(image)
                quality = 0.8 if roi is not None else 0.0
            if roi is None:
                return None, 0.0
        else:
            roi = image
            quality = 0.95

        # Step 2: Preprocessing
        if self.preprocessor is not None:
            processed = self.preprocessor.preprocess(roi)
            batch = self.preprocessor.normalize_for_model(processed)
        else:
            batch = roi

        # Step 3: Feature extraction
        device = next(self.feature_extractor.parameters()).device
        tensor = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            feat = self.feature_extractor(tensor)

        return feat.squeeze(0).cpu().numpy(), quality

    def enroll(self, user_id: str, image: np.ndarray) -> bool:
        """
        Enroll a user with a palm image.

        Args:
            user_id: Unique user identifier
            image: Palm image

        Returns:
            True if enrollment succeeded, False otherwise
        """
        feat, quality = self.extract_feature(image)
        if feat is None or quality < self.config.min_quality_threshold:
            return False

        template = {
            'user_id': user_id,
            'feature': feat,
            'quality': float(quality),
        }
        self._templates.append(template)
        return True

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """
        Recognize a user from a palm image.

        Args:
            image: Palm image

        Returns:
            RecognitionResult with match status and details
        """
        feat, quality = self.extract_feature(image)
        if feat is None:
            return RecognitionResult(
                matched=False,
                user_id=None,
                similarity=0.0,
                confidence='low',
                details=[],
                quality=quality,
                message="Feature extraction failed",
            )

        if self.matcher is None:
            return RecognitionResult(
                matched=False,
                user_id=None,
                similarity=0.0,
                confidence='low',
                details=[],
                quality=quality,
                message="No matcher configured",
            )

        result = self.matcher.match(feat, self._templates)
        result.quality = quality
        return result

    def get_templates(self) -> List[Dict[str, Any]]:
        """Return all enrolled templates."""
        return self._templates

    def clear_templates(self) -> None:
        """Clear all enrolled templates."""
        self._templates.clear()
