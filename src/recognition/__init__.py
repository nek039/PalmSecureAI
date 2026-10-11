"""
PalmSecureAI 本地掌纹识别模块包

入口:
    PalmRecognizer - 端到端掌纹识别器 (palm_recognizer.py)

核心组件:
    - feature_extractor: 基于 ResNet18 的特征提取 (512/256/128维)
    - database: SQLite 模板存储
    - multi_template_matcher: 多模板匹配

预处理 (由 PalmRecognizer 根据配置选择):
    - PLEPreprocessor: 掌纹线增强预处理 (训练时使用 PLE 模型)
    - EnhancedPreprocessor: CLAHE + Gabor 增强预处理
    - ImagePreprocessor: 简单预处理 (fallback)

ROI 提取:
    - AdaptiveROIExtractor: 自适应 ROI 提取
    - TextureBasedROIExtractor: 基于纹理分析的 ROI 提取

辅助:
    - losses: 损失函数 (ArcFace, Triplet, CosFace)
    - palm_line_enhancement: PLE 掌纹线增强实现
"""

from .palm_recognizer import PalmRecognizer
from .feature_extractor import create_feature_extractor
from .database import PalmDatabase
from .multi_template_matcher import MultiTemplateMatcher
from .losses import ArcFaceLoss, CosFaceLoss, TripletLoss, BatchHardTripletLoss
