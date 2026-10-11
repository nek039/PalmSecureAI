"""
训练模块

提供模型训练功能。
"""

from .training import (
    TripletLoss,
    Trainer,
    WarmupCosineScheduler,
)

__all__ = [
    "TripletLoss",
    "Trainer",
    "WarmupCosineScheduler",
]
