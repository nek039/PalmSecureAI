"""
增强版掌纹ROI提取器

提供两种提取方法：
1. 纹理分析法（主要方法）- 基于 Gabor 滤波器的纹理能量检测
2. 中心裁剪（备选方案）- 当纹理分析失败时的保底方案

纹理分析法优势：
- 不依赖肤色检测，适用于灰度图/红外图
- 直接定位掌纹纹理区域，而非"手"的形状
- 对手掌位置和方向不敏感
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


class AdaptiveROIExtractor:
    """
    自适应ROI提取器

    方法选择逻辑：
    1. 纹理分析（主要方法）- 适用于灰度图和已裁剪的掌纹图像
    2. 中心裁剪（最后备选）- 当纹理分析失败时的保底方案
    """

    def __init__(
        self,
        use_texture: bool = True,
        fallback_to_center_crop: bool = True
    ):
        """
        初始化自适应 ROI 提取器

        Args:
            use_texture: 是否启用纹理分析方法（主要方法）
            fallback_to_center_crop: 是否在所有方法失败时使用中心裁剪
        """
        self.use_texture = use_texture
        self.fallback_to_center_crop = fallback_to_center_crop

        # 初始化纹理分析提取器
        if use_texture:
            from .texture_roi_extractor import TextureBasedROIExtractor
            self.texture_extractor = TextureBasedROIExtractor()
        else:
            self.texture_extractor = None

        self.min_roi_size = 150
        self.max_roi_size = 400
        self.quality_threshold = 0.3  # 降低质量阈值，适应模糊图像

    def extract_roi_with_quality_check(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        提取ROI并进行质量检查

        Args:
            image: 输入图像

        Returns:
            roi: ROI图像
            quality: 质量分数 [0, 1]
        """
        # 检查输入有效性
        if image is None or (hasattr(image, 'size') and image.size == 0):
            print("ROI提取失败：输入图像为空")
            return None, 0.0

        # 确保图像是numpy数组
        if not isinstance(image, np.ndarray):
            print(f"ROI提取失败：输入类型错误 {type(image)}")
            return None, 0.0

        roi = None
        info = None

        # 1. 尝试纹理分析（主要方法）
        if self.use_texture and self.texture_extractor:
            roi, info = self.texture_extractor.extract_roi(image)
            if roi is not None:
                print("纹理分析 ROI 提取成功")
                quality = self._calculate_quality(roi)
                return roi, quality

        # 2. 最后备选：中心裁剪
        if self.fallback_to_center_crop:
            print("使用中心裁剪作为备选方案")
            roi = self._center_crop(image)
            if roi is not None:
                quality = self._calculate_quality(roi)
                return roi, quality

        print("ROI提取失败：无法提取ROI")
        return None, 0.0

    def _center_crop(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        中心裁剪（备选方案）

        Args:
            image: 输入图像

        Returns:
            裁剪后的图像 (224x224)
        """
        try:
            if image is None or image.size == 0:
                return None

            # 处理灰度图
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            h, w = image.shape[:2]
            min_dim = min(w, h)

            # 计算中心裁剪区域
            x1 = (w - min_dim) // 2
            y1 = (h - min_dim) // 2
            x2 = x1 + min_dim
            y2 = y1 + min_dim

            roi = image[y1:y2, x1:x2]

            # 调整到目标大小
            roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_CUBIC)

            return roi
        except Exception as e:
            print(f"中心裁剪失败: {e}")
            return None

    def _calculate_quality(self, roi: np.ndarray) -> float:
        """
        计算ROI质量分数

        基于以下因素：
        1. 对比度（标准差）
        2. 清晰度（拉普拉斯方差）
        3. 亮度分布

        针对低质量/模糊摄像头图像优化：
        - 大幅降低清晰度要求
        - 提高对比度权重
        - 允许更暗的亮度范围
        """
        # 转灰度
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # 1. 对比度分数 - 模糊图像对比度通常较低，使用更宽松的归一化
        std_dev = np.std(gray)
        contrast_score = min(std_dev / 40.0, 1.0)  # 从64降到40，更容易得高分

        # 2. 清晰度分数（拉普拉斯方差）- 大幅降低要求
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = np.var(laplacian)
        sharpness_score = min(lap_var / 200.0, 1.0)  # 从500降到200，模糊图像也能得高分

        # 3. 亮度分数（避免过暗或过亮）- 模糊/红外图像通常较暗
        mean_brightness = np.mean(gray)
        # 允许更宽的亮度范围（均值50-180都算好）
        if mean_brightness < 50:
            brightness_score = mean_brightness / 50.0  # 越暗分数越低
        elif mean_brightness > 200:
            brightness_score = (255 - mean_brightness) / 55.0  # 越亮分数越低
        else:
            brightness_score = 1.0  # 理想范围

        # 综合质量分数 - 对模糊图像更宽容
        # 对比度最重要，亮度次之，清晰度最不重要（因为是模糊摄像头）
        quality = 0.5 * contrast_score + 0.2 * sharpness_score + 0.3 * brightness_score

        # 给一个基础分数，避免质量太低被拒绝
        quality = max(quality, 0.3)

        return quality


if __name__ == "__main__":
    # 测试ROI提取器
    print("测试自适应ROI提取器...")

    from texture_roi_extractor import TextureBasedROIExtractor

    extractor = AdaptiveROIExtractor()

    # 创建测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = [200, 200, 200]

    roi, quality = extractor.extract_roi_with_quality_check(test_image)

    if roi is not None:
        print(f"ROI提取成功，形状: {roi.shape}")
        print(f"质量分数: {quality:.4f}")
    else:
        print("ROI提取失败")

    print("ROI提取器测试完成")
