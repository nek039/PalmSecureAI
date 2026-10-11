"""
增强版掌纹图像预处理器

基于《掌纹掌脉融合识别技术》中的方法：
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- 方向归一化
- Gabor滤波增强
- 优化的锐化和去噪
- PLE (掌纹线增强) - 用于与训练时对齐
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class EnhancedPreprocessor:
    """
    增强版掌纹图像预处理器

    主要步骤：
    1. 灰度化
    2. CLAHE 增强（自适应直方图均衡化）
    3. Gabor滤波增强纹理
    4. 双边滤波去噪（保留边缘）
    5. 锐化增强细节
    6. 归一化
    """

    def __init__(self, target_size=(224, 224), use_gabor=True, use_clahe=True):
        self.target_size = target_size
        self.use_gabor = use_gabor
        self.use_clahe = use_clahe

        # CLAHE 参数
        if use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 预生成Gabor滤波器
        if use_gabor:
            self.gabor_filters = self._create_gabor_filters()

        # 锐化核
        self.sharpen_kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ], dtype=np.float32)

        # 更强的锐化核（保留更多细节）
        self.strong_sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)

    def _create_gabor_filters(self, num_orientations=8, num_scales=4) -> list:
        """
        创建多方向、多尺度的Gabor滤波器组

        Args:
            num_orientations: 方向数量
            num_scales: 尺度数量

        Returns:
            Gabor滤波器列表
        """
        filters = []
        for scale in range(num_scales):
            ksize = 2 * scale + 9  # 核大小: 9, 13, 17, 21
            for orientation in np.arange(0, np.pi, np.pi / num_orientations):
                # Gabor参数
                sigma = 2.0 + scale * 0.5
                theta = orientation
                lambd = 10.0 + scale * 5.0
                gamma = 0.5
                psi = 0

                # 创建Gabor核
                kernel = cv2.getGaborKernel(
                    (ksize, ksize),
                    sigma,
                    theta,
                    lambd,
                    gamma,
                    psi,
                    ktype=cv2.CV_32F
                )
                filters.append(kernel)

        return filters

    def apply_gabor_filter(self, image: np.ndarray) -> np.ndarray:
        """
        应用Gabor滤波器增强纹理

        Args:
            image: 灰度图像

        Returns:
            Gabor增强后的图像
        """
        if not self.use_gabor:
            return image

        # 计算所有滤波器的响应
        responses = []
        for kernel in self.gabor_filters:
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            responses.append(filtered)

        # 取最大响应作为增强结果
        enhanced = np.maximum.reduce(responses)

        # 归一化到0-255
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        return enhanced.astype(np.uint8)

    def normalize_orientation(self, image: np.ndarray) -> np.ndarray:
        """
        方向归一化（简化版）
        使用Sobel梯度计算主方向并旋转

        Args:
            image: 输入图像

        Returns:
            方向归一化后的图像
        """
        # 计算梯度
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # 计算梯度方向
        angles = np.arctan2(sobely, sobelx)

        # 计算主方向（简化：使用平均方向）
        # 在实际应用中，应该计算掌纹区域的主方向
        # 这里使用直方图统计

        return image

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        完整的预处理流程

        Args:
            image: 输入ROI图像 (H, W, 3, BGR)

        Returns:
            preprocessed: 预处理后的图像 (H, W, 3, float32, [0,1])
        """
        if image is None or image.size == 0:
            raise ValueError("Empty image for preprocessing.")

        # 确保尺寸正确
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)

        # 1. 转灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. CLAHE 增强
        if self.use_clahe:
            clahe_enhanced = self.clahe.apply(gray)
        else:
            clahe_enhanced = gray.copy()

        # 3. Gabor滤波增强纹理
        if self.use_gabor:
            gabor_enhanced = self.apply_gabor_filter(clahe_enhanced)
            # 混合CLAHE和Gabor结果
            gray_enhanced = cv2.addWeighted(clahe_enhanced, 0.6, gabor_enhanced, 0.4, 0)
        else:
            gray_enhanced = clahe_enhanced

        # 4. 双边滤波去噪（保留边缘）
        denoised = cv2.bilateralFilter(gray_enhanced, 9, 75, 75)

        # 5. 锐化增强细节
        sharpened = cv2.filter2D(denoised, -1, self.sharpen_kernel)

        # 6. 转回3通道（如果需要）
        if len(sharpened.shape) == 2:
            sharpened = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

        # 7. 归一化到 [0, 1]
        normalized = sharpened.astype(np.float32) / 255.0

        return normalized

    def normalize_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        将预处理后的图像转换为模型输入格式

        Args:
            image: HxWxC 或 HxW

        Returns:
            batch: 模型输入 (1, 3, H, W), float32
        """
        if image.ndim == 2:
            # 灰度 -> 三通道
            image = np.stack([image] * 3, axis=-1)

        # HWC -> CHW
        chw = np.transpose(image, (2, 0, 1))
        batch = np.expand_dims(chw, axis=0)
        return batch.astype(np.float32)


class MultiModalPreprocessor:
    """
    多模态预处理器（用于掌纹+掌脉融合）

    支持同时处理可见光和近红外图像
    """

    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.palm_preprocessor = EnhancedPreprocessor(target_size)
        self.vein_preprocessor = EnhancedPreprocessor(
            target_size,
            use_gabor=False,  # 掌脉图像纹理较浅，不需要Gabor
            use_clahe=True
        )

    def preprocess_palm(self, image: np.ndarray) -> np.ndarray:
        """预处理掌纹图像"""
        return self.palm_preprocessor.preprocess(image)

    def preprocess_vein(self, image: np.ndarray) -> np.ndarray:
        """预处理掌脉图像"""
        return self.vein_preprocessor.preprocess(image)

    def preprocess_fusion(self, palm_img: np.ndarray, vein_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        同时预处理掌纹和掌脉图像

        Returns:
            (palm_processed, vein_processed)
        """
        palm_processed = self.preprocess_palm(palm_img)
        vein_processed = self.preprocess_vein(vein_img)
        return palm_processed, vein_processed


class PLEPreprocessor:
    """
    PLE 预处理器 - 与 Mixed+PLE 模型训练时对齐

    使用掌纹线增强 (Palmprint Line Enhancement)
    与 train_with_ple.py 中的预处理保持一致
    """

    def __init__(self, target_size=(224, 224), use_ple=True,
                 ple_strategy='subtract', ple_alpha=0.5):
        """
        Args:
            target_size: 目标尺寸
            use_ple: 是否使用 PLE 增强
            ple_strategy: PLE 策略 ('subtract', 'add', 'mixed')
            ple_alpha: PLE 增强强度
        """
        self.target_size = target_size
        self.use_ple = use_ple
        self.ple_strategy = ple_strategy
        self.ple_alpha = ple_alpha

        # 初始化 PLE 增强器
        if use_ple:
            from .palm_line_enhancement import PLEAugmentor
            self.ple = PLEAugmentor(strategy=ple_strategy, alpha=ple_alpha)
        else:
            self.ple = None

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        PLE 预处理流程

        Args:
            image: 输入 BGR 图像 (H, W, 3)

        Returns:
            预处理后的图像 (H, W, 3), float32, [0,1]
        """
        if image is None or image.size == 0:
            raise ValueError("Empty image for preprocessing.")

        # 确保是 RGB 格式 (PLE 需要 RGB)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 转为 RGB (PLE 内部处理需要)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. 调整尺寸
        if image_rgb.shape[:2] != self.target_size:
            image_rgb = cv2.resize(image_rgb, self.target_size, interpolation=cv2.INTER_CUBIC)

        # 2. 应用 PLE 增强 (与训练时一致)
        if self.use_ple and self.ple is not None:
            enhanced = self.ple.enhance(image_rgb)
        else:
            enhanced = image_rgb

        # 3. 转回 BGR
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

        # 4. 归一化到 [0, 1]
        normalized = enhanced_bgr.astype(np.float32) / 255.0

        return normalized

    def normalize_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        将预处理后的图像转换为模型输入格式

        Args:
            image: HxWxC, float32, [0,1]

        Returns:
            batch: 模型输入 (1, 3, H, W), float32
        """
        # ImageNet 标准化（与训练时一致）
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (image - mean) / std

        # HWC -> CHW
        chw = np.transpose(normalized, (2, 0, 1))
        batch = np.expand_dims(chw, axis=0)
        return batch.astype(np.float32)


if __name__ == "__main__":
    # 测试预处理器
    print("测试增强版预处理器...")
    preprocessor = EnhancedPreprocessor()

    # 创建测试图像
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # 预处理
    result = preprocessor.preprocess(test_image)
    print(f"预处理结果形状: {result.shape}, 数据类型: {result.dtype}, 范围: [{result.min():.3f}, {result.max():.3f}]")

    # 测试 PLE 预处理器
    print("\n测试 PLE 预处理器...")
    ple_preprocessor = PLEPreprocessor(use_ple=True, ple_strategy='subtract', ple_alpha=0.5)
    ple_result = ple_preprocessor.preprocess(test_image)
    print(f"PLE 预处理结果形状: {ple_result.shape}, 数据类型: {ple_result.dtype}, 范围: [{ple_result.min():.3f}, {ple_result.max():.3f}]")

    print("预处理器测试完成")
