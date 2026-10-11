"""
统一预处理模块 - 训练和推理使用相同的预处理流程

核心流程：
1. 灰度转换 + 直方图均衡化（增强对比度）
2. 高斯滤波（去噪）
3. 锐化（增强纹理）
4. 归一化到 [0, 1]

训练时可选的数据增强：
- 随机旋转
- 亮度/对比度调整
- 高斯噪声
"""

from __future__ import annotations

from typing import Tuple, Optional

import cv2
import numpy as np
import torch


class PalmUnifiedPreprocessor:
    """
    统一的掌纹图像预处理器

    用于训练和推理，确保输入分布一致
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        augment_config: Optional[dict] = None,
    ):
        """
        初始化预处理器

        Args:
            target_size: 目标图像尺寸 (H, W)
            augment: 是否启用数据增强（训练时启用）
            augment_config: 数据增强配置
        """
        self.target_size = target_size
        self.augment = augment

        # 默认增强配置
        self.augment_config = augment_config or {
            'rotation_limit': 15,       # 旋转角度范围
            'brightness_limit': 0.2,    # 亮度调整范围
            'contrast_limit': 0.2,      # 对比度调整范围
            'noise_var': 20,            # 噪声方差
            'flip_prob': 0.5,           # 水平翻转概率
            'augment_prob': 0.5,        # 应用增强的概率
        }

        # 锐化卷积核
        self.sharpen_kernel = np.array(
            [[-1, -1, -1],
             [-1,  9, -1],
             [-1, -1, -1]],
            dtype=np.float32
        )

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        对图像进行预处理

        Args:
            image: 输入图像 (H, W, 3) BGR格式

        Returns:
            preprocessed: 预处理后的图像 (H, W, 3) float32 [0, 1]
        """
        if image is None or image.size == 0:
            raise ValueError("Empty image for preprocessing.")

        # 确保尺寸正确
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)

        # 1. 转灰度并直方图均衡化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)

        # 转回 3 通道以兼容模型输入
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # 2. 数据增强（仅在训练时）
        if self.augment:
            enhanced = self._apply_augmentation(enhanced)

        # 3. 高斯滤波去噪
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # 4. 锐化增强纹理
        sharpened = cv2.filter2D(denoised, -1, self.sharpen_kernel)

        # 5. 归一化到 [0, 1]
        normalized = sharpened.astype(np.float32) / 255.0

        return normalized

    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """应用数据增强"""
        config = self.augment_config

        # 随机决定是否应用增强
        if np.random.random() > config['augment_prob']:
            return image

        # 随机旋转
        if np.random.random() < 0.5:
            angle = np.random.uniform(-config['rotation_limit'], config['rotation_limit'])
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        # 随机水平翻转（左右手镜像）
        if np.random.random() < config['flip_prob']:
            image = cv2.flip(image, 1)

        # 随机亮度/对比度调整
        if np.random.random() < 0.5:
            brightness = np.random.uniform(-config['brightness_limit'], config['brightness_limit'])
            contrast = np.random.uniform(1 - config['contrast_limit'], 1 + config['contrast_limit'])

            image = image.astype(np.float32)
            image = image * contrast + brightness * 255
            image = np.clip(image, 0, 255).astype(np.uint8)

        # 随机高斯噪声
        if np.random.random() < 0.3:
            noise = np.random.normal(0, config['noise_var'], image.shape).astype(np.float32)
            image = image.astype(np.float32) + noise
            image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理并转换为 Tensor

        Args:
            image: 输入图像 (H, W, 3) BGR格式

        Returns:
            tensor: (3, H, W) float32 tensor
        """
        preprocessed = self.preprocess(image)

        # HWC -> CHW
        tensor = torch.from_numpy(preprocessed).permute(2, 0, 1)

        return tensor

    def normalize_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        将预处理后的图像转换为模型输入格式 (1, 3, H, W)

        Args:
            image: HxWxC 或 HxW

        Returns:
            batch: 模型输入 (1, 3, H, W), float32
        """
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # HWC -> CHW -> NCHW
        chw = np.transpose(image, (2, 0, 1))
        batch = np.expand_dims(chw, axis=0)
        return batch.astype(np.float32)


class PalmTrainingTransform:
    """
    训练时使用的预处理 Transform

    兼容 torchvision.transforms 接口
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = True,
    ):
        self.preprocessor = PalmUnifiedPreprocessor(
            target_size=image_size,
            augment=augment,
        )

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        应用预处理

        Args:
            image: 输入图像 (H, W, 3) BGR格式

        Returns:
            tensor: (3, H, W) float32 tensor
        """
        return self.preprocessor(image)


class PalmInferenceTransform:
    """
    推理时使用的预处理 Transform（无数据增强）
    """

    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.preprocessor = PalmUnifiedPreprocessor(
            target_size=image_size,
            augment=False,
        )

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        return self.preprocessor(image)


# 测试代码
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("统一预处理模块测试")
    print("=" * 60)

    # 测试预处理
    preprocessor = PalmUnifiedPreprocessor(augment=False)

    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 预处理
    result = preprocessor(test_image)
    print(f"输入形状: {test_image.shape}")
    print(f"输出形状: {result.shape}")
    print(f"输出范围: [{result.min():.4f}, {result.max():.4f}]")

    # 测试训练 Transform
    train_transform = PalmTrainingTransform(augment=True)
    tensor = train_transform(test_image)
    print(f"\n训练 Transform 输出: {tensor.shape}, dtype={tensor.dtype}")

    # 测试推理 Transform
    infer_transform = PalmInferenceTransform()
    tensor = infer_transform(test_image)
    print(f"推理 Transform 输出: {tensor.shape}, dtype={tensor.dtype}")

    print("\n测试完成!")
