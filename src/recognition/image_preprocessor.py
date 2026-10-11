import cv2
import numpy as np


class ImagePreprocessor:
    """
    图像预处理器

    主要步骤：
    - 灰度化 + 直方图均衡化（增强对比度）
    - 高斯滤波（去噪）
    - 锐化（增强纹理）
    - 归一化到 [0, 1]
    """

    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        对 ROI 图像进行预处理。

        Args:
            image: 输入 ROI 图像 (H, W, 3, BGR)

        Returns:
            preprocessed: 预处理后的图像 (H, W, 3, float32, [0,1])
        """
        if image is None or image.size == 0:
            raise ValueError("Empty image for preprocessing.")

        # 确保尺寸正确
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)

        # 1. 转灰度并直方图均衡化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)

        # 转回 3 通道以兼容 ResNet 输入
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # 2. 高斯滤波去噪
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # 3. 锐化增强纹理
        kernel = np.array(
            [
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1],
            ],
            dtype=np.float32,
        )
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # 4. 归一化到 [0, 1]
        normalized = sharpened.astype(np.float32) / 255.0
        return normalized

    def normalize_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        将预处理后的图像转换为模型输入格式 (1, 3, H, W)。

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

