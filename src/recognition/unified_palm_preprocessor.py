# -*- coding: utf-8 -*-
"""
统一掌纹预处理模块

解决 CASIA 和 PolyU 图像风格差异问题：
1. 手掌检测与对齐
2. ROI 提取
3. 统一尺寸和归一化
"""

import cv2
import numpy as np
from pathlib import Path


class UnifiedPalmPreprocessor:
    """统一掌纹预处理器"""

    def __init__(self, output_size=224, padding_ratio=0.1):
        """
        Args:
            output_size: 输出图像尺寸
            padding_ratio: 边缘填充比例
        """
        self.output_size = output_size
        self.padding_ratio = padding_ratio

        # 手掌检测器（使用 Haar 或基于肤色）
        self.use_skin_detection = True

    def detect_palm_hsv(self, image):
        """基于肤色检测手掌区域"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # 肤色范围 (HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower_skin, upper_skin)

        # 另一个肤色范围
        lower_skin2 = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([15, 255, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)

        mask = mask1 | mask2

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 找最大轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 取最大轮廓
        max_contour = max(contours, key=cv2.contourArea)

        # 获取边界框
        x, y, w, h = cv2.boundingRect(max_contour)

        # 添加填充
        pad_w = int(w * self.padding_ratio)
        pad_h = int(h * self.padding_ratio)

        x = max(0, x - pad_w)
        y = max(0, y - pad_h)
        w = min(image.shape[1] - x, w + 2 * pad_w)
        h = min(image.shape[0] - y, h + 2 * pad_h)

        return (x, y, w, h)

    def detect_palm_otsu(self, image):
        """基于 Otsu 阈值检测手掌（适用于对比度高的图像）"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu 阈值
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 如果背景是亮的，反转
        if np.mean(binary) > 127:
            binary = 255 - binary

        # 找最大轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        # 添加填充
        pad_w = int(w * self.padding_ratio)
        pad_h = int(h * self.padding_ratio)

        x = max(0, x - pad_w)
        y = max(0, y - pad_h)
        w = min(image.shape[1] - x, w + 2 * pad_w)
        h = min(image.shape[0] - y, h + 2 * pad_h)

        return (x, y, w, h)

    def extract_roi(self, image):
        """
        提取手掌 ROI

        Args:
            image: RGB 图像

        Returns:
            roi: 提取的 ROI 区域
            bbox: 边界框 (x, y, w, h)
        """
        # 尝试多种方法
        bbox = self.detect_palm_otsu(image)

        if bbox is None or bbox[2] < 50 or bbox[3] < 50:
            bbox = self.detect_palm_hsv(image)

        if bbox is None:
            # 如果都失败，使用中心裁剪
            h, w = image.shape[:2]
            size = min(h, w)
            x = (w - size) // 2
            y = (h - size) // 2
            bbox = (x, y, size, size)

        x, y, bw, bh = bbox
        roi = image[y:y+bh, x:x+bw]

        return roi, bbox

    def align_palm(self, image):
        """
        对齐手掌（基于主方向）

        Args:
            image: RGB 图像

        Returns:
            aligned: 对齐后的图像
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 计算梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # 计算主方向
        angles = np.arctan2(grad_y, grad_x)
        magnitudes = np.sqrt(grad_x**2 + grad_y**2)

        # 加权平均角度
        mask = magnitudes > np.percentile(magnitudes, 50)
        if np.any(mask):
            mean_angle = np.arctan2(
                np.sum(np.sin(angles[mask]) * magnitudes[mask]),
                np.sum(np.cos(angles[mask]) * magnitudes[mask])
            )

            # 旋转图像使主方向垂直
            if abs(mean_angle) > 0.1:  # 大于 5.7 度才旋转
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, -np.degrees(mean_angle), 1)
                image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        return image

    def normalize_illumination(self, image):
        """
        光照归一化

        使用 DoG (Difference of Gaussians) 滤波
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float64)

        # DoG 滤波
        sigma1 = 1.0
        sigma2 = 2.0
        blur1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
        blur2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
        dog = blur1 - blur2

        # 归一化到 0-255
        dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
        dog = dog.astype(np.uint8)

        # 转回 RGB
        result = cv2.cvtColor(dog, cv2.COLOR_GRAY2RGB)

        return result

    def preprocess(self, image, normalize_illum=True, align=True):
        """
        完整预处理流程

        Args:
            image: 输入 RGB 图像
            normalize_illum: 是否进行光照归一化
            align: 是否进行对齐

        Returns:
            processed: 预处理后的图像 (output_size x output_size)
        """
        # 1. 提取 ROI
        roi, bbox = self.extract_roi(image)

        # 2. 对齐（可选）
        if align:
            roi = self.align_palm(roi)

        # 3. 光照归一化（可选）
        if normalize_illum:
            roi = self.normalize_illumination(roi)

        # 4. 调整大小
        processed = cv2.resize(roi, (self.output_size, self.output_size))

        return processed

    def preprocess_batch(self, images, **kwargs):
        """批量预处理"""
        return np.array([self.preprocess(img, **kwargs) for img in images])


def visualize_preprocessing(image_path, output_path=None):
    """可视化预处理效果"""
    import matplotlib.pyplot as plt

    preprocessor = UnifiedPalmPreprocessor(output_size=224)

    # 读取图像
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 各阶段
    roi, bbox = preprocessor.extract_roi(image)
    aligned = preprocessor.align_palm(roi.copy())
    normalized = preprocessor.normalize_illumination(aligned)
    final = cv2.resize(normalized, (224, 224))

    # 可视化
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(image)
    axes[0].set_title('原图')
    axes[0].axis('off')

    # 画边界框
    img_with_bbox = image.copy()
    x, y, w, h = bbox
    cv2.rectangle(img_with_bbox, (x, y), (x+w, y+h), (255, 0, 0), 2)
    axes[1].imshow(img_with_bbox)
    axes[1].set_title('手掌检测')
    axes[1].axis('off')

    axes[2].imshow(roi)
    axes[2].set_title('ROI提取')
    axes[2].axis('off')

    axes[3].imshow(normalized)
    axes[3].set_title('光照归一化')
    axes[3].axis('off')

    axes[4].imshow(final)
    axes[4].set_title('最终结果 (224x224)')
    axes[4].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"保存到: {output_path}")

    plt.show()
    return fig


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 测试图像
        test_images = list(Path('data/casia_850/train').glob('*.jpg'))
        if test_images:
            image_path = str(test_images[0])
        else:
            print("请提供图像路径")
            sys.exit(1)

    print(f"处理图像: {image_path}")
    visualize_preprocessing(image_path, 'preprocessing_example.png')
