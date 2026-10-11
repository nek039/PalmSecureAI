# -*- coding: utf-8 -*-
"""
掌纹线增强 (Palmprint Line Enhancement - PLE)

基于论文: "使用掌纹线对基于深度学习的掌纹识别进行数据增强"
核心思路: 通过 MFRAT + Gabor 滤波器提取掌纹主线，用于数据增强
"""

import cv2
import numpy as np
from scipy import ndimage


class PalmLineExtractor:
    """掌纹线提取器 - 使用 MFRAT + Gabor"""

    def __init__(self, mfrat_scales=[3, 5, 7, 9], gabor_params=None):
        """
        Args:
            mfrat_scales: MFRAT 多尺度参数
            gabor_params: Gabor 滤波器参数
        """
        self.mfrat_scales = mfrat_scales
        self.gabor_params = gabor_params or {
            'ksize': (31, 31),
            'sigma': 4.0,
            'lambd': 10.0,
            'gamma': 0.5,
            'psi': 0,
            'angles': [0, np.pi/4, np.pi/2, 3*np.pi/4]
        }

    def mfrat(self, image, scale):
        """
        Modified Finite Radon Transform (MFRAT)

        计算图像在不同方向上的径向对称响应
        """
        h, w = image.shape
        n = scale * 2 + 1

        # 创建径向模板
        result = np.zeros_like(image, dtype=np.float64)

        for angle_idx in range(8):
            angle = angle_idx * np.pi / 8

            # 计算该方向上的响应
            dx = np.cos(angle)
            dy = np.sin(angle)

            # 在该方向上进行积分
            for i in range(-scale, scale + 1):
                for j in range(-scale, scale + 1):
                    # 检查是否在径向线上
                    if abs(j * dx - i * dy) < 1.5:
                        y_shift = int(round(i))
                        x_shift = int(round(j))

                        # 移动图像并累加
                        shifted = np.roll(image, (y_shift, x_shift), axis=(0, 1))
                        result += shifted

        # 归一化
        result = result / (2 * scale + 1)
        return result

    def multi_scale_mfrat(self, image):
        """多尺度 MFRAT"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.float64)

        # 归一化到 0-1
        gray = gray.astype(np.float64) / 255.0

        mfrat_responses = []
        for scale in self.mfrat_scales:
            response = self.mfrat(gray, scale)
            mfrat_responses.append(response)

        # 融合多尺度响应
        mfrat_result = np.mean(mfrat_responses, axis=0)
        return mfrat_result

    def gabor_filter_bank(self, image):
        """Gabor 滤波器组"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        gray = gray.astype(np.float64)

        params = self.gabor_params
        gabor_responses = []

        for theta in params['angles']:
            kernel = cv2.getGaborKernel(
                ksize=params['ksize'],
                sigma=params['sigma'],
                theta=theta,
                lambd=params['lambd'],
                gamma=params['gamma'],
                psi=params['psi']
            )

            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            gabor_responses.append(filtered)

        # 取最大响应
        gabor_result = np.max(np.abs(gabor_responses), axis=0)
        return gabor_result

    def extract_palm_lines(self, image):
        """
        提取掌纹线

        Args:
            image: 输入图像 (RGB 或灰度)

        Returns:
            palm_lines: 掌纹线图像
        """
        # 第一阶段: MFRAT 提取主方向线
        mfrat_lines = self.multi_scale_mfrat(image)

        # 第二阶段: Gabor 滤波增强细节
        gabor_lines = self.gabor_filter_bank(image)

        # 融合
        palm_lines = 0.5 * mfrat_lines + 0.5 * gabor_lines

        # 归一化
        palm_lines = (palm_lines - palm_lines.min()) / (palm_lines.max() - palm_lines.min() + 1e-7)
        palm_lines = (palm_lines * 255).astype(np.uint8)

        return palm_lines


class PLEAugmentor:
    """
    PLE 数据增强器

    支持多种增强策略:
    1. subtract: 增强图像 = 原图 - 掌纹线
    2. add: 增强图像 = 原图 + 掌纹线
    3. concat: 将增强图像作为额外通道
    4. mixed: 随机选择策略
    """

    def __init__(self, strategy='subtract', alpha=0.5):
        """
        Args:
            strategy: 增强策略 ('subtract', 'add', 'concat', 'mixed')
            alpha: 增强强度 (0-1)
        """
        self.strategy = strategy
        self.alpha = alpha
        self.extractor = PalmLineExtractor()

    def enhance(self, image):
        """
        应用 PLE 增强

        Args:
            image: 输入 RGB 图像 (H, W, 3), 值域 0-255

        Returns:
            enhanced: 增强后的图像
        """
        # 提取掌纹线
        palm_lines = self.extractor.extract_palm_lines(image)

        # 转换为 float
        image_float = image.astype(np.float64)

        # 扩展掌纹线到 3 通道
        if len(palm_lines.shape) == 2:
            palm_lines_3ch = np.stack([palm_lines] * 3, axis=-1)
        else:
            palm_lines_3ch = palm_lines

        strategy = self.strategy
        if strategy == 'mixed':
            strategy = np.random.choice(['subtract', 'add', 'original'])

        if strategy == 'subtract':
            # 论文推荐: X = X_ROI - X_ROI_L
            enhanced = image_float - self.alpha * palm_lines_3ch
            enhanced = np.clip(enhanced, 0, 255)
        elif strategy == 'add':
            enhanced = image_float + self.alpha * palm_lines_3ch
            enhanced = np.clip(enhanced, 0, 255)
        elif strategy == 'concat':
            # 返回 4 通道 (R, G, B, palm_lines)
            enhanced = np.dstack([image, palm_lines])
            return enhanced.astype(np.uint8)
        else:  # original
            enhanced = image_float

        return enhanced.astype(np.uint8)

    def augment_batch(self, images, apply_prob=0.5):
        """
        批量增强

        Args:
            images: 图像批次 (N, H, W, 3)
            apply_prob: 应用增强的概率

        Returns:
            augmented: 增强后的批次
        """
        augmented = []
        for img in images:
            if np.random.random() < apply_prob:
                img = self.enhance(img)
            augmented.append(img)
        return np.array(augmented)


def visualize_ple(image_path, output_path=None):
    """可视化 PLE 效果"""
    import matplotlib.pyplot as plt

    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 创建增强器
    augmentor = PLEAugmentor(strategy='subtract', alpha=0.5)

    # 提取掌纹线
    palm_lines = augmentor.extractor.extract_palm_lines(image)

    # 增强
    enhanced = augmentor.enhance(image)

    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image)
    axes[0].set_title('原图')
    axes[0].axis('off')

    axes[1].imshow(palm_lines, cmap='gray')
    axes[1].set_title('掌纹线提取')
    axes[1].axis('off')

    axes[2].imshow(enhanced)
    axes[2].set_title('PLE 增强 (subtract)')
    axes[2].axis('off')

    # 差异图
    diff = np.abs(image.astype(np.float64) - enhanced.astype(np.float64)).astype(np.uint8)
    axes[3].imshow(diff)
    axes[3].set_title('差异图')
    axes[3].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"保存到: {output_path}")

    plt.show()
    return fig


if __name__ == '__main__':
    import sys
    from pathlib import Path

    # 测试
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 使用示例图像
        test_images = list(Path('data/processed_polyu/train').glob('*.png'))
        if test_images:
            image_path = str(test_images[0])
        else:
            print("请提供图像路径或确保 data/processed_polyu/train 中有图像")
            sys.exit(1)

    print(f"处理图像: {image_path}")
    visualize_ple(image_path, 'ple_example.png')
