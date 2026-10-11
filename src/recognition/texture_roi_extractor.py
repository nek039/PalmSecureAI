"""
基于纹理分析的掌纹 ROI 提取器

核心思路：
- 掌纹区域有丰富的纹理（主线、皱褶、细节线）
- 使用 Gabor 滤波器计算纹理能量图
- 通过阈值分割找到纹理丰富区域
- 提取最大连通区域作为掌纹 ROI

适用于：
- 灰度图/红外图（不依赖肤色检测）
- 已裁剪或未裁剪的掌纹图像
- 手掌位置和方向不确定的场景
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


class TextureBasedROIExtractor:
    """
    基于纹理分析的掌纹 ROI 提取器

    方案 A：纹理能量图 + 最大连通区域
    1. 使用多方向 Gabor 滤波器计算纹理能量
    2. 阈值分割分离纹理区域
    3. 找到最大连通区域
    4. 提取 ROI
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        num_orientations: int = 8,
        num_scales: int = 4,
        texture_threshold: Optional[float] = None,  # None 表示使用 Otsu 自动阈值
        min_region_area: float = 0.1,  # 最小区域占比
    ):
        """
        初始化纹理分析 ROI 提取器

        Args:
            target_size: 目标 ROI 尺寸
            num_orientations: Gabor 滤波器方向数
            num_scales: Gabor 滤波器尺度数
            texture_threshold: 纹理能量阈值，None 则使用 Otsu 自动计算
            min_region_area: 最小有效区域的面积占比（相对于图像总面积）
        """
        self.target_size = target_size
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.texture_threshold = texture_threshold
        self.min_region_area = min_region_area

        # 预生成 Gabor 滤波器组
        self.gabor_filters = self._create_gabor_filters()

    def _create_gabor_filters(self) -> list:
        """
        创建多方向、多尺度的 Gabor 滤波器组

        Returns:
            Gabor 滤波器核列表
        """
        filters = []
        for scale in range(self.num_scales):
            ksize = 2 * scale + 9  # 核大小: 9, 13, 17, 21
            for i in range(self.num_orientations):
                theta = i * np.pi / self.num_orientations

                # Gabor 参数
                sigma = 2.0 + scale * 0.5
                lambd = 10.0 + scale * 5.0
                gamma = 0.5
                psi = 0

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

    def extract_roi(
        self,
        image: np.ndarray,
        return_info: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        从图像中提取掌纹 ROI

        Args:
            image: 输入图像（BGR 或灰度）
            return_info: 是否返回提取信息

        Returns:
            roi: 提取的 ROI 图像 (224x224)，失败时为 None
            info: 提取信息字典，包含中心点、方法等
        """
        if image is None or image.size == 0:
            return None, None

        # 1. 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape[:2]

        # 2. 计算纹理能量图
        texture_map = self._compute_texture_energy(gray)

        if texture_map is None:
            return None, None

        # 3. 阈值分割
        binary = self._threshold_texture(texture_map)

        # 4. 形态学处理（去噪、填充空洞）
        binary = self._morphology_cleanup(binary)

        # 5. 找到最大连通区域
        center, region_info = self._find_largest_region_center(binary)

        if center is None:
            # 没有找到有效区域，尝试中心裁剪
            center = (w // 2, h // 2)
            region_info = {'area_ratio': 0.0, 'fallback': True}

        # 6. 计算自适应 ROI 大小
        roi_size = self._calculate_roi_size(binary, center, region_info)

        # 7. 提取 ROI
        roi = self._extract_roi_centered(image, center, roi_size, w, h)

        if roi is None:
            # 最后备选：中心裁剪
            roi = self._center_crop(image)

        info = {
            'center': center,
            'roi_size': roi_size,
            'method': 'texture',
            'region_info': region_info,
            'texture_map': texture_map if return_info else None,
            'binary': binary if return_info else None
        }

        return roi, info

    def _compute_texture_energy(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        使用 Gabor 滤波器组计算纹理能量图

        Args:
            gray: 灰度图像

        Returns:
            纹理能量图（高值表示纹理丰富区域）
        """
        try:
            # 确保图像是 uint8
            if gray.dtype != np.uint8:
                gray = (gray * 255).astype(np.uint8) if gray.max() <= 1 else gray.astype(np.uint8)

            # 计算所有 Gabor 滤波器的响应能量
            energy = np.zeros_like(gray, dtype=np.float32)

            for kernel in self.gabor_filters:
                # 应用滤波器
                response = cv2.filter2D(gray, cv2.CV_32F, kernel)
                # 累加能量（取绝对值）
                energy += np.abs(response)

            # 归一化到 0-255
            energy = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX)

            return energy.astype(np.uint8)

        except Exception as e:
            print(f"计算纹理能量失败: {e}")
            return None

    def _threshold_texture(self, texture_map: np.ndarray) -> np.ndarray:
        """
        对纹理能量图进行阈值分割

        Args:
            texture_map: 纹理能量图

        Returns:
            二值图像（纹理区域为白色）
        """
        if self.texture_threshold is not None:
            # 使用固定阈值
            _, binary = cv2.threshold(
                texture_map,
                self.texture_threshold * 255,
                255,
                cv2.THRESH_BINARY
            )
        else:
            # 使用 Otsu 自适应阈值
            _, binary = cv2.threshold(
                texture_map,
                0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        return binary

    def _morphology_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """
        形态学处理：去噪、填充空洞

        Args:
            binary: 二值图像

        Returns:
            处理后的二值图像
        """
        # 定义核
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 闭运算：填充小空洞
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)

        # 开运算：去除小噪点
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small)

        # 再次闭运算：连接相邻区域
        result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium)

        return result

    def _find_largest_region_center(
        self,
        binary: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int]], Dict[str, Any]]:
        """
        找到最大连通区域并计算其中心点

        Args:
            binary: 二值图像

        Returns:
            center: 中心点坐标 (x, y)
            region_info: 区域信息
        """
        # 查找轮廓
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, {'area_ratio': 0.0, 'num_regions': 0}

        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)

        # 计算面积占比
        total_area = binary.shape[0] * binary.shape[1]
        area_ratio = largest_area / total_area

        # 检查是否满足最小区域要求
        if area_ratio < self.min_region_area:
            # 尝试取前几个大轮廓的并集
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            combined_mask = np.zeros_like(binary)

            for contour in sorted_contours[:3]:  # 取前3个
                if cv2.contourArea(contour) / total_area >= self.min_region_area / 3:
                    cv2.drawContours(combined_mask, [contour], -1, 255, -1)

            if np.sum(combined_mask > 0) / total_area < self.min_region_area:
                return None, {'area_ratio': area_ratio, 'num_regions': len(contours)}

            # 使用合并后的掩码
            largest_contour = max(
                cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                key=cv2.contourArea
            )
            largest_area = cv2.contourArea(largest_contour)
            area_ratio = largest_area / total_area

        # 计算轮廓的质心
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            # 使用外接矩形中心
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x = x + w // 2
            center_y = y + h // 2

        region_info = {
            'area_ratio': area_ratio,
            'num_regions': len(contours),
            'contour_area': largest_area,
            'bounding_rect': cv2.boundingRect(largest_contour)
        }

        return (center_x, center_y), region_info

    def _calculate_roi_size(
        self,
        binary: np.ndarray,
        center: Tuple[int, int],
        region_info: Dict[str, Any]
    ) -> int:
        """
        计算自适应 ROI 大小

        Args:
            binary: 二值图像
            center: 中心点
            region_info: 区域信息

        Returns:
            ROI 尺寸（正方形边长）
        """
        h, w = binary.shape[:2]
        min_dim = min(w, h)

        if 'bounding_rect' in region_info:
            # 基于外接矩形大小计算
            rx, ry, rw, rh = region_info['bounding_rect']
            rect_size = max(rw, rh)

            # ROI 大小为外接矩形的 1.2-1.5 倍
            roi_size = int(rect_size * 1.3)
        else:
            # 基于图像尺寸计算
            roi_size = int(min_dim * 0.7)

        # 限制在合理范围内
        roi_size = max(150, min(roi_size, min_dim - 20))

        return roi_size

    def _extract_roi_centered(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        roi_size: int,
        img_w: int,
        img_h: int
    ) -> Optional[np.ndarray]:
        """
        以中心点为基准提取正方形 ROI

        Args:
            image: 输入图像
            center: 中心点 (x, y)
            roi_size: ROI 大小
            img_w, img_h: 图像尺寸

        Returns:
            ROI 图像 (224x224)
        """
        center_x, center_y = center
        half_size = roi_size // 2

        # 计算边界
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(img_w, center_x + half_size)
        y2 = min(img_h, center_y + half_size)

        # 提取 ROI
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        # 如果提取的区域小于目标大小，进行填充
        if roi.shape[0] < roi_size or roi.shape[1] < roi_size:
            # 填充到正方形
            pad_h = roi_size - roi.shape[0]
            pad_w = roi_size - roi.shape[1]

            if len(roi.shape) == 3:
                roi = cv2.copyMakeBorder(
                    roi,
                    pad_h // 2, pad_h - pad_h // 2,
                    pad_w // 2, pad_w - pad_w // 2,
                    cv2.BORDER_REPLICATE
                )
            else:
                roi = cv2.copyMakeBorder(
                    roi,
                    pad_h // 2, pad_h - pad_h // 2,
                    pad_w // 2, pad_w - pad_w // 2,
                    cv2.BORDER_REPLICATE
                )

        # 调整到目标大小
        roi = cv2.resize(roi, self.target_size, interpolation=cv2.INTER_CUBIC)

        return roi

    def _center_crop(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        中心裁剪（最后备选方案）

        Args:
            image: 输入图像

        Returns:
            中心裁剪的 ROI
        """
        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]
        min_dim = min(w, h)

        # 计算中心裁剪区域
        x1 = (w - min_dim) // 2
        y1 = (h - min_dim) // 2
        x2 = x1 + min_dim
        y2 = y1 + min_dim

        roi = image[y1:y2, x1:x2]

        # 调整到目标大小
        roi = cv2.resize(roi, self.target_size, interpolation=cv2.INTER_CUBIC)

        return roi

    def visualize_detection(
        self,
        image: np.ndarray,
        info: Dict[str, Any]
    ) -> np.ndarray:
        """
        可视化检测结果（调试用）

        Args:
            image: 原始图像
            info: extract_roi 返回的信息

        Returns:
            可视化图像
        """
        result = image.copy()
        h, w = result.shape[:2]

        center = info.get('center', (w // 2, h // 2))
        roi_size = info.get('roi_size', 200)

        # 绘制中心点
        cv2.circle(result, center, 10, (0, 255, 0), -1)

        # 绘制 ROI 边界
        half_size = roi_size // 2
        x1 = max(0, center[0] - half_size)
        y1 = max(0, center[1] - half_size)
        x2 = min(w, center[0] + half_size)
        y2 = min(h, center[1] + half_size)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 添加文字信息
        method = info.get('method', 'unknown')
        area_ratio = info.get('region_info', {}).get('area_ratio', 0)
        cv2.putText(
            result,
            f"Method: {method}, Area: {area_ratio:.2%}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        return result


if __name__ == "__main__":
    # 测试纹理分析 ROI 提取器
    print("测试纹理分析 ROI 提取器...")

    extractor = TextureBasedROIExtractor()

    # 测试 1：随机图像
    print("\n1. 测试随机图像")
    random_image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    roi, info = extractor.extract_roi(random_image)
    if roi is not None:
        print(f"   成功！ROI 形状: {roi.shape}")
        print(f"   中心点: {info['center']}")
        print(f"   区域面积比: {info['region_info']['area_ratio']:.2%}")
    else:
        print("   失败！")

    # 测试 2：创建模拟掌纹图像
    print("\n2. 测试模拟掌纹图像")
    palm_image = np.ones((400, 400, 3), dtype=np.uint8) * 200

    # 在中心区域添加纹理
    for i in range(50, 350):
        for j in range(50, 350):
            # 添加一些随机纹理
            palm_image[i, j] = [
                200 + int(30 * np.sin(i * 0.1) * np.cos(j * 0.1)),
                200 + int(30 * np.cos(i * 0.1) * np.sin(j * 0.1)),
                200 + int(30 * np.sin((i + j) * 0.05))
            ]

    roi, info = extractor.extract_roi(palm_image)
    if roi is not None:
        print(f"   成功！ROI 形状: {roi.shape}")
        print(f"   中心点: {info['center']}")
        print(f"   区域面积比: {info['region_info']['area_ratio']:.2%}")

        # 保存可视化结果
        vis = extractor.visualize_detection(palm_image, info)
        cv2.imwrite("test_texture_roi.png", vis)
        print("   已保存可视化结果到 test_texture_roi.png")
    else:
        print("   失败！")

    print("\n测试完成！")
