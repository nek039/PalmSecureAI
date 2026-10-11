"""
增强版掌纹识别器 (Palm Recognizer V2)

基于《掌纹掌脉融合识别技术》中的方法改进：
1. 多模板注册 - 每个用户可注册多个样本
2. 改进的ROI提取 - 基于纹理分析的掌纹定位（不依赖MediaPipe）
3. 增强的预处理 - CLAHE + Gabor滤波 或 PLE (掌纹线增强)
4. 多模板融合匹配
5. 质量评估和自适应阈值
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import cv2

from .enhanced_roi_extractor import AdaptiveROIExtractor
from .enhanced_preprocessor import EnhancedPreprocessor, MultiModalPreprocessor, PLEPreprocessor
from .feature_extractor import create_feature_extractor
from .multi_template_matcher import MultiTemplateMatcher, AdaptiveThresholdMatcher
from .database import PalmDatabase, DatabaseConfig


class PalmRecognizer:
    """
    增强版掌纹识别器

    主要改进：
    - 支持多模板注册（每个用户可注册多个样本）
    - ROI提取使用纹理分析法（适用于灰度图，不依赖MediaPipe）
    - 增强的预处理（CLAHE、Gabor滤波 或 PLE）
    - 多模板融合匹配
    - 质量评估和自适应阈值
    """

    def __init__(
        self,
        model_checkpoint: Optional[str] = None,
        templates_path: str = "templates.pkl",
        db_path: Optional[str] = None,
        match_threshold: float = 0.80,  # 折中阈值 (识别率80%, 拒绝率60%)
        device: Optional[torch.device] = None,
        use_database: bool = True,
        use_enhanced_preprocessing: bool = True,
        use_multi_template: bool = True,
        min_samples_per_user: int = 1,
        max_samples_per_user: int = 10,
        feat_dim: int = 512,  # Mixed+PLE 模型使用 512 维特征
        use_mediapipe_roi: bool = False,  # 默认禁用 MediaPipe
        skip_roi: bool = False,  # 跳过 ROI 提取，直接 resize（适用于已处理的数据集）
        use_ple_preprocessing: bool = True,  # 默认启用 PLE 预处理
        ple_strategy: str = 'subtract',  # PLE 策略
        ple_alpha: float = 0.5,  # PLE 增强强度
    ):
        """
        Args:
            model_checkpoint: 模型检查点路径
            templates_path: 模板文件路径
            db_path: 数据库路径
            match_threshold: 匹配阈值
            device: 设备
            use_database: 是否使用数据库
            use_enhanced_preprocessing: 是否使用增强预处理
            use_multi_template: 是否使用多模板
            min_samples_per_user: 每个用户最少样本数
            max_samples_per_user: 每个用户最大样本数
            feat_dim: 特征维度（默认256）
            use_mediapipe_roi: 是否使用 MediaPipe 进行 ROI 提取（默认 False，使用纹理分析）
            skip_roi: 跳过 ROI 提取，直接 resize（适用于已处理的数据集如 CASIA/PolyU）
            use_ple_preprocessing: 使用 PLE 预处理（用于 Mixed+PLE 模型）
            ple_strategy: PLE 策略 ('subtract', 'add', 'mixed')
            ple_alpha: PLE 增强强度 (0-1)
        """
        # 配置
        self.use_enhanced_preprocessing = use_enhanced_preprocessing
        self.use_ple_preprocessing = use_ple_preprocessing
        self.use_multi_template = use_multi_template
        self.min_samples_per_user = min_samples_per_user
        self.max_samples_per_user = max_samples_per_user
        self.templates_path = templates_path
        self.use_database = use_database
        self.feat_dim = feat_dim
        self.use_mediapipe_roi = use_mediapipe_roi
        self.skip_roi = skip_roi

        # 子模块初始化
        # 使用纹理分析作为主要 ROI 提取方法（不依赖 MediaPipe）
        # 注意：图像增强会降低特征区分度，建议在高质量图像时禁用
        self.roi_extractor = AdaptiveROIExtractor(
            use_texture=True,
            fallback_to_center_crop=True,
        )

        # 预处理器选择
        if use_ple_preprocessing:
            # 使用 PLE 预处理（与 Mixed+PLE 模型训练时对齐）
            print("[INFO] 使用 PLE 预处理 (strategy={}, alpha={})".format(ple_strategy, ple_alpha))
            self.preprocessor = PLEPreprocessor(
                use_ple=True,
                ple_strategy=ple_strategy,
                ple_alpha=ple_alpha
            )
        elif use_enhanced_preprocessing:
            self.preprocessor = EnhancedPreprocessor()
        else:
            from .image_preprocessor import ImagePreprocessor
            self.preprocessor = ImagePreprocessor()

        # 特征提取器（使用工厂函数自动检测模型类型）
        self.feature_extractor = create_feature_extractor(
            checkpoint_path=model_checkpoint,
            device=device,
            feat_dim=feat_dim,
        )
        self.feature_extractor.eval()

        # 匹配器
        if use_multi_template:
            self.matcher = MultiTemplateMatcher(threshold=match_threshold)
        else:
            from .similarity_matcher import SimilarityMatcher
            self.matcher = SimilarityMatcher(threshold=match_threshold)

        # 存储配置
        if self.use_database:
            db_config = DatabaseConfig(db_path=db_path) if db_path else DatabaseConfig()
            self.database = PalmDatabase(db_config)
            self.templates: List[Dict[str, Any]] = self._load_templates_from_db()
            # 如果存在 pickle 文件，尝试导入
            if os.path.isfile(self.templates_path):
                self._import_from_pickle_if_needed()
        else:
            self.database = None
            self.templates: List[Dict[str, Any]] = []
            self._load_templates_if_exists()

    # ==================== 模板加载/保存 ====================

    def _load_templates_if_exists(self) -> None:
        """从 pickle 文件加载模板"""
        try:
            import pickle
            if os.path.isfile(self.templates_path):
                with open(self.templates_path, "rb") as f:
                    self.templates = pickle.load(f)
            else:
                self.templates = []
        except Exception:
            self.templates = []

    def _load_templates_from_db(self) -> List[Dict[str, Any]]:
        """从数据库加载模板"""
        if self.database:
            return self.database.load_templates()
        return []

    def _import_from_pickle_if_needed(self) -> None:
        """如果 pickle 文件存在且数据库为空，从 pickle 导入"""
        if self.database and os.path.isfile(self.templates_path):
            if self.database.get_user_count() == 0:
                print(f"检测到 {self.templates_path}，正在导入到数据库...")
                count = self.database.import_from_pickle(self.templates_path)
                print(f"已导入 {count} 个用户模板")
                self.templates = self.database.load_templates()
            else:
                print(f"数据库中已有 {self.database.get_user_count()} 个用户，跳过导入")

    def save_templates(self) -> None:
        """保存模板"""
        if self.use_database and self.database:
            self.templates = self.database.load_templates()
        else:
            try:
                import pickle
                with open(self.templates_path, "wb") as f:
                    pickle.dump(self.templates, f)
            except Exception as e:
                print(f"保存模板失败: {e}")

    # ==================== 特征提取 ====================

    def extract_feature(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        提取特征并返回质量分数

        Args:
            image: 输入 BGR 图像

        Returns:
            (feature, quality): 特征向量和质量分数 [0,1]
        """
        try:
            # 如果跳过 ROI，直接 resize
            if self.skip_roi:
                roi = cv2.resize(image, (224, 224))
                quality = 0.95  # 默认高质量
            else:
                # 1. ROI 提取（带质量检查）
                if hasattr(self.roi_extractor, 'extract_roi_with_quality_check'):
                    roi, quality = self.roi_extractor.extract_roi_with_quality_check(image)
                else:
                    roi, info = self.roi_extractor.base_extractor.extract_roi(image)
                    quality = 0.8 if roi is not None else 0.0

                if roi is None:
                    print("ROI 提取失败：未检测到手掌")
                    return None, 0.0

            # 2. 预处理
            preprocessed = self.preprocessor.preprocess(roi)

            # 3. 转为模型输入
            batch = self.preprocessor.normalize_for_model(preprocessed)

            # 4. 特征提取
            tensor = torch.from_numpy(batch).to(
                next(self.feature_extractor.parameters()).device
            )
            with torch.no_grad():
                feat = self.feature_extractor(tensor)

            return feat.squeeze(0).cpu().numpy(), quality
        except Exception as e:
            import traceback
            print(f"特征提取失败: {e}")
            traceback.print_exc()
            return None, 0.0

    # ==================== 注册 ====================

    def enroll(
        self,
        user_id: str,
        image: np.ndarray,
        force: bool = False,
        station: Optional[str] = None
    ) -> bool:
        """
        注册用户

        Args:
            user_id: 用户ID
            image: 掌纹图像
            force: 是否强制注册（即使已存在）
            station: 所属站点（可选）

        Returns:
            是否注册成功
        """
        try:
            # 检查用户是否已存在
            existing_count = self._get_user_sample_count(user_id)
            if existing_count >= self.max_samples_per_user and not force:
                print(f"注册失败：用户 {user_id} 已达到最大样本数 {self.max_samples_per_user}")
                return False

            # 提取特征
            feat, quality = self.extract_feature(image)
            if feat is None:
                print("注册失败：特征提取失败")
                return False

            # 质量检查 - 确保质量是标量
            try:
                quality_scalar = float(quality)
            except Exception:
                quality_scalar = 0.0

            if quality_scalar < 0.10:  # 质量阈值（大幅降低以适应模糊摄像头）
                print(f"注册失败：图像质量过低 (quality={quality_scalar:.2f})")
                return False

            # 创建模板
            template = {
                'user_id': user_id,
                'feature': feat,
                'quality': quality_scalar,  # 使用已转换的标量
                'timestamp': int(np.datetime64('now').astype(int))  # 转为 Python int
            }

            # 保存模板
            if self.use_database and self.database:
                # 数据库模式：添加新样本
                success = self.database.add_template(template, station=station)
                if success:
                    self.templates.append(template)
                    print(f"注册成功，用户 {user_id} 现有样本数: {self._get_user_sample_count(user_id)}")
                else:
                    print(f"注册失败：数据库操作失败")
                return success
            else:
                # pickle 模式
                self.templates.append(template)
                self.save_templates()
                print(f"注册成功，用户 {user_id} 现有样本数: {self._get_user_sample_count(user_id)}")
                return True
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False

    def enroll_multiple(
        self,
        user_id: str,
        images: List[np.ndarray],
        required_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        多次注册（批量注册）

        Args:
            user_id: 用户ID
            images: 图像列表
            required_samples: 需要的样本数（默认使用min_samples_per_user）

        Returns:
            {'success': bool, 'count': int, 'quality_scores': List[float]}
        """
        if required_samples is None:
            required_samples = self.min_samples_per_user

        print(f"批量注册用户 {user_id}，目标样本数: {required_samples}")

        success_count = 0
        quality_scores = []

        for i, image in enumerate(images):
            print(f"  处理第 {i+1}/{len(images)} 张图像...")
            if self.enroll(user_id, image):
                success_count += 1
                _, quality = self.extract_feature(image)
                quality_scores.append(quality)

        current_count = self._get_user_sample_count(user_id)

        result = {
            'success': current_count >= required_samples,
            'count': current_count,
            'required': required_samples,
            'quality_scores': quality_scores,
            'avg_quality': np.mean(quality_scores) if quality_scores else 0.0
        }

        print(f"批量注册完成：{result['count']}/{result['required']} 样本")
        print(f"平均质量: {result['avg_quality']:.3f}")

        return result

    # ==================== 识别 ====================

    def recognize(self, image: np.ndarray, method: str = "cosine") -> Dict[str, Any]:
        """
        识别用户

        Args:
            image: 掌纹图像
            method: 相似度计算方法

        Returns:
            识别结果
        """
        feat, quality = self.extract_feature(image)
        if feat is None:
            return {
                "matched": False,
                "user_id": None,
                "similarity": 0.0,
                "message": "特征提取失败",
                "quality": 0.0
            }

        # 匹配
        if self.use_multi_template and isinstance(self.matcher, MultiTemplateMatcher):
            result = self.matcher.match(feat, self.templates, method=method)
        else:
            # 单模板匹配
            result = self.matcher.match(feat, self.templates)
            if isinstance(result, dict) and 'details' not in result:
                # 转换为统一格式
                result['details'] = [(result.get('user_id', ''), result.get('similarity', 0.0))]

        # 添加质量信息
        result['quality'] = quality

        # 生成消息
        if result["matched"]:
            msg = f"识别成功：user_id={result['user_id']}, sim={result['similarity']:.4f}, quality={quality:.2f}"
        else:
            msg = f"识别失败，最高相似度={result['similarity']:.4f}, 质量={quality:.2f}"

        result['message'] = msg
        print(msg)

        return result

    # ==================== 用户管理 ====================

    def _get_user_sample_count(self, user_id: str) -> int:
        """获取用户的样本数"""
        if self.use_database and self.database:
            return self.database.get_user_sample_count(user_id)
        return sum(1 for t in self.templates if t.get('user_id') == user_id)

    def list_users(self) -> List[str]:
        """列出所有已注册的用户ID"""
        if self.use_database and self.database:
            return self.database.get_user_ids()
        return list(set(t.get('user_id', '') for t in self.templates))

    def get_user_templates(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户的所有模板"""
        if self.use_database and self.database:
            return self.database.get_user_templates(user_id)
        return [t for t in self.templates if t.get('user_id') == user_id]

    def get_user_count(self) -> int:
        """获取已注册用户数量"""
        if self.use_database and self.database:
            return self.database.get_user_count()
        return len(set(t.get('user_id', '') for t in self.templates))

    def get_total_templates(self) -> int:
        """获取总模板数"""
        if self.use_database and self.database:
            return self.database.get_total_templates()
        return len(self.templates)

    def delete_user(self, user_id: str) -> bool:
        """删除用户"""
        if self.use_database and self.database:
            success = self.database.delete_user(user_id)
            if success:
                self.templates = self.database.load_templates()
            return success
        else:
            original_count = len(self.templates)
            self.templates = [t for t in self.templates if t.get('user_id') != user_id]
            if len(self.templates) < original_count:
                self.save_templates()
                return True
            return False

    def clear_users(self) -> int:
        """清空所有用户"""
        if self.use_database and self.database:
            count = self.database.delete_all_users()
            self.templates = []
            return count
        else:
            count = len(self.templates)
            self.templates = []
            self.save_templates()
            return count

    def get_all_users(self) -> List[Dict[str, Any]]:
        """获取所有用户信息"""
        if self.use_database and self.database:
            users = self.database.get_all_users()
            # 添加样本数
            for user in users:
                user['sample_count'] = self.database.get_user_sample_count(user['user_id'])
            return users
        return [
            {
                'user_id': uid,
                'sample_count': self._get_user_sample_count(uid),
                'avg_quality': np.mean([t.get('quality', 0.5) for t in self.templates if t.get('user_id') == uid])
            }
            for uid in set(t.get('user_id', '') for t in self.templates)
        ]


if __name__ == "__main__":
    # 快速测试
    recognizer = PalmRecognizer(
        model_checkpoint=None,
        match_threshold=0.85,
        use_enhanced_preprocessing=True,
        use_multi_template=True
    )

    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    recognizer.enroll("user001", test_img)
    result = recognizer.recognize(test_img)
    print(f"用户数: {recognizer.get_user_count()}")
