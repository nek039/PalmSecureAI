"""
多模板匹配器

基于《掌纹掌脉融合识别技术》中的方法：
- 每个用户可注册多个样本
- 多模板融合匹配
- 自适应阈值
- 质量加权匹配
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


class MultiTemplateMatcher:
    """
    多模板匹配器

    支持每个用户注册多个样本，提高识别准确率
    """

    # 安全验证常量（根据 unified evaluation 结果调整）
    MIN_GAP_TO_SECOND = 0.0  # 禁用差距检查（跨数据集场景差距不明显）
    ABSOLUTE_MIN_THRESHOLD = 0.55  # 绝对最低阈值（允许 0.60 最优阈值生效）
    LOW_CONFIDENCE_THRESHOLD = 0.65  # 低置信度警告阈值（与 0.60 阈值对齐）

    def __init__(self, threshold: float = 0.80, fusion_method: str = "average"):
        """
        Args:
            threshold: 匹配阈值（注意：实际使用时会与 ABSOLUTE_MIN_THRESHOLD 取最大值）
            fusion_method: 融合方法，可选：
                - "max": 取最大相似度
                - "average": 取平均相似度
                - "weighted": 质量加权平均
        """
        self.threshold = max(threshold, self.ABSOLUTE_MIN_THRESHOLD)
        self.fusion_method = fusion_method
        print(f"[MultiTemplateMatcher] 初始化: 输入阈值={threshold}, 实际阈值={self.threshold}, ABSOLUTE_MIN={self.ABSOLUTE_MIN_THRESHOLD}")

    @staticmethod
    def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """计算余弦相似度 [0,1]"""
        f1 = np.asarray(feat1, dtype=np.float32)
        f2 = np.asarray(feat2, dtype=np.float32)

        dot = float(np.dot(f1, f2))
        norm1 = float(np.linalg.norm(f1) + 1e-8)
        norm2 = float(np.linalg.norm(f2) + 1e-8)
        return dot / (norm1 * norm2)

    @staticmethod
    def euclidean_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """计算欧氏距离"""
        f1 = np.asarray(feat1, dtype=np.float32)
        f2 = np.asarray(feat2, dtype=np.float32)
        return float(np.linalg.norm(f1 - f2))

    @staticmethod
    def hamming_distance(feat1: np.ndarray, feat2: np.ndarray, threshold: float = 0) -> float:
        """
        计算汉明距离（用于二值编码）

        Args:
            feat1, feat2: 特征向量
            threshold: 二值化阈值

        Returns:
            归一化的汉明距离 [0, 1]
        """
        f1 = np.asarray(feat1, dtype=np.float32)
        f2 = np.asarray(feat2, dtype=np.float32)

        # 二值化
        binary1 = (f1 > threshold).astype(np.int8)
        binary2 = (f2 > threshold).astype(np.int8)

        # 计算汉明距离
        hamming = np.sum(binary1 != binary2) / len(f1)
        return hamming

    def match_single(
        self,
        query_feature: np.ndarray,
        template_feature: np.ndarray,
        method: str = "cosine"
    ) -> float:
        """
        计算查询特征与单个模板的相似度

        Args:
            query_feature: 查询特征
            template_feature: 模板特征
            method: 相似度计算方法，可选 "cosine", "euclidean", "hamming"

        Returns:
            相似度分数 [0, 1]
        """
        # 检查特征维度是否匹配
        if query_feature.shape != template_feature.shape:
            print(f"[WARN] 特征维度不匹配: query={query_feature.shape}, template={template_feature.shape}，跳过该模板")
            return -1.0  # 返回负值表示维度不匹配

        if method == "cosine":
            return self.cosine_similarity(query_feature, template_feature)
        elif method == "euclidean":
            # 转换为相似度：距离越小，相似度越高
            dist = self.euclidean_distance(query_feature, template_feature)
            return np.exp(-dist)  # 指数衰减
        elif method == "hamming":
            dist = self.hamming_distance(query_feature, template_feature)
            return 1.0 - dist
        else:
            return self.cosine_similarity(query_feature, template_feature)

    def match_multi_template(
        self,
        query_feature: np.ndarray,
        templates: List[Dict[str, Any]],
        method: str = "cosine"
    ) -> Dict[str, float]:
        """
        多模板匹配：计算查询特征与每个用户所有模板的融合相似度

        Args:
            query_feature: 查询特征
            templates: 模板列表，每个模板包含 {'user_id': str, 'feature': np.ndarray, 'quality': float}
            method: 相似度计算方法

        Returns:
            {user_id: similarity} 字典
        """
        # 记录查询特征的维度
        query_dim = query_feature.shape[0] if hasattr(query_feature, 'shape') else len(query_feature)
        valid_count = 0
        skipped_count = 0

        # 按用户分组
        user_templates = defaultdict(list)
        for template in templates:
            user_id = template.get('user_id')
            if user_id:
                user_templates[user_id].append(template)

        # 计算每个用户的融合相似度
        results = {}
        for user_id, user_templs in user_templates.items():
            similarities = []
            qualities = []

            for tmpl in user_templs:
                sim = self.match_single(query_feature, tmpl['feature'], method)
                if sim < 0:
                    # 维度不匹配，跳过该模板
                    skipped_count += 1
                    continue
                similarities.append(sim)
                qualities.append(tmpl.get('quality', 1.0))
                valid_count += 1

            # 只有有有效相似度时才融合
            if similarities:
                fused_sim = self._fuse_similarities(similarities, qualities)
                results[user_id] = fused_sim
            else:
                # 该用户没有有效模板（可能是维度不匹配）
                results[user_id] = 0.0

        if skipped_count > 0:
            print(f"[INFO] 跳过 {skipped_count} 个维度不匹配的模板 (查询维度: {query_dim})")

        return results

    def _fuse_similarities(
        self,
        similarities: List[float],
        qualities: Optional[List[float]] = None
    ) -> float:
        """
        融合多个相似度分数

        Args:
            similarities: 相似度列表
            qualities: 质量分数列表（可选）

        Returns:
            融合后的相似度
        """
        if not similarities:
            return 0.0

        if self.fusion_method == "max":
            return max(similarities)
        elif self.fusion_method == "average":
            return sum(similarities) / len(similarities)
        elif self.fusion_method == "weighted":
            if qualities is None or len(qualities) != len(similarities):
                return sum(similarities) / len(similarities)
            # 质量加权
            weighted_sum = sum(s * q for s, q in zip(similarities, qualities))
            total_weight = sum(qualities)
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            return max(similarities)

    def match(
        self,
        query_feature: np.ndarray,
        templates: List[Dict[str, Any]],
        method: str = "cosine"
    ) -> Dict[str, Any]:
        """
        在模板库中查找最佳匹配（完整版）

        Args:
            query_feature: 查询特征
            templates: 模板列表
            method: 相似度计算方法

        Returns:
            result: {
                'matched': bool,
                'user_id': Optional[str],
                'similarity': float,
                'rank': int,
                'details': List[Tuple[str, float]],  # (user_id, similarity)
            }
        """
        if not templates:
            return {
                "matched": False,
                "user_id": None,
                "similarity": 0.0,
                "rank": -1,
                "details": []
            }

        # 多模板匹配
        user_similarities = self.match_multi_template(query_feature, templates, method)

        # 排序
        ranked = sorted(user_similarities.items(), key=lambda x: x[1], reverse=True)

        # 获取最佳匹配
        best_user, best_sim = ranked[0]

        # === 安全验证机制 ===
        print(f"[MATCH] best_sim={best_sim:.4f}, threshold={self.threshold}, ABSOLUTE_MIN={self.ABSOLUTE_MIN_THRESHOLD}")

        # 验证 1：绝对阈值检查
        if best_sim < self.ABSOLUTE_MIN_THRESHOLD:
            print(f"[MATCH] 拒绝: 相似度 {best_sim:.4f} < 绝对阈值 {self.ABSOLUTE_MIN_THRESHOLD}")
            return {
                "matched": False,
                "user_id": None,
                "similarity": best_sim,
                "rank": 1,
                "details": ranked,
                "rejection_reason": "相似度低于阈值",
                "confidence": "low"
            }

        # 验证 2：双重检查（第一名必须显著高于第二名）
        if len(ranked) >= 2:
            second_sim = ranked[1][1]
            if best_sim - second_sim < self.MIN_GAP_TO_SECOND:
                return {
                    "matched": False,
                    "user_id": None,
                    "similarity": best_sim,
                    "rank": 1,
                    "details": ranked,
                    "rejection_reason": "匹配不显著，第一名与第二名差距过小",
                    "confidence": "low"
                }

        # 验证 3：计算置信度
        confidence_level = "high" if best_sim >= self.LOW_CONFIDENCE_THRESHOLD else "medium"

        result = {
            "matched": best_sim >= self.threshold,
            "user_id": best_user if best_sim >= self.threshold else None,
            "similarity": best_sim,
            "rank": 1,
            "details": ranked,
            "confidence": confidence_level
        }

        return result


class AdaptiveThresholdMatcher(MultiTemplateMatcher):
    """
    自适应阈值匹配器

    根据模板库的分布动态调整阈值
    """

    def __init__(self, initial_threshold: float = 0.85, adapt_rate: float = 0.1):
        super().__init__(threshold=initial_threshold)
        self.adapt_rate = adapt_rate
        self.similarity_history: List[float] = []

    def update_threshold(self, is_correct: bool, similarity: float):
        """
        根据反馈更新阈值

        Args:
            is_correct: 识别是否正确
            similarity: 相似度分数
        """
        self.similarity_history.append(similarity)

        if len(self.similarity_history) < 10:
            return

        # 如果最近识别都是正确的，可以降低阈值
        # 如果有错误，可以提高阈值
        recent_correct = sum(1 for _ in self.similarity_history[-10:])  # 这里简化了

        if is_correct:
            self.threshold = max(0.7, self.threshold - self.adapt_rate * 0.1)
        else:
            self.threshold = min(0.95, self.threshold + self.adapt_rate * 0.1)

    def get_threshold(self) -> float:
        """获取当前阈值"""
        return self.threshold


class DecisionLevelFusionMatcher:
    """
    决策级融合匹配器

    用于融合多个匹配器的结果（如掌纹+掌脉）
    """

    def __init__(self, weights: Optional[List[float]] = None):
        """
        Args:
            weights: 各个匹配器的权重，默认均等
        """
        self.weights = weights

    def fuse_decisions(
        self,
        decisions: List[Dict[str, Any]],
        method: str = "weighted_vote"
    ) -> Dict[str, Any]:
        """
        融合多个匹配器的决策

        Args:
            decisions: 多个匹配器的决策结果列表
            method: 融合方法
                - "weighted_vote": 加权投票
                - "weighted_score": 加权分数
                - "max_score": 最大分数

        Returns:
            融合后的决策结果
        """
        if not decisions:
            return {
                "matched": False,
                "user_id": None,
                "similarity": 0.0,
                "method": method
            }

        if method == "max_score":
            # 取最大分数
            best = max(decisions, key=lambda x: x.get('similarity', 0))
            return best

        elif method == "weighted_score":
            # 加权分数
            if self.weights is None:
                self.weights = [1.0 / len(decisions)] * len(decisions)

            # 聚合相似度分数
            user_scores = defaultdict(float)
            user_counts = defaultdict(int)

            for decision, weight in zip(decisions, self.weights):
                user_id = decision.get('user_id')
                similarity = decision.get('similarity', 0)
                if user_id:
                    user_scores[user_id] += similarity * weight
                    user_counts[user_id] += 1

            if not user_scores:
                return {
                    "matched": False,
                    "user_id": None,
                    "similarity": 0.0,
                    "method": method
                }

            # 取平均分数最高的用户
            best_user = max(user_scores.items(), key=lambda x: x[1])
            return {
                "matched": True,
                "user_id": best_user[0],
                "similarity": best_user[1] / user_counts[best_user[0]],
                "method": method
            }

        else:  # weighted_vote
            # 加权投票
            votes = defaultdict(float)
            for decision, weight in zip(decisions, self.weights):
                user_id = decision.get('user_id')
                if user_id and decision.get('matched', False):
                    votes[user_id] += weight

            if not votes:
                return {
                    "matched": False,
                    "user_id": None,
                    "similarity": 0.0,
                    "method": method
                }

            best_user = max(votes.items(), key=lambda x: x[1])
            return {
                "matched": True,
                "user_id": best_user[0],
                "similarity": votes[best_user[0]] / sum(self.weights),
                "method": method
            }


if __name__ == "__main__":
    # 测试多模板匹配器
    print("测试多模板匹配器...")

    # 创建匹配器
    matcher = MultiTemplateMatcher(threshold=0.85, fusion_method="max")

    # 创建测试模板
    templates = [
        {'user_id': 'user1', 'feature': np.random.randn(128), 'quality': 0.9},
        {'user_id': 'user1', 'feature': np.random.randn(128), 'quality': 0.85},
        {'user_id': 'user2', 'feature': np.random.randn(128), 'quality': 0.92},
        {'user_id': 'user3', 'feature': np.random.randn(128), 'quality': 0.88},
    ]

    # 查询特征
    query = templates[0]['feature'] * 0.9 + np.random.randn(128) * 0.1  # 与user1的第一个模板相似

    # 匹配
    result = matcher.match(query, templates)

    print(f"匹配结果: {result}")
    print("多模板匹配器测试完成")
