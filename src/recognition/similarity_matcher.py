from typing import List, Dict, Any

import numpy as np


class SimilarityMatcher:
    """
    相似度匹配器

    - 使用余弦相似度进行匹配
    - 输入特征向量应为 L2 归一化后的 128 维向量
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    @staticmethod
    def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """计算余弦相似度 [0,1]。"""
        f1 = np.asarray(feat1, dtype=np.float32)
        f2 = np.asarray(feat2, dtype=np.float32)

        dot = float(np.dot(f1, f2))
        norm1 = float(np.linalg.norm(f1) + 1e-8)
        norm2 = float(np.linalg.norm(f2) + 1e-8)
        return dot / (norm1 * norm2)

    @staticmethod
    def euclidean_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """计算欧氏距离。"""
        f1 = np.asarray(feat1, dtype=np.float32)
        f2 = np.asarray(feat2, dtype=np.float32)
        return float(np.linalg.norm(f1 - f2))

    def match(
        self,
        query_feature: np.ndarray,
        template_features: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        在模板库中查找最佳匹配。

        Args:
            query_feature: 查询特征 (128,)
            template_features: 模板库 [{'user_id': str, 'feature': np.array}, ...]

        Returns:
            result: {
                'matched': bool,
                'user_id': Optional[str],
                'similarity': float,
                'rank': int,
            }
        """
        if not template_features:
            return {
                "matched": False,
                "user_id": None,
                "similarity": 0.0,
                "rank": -1,
            }

        sims = []
        for t in template_features:
            sim = self.cosine_similarity(query_feature, t["feature"])
            sims.append({"user_id": t["user_id"], "similarity": float(sim)})

        sims.sort(key=lambda x: x["similarity"], reverse=True)
        best = sims[0]

        if best["similarity"] >= self.threshold:
            return {
                "matched": True,
                "user_id": best["user_id"],
                "similarity": best["similarity"],
                "rank": 1,
            }

        return {
            "matched": False,
            "user_id": None,
            "similarity": best["similarity"],
            "rank": 1,
        }

