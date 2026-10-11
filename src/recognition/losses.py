# -*- coding: utf-8 -*-
"""
掌纹识别损失函数模块

包含:
- ArcFace Loss: 加性角度间隔损失
- CosFace Loss: 加性余弦间隔损失
- Triplet Loss: 三元组损失
- Contrastive Loss: 对比损失
- Focal Loss: 焦点损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

__all__ = ["ArcFaceLoss", "CosFaceLoss", "TripletLoss", "BatchHardTripletLoss"]


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss - 加性角度间隔损失

    论文: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019)

    原理:
        在角度空间中增加间隔 m，使得同一类别的特征更加紧凑，
        不同类别的特征更加分散。

    公式:
        L = -log(exp(s * cos(θ_yi + m)) / Σexp(s * cos(θ_j)))

    参数:
        in_features: 输入特征维度
        out_features: 类别数（用户数）
        s: 缩放因子（默认30.0）
        m: 角度间隔（默认0.5，约28.6°）
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.5,
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features

        # 可学习的类别中心
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: 输入特征 (B, in_features)，应该已经 L2 归一化
            label: 类别标签 (B,)

        Returns:
            loss: 标量损失值
        """
        # L2 归一化
        input = F.normalize(input, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # 计算余弦相似度 (B, out_features)
        cosine = F.linear(input, weight)

        # 避免数值问题
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # 转换为角度
        theta = torch.acos(cosine)

        # 创建 one-hot 标签
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # 对目标类别添加角度间隔 m
        theta_yi = theta + one_hot * self.m

        # 转回余弦空间并缩放
        output = torch.cos(theta_yi) * self.s

        # 交叉熵损失
        loss = F.cross_entropy(output, label)

        return loss


class CosFaceLoss(nn.Module):
    """
    CosFace Loss - 加性余弦间隔损失

    论文: "CosFace: Large Margin Cosine Loss for Deep Face Recognition" (CVPR 2018)

    与 ArcFace 的区别:
        - ArcFace 在角度空间加间隔
        - CosFace 在余弦空间加间隔

    公式:
        L = -log(exp(s * (cos(θ_yi) - m)) / Σexp(s * cos(θ_j)))

    参数:
        in_features: 输入特征维度
        out_features: 类别数
        s: 缩放因子（默认30.0）
        m: 余弦间隔（默认0.35）
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.35,
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # L2 归一化
        input = F.normalize(input, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # 计算余弦相似度
        cosine = F.linear(input, weight)

        # 创建 one-hot 标签
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # 对目标类别减去余弦间隔 m
        output = (cosine - one_hot * self.m) * self.s

        loss = F.cross_entropy(output, label)

        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss - 三元组损失

    论文: "FaceNet: A Unified Embedding for Face Recognition" (CVPR 2015)

    原理:
        使锚点与正样本的距离小于锚点与负样本的距离，且至少相差 margin

    公式:
        L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    参数:
        margin: 间隔（默认0.5）
        p: 距离度量（1=曼哈顿, 2=欧氏）
        mining_strategy: 挖掘策略
            - 'all': 所有有效三元组
            - 'hard': 只用最难的三元组
            - 'semi_hard': 用半难三元组
    """

    def __init__(
        self,
        margin: float = 0.5,
        p: int = 2,
        mining_strategy: str = 'semi_hard',
    ):
        super().__init__()
        self.margin = margin
        self.p = p
        self.mining_strategy = mining_strategy

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor: 锚点特征 (B, D)
            positive: 正样本特征 (B, D)
            negative: 负样本特征 (B, D)

        Returns:
            loss: 标量损失值
        """
        # 计算距离
        pos_dist = F.pairwise_distance(anchor, positive, p=self.p)
        neg_dist = F.pairwise_distance(anchor, negative, p=self.p)

        # 基本损失
        losses = F.relu(pos_dist - neg_dist + self.margin)

        if self.mining_strategy == 'hard':
            # 只用最难的三元组（损失最大的）
            loss = losses.max()
        elif self.mining_strategy == 'semi_hard':
            # 用半难三元组（负样本距离大于正样本但小于正样本+margin）
            mask = (neg_dist > pos_dist) & (neg_dist < pos_dist + self.margin)
            if mask.sum() > 0:
                loss = losses[mask].mean()
            else:
                loss = losses.mean()
        else:
            # 所有三元组的平均
            loss = losses.mean()

        return loss


class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss

    论文: "In Defense of the Triplet Loss for Person Re-Identification" (2017)

    在一个 batch 中为每个样本选择最难的正样本和负样本
    """

    def __init__(self, margin: float = 0.5, p: int = 2):
        super().__init__()
        self.margin = margin
        self.p = p

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: 特征向量 (B, D)
            labels: 类别标签 (B,)

        Returns:
            loss: 标量损失值
        """
        # 计算距离矩阵
        dist_mat = self._pairwise_distance(embeddings)

        # 为每个样本找最难的正样本和负样本
        labels = labels.unsqueeze(1)
        mask_pos = labels == labels.t()  # 正样本掩码
        mask_neg = labels != labels.t()  # 负样本掩码

        # 最难正样本（距离最大的正样本）
        hardest_pos_dist = (dist_mat * mask_pos.float()).max(dim=1)[0]

        # 最难负样本（距离最小的负样本）
        # 将正样本距离设为很大的值，这样 min 就会选择负样本
        dist_mat_neg = dist_mat.clone()
        dist_mat_neg[mask_pos] = float('inf')
        hardest_neg_dist = dist_mat_neg.min(dim=1)[0]

        # 计算损失
        loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)

        return loss.mean()

    def _pairwise_distance(self, x: torch.Tensor) -> torch.Tensor:
        """计算成对欧氏距离"""
        dot = torch.mm(x, x.t())
        sq_norm = torch.diag(dot)
        dist = sq_norm.unsqueeze(1) - 2 * dot + sq_norm.unsqueeze(0)
        dist = F.relu(dist)  # 数值稳定性
        return torch.sqrt(dist + 1e-8)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss - 对比损失

    论文: "Dimensionality Reduction by Learning an Invariant Mapping" (2005)

    参数:
        margin: 负样本对的间隔
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x1: 第一个特征 (B, D)
            x2: 第二个特征 (B, D)
            label: 是否为同一类 (B,) 1=同类, 0=不同类

        Returns:
            loss: 标量损失值
        """
        dist = F.pairwise_distance(x1, x2)

        # 同类：距离越小越好
        # 不同类：距离越大越好（至少大于 margin）
        loss = label * dist.pow(2) + (1 - label) * F.relu(self.margin - dist).pow(2)

        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss - 焦点损失

    论文: "Focal Loss for Dense Object Detection" (ICCV 2017)

    用于处理类别不平衡问题

    公式:
        L = -α(1 - p_t)^γ log(p_t)

    参数:
        alpha: 类别权重
        gamma: 聚焦参数（默认2.0）
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: 模型输出 logits (B, C)
            target: 目标类别 (B,)

        Returns:
            loss: 标量损失值
        """
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    组合损失函数

    可以同时使用多种损失函数

    示例:
        loss_fn = CombinedLoss([
            (ArcFaceLoss(128, num_classes), 1.0),
            (BatchHardTripletLoss(), 0.5),
        ])
    """

    def __init__(self, loss_fns_with_weights: list):
        """
        Args:
            loss_fns_with_weights: [(loss_fn, weight), ...]
        """
        super().__init__()
        self.loss_fns = nn.ModuleList([fn for fn, _ in loss_fns_with_weights])
        self.weights = [w for _, w in loss_fns_with_weights]

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: 总损失
            losses_dict: 各损失分量
        """
        total_loss = 0.0
        losses_dict = {}

        for i, (fn, w) in enumerate(zip(self.loss_fns, self.weights)):
            try:
                loss = fn(*args, **kwargs)
            except TypeError:
                # 某些损失函数可能需要不同的参数
                continue

            total_loss = total_loss + w * loss
            losses_dict[f'loss_{i}_{type(fn).__name__}'] = loss.item()

        return total_loss, losses_dict


# ========== 工厂函数 ==========

def create_loss_function(
    loss_type: str,
    in_features: Optional[int] = None,
    out_features: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """
    创建损失函数

    Args:
        loss_type: 损失函数类型
            - 'arcface': ArcFace Loss
            - 'cosface': CosFace Loss
            - 'triplet': Triplet Loss
            - 'batch_hard_triplet': Batch Hard Triplet Loss
            - 'contrastive': Contrastive Loss
            - 'focal': Focal Loss
            - 'cross_entropy': Cross Entropy Loss
        in_features: 输入特征维度（某些损失需要）
        out_features: 类别数（某些损失需要）

    Returns:
        损失函数模块
    """
    loss_type = loss_type.lower()

    if loss_type == 'arcface':
        return ArcFaceLoss(
            in_features=in_features,
            out_features=out_features,
            s=kwargs.get('s', 30.0),
            m=kwargs.get('m', 0.5),
        )

    elif loss_type == 'cosface':
        return CosFaceLoss(
            in_features=in_features,
            out_features=out_features,
            s=kwargs.get('s', 30.0),
            m=kwargs.get('m', 0.35),
        )

    elif loss_type == 'triplet':
        return TripletLoss(
            margin=kwargs.get('margin', 0.5),
            p=kwargs.get('p', 2),
            mining_strategy=kwargs.get('mining_strategy', 'semi_hard'),
        )

    elif loss_type == 'batch_hard_triplet':
        return BatchHardTripletLoss(
            margin=kwargs.get('margin', 0.5),
            p=kwargs.get('p', 2),
        )

    elif loss_type == 'contrastive':
        return ContrastiveLoss(
            margin=kwargs.get('margin', 1.0),
        )

    elif loss_type == 'focal':
        return FocalLoss(
            alpha=kwargs.get('alpha', 0.25),
            gamma=kwargs.get('gamma', 2.0),
        )

    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ========== 测试代码 ==========

if __name__ == "__main__":
    batch_size = 32
    feat_dim = 128
    num_classes = 100

    # 生成测试数据
    features = torch.randn(batch_size, feat_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    print("Testing loss functions...")
    print(f"  Batch size: {batch_size}")
    print(f"  Feature dim: {feat_dim}")
    print(f"  Num classes: {num_classes}")
    print()

    # 测试 ArcFace
    arcface = ArcFaceLoss(feat_dim, num_classes)
    loss = arcface(features, labels)
    print(f"ArcFace Loss: {loss.item():.4f}")

    # 测试 CosFace
    cosface = CosFaceLoss(feat_dim, num_classes)
    loss = cosface(features, labels)
    print(f"CosFace Loss: {loss.item():.4f}")

    # 测试 Batch Hard Triplet
    triplet = BatchHardTripletLoss()
    loss = triplet(features, labels)
    print(f"Batch Hard Triplet Loss: {loss.item():.4f}")

    print("\nAll tests passed!")
