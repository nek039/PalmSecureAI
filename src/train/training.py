"""
掌纹特征提取模型训练脚本

功能：
1. 加载掌纹数据集
2. 使用 Triplet Loss 训练 ResNet18 特征提取器
3. 支持混合精度训练
4. 支持学习率调度
5. 支持早停和模型保存

使用示例：
    python train/training.py --data-dir ./data --epochs 100 --batch-size 32
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 导入项目模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SystemConfig, TrainingConfig, Config
from data.dataset_loader import TripletPalmDataset, create_dataloader, PalmDataset

# 导入特征提取模型
from src.recognition.feature_extractor import PalmFeatureExtractor, ResNet18FeatureExtractor

# 导入权威损失函数（定义在 src/recognition/losses.py）
# 本地的 TripletLoss 是训练专用变体（内部做归一化），与 losses.py 中的实现略有不同
from src.recognition.losses import TripletLoss as CanonicalTripletLoss

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ========== 损失函数 ==========


class TripletLoss(nn.Module):
    """Triplet 损失函数"""

    def __init__(self, margin: float = 0.5, p: float = 2.0, mining: str = "hard"):
        """
        初始化 Triplet Loss

        Args:
            margin: 边界值
            p: 范数类型 (1=曼哈顿, 2=欧氏)
            mining: 难样本挖掘方式 ("random", "semihard", "hard")
        """
        super().__init__()
        self.margin = margin
        self.p = p
        self.mining = mining

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Triplet Loss

        Args:
            anchor: 锚点特征 (B, D)
            positive: 正样本特征 (B, D)
            negative: 负样本特征 (B, D)

        Returns:
            loss: 损失值
        """
        # L2 归一化
        anchor = nn.functional.normalize(anchor, p=2, dim=1)
        positive = nn.functional.normalize(positive, p=2, dim=1)
        negative = nn.functional.normalize(negative, p=2, dim=1)

        # 计算距离
        pos_dist = torch.norm(anchor - positive, p=self.p, dim=1)
        neg_dist = torch.norm(anchor - negative, p=self.p, dim=1)

        # 难样本挖掘
        if self.mining == "random":
            # 随机选择负样本
            pass
        elif self.mining == "semihard":
            # 选择半难样本：负样本距离在 (pos_dist, pos_dist + margin) 之间
            semihard_mask = (neg_dist > pos_dist) & (neg_dist < pos_dist + self.margin)
            if semihard_mask.any():
                neg_dist = neg_dist * semihard_mask + neg_dist * (~semihard_mask)
        elif self.mining == "hard":
            # 选择最难负样本：负样本距离小于正样本距离
            hard_mask = neg_dist < pos_dist
            neg_dist = neg_dist * hard_mask + neg_dist * (~hard_mask)
        else:
            raise ValueError(f"未知的挖掘方式: {self.mining}")

        # 计算 Triplet Loss
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)

        return loss.mean()


# ========== 训练器 ==========


class Trainer:
    """训练器类"""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ):
        """
        初始化训练器

        Args:
            model: 训练模型
            config: 训练配置
            device: 训练设备
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.scheduler = self._create_scheduler()

        # 损失函数
        if config.loss_type == "triplet":
            self.criterion = TripletLoss(
                margin=config.triplet_margin,
                mining=config.triplet_mining
            )
        else:
            raise ValueError(f"未知的损失函数: {config.loss_type}")

        # 混合精度训练
        self.use_amp = config.use_amp
        self.scaler = GradScaler() if config.use_amp else None

        # TensorBoard
        self.writer = SummaryWriter(config.tensorboard_dir)

        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.global_step = 0

        # 早停机制
        self.patience = getattr(config, 'patience', 10)
        self.patience_counter = 0
        self.early_stop = False

        # 创建保存目录
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        if self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"未知的优化器: {self.config.optimizer}")

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=1e-6,
            )
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif self.config.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.scheduler_patience,
            )
        elif self.config.scheduler == "warmup_cosine":
            # 自定义 warmup + cosine 调度器
            return WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=self.config.warmup_epochs,
                total_epochs=self.config.num_epochs,
            )
        elif self.config.scheduler == "none":
            return None
        else:
            raise ValueError(f"未知的调度器: {self.config.scheduler}")

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> float:
        """
        训练一个 epoch

        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.num_epochs}")

        for batch in pbar:
            # 移动数据到设备
            anchor = batch['anchor'].to(self.device)
            positive = batch['positive'].to(self.device)
            negative = batch['negative'].to(self.device)

            # 前向传播
            with autocast(enabled=self.use_amp):
                anchor_feat = self.model(anchor)
                positive_feat = self.model(positive)
                negative_feat = self.model(negative)

                loss = self.criterion(anchor_feat, positive_feat, negative_feat)

            # 反向传播
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.clip_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.clip_grad_norm
                )
                self.optimizer.step()

            # 记录
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

            # TensorBoard 记录
            if self.global_step % 10 == 0:
                self.writer.add_scalar('Train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> float:
        """
        验证模型

        Returns:
            平均损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validation {epoch}")

            for batch in pbar:
                anchor = batch['anchor'].to(self.device)
                positive = batch['positive'].to(self.device)
                negative = batch['negative'].to(self.device)

                anchor_feat = self.model(anchor)
                positive_feat = self.model(positive)
                negative_feat = self.model(negative)

                loss = self.criterion(anchor_feat, positive_feat, negative_feat)

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / num_batches

        # TensorBoard 记录
        self.writer.add_scalar('Val/loss', avg_loss, epoch)

        return avg_loss

    def save_checkpoint(
        self,
        epoch: int,
        loss: float,
        is_best: bool = False,
    ) -> None:
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
        }

        # 保存当前检查点
        checkpoint_path = Path(self.config.save_dir) / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"保存检查点: {checkpoint_path}")

        # 保存最佳模型
        if is_best:
            best_path = Path(self.config.save_dir) / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型: {best_path}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """
        完整训练流程

        Returns:
            训练结果字典
        """
        logger.info("=" * 60)
        logger.info("开始训练")
        logger.info(f"  设备: {self.device}")
        logger.info(f"  Epochs: {self.config.num_epochs}")
        logger.info(f"  混合精度: {self.use_amp}")
        logger.info(f"  早停耐心: {self.patience}")
        logger.info("=" * 60)

        start_time = time.time()

        for epoch in range(1, self.config.num_epochs + 1):
            self.current_epoch = epoch

            # 训练
            train_loss = self.train_epoch(train_loader, epoch)

            # 验证
            if val_loader is not None:
                val_loss = self.validate(val_loader, epoch)
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    logger.info(f"验证损失未改善，早停计数: {self.patience_counter}/{self.patience}")
            else:
                val_loss = 0.0
                is_best = train_loss < self.best_loss
                if is_best:
                    self.best_loss = train_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}")

            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loader else train_loss)
                else:
                    self.scheduler.step()

            # 保存检查点
            if epoch % self.config.save_frequency == 0:
                self.save_checkpoint(epoch, train_loss, is_best)

            # 早停检查
            if self.patience_counter >= self.patience:
                logger.info(f"早停触发！已连续 {self.patience} 个 epoch 验证损失未改善")
                self.early_stop = True
                break

        # 保存最终模型
        final_epoch = self.current_epoch if self.early_stop else self.config.num_epochs
        self.save_checkpoint(final_epoch, self.best_loss, is_best=True)

        total_time = time.time() - start_time

        # 关闭 TensorBoard
        self.writer.close()

        logger.info("=" * 60)
        logger.info("训练完成")
        logger.info(f"  总时间: {total_time / 60:.2f} 分钟")
        logger.info(f"  最佳损失: {self.best_loss:.6f}")
        logger.info(f"  训练轮数: {final_epoch}/{self.config.num_epochs}")
        if self.early_stop:
            logger.info(f"  (早停)")
        logger.info("=" * 60)

        return {
            'train_loss': self.best_loss,
            'val_loss': val_loss if val_loader else 0.0,
            'total_time': total_time,
            'epochs_trained': final_epoch,
            'early_stopped': self.early_stop,
        }


class WarmupCosineScheduler:
    """Warmup + Cosine 学习率调度器"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self) -> None:
        if self.current_epoch < self.warmup_epochs:
            # Warmup 阶段：线性增长
            alpha = (self.current_epoch + 1) / self.warmup_epochs
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = self.min_lr + alpha * (base_lr - self.min_lr)
        else:
            # Cosine 退火阶段
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))

            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = self.min_lr + cosine_factor * (base_lr - self.min_lr)

        self.current_epoch += 1


# ========== 主函数 ==========


def main():
    parser = argparse.ArgumentParser(description="训练掌纹特征提取模型")

    # 数据参数
    parser.add_argument("--data-dir", type=str, default="./data",
                      help="数据集目录")
    parser.add_argument("--train-users-ratio", type=float, default=0.8,
                      help="用于训练的用户比例")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100,
                      help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                      help="学习率")
    parser.add_argument("--triplet-margin", type=float, default=0.5,
                      help="Triplet Loss 边界值")
    parser.add_argument("--mining", type=str, default="hard",
                      choices=["random", "semihard", "hard"],
                      help="难样本挖掘方式")

    # 模型参数
    parser.add_argument("--feature-dim", type=int, default=128,
                      help="特征维度")
    parser.add_argument("--pretrained", action="store_true",
                      help="使用预训练权重")

    # 系统参数
    parser.add_argument("--device", type=str, default=None,
                      choices=["cuda", "cpu"],
                      help="训练设备")
    parser.add_argument("--num-workers", type=int, default=4,
                      help="数据加载进程数")
    parser.add_argument("--seed", type=int, default=42,
                      help="随机种子")
    parser.add_argument("--save-dir", type=str, default="./models",
                      help="模型保存目录")
    parser.add_argument("--resume", type=str, default=None,
                      help="恢复训练的检查点路径")

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 确定设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"使用设备: {device}")

    # 加载数据集
    logger.info(f"加载数据集: {args.data_dir}")

    train_dataset = TripletPalmDataset(
        args.data_dir,
        image_size=(224, 224),
        augment=True,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # 创建模型
    logger.info("创建模型...")
    model = PalmFeatureExtractor(
        checkpoint_path=None,  # 不使用预训练，从头训练
        feat_dim=args.feature_dim,
        device=device,
    )

    # 恢复训练
    start_epoch = 1
    if args.resume:
        logger.info(f"从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # 创建配置
    config = TrainingConfig(
        num_epochs=args.epochs,
        local_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        triplet_margin=args.triplet_margin,
        triplet_mining=args.mining,
        save_dir=args.save_dir,
        use_amp=True,
    )

    # 创建训练器
    trainer = Trainer(model, config, device)

    # 开始训练
    results = trainer.train(train_loader)

    logger.info("训练完成！")
    logger.info(f"最佳模型保存在: {Path(args.save_dir) / 'best_model.pth'}")


if __name__ == "__main__":
    main()
