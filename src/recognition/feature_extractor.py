import os
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 掌纹特征提取 Backbone ==========

class PalmBackbone(nn.Module):
    """
    掌纹特征提取 Backbone（不依赖预训练模型）

    架构：
    - 输入: 224x224 RGB
    - 输出: 512 维特征
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()

        # 自定义卷积网络（从头训练）
        self.features = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Layer 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Layer 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Layer 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.features(x)


# ========== 掌纹特征提取器 ==========

class PalmFeatureExtractor(nn.Module):
    """
    掌纹特征提取器（128维特征）

    架构：Backbone + 特征投影层
    - 不依赖预训练模型
    - 可加载训练好的检查点
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        feat_dim: int = 128,
    ):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feat_dim = feat_dim

        # 初始化 backbone
        self.backbone = PalmBackbone()

        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, feat_dim),
            nn.BatchNorm1d(feat_dim),
        )

        self.to(self.device)

        # 可选：加载训练好的权重
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """加载检查点"""
        if not os.path.isfile(checkpoint_path):
            print(f"警告: 检查点不存在 {checkpoint_path}，使用随机初始化")
            return

        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")

            # 尝试加载模型权重
            if "model_state_dict" in ckpt:
                self.load_state_dict(ckpt["model_state_dict"])
            else:
                # 兼容：直接加载权重
                self.load_state_dict(ckpt)

            print(f"已加载模型权重: {checkpoint_path}")
        except Exception as e:
            print(f"加载检查点失败: {e}，使用随机初始化")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, 224, 224), float32, [0,1]

        Returns:
            features: (B, 128) 的 L2 归一化特征
        """
        x = self.backbone(x)  # (B, 512, 7, 7)
        features = self.feature_projection(x)  # (B, 128)

        # L2 归一化
        features = nn.functional.normalize(features, p=2, dim=1)

        return features

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        从单张图像提取 128 维特征向量

        Args:
            image: numpy 数组，形状为 (H, W, 3)，值范围 [0, 1] 或 [0, 255]

        Returns:
            feature: (128,) 的 numpy 向量
        """
        # 归一化到 [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        # 处理图像维度
        if len(image.shape) == 2:
            # 灰度图像，扩展为 3 通道
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 3:
            # RGB 图像，确保是 (H, W, 3)
            pass
        else:
            # 其他格式，尝试处理
            image = np.stack([image] * 3, axis=-1)

        # 调整为 (224, 224, 3)
        if image.shape[0] != 224 or image.shape[1] != 224:
            image = cv2.resize(image, (224, 224))

        # 转换为张量并添加 batch 维
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 224, 224)
        tensor = tensor.to(self.device, dtype=torch.float32)

        # 提取特征
        self.eval()
        with torch.no_grad():
            features = self.forward(tensor)

        return features.squeeze(0).cpu().numpy()


# ========== ResNet18 特征提取器（兼容已训练模型） ==========

class ResNet18FeatureExtractor(nn.Module):
    """
    基于 ResNet18 的掌纹特征提取器

    用于加载已训练的模型权重（backbone.0.weight 格式）
    支持两种架构:
    - ReLU 版本 (旧模型)
    - PReLU 版本 (mixed_ple_final 等新模型)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        feat_dim: int = 128,
        use_prelu: bool = False,
    ):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feat_dim = feat_dim
        self.use_prelu = use_prelu

        # 使用 torchvision 的 ResNet18 作为 backbone
        from torchvision.models import resnet18, ResNet18_Weights
        resnet = resnet18(weights=None)

        # 移除最后的全连接层，保留到 AdaptiveAvgPool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 输出 (B, 512, 1, 1)

        # 特征投影层
        if use_prelu:
            # PReLU 版本 (mixed_ple_final 等新模型)
            self.feature_projection = nn.Sequential(
                nn.Flatten(),                    # 0
                nn.Dropout(0.5),                 # 1
                nn.Linear(512, 1024),            # 2
                nn.BatchNorm1d(1024),            # 3
                nn.PReLU(),                      # 4 (有权重)
                nn.Dropout(0.5),                 # 5
                nn.Linear(1024, feat_dim),       # 6
                nn.BatchNorm1d(feat_dim),        # 7
            )
        else:
            # ReLU 版本 (旧模型)
            self.feature_projection = nn.Sequential(
                nn.Flatten(),                    # 0
                nn.Dropout(0.3),                 # 1
                nn.Linear(512, 1024),            # 2
                nn.BatchNorm1d(1024),            # 3
                nn.ReLU(inplace=True),           # 4 (无权重)
                nn.Dropout(0.3),                 # 5
                nn.Linear(1024, feat_dim),       # 6
                nn.BatchNorm1d(feat_dim),        # 7
            )

        self.to(self.device)

        # 可选：加载训练好的权重
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """加载检查点（支持多种格式，自动检测架构）"""
        if not os.path.isfile(checkpoint_path):
            print(f"警告: 检查点不存在 {checkpoint_path}，使用随机初始化")
            return

        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            # 提取 state_dict
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
                if "epoch" in ckpt:
                    print(f"加载模型: epoch={ckpt['epoch']}, loss={ckpt.get('loss', 'N/A')}")
            else:
                state_dict = ckpt

            # 检测是否需要重建模型
            needs_rebuild = False
            detected_feat_dim = self.feat_dim
            detected_prelu = self.use_prelu

            # 检测 PReLU
            if "feature_projection.4.weight" in state_dict:
                if not self.use_prelu:
                    print("检测到 PReLU 架构，重建模型...")
                    detected_prelu = True
                    needs_rebuild = True

            # 检测特征维度
            for k in state_dict.keys():
                if "feature_projection.6.weight" in k:
                    detected_feat_dim = state_dict[k].shape[0]
                    if detected_feat_dim != self.feat_dim:
                        print(f"检测到特征维度: {detected_feat_dim}")
                        needs_rebuild = True
                    break

            # 如果需要，重建模型
            if needs_rebuild:
                self.feat_dim = detected_feat_dim
                self.use_prelu = detected_prelu

                # 重建投影层
                if detected_prelu:
                    self.feature_projection = nn.Sequential(
                        nn.Flatten(),
                        nn.Dropout(0.5),
                        nn.Linear(512, 1024),
                        nn.BatchNorm1d(1024),
                        nn.PReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(1024, detected_feat_dim),
                        nn.BatchNorm1d(detected_feat_dim),
                    ).to(self.device)
                else:
                    self.feature_projection = nn.Sequential(
                        nn.Flatten(),
                        nn.Dropout(0.3),
                        nn.Linear(512, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.3),
                        nn.Linear(1024, detected_feat_dim),
                        nn.BatchNorm1d(detected_feat_dim),
                    ).to(self.device)

            # 加载权重
            self.load_state_dict(state_dict)
            print(f"已加载模型权重: {checkpoint_path} (feat_dim={self.feat_dim}, prelu={self.use_prelu})")

        except Exception as e:
            print(f"加载检查点失败: {e}，使用随机初始化")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, 224, 224), float32, [0,1]

        Returns:
            features: (B, 128) 的 L2 归一化特征
        """
        x = self.backbone(x)  # (B, 512, 1, 1)
        features = self.feature_projection(x)  # (B, 128)

        # L2 归一化
        features = F.normalize(features, p=2, dim=1)

        return features

    def extract(self, image: np.ndarray) -> np.ndarray:
        """从单张图像提取 128 维特征向量"""
        import cv2

        # 归一化到 [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        # 处理图像维度
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        # 调整为 (224, 224, 3)
        if image.shape[0] != 224 or image.shape[1] != 224:
            image = cv2.resize(image, (224, 224))

        # 转换为张量
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=torch.float32)

        # 提取特征
        self.eval()
        with torch.no_grad():
            features = self.forward(tensor)

        return features.squeeze(0).cpu().numpy()


# ========== 云端训练的特征提取器 (256维) ==========

class CloudTrainedFeatureExtractor(nn.Module):
    """
    云端训练的 ResNet18 特征提取器 (256维)

    用于加载 models/federated_global_model.pth 模型
    架构与训练时完全一致
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        feat_dim: int = 256,
    ):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feat_dim = feat_dim

        # 使用 torchvision 的 ResNet18 作为 backbone
        from torchvision.models import resnet18
        resnet = resnet18(weights=None)

        # 移除最后的全连接层，保留到 AdaptiveAvgPool
        # 与训练时结构一致
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 输出 (B, 512, 1, 1)

        # 特征投影层（与云端训练时的结构完全一致）
        # checkpoint中的结构: Linear(512,1024) -> BN(1024) -> PReLU(1) -> Linear(1024,512) -> BN(512)
        self.proj = nn.Sequential(
            nn.Flatten(),                    # 0
            nn.Linear(512, 1024),            # 1 - feature_projection.2
            nn.BatchNorm1d(1024),           # 2 - feature_projection.3
            nn.PReLU(),                     # 3 - feature_projection.4
            nn.Linear(1024, feat_dim),       # 4 - feature_projection.6
            nn.BatchNorm1d(feat_dim),       # 5 - feature_projection.7
        )

        self.to(self.device)

        # 可选：加载训练好的权重
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """加载检查点（支持云端训练格式）"""
        if not os.path.isfile(checkpoint_path):
            print(f"警告: 检查点不存在 {checkpoint_path}，使用随机初始化")
            return

        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            # 提取 state_dict
            if "model" in ckpt:
                # 云端训练格式
                state_dict = ckpt["model"]
                if "epoch" in ckpt:
                    print(f"加载云端训练模型: epoch={ckpt['epoch']}, loss={ckpt.get('loss', 'N/A'):.4f}")
            elif "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
                if "epoch" in ckpt:
                    print(f"加载模型: epoch={ckpt['epoch']}, loss={ckpt.get('loss', 'N/A')}")
            else:
                state_dict = ckpt

            # 检查checkpoint的feat_dim
            ckpt_args = ckpt.get("args", {})
            ckpt_feat_dim = ckpt_args.get("feat_dim", self.feat_dim)
            if ckpt_feat_dim != self.feat_dim:
                print(f"警告: checkpoint feat_dim={ckpt_feat_dim}, 当前模型 feat_dim={self.feat_dim}")
                # 需要重建proj层
                print(f"重建proj层，使用feat_dim={ckpt_feat_dim}")
                self.feat_dim = ckpt_feat_dim
                self.proj = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(512, 1024),
                    nn.BatchNorm1d(1024),
                    nn.PReLU(),
                    nn.Linear(1024, self.feat_dim),
                    nn.BatchNorm1d(self.feat_dim),
                )
                self.proj.to(self.device)

            # 处理 key 名称映射（兼容不同的命名习惯）
            # checkpoint: feature_projection.N.* -> proj.(N-1).*
            # feature_projection.2.* -> proj.1.* (Linear 512->1024)
            # feature_projection.3.* -> proj.2.* (BN 1024)
            # feature_projection.4.* -> proj.3.* (PReLU)
            # feature_projection.6.* -> proj.4.* (Linear 1024->feat_dim)
            # feature_projection.7.* -> proj.5.* (BN feat_dim)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("feature_projection."):
                    # 提取数字部分
                    suffix = k[len("feature_projection."):]
                    # .2 -> .1, .3 -> .2, .4 -> .3, .6 -> .4, .7 -> .5
                    num_part = suffix.split('.')[0]
                    rest = '.' + '.'.join(suffix.split('.')[1:])
                    mapping = {'2': '1', '3': '2', '4': '3', '6': '4', '7': '5'}
                    if num_part in mapping:
                        new_key = "proj." + mapping[num_part] + rest
                        new_state_dict[new_key] = v
                    else:
                        # 跳过不认识的key
                        pass
                else:
                    new_state_dict[k] = v

            # 加载权重（允许不严格匹配）
            load_result = self.load_state_dict(new_state_dict, strict=False)
            if load_result.missing_keys:
                print(f"警告: 缺少的key: {load_result.missing_keys[:3]}...")
            if load_result.unexpected_keys:
                print(f"警告: 多余的key: {load_result.unexpected_keys[:3]}...")
            print(f"已加载模型权重: {checkpoint_path}")

        except Exception as e:
            print(f"加载检查点失败: {e}，使用随机初始化")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, 224, 224), float32, [0,1]

        Returns:
            features: (B, 256) 的 L2 归一化特征
        """
        x = self.backbone(x)  # (B, 512, 1, 1)
        features = self.proj(x)  # (B, 256)

        # L2 归一化
        features = F.normalize(features, p=2, dim=1)

        return features

    def extract(self, image: np.ndarray) -> np.ndarray:
        """从单张图像提取 256 维特征向量"""
        import cv2

        # 归一化到 [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        # 处理图像维度
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        # 调整为 (224, 224, 3)
        if image.shape[0] != 224 or image.shape[1] != 224:
            image = cv2.resize(image, (224, 224))

        # 转换为张量
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=torch.float32)

        # 提取特征
        self.eval()
        with torch.no_grad():
            features = self.forward(tensor)

        return features.squeeze(0).cpu().numpy()


# ========== 模型工厂函数 ==========

def create_feature_extractor(
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    feat_dim: int = 256,
    model_type: str = "auto",
) -> nn.Module:
    """
    创建特征提取器（自动检测模型类型）

    Args:
        checkpoint_path: 模型检查点路径
        device: 设备
        feat_dim: 特征维度（默认256，与新模型一致）
        model_type: 模型类型
            - "auto": 自动检测（推荐）
            - "cloud_trained": 云端训练的 ResNet18 (256维)
            - "resnet18": 旧版 ResNet18 架构 (128维)
            - "palm_backbone": 使用自定义 PalmBackbone

    Returns:
        特征提取器模型
    """
    if model_type == "auto" and checkpoint_path and os.path.isfile(checkpoint_path):
        # 自动检测模型类型
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # 检查是否是云端训练格式
        if "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt.get("model_state_dict", ckpt)

        # 检查 key 格式判断模型类型
        keys = list(state_dict.keys())

        # 检查是否有 proj 层（云端训练格式）
        if any(k.startswith("proj.") for k in keys):
            print("检测到云端训练格式模型 (CloudTrained)")
            model_type = "cloud_trained"
            # 从模型权重推断特征维度
            for k in keys:
                if "proj.5.weight" in k:
                    feat_dim = state_dict[k].shape[0]
                    print(f"  特征维度: {feat_dim}")
                    break
        elif any(k.startswith("backbone.0.") for k in keys):
            # 旧版 ResNet18 格式
            print("检测到 ResNet18 格式模型")
            model_type = "resnet18"
        elif any(k.startswith("backbone.features.") for k in keys):
            # PalmBackbone 格式
            print("检测到 PalmBackbone 格式模型")
            model_type = "palm_backbone"
        else:
            # 默认使用云端训练格式
            print("使用默认云端训练格式模型")
            model_type = "cloud_trained"

    # 创建对应类型的模型
    if model_type == "cloud_trained":
        return CloudTrainedFeatureExtractor(
            checkpoint_path=checkpoint_path,
            device=device,
            feat_dim=feat_dim,
        )
    elif model_type == "resnet18":
        return ResNet18FeatureExtractor(
            checkpoint_path=checkpoint_path,
            device=device,
            feat_dim=feat_dim,
        )
    else:
        return PalmFeatureExtractor(
            checkpoint_path=checkpoint_path,
            device=device,
            feat_dim=feat_dim,
        )


def get_model_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    获取模型检查点信息

    Args:
        checkpoint_path: 模型检查点路径

    Returns:
        模型信息字典
    """
    if not os.path.isfile(checkpoint_path):
        return {"error": "文件不存在"}

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        info = {
            "path": checkpoint_path,
            "file_size_mb": os.path.getsize(checkpoint_path) / (1024 * 1024),
        }

        # 基本信息
        if "epoch" in ckpt:
            info["epoch"] = ckpt["epoch"]
        if "loss" in ckpt:
            info["loss"] = ckpt["loss"]
        if "config" in ckpt:
            info["config"] = ckpt["config"]

        # 模型结构信息
        if "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt.get("model_state_dict", ckpt)

        if isinstance(state_dict, dict):
            keys = list(state_dict.keys())
            info["num_parameters"] = len(keys)
            info["sample_keys"] = keys[:5]

            # 检测模型类型
            if any(k.startswith("proj.") for k in keys):
                info["model_type"] = "CloudTrained (ResNet18, 256维)"
                # 提取特征维度
                for k in keys:
                    if "proj.5.weight" in k:
                        info["feature_dim"] = state_dict[k].shape[0]
                        break
            elif any(k.startswith("backbone.0.") for k in keys):
                info["model_type"] = "ResNet18"
            elif any(k.startswith("backbone.features.") for k in keys):
                info["model_type"] = "PalmBackbone"
            else:
                info["model_type"] = "Unknown"

        return info

    except Exception as e:
        return {"error": str(e)}
