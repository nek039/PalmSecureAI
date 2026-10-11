"""
PalmSecureAI 全局配置文件

包含系统所有组件的配置参数：
- 掌纹识别模块配置
- 联邦学习配置
- 云端协调器配置
- 终端客户端配置
- 边缘部署配置
- 数据集配置
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ========== 项目根目录 ==========

PROJECT_ROOT = Path(__file__).parent.absolute()


# ========== 掌纹识别配置 ==========


@dataclass
class RecognitionConfig:
    """掌纹识别配置"""

    # 图像配置
    input_size: tuple[int, int] = (224, 224)
    image_channels: int = 3

    # 模型配置
    model_type: str = "mixed_ple"  # 使用 Mixed+PLE 训练的模型
    feature_dim: int = 512  # Mixed+PLE 模型使用 512 维特征

    # 模型路径（默认使用 Mixed+PLE 模型）
    model_checkpoint: Optional[str] = str(PROJECT_ROOT / "models" / "mixed_ple_final" / "palm_recognizer.pth")
    templates_path: str = str(PROJECT_ROOT / "templates.pkl")

    # 匹配配置
    similarity_threshold: float = 0.80  # 折中阈值 (识别率80%, 拒绝率60%)

    # ROI提取配置
    roi_confidence: float = 0.7
    roi_padding: int = 30

    # 预处理配置
    enable_histogram_equalization: bool = True
    gaussian_blur_kernel: int = 3
    enable_sharpening: bool = True

    # 设备配置
    device: str = "cuda"  # "cuda" or "cpu"


# ========== 联邦学习配置 ==========


@dataclass
class FederatedLearningConfig:
    """联邦学习配置"""

    # 终端配置
    num_terminals: int = 3
    terminal_ids: list[int] = field(default_factory=lambda: [1, 2, 3])

    # 训练配置
    num_rounds: int = 5
    local_epochs: int = 1
    local_batch_size: int = 32
    learning_rate: float = 0.01

    # 聚合配置
    aggregation_algorithm: str = "FedAvg"  # "FedAvg", "FedProx", "FedNova"
    min_update_count: int = 2  # 最少需要多少终端更新才聚合

    # 隐私配置
    enable_differential_privacy: bool = False
    dp_noise_multiplier: float = 0.1
    dp_max_grad_norm: float = 1.0


# ========== 云端协调器配置 ==========


@dataclass
class CloudCoordinatorConfig:
    """云端协调器配置"""

    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False

    # 模型配置
    model_dim: int = 10  # 简化演示用的模型维度
    model_save_path: str = str(PROJECT_ROOT / "global_model.pth")

    # 日志配置
    log_level: str = "INFO"
    log_file: str = str(PROJECT_ROOT / "logs" / "cloud.log")

    # 超时配置
    request_timeout: int = 30

    # 备份配置
    auto_backup: bool = True
    backup_interval: int = 10  # 每N轮备份一次
    backup_dir: str = str(PROJECT_ROOT / "backups")


# ========== 终端客户端配置 ==========


@dataclass
class TerminalClientConfig:
    """终端客户端配置"""

    # 云端连接配置
    cloud_url: str = "http://localhost:5000"
    terminal_id: int = 1

    # 本地数据配置
    local_data_dir: str = str(PROJECT_ROOT / "data" / "local" / "terminal_1")
    local_cache_dir: str = str(PROJECT_ROOT / "cache" / "terminal_1")

    # 训练配置
    training_interval: int = 300  # 每隔多少秒进行一次训练（秒）

    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None  # 动态生成

    # 重试配置
    max_retries: int = 3
    retry_delay: float = 2.0  # 秒


# ========== 边缘部署配置 ==========


@dataclass
class EdgeDeploymentConfig:
    """边缘部署配置"""

    # TFLite配置
    tflite_model_path: str = str(PROJECT_ROOT / "models" / "palm_model.tflite")
    tflite_quantization: str = "none"  # "none", "float16", "int8"

    # 推理配置
    inference_threads: int = 1
    enable_delegate: bool = False  # 是否使用硬件加速（如GPU/NPU）
    delegate_path: Optional[str] = None

    # 性能配置
    max_batch_size: int = 1
    enable_profiling: bool = False


# ========== 数据集配置 ==========


@dataclass
class DatasetConfig:
    """数据集配置"""

    # 数据目录
    data_dir: str = str(PROJECT_ROOT / "data")
    train_dir: str = ""
    val_dir: str = ""
    test_dir: str = ""

    # 数据集类型
    dataset_type: str = "custom"  # "polyu", "casia", "custom"

    # 数据加载配置
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # 数据增强配置
    enable_augmentation: bool = True
    augmentation_prob: float = 0.5

    # 数据格式
    image_size: tuple[int, int] = (224, 224)
    color_space: str = "RGB"


# ========== 训练配置 ==========


@dataclass
class TrainingConfig:
    """训练配置"""

    # 模型配置
    model_type: str = "resnet18"
    feature_dim: int = 256  # 云端训练模型使用 256 维特征
    pretrained: bool = True

    # 优化器配置
    optimizer: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9

    # 学习率调度器配置
    scheduler: str = "cosine"  # "step", "cosine", "plateau"
    warmup_epochs: int = 5
    scheduler_patience: int = 10

    # 损失函数配置
    loss_type: str = "triplet"  # "triplet", "arcface", "contrastive"
    triplet_margin: float = 0.5
    triplet_mining: str = "hard"  # "random", "semihard", "hard"

    # 训练配置
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    clip_grad_norm: float = 5.0
    local_batch_size: int = 32  # 批次大小

    # 保存配置
    save_dir: str = str(PROJECT_ROOT / "models")
    save_frequency: int = 5  # 每N个epoch保存一次

    # 日志配置
    log_dir: str = str(PROJECT_ROOT / "logs" / "training")
    tensorboard_dir: str = str(PROJECT_ROOT / "logs" / "tensorboard")
    wandb_project: Optional[str] = None

    # 混合精度训练
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16", "bfloat16"


# ========== 系统配置 ==========


@dataclass
class SystemConfig:
    """系统全局配置"""

    # 掌纹识别
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)

    # 联邦学习
    federated: FederatedLearningConfig = field(default_factory=FederatedLearningConfig)

    # 云端协调器
    cloud: CloudCoordinatorConfig = field(default_factory=CloudCoordinatorConfig)

    # 终端客户端
    terminal: TerminalClientConfig = field(default_factory=TerminalClientConfig)

    # 边缘部署
    edge: EdgeDeploymentConfig = field(default_factory=EdgeDeploymentConfig)

    # 数据集
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # 训练
    training: TrainingConfig = field(default_factory=TrainingConfig)


# ========== 配置管理 ==========


class Config:
    """配置管理器（单例模式）"""

    _instance: Optional[SystemConfig] = None

    @classmethod
    def get(cls) -> SystemConfig:
        """获取全局配置实例"""
        if cls._instance is None:
            cls._instance = SystemConfig()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """重置配置（主要用于测试）"""
        cls._instance = None

    @classmethod
    def from_env(cls) -> SystemConfig:
        """从环境变量加载配置"""
        config = SystemConfig()

        # 云端配置
        if "CLOUD_HOST" in os.environ:
            config.cloud.host = os.environ["CLOUD_HOST"]
        if "CLOUD_PORT" in os.environ:
            config.cloud.port = int(os.environ["CLOUD_PORT"])

        # 终端配置
        if "TERMINAL_ID" in os.environ:
            config.terminal.terminal_id = int(os.environ["TERMINAL_ID"])
        if "CLOUD_URL" in os.environ:
            config.terminal.cloud_url = os.environ["CLOUD_URL"]

        # 识别配置
        if "SIMILARITY_THRESHOLD" in os.environ:
            config.recognition.similarity_threshold = float(os.environ["SIMILARITY_THRESHOLD"])

        # 设备配置
        if "DEVICE" in os.environ:
            config.recognition.device = os.environ["DEVICE"]

        return config

    @classmethod
    def from_file(cls, config_path: str) -> SystemConfig:
        """
        从配置文件加载配置（支持YAML/JSON格式）

        Args:
            config_path: 配置文件路径

        Returns:
            SystemConfig 实例
        """
        import json
        from typing import Any

        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        config_dict: dict[str, Any]

        if config_file.suffix == ".json":
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        elif config_file.suffix in (".yaml", ".yml"):
            try:
                import yaml

                with open(config_file, "r", encoding="utf-8") as f:
                    config_dict = yaml.safe_load(f)
            except ImportError:
                raise ImportError(
                    "需要安装 PyYAML: pip install pyyaml"
                )
        else:
            raise ValueError(f"不支持的配置文件格式: {config_file.suffix}")

        # 应用配置
        config = SystemConfig()
        cls._apply_dict(config, config_dict)

        return config

    @classmethod
    def _apply_dict(cls, config: SystemConfig, config_dict: dict[str, Any]) -> None:
        """递归应用字典配置到 dataclass 对象"""
        for key, value in config_dict.items():
            if hasattr(config, key):
                obj = getattr(config, key)
                if isinstance(value, dict) and hasattr(obj, "__dataclass_fields__"):
                    cls._apply_dict(obj, value)
                else:
                    setattr(config, key, value)

    @classmethod
    def save(cls, config: SystemConfig, save_path: str, format: str = "yaml") -> None:
        """
        保存配置到文件

        Args:
            config: 配置对象
            save_path: 保存路径
            format: 保存格式 ("yaml" 或 "json")
        """
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        config_dict = cls._to_dict(config)

        if format == "yaml":
            try:
                import yaml

                with open(save_file, "w", encoding="utf-8") as f:
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            except ImportError:
                raise ImportError("需要安装 PyYAML: pip install pyyaml")
        elif format == "json":
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的格式: {format}")

    @classmethod
    def _to_dict(cls, obj: Any) -> Any:
        """递归将 dataclass 转换为字典"""
        from dataclasses import is_dataclass, asdict

        if is_dataclass(obj):
            return {k: cls._to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, (list, tuple)):
            return [cls._to_dict(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: cls._to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj


# ========== 便捷函数 ==========


def get_config() -> SystemConfig:
    """获取全局配置的便捷函数"""
    return Config.get()


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    设置日志配置

    Args:
        level: 日志级别
        log_file: 日志文件路径
    """
    import logging

    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers,
    )


# ========== 环境变量预设 ==========

# 开发环境预设
DEV_CONFIG = {
    "CLOUD_HOST": "localhost",
    "CLOUD_PORT": "5000",
    "DEVICE": "cpu",
    "LOG_LEVEL": "DEBUG",
}

# 生产环境预设
PROD_CONFIG = {
    "CLOUD_HOST": "0.0.0.0",
    "CLOUD_PORT": "5000",
    "DEVICE": "cuda",
    "LOG_LEVEL": "INFO",
}


if __name__ == "__main__":
    # 测试配置功能
    print("=" * 60)
    print("PalmSecureAI 配置测试")
    print("=" * 60)

    # 获取默认配置
    config = Config.get()
    print("\n默认配置:")
    print(f"  - 终端数量: {config.federated.num_terminals}")
    print(f"  - 训练轮数: {config.federated.num_rounds}")
    print(f"  - 相似度阈值: {config.recognition.similarity_threshold}")
    print(f"  - 云端端口: {config.cloud.port}")

    # 保存配置示例
    print("\n保存配置到 config.yaml...")
    Config.save(config, "config.yaml")
    Config.save(config, "config.json", format="json")
    print("配置已保存")

    # 加载配置示例
    print("\n从文件加载配置...")
    loaded_config = Config.from_file("config.yaml")
    print(f"  - 加载的终端数量: {loaded_config.federated.num_terminals}")
    print(f"  - 加载的云端端口: {loaded_config.cloud.port}")

    print("\n配置测试完成")
