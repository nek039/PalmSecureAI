# -*- coding: utf-8 -*-
"""
终端客户端 - 联邦学习训练客户端 (v2)

功能：
1. 从云端下载全局模型 (ResNet18)
2. 使用本地掌纹数据进行真实训练
3. 计算模型参数差异并上传到云端

使用场景：
- 通关系统终端（自助通关闸机、边境检查站）
- 本地存储旅客掌纹特征数据
- 定期参与联邦学习，改进全局识别模型
- 保障旅客隐私，数据不离开本地终端
"""

import os
import sys
import requests
import numpy as np
import time
import logging
import argparse
import copy
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
# __file__ = federated/terminal/client.py
# parent = federated/terminal
# parent.parent = federated
# parent.parent.parent = 项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ========== 配置 ==========

# 云端服务器地址
CLOUD_URL = 'http://localhost:5002'

# 训练配置
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# 数据路径
DATA_DIR = 'data/federated/terminals'

# 确保日志目录存在
os.makedirs('logs', exist_ok=True)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/terminal.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ========== 联邦学习客户端 ==========

class FederatedClient:
    """联邦学习客户端"""

    def __init__(self, terminal_id, cloud_url, data_dir=None):
        self.terminal_id = terminal_id
        self.cloud_url = cloud_url
        self.data_dir = data_dir or os.path.join(DATA_DIR, f'terminal_{terminal_id}')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 模型相关
        self.model = None
        self.initial_state_dict = None
        self.optimizer = None
        self.current_round = 0  # 跟踪协调器的当前轮次

        # 数据相关
        self.train_loader = None
        self.num_samples = 0

        logger.info("=" * 70)
        logger.info(f"联邦学习客户端 v2 初始化")
        logger.info(f"  终端ID: {terminal_id}")
        logger.info(f"  云端地址: {cloud_url}")
        logger.info(f"  数据目录: {self.data_dir}")
        logger.info(f"  设备: {self.device}")
        logger.info("=" * 70)

    def download_global_model(self):
        """从云端下载全局模型"""
        try:
            logger.info("下载全局模型...")

            response = requests.get(
                f'{self.cloud_url}/global_model',
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"下载失败: HTTP {response.status_code}")
                return False

            data = response.json()

            if data.get('status') != 'success':
                logger.error(f"云端返回错误: {data.get('message')}")
                return False

            model_data = data['model']
            model_info = data['model_info']
            round_num = data['round']

            # 保存协调器当前轮次
            self.current_round = round_num

            logger.info(f"下载成功 - 轮次: {round_num}")
            logger.info(f"模型信息: {model_info}")

            # 创建模型实例
            from src.recognition.feature_extractor import CloudTrainedFeatureExtractor

            self.model = CloudTrainedFeatureExtractor(device=self.device)

            # 加载 state_dict
            state_dict = {}
            for key, value in model_data.items():
                tensor_data = np.array(value['data'], dtype=np.float32)
                state_dict[key] = torch.tensor(
                    tensor_data,
                    dtype=getattr(torch, value['dtype'].replace('torch.', '')),
                    device=self.device
                ).reshape(value['shape'])

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)

            # 保存初始参数（用于计算更新）
            self.initial_state_dict = copy.deepcopy(self.model.state_dict())

            # 设置优化器
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=LEARNING_RATE,
                momentum=0.9,
                weight_decay=1e-4
            )

            logger.info("全局模型加载完成")
            return True

        except Exception as e:
            logger.error(f"下载全局模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_local_data(self):
        """加载本地训练数据"""
        try:
            from data.dataset_loader import CSVLabelDataset, create_dataloader

            train_dir = os.path.join(self.data_dir, 'train')
            labels_file = os.path.join(self.data_dir, 'train_labels.csv')

            if not os.path.exists(labels_file):
                logger.warning(f"标签文件不存在: {labels_file}")
                logger.info("使用模拟数据进行测试")
                return self._create_mock_dataloader()

            logger.info(f"加载本地数据: {train_dir}")

            dataset = CSVLabelDataset(
                csv_path=labels_file,
                image_dir=train_dir,
                image_size=(224, 224),
                augment=False
            )

            self.train_loader = create_dataloader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=True  # 避免最后一个批次大小为1导致BatchNorm错误
            )

            self.num_samples = len(dataset)
            logger.info(f"加载完成 - 样本数: {self.num_samples}")

            return True

        except Exception as e:
            logger.error(f"加载本地数据失败: {e}")
            logger.info("使用模拟数据进行测试")
            return self._create_mock_dataloader()

    def _create_mock_dataloader(self):
        """创建模拟数据加载器（用于测试）"""
        logger.info("创建模拟数据加载器...")

        # 生成模拟数据
        self.num_samples = 100

        class MockDataset:
            def __init__(self, num_samples=100):
                self.num_samples = num_samples

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                # 返回模拟图像和标签
                image = torch.randn(3, 224, 224)
                label = idx % 10
                return {'image': image, 'user_id': label}

        dataset = MockDataset(self.num_samples)
        self.train_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True  # 避免最后一个批次大小为1导致BatchNorm错误
        )

        logger.info(f"模拟数据 - 样本数: {self.num_samples}")
        return True

    def local_train(self, epochs=LOCAL_EPOCHS):
        """本地训练"""
        if self.model is None:
            logger.error("模型未加载")
            return None, None

        logger.info("=" * 70)
        logger.info(f"开始本地训练 - Epochs: {epochs}, 样本数: {self.num_samples}")
        logger.info("=" * 70)

        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            for batch_idx, batch in enumerate(self.train_loader):
                # 获取数据
                if isinstance(batch, dict):
                    images = batch['image']
                    labels = batch.get('user_id', torch.zeros(images.size(0), dtype=torch.long))
                else:
                    images, labels = batch

                images = images.to(self.device)
                labels = labels.to(self.device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=self.device)

                # 前向传播
                self.optimizer.zero_grad()
                features = self.model(images)

                # 计算损失 (使用简化的对比损失)
                loss = self._compute_loss(features, labels)

                # 反向传播
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_batches += 1

                if (batch_idx + 1) % 5 == 0:
                    logger.info(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
            logger.info(f"Epoch {epoch+1} 完成 - 平均损失: {avg_epoch_loss:.4f}")

            total_loss += epoch_loss
            total_batches += epoch_batches

        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        logger.info(f"训练完成 - 总平均损失: {avg_loss:.4f}")

        # 计算模型更新
        update = self._compute_update()

        metrics = {
            'loss': avg_loss,
            'epochs': epochs,
            'samples': self.num_samples,
            'batches': total_batches
        }

        return update, metrics

    def _compute_loss(self, features, labels):
        """计算训练损失"""
        # 归一化特征
        features = F.normalize(features, p=2, dim=1)

        # 简化的对比损失
        # 计算相似度矩阵
        similarity = torch.mm(features, features.t())

        # 创建标签掩码
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(self.device)

        # InfoNCE 风格的损失
        batch_size = features.size(0)
        positives = (similarity * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        negatives = (similarity * (1 - mask)).sum(dim=1) / (batch_size - mask.sum(dim=1) + 1e-6)

        loss = -torch.log(torch.exp(positives) / (torch.exp(positives) + torch.exp(negatives) + 1e-6) + 1e-6)
        loss = loss.mean()

        return loss

    def _compute_update(self):
        """计算模型参数更新（新参数 - 旧参数）"""
        if self.initial_state_dict is None or self.model is None:
            return None

        update = {}
        new_state_dict = self.model.state_dict()

        for key in new_state_dict.keys():
            if key in self.initial_state_dict:
                # 计算差异
                diff = new_state_dict[key].cpu() - self.initial_state_dict[key].cpu()
                update[key] = diff.numpy().tolist()

        logger.info(f"计算参数更新完成 - {len(update)} 个参数")
        return update

    def _sign_update(self, update):
        """Sign update with HMAC-SHA256"""
        try:
            import hmac
            import hashlib
            import json
            import os

            secret = os.getenv('FEDERATED_SHARED_SECRET', '').encode('utf-8')
            if not secret:
                return None

            # Create canonical representation
            canonical = json.dumps(update, sort_keys=True, separators=(',', ':'))
            message = f"{self.terminal_id}:{canonical}".encode('utf-8')

            # Generate HMAC
            signature = hmac.new(secret, message, hashlib.sha256).hexdigest()
            return signature
        except Exception as e:
            logger.warning(f"Failed to sign update: {e}")
            return None

    def upload_update(self, update, metrics):
        """上传模型更新到云端"""
        try:
            logger.info("上传模型更新...")

            request_data = {
                'terminal_id': self.terminal_id,
                'update': update,
                'sample_count': self.num_samples,
                'metrics': metrics
            }

            # Add HMAC signature for verification
            signature = self._sign_update(update)
            if signature:
                request_data['signature'] = signature
                logger.info("Update signed with HMAC-SHA256")

            response = requests.post(
                f'{self.cloud_url}/upload_update',
                json=request_data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"上传成功 - 云端状态: {result.get('status')}")
                logger.info(f"  消息: {result.get('message')}")
                return True
            else:
                logger.error(f"上传失败: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"上传更新失败: {e}")
            return False

    def run_federated_round(self, round_num):
        """执行一轮联邦学习"""
        # 下载全局模型，获取协调器当前轮次
        logger.info("步骤1：下载全局模型")
        if not self.download_global_model():
            logger.error(f"第 {round_num} 轮失败：无法下载全局模型")
            return False, None

        # 更新要上传的轮次 = 协调器当前轮次 + 1
        upload_round = self.current_round + 1
        logger.info("=" * 70)
        logger.info(f"开始第 {upload_round} 轮联邦学习 (协调器当前轮次: {self.current_round})")
        logger.info("=" * 70)

        # 步骤2：加载本地数据
        logger.info("步骤2：加载本地数据")
        if not self.load_local_data():
            logger.error(f"第 {upload_round} 轮失败：无法加载本地数据")
            return False, None

        # 步骤3：本地训练
        logger.info("步骤3：本地训练")
        update, metrics = self.local_train()

        if update is None:
            logger.error(f"第 {upload_round} 轮失败：本地训练失败")
            return False, None

        # 步骤4：上传更新
        logger.info("步骤4：上传模型更新")
        # 在 metrics 中添加轮次信息（应该是 current_round + 1）
        metrics['round'] = upload_round
        if not self.upload_update(update, metrics):
            logger.error(f"第 {upload_round} 轮失败：无法上传更新")
            return False, None

        logger.info(f"第 {upload_round} 轮完成")
        return True, metrics

    def run_federated_learning(self, num_rounds=NUM_ROUNDS):
        """运行完整的联邦学习训练"""
        logger.info("=" * 70)
        logger.info(f"开始联邦学习训练 - 总轮数: {num_rounds}")
        logger.info("=" * 70)

        results = []

        for round_idx in range(num_rounds):
            # 下载全局模型，获取协调器当前轮次
            logger.info(f"[Round {round_idx + 1}] 下载全局模型...")
            if not self.download_global_model():
                logger.error(f"第 {round_idx + 1} 轮失败：无法下载全局模型")
                return False, results

            # 确认这是我们应该在的轮次
            expected_round = self.current_round + 1
            logger.info(f"[Round {round_idx + 1}] 协调器当前轮次: {self.current_round}, 将上传轮次: {expected_round}")

            # 训练
            logger.info(f"[Round {round_idx + 1}] 加载本地数据...")
            if not self.load_local_data():
                logger.error(f"第 {round_idx + 1} 轮失败：无法加载本地数据")
                return False, results

            logger.info(f"[Round {round_idx + 1}] 开始本地训练...")
            update, metrics = self.local_train()

            if update is None:
                logger.error(f"第 {round_idx + 1} 轮失败：本地训练失败")
                return False, results

            # 上传
            logger.info(f"[Round {round_idx + 1}] 上传模型更新...")
            metrics['round'] = expected_round
            if not self.upload_update(update, metrics):
                logger.error(f"第 {round_idx + 1} 轮失败：无法上传更新")
                return False, results

            results.append(metrics)

            # 等待协调器处理完成（轮次更新）
            logger.info(f"[Round {round_idx + 1}] 等待协调器处理...")
            max_wait = 60  # 最多等待60秒
            waited = 0
            while waited < max_wait:
                time.sleep(2)
                waited += 2

                try:
                    response = requests.get(f'{self.cloud_url}/status', timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('status') == 'success':
                            coordinator_round = data['data'].get('round', 0)
                            logger.info(f"[Round {round_idx + 1}] 协调器轮次: {coordinator_round}, 目标: {expected_round}")

                            # 如果协调器轮次已经更新到我们上传的轮次，说明可以开始下一轮了
                            if coordinator_round >= expected_round:
                                logger.info(f"[Round {round_idx + 1}] 协调器已处理，开始下一轮")
                                break
                except Exception as e:
                    logger.warning(f"检查协调器状态失败: {e}")

            if waited >= max_wait:
                logger.warning(f"[Round {round_idx + 1}] 等待协调器超时，继续下一轮")

        logger.info("=" * 70)
        logger.info(f"联邦学习训练完成 - 完成轮数: {len(results)}")
        logger.info("=" * 70)

        # 输出结果摘要
        logger.info("训练结果摘要:")
        for i, metrics in enumerate(results, 1):
            logger.info(f"  第 {i} 轮: 损失={metrics['loss']:.4f}, 样本数={metrics['samples']}")

        return True, results


# ========== 主程序 ==========

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='联邦学习终端客户端')
    parser.add_argument('terminal_id', type=int, nargs='?', default=1, help='终端ID')
    parser.add_argument('--cloud_url', type=str, default=CLOUD_URL, help='云端地址')
    parser.add_argument('--data_dir', type=str, default=None, help='数据目录')
    parser.add_argument('--rounds', type=int, default=NUM_ROUNDS, help='训练轮数')
    args = parser.parse_args()

    # 设置编码
    if sys.platform == 'win32':
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')

    # 创建客户端
    client = FederatedClient(
        terminal_id=args.terminal_id,
        cloud_url=args.cloud_url,
        data_dir=args.data_dir
    )

    # 运行联邦学习
    try:
        success, results = client.run_federated_learning(num_rounds=args.rounds)

        if success:
            logger.info("联邦学习训练成功完成")
        else:
            logger.error("联邦学习训练失败")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("收到中断信号，正在退出...")
        sys.exit(0)

    except Exception as e:
        logger.error(f"运行异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
