# -*- coding: utf-8 -*-
"""
云端协调器 - 联邦学习模型协调中心 (v2)

功能：
1. 加载真实的 ResNet18 掌纹模型
2. 接收各终端的模型更新（state_dict 差异）
3. 使用 FedAvg 算法聚合更新
4. 分发更新后的全局模型

技术栈：
- Flask HTTP服务
- PyTorch 模型管理
- FedAvg 聚合算法
"""

import os
import sys
from flask import Flask, request, jsonify
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
import json
import copy

# 添加项目根目录到路径
# __file__ = federated/cloud/coordinator.py
# dirname(__file__) = federated/cloud
# dirname(dirname(__file__)) = federated
# dirname(dirname(dirname(__file__))) = 项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Security imports
from federated.cloud.security import (
    node_auth, model_signature, require_node_auth, verify_update_signature,
    secure_torch_load, SecurityError
)
from config import SecurityConfig

# ========== 配置 ==========

# 终端配置文件路径
TERMINAL_CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'terminals.json')

def load_terminal_config():
    """加载终端配置"""
    if os.path.exists(TERMINAL_CONFIG_FILE):
        try:
            with open(TERMINAL_CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载终端配置失败: {e}")
    # 返回默认配置
    return {"terminals": [
        {"id": 1, "name": "总站"},
        {"id": 2, "name": "口岸1"},
        {"id": 3, "name": "口岸2"}
    ]}

def save_terminal_config(config):
    """保存终端配置"""
    os.makedirs(os.path.dirname(TERMINAL_CONFIG_FILE), exist_ok=True)
    with open(TERMINAL_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"终端配置已保存: {TERMINAL_CONFIG_FILE}")

def get_terminal_name(terminal_id):
    """获取终端名称"""
    config = load_terminal_config()
    for t in config.get('terminals', []):
        if t['id'] == terminal_id:
            return t['name']
    return f'终端 {terminal_id}'

def get_next_terminal_id():
    """获取下一个可用的终端ID"""
    config = load_terminal_config()
    terminals = config.get('terminals', [])
    if not terminals:
        return 1
    return max(t['id'] for t in terminals) + 1

# 确保日志目录存在
os.makedirs('logs', exist_ok=True)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/cloud.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局状态
global_model = None
initial_state_dict = None  # 保存初始参数，用于计算更新
pending_updates = []
pending_sample_counts = []
pending_update_rounds = []  # 跟踪每个更新属于哪一轮
updates_received = 0
current_waiting_round = 0  # 当前正在等待的轮次
NUM_TERMINALS = 3
NUM_ROUNDS = 5

# 终端状态跟踪
terminal_status = {}  # {terminal_id: {'online': bool, 'last_seen': datetime, 'samples': int, 'loss': float, 'training': bool}}
is_training_active = False
training_rounds = 0
active_terminals = 0  # 正在训练的终端数量
terminal_processes = {}  # 存储终端进程 {terminal_id: process}

# Flask应用
app = Flask(__name__)

# 配置 CORS - 限制允许的源
allowed_origins = SecurityConfig.get_allowed_origins()

@app.after_request
def after_request(response):
    # CORS: 允许所有来源（论文项目便利性）
    # 注意：生产环境应限制为特定域名
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Node-API-Key,X-Terminal-ID')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    # 添加安全响应头
    response.headers.add('X-Content-Type-Options', 'nosniff')
    response.headers.add('X-Frame-Options', 'DENY')
    return response

# 模型配置
MODEL_CHECKPOINT = 'models/palm_recognizer.pth'
MODEL_SAVE_PATH = 'models/federated_global_model.pth'
MODEL_BACKUP_DIR = 'models/federated_backups'
MODEL_VERSION = '2.0.0'

# 演示模式配置 - 启用后训练结果保存在副本目录，不影响原始模型
DEMO_MODE = True  # 演示模式：训练在副本上进行，原始模型不变
DEMO_BACKUP_DIR = 'models/federated_demo'
DEMO_CHECKPOINT = f'{DEMO_BACKUP_DIR}/checkpoint.pth'
DEMO_SAVE_PATH = f'{DEMO_BACKUP_DIR}/global_model.pth'
DEMO_MODEL_BACKUP_DIR = f'{DEMO_BACKUP_DIR}/rounds'

# 保护机制配置
ENABLE_PROTECTION = False  # 暂时禁用保护机制，方便测试
MIN_ACCURACY_THRESHOLD = 0.90  # 最低准确率阈值 (90%)
TEST_DATA_DIR = 'data/federated/global_test'  # 测试集目录


# ========== 模型定义 ==========

class GlobalModel:
    """全局模型管理类"""

    def __init__(self, checkpoint_path=None):
        self.model = None
        self.round = 0
        self.version = MODEL_VERSION
        self.created_at = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if checkpoint_path:
            self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path):
        """加载预训练模型"""
        from src.recognition.feature_extractor import CloudTrainedFeatureExtractor

        logger.info(f"加载全局模型: {checkpoint_path}")

        try:
            self.model = CloudTrainedFeatureExtractor(
                checkpoint_path=checkpoint_path,
                device=self.device
            )
            self.model.eval()  # 设置为评估模式
            self.created_at = datetime.now()

            # 统计模型信息
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            logger.info(f"模型加载成功!")
            logger.info(f"  设备: {self.device}")
            logger.info(f"  总参数: {total_params:,}")
            logger.info(f"  可训练参数: {trainable_params:,}")
            logger.info(f"  特征维度: {self.model.feat_dim}")

            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def get_state_dict_serializable(self):
        """获取可序列化的 state_dict (转换为 list)"""
        if self.model is None:
            return None

        state_dict = self.model.state_dict()
        serializable = {}
        for key, value in state_dict.items():
            serializable[key] = {
                'data': value.cpu().numpy().tolist(),
                'dtype': str(value.dtype),
                'shape': list(value.shape)
            }
        return serializable

    def get_model_info(self):
        """获取模型信息"""
        if self.model is None:
            return {'loaded': False}

        return {
            'loaded': True,
            'round': self.round,
            'version': self.version,
            'feature_dim': self.model.feat_dim,
            'device': str(self.device),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'total_params': sum(p.numel() for p in self.model.parameters())
        }

    def apply_aggregated_update(self, aggregated_update, enable_protection=ENABLE_PROTECTION):
        """
        应用聚合后的更新（带保护机制）

        Args:
            aggregated_update: 聚合后的参数更新
            enable_protection: 是否启用退化保护

        Returns:
            success: 是否成功
            message: 结果消息
        """
        if self.model is None:
            logger.error("模型未初始化")
            return False, "模型未初始化"

        try:
            # 保存旧模型状态（用于回滚）
            old_state_dict = copy.deepcopy(self.model.state_dict())
            old_accuracy = getattr(self, 'current_accuracy', None)

            # 应用更新
            state_dict = self.model.state_dict()
            for key, update_data in aggregated_update.items():
                if key in state_dict:
                    update_tensor = torch.tensor(update_data, dtype=state_dict[key].dtype, device=self.device)
                    state_dict[key] = state_dict[key] + update_tensor

            self.model.load_state_dict(state_dict)

            # 评估新模型（保护机制或演示模式都需要）
            new_accuracy = protection_manager.evaluate_model(self.model)

            if enable_protection:
                accept, reason = protection_manager.should_accept_update(old_accuracy, new_accuracy, self.round + 1)

                logger.info(f"保护检查: {reason}")

                if not accept:
                    # 回滚到旧模型
                    self.model.load_state_dict(old_state_dict)
                    logger.warning(f"模型更新被拒绝，已回滚: {reason}")
                    return False, reason

            # 更新准确率记录（演示模式和保护模式都需要）
            if new_accuracy is not None:
                self.current_accuracy = new_accuracy
                protection_manager.accuracy_history.append({
                    'round': self.round + 1,
                    'accuracy': new_accuracy
                })
                logger.info(f"模型评估完成 - 第 {self.round + 1} 轮准确率: {new_accuracy:.4f}")
            else:
                logger.warning("无法评估模型准确率")

            self.round += 1
            logger.info(f"全局模型已更新到轮次 {self.round}")

            # 演示模式：保存每轮的模型快照
            if DEMO_MODE:
                new_accuracy = getattr(self, 'current_accuracy', None)
                demo_manager.save_round_model(
                    self.model.state_dict(),
                    self.round,
                    new_accuracy
                )

            return True, "更新成功"

        except Exception as e:
            logger.error(f"应用更新失败: {e}")
            return False, str(e)

    def save_model(self, path=None):
        """保存模型"""
        if self.model is None:
            return False

        path = path or MODEL_SAVE_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            'model': self.model.state_dict(),
            'round': self.round,
            'version': self.version,
            'timestamp': datetime.now().isoformat()
        }, path)

        logger.info(f"模型已保存: {path}")
        return True


# ========== 演示模式管理器 ==========

class DemoModeManager:
    """演示模式管理器 - 训练在副本上进行，不影响原始模型"""

    def __init__(self):
        self.backup_dir = DEMO_BACKUP_DIR
        self.checkpoint_path = DEMO_CHECKPOINT
        self.save_path = DEMO_SAVE_PATH
        self.rounds_dir = DEMO_MODEL_BACKUP_DIR

    def prepare_demo_model(self):
        """准备演示模式模型：如果不存在则从原始模型复制"""
        if not DEMO_MODE:
            logger.info("演示模式未启用，使用原始模型")
            return MODEL_CHECKPOINT

        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.rounds_dir, exist_ok=True)

        # 如果演示模型已存在，直接使用
        if os.path.exists(self.checkpoint_path):
            logger.info(f"使用已有的演示模型: {self.checkpoint_path}")
            return self.checkpoint_path

        # 从原始模型复制到演示目录
        if not os.path.exists(MODEL_CHECKPOINT):
            logger.warning(f"原始模型不存在: {MODEL_CHECKPOINT}")
            return MODEL_CHECKPOINT

        import shutil
        shutil.copy2(MODEL_CHECKPOINT, self.checkpoint_path)
        logger.info(f"已创建演示模型副本: {self.checkpoint_path}")

        # 同时复制到 save_path（这是训练会保存的地方）
        shutil.copy2(MODEL_CHECKPOINT, self.save_path)
        logger.info(f"演示模型保存路径: {self.save_path}")

        return self.checkpoint_path

    def get_save_path(self):
        """获取演示模式的保存路径"""
        if DEMO_MODE:
            return self.save_path
        return MODEL_SAVE_PATH

    def get_checkpoint_path(self):
        """获取演示模式的检查点路径"""
        if DEMO_MODE:
            return self.checkpoint_path
        return MODEL_CHECKPOINT

    def save_round_model(self, state_dict, round_num, accuracy=None):
        """保存每轮的模型快照"""
        if not DEMO_MODE:
            return

        os.makedirs(self.rounds_dir, exist_ok=True)
        path = os.path.join(self.rounds_dir, f'round_{round_num}.pth')

        torch.save({
            'model': state_dict,
            'round': round_num,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat()
        }, path)
        logger.info(f"演示模式：已保存第 {round_num} 轮模型: {path}")

    def get_demo_info(self):
        """获取演示模式信息"""
        if not DEMO_MODE:
            return {'enabled': False}

        rounds = []
        if os.path.exists(self.rounds_dir):
            for f in sorted(os.listdir(self.rounds_dir)):
                if f.endswith('.pth'):
                    rounds.append(f)

        return {
            'enabled': True,
            'backup_dir': self.backup_dir,
            'checkpoint_exists': os.path.exists(self.checkpoint_path),
            'saved_rounds': len(rounds)
        }


# 创建演示模式管理器实例
demo_manager = DemoModeManager()


# ========== 聚合管理器 ==========

class AggregationManager:
    """FedAvg 聚合管理器"""

    def aggregate_updates(self, updates, sample_counts):
        """
        FedAvg 聚合算法

        Args:
            updates: 各终端的参数更新列表
            sample_counts: 各终端的样本数量

        Returns:
            aggregated: 聚合后的更新
        """
        if not updates:
            logger.warning("没有待聚合的更新")
            return None

        logger.info(f"开始 FedAvg 聚合 - {len(updates)} 个终端")

        # 计算总样本数
        total_samples = sum(sample_counts)
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"各终端样本数: {sample_counts}")

        # 计算权重
        weights = [n / total_samples for n in sample_counts]
        logger.info(f"聚合权重: {[f'{w:.3f}' for w in weights]}")

        # 加权平均
        aggregated = {}
        first_update = updates[0]

        # 检查更新格式：字典还是列表
        if isinstance(first_update, dict):
            # 字典格式：{'weights': [...], 'bias': [...]}
            for key in first_update.keys():
                weighted_sum = np.zeros_like(first_update[key], dtype=np.float64)
                for update, weight in zip(updates, weights):
                    weighted_sum += np.array(update[key], dtype=np.float64) * weight
                aggregated[key] = weighted_sum.tolist()
        elif isinstance(first_update, list):
            # 列表格式：[0.001, 0.002, ...]
            weighted_sum = np.zeros_like(first_update, dtype=np.float64)
            for update, weight in zip(updates, weights):
                weighted_sum += np.array(update, dtype=np.float64) * weight
            aggregated['weights'] = weighted_sum.tolist()
        else:
            logger.error(f"未知的更新格式: {type(first_update)}")
            return None

        logger.info("FedAvg 聚合完成")
        return aggregated


# 创建全局实例
aggregator = AggregationManager()


# ========== 模型保护管理器 ==========

class ModelProtectionManager:
    """模型退化保护管理器"""

    def __init__(self, backup_dir=MODEL_BACKUP_DIR):
        self.backup_dir = backup_dir
        self.accuracy_history = []
        os.makedirs(backup_dir, exist_ok=True)

    def evaluate_model(self, model, test_dir=None):
        """
        评估模型准确率

        Returns:
            accuracy: 准确率 (0-1)
        """
        if test_dir is None:
            test_dir = TEST_DATA_DIR

        if not os.path.exists(test_dir):
            logger.warning(f"测试目录不存在: {test_dir}")
            return None

        try:
            # 尝试加载测试数据
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from data.dataset_loader import CSVLabelDataset, create_dataloader

            # 查找标签文件
            labels_file = None
            for possible in [
                os.path.join(test_dir, 'test_labels.csv'),
                os.path.join(os.path.dirname(test_dir), 'global_test_labels.csv')
            ]:
                if os.path.exists(possible):
                    labels_file = possible
                    break

            if labels_file is None:
                logger.warning("找不到测试标签文件")
                return None

            dataset = CSVLabelDataset(
                csv_path=labels_file,
                image_dir=test_dir,
                image_size=(224, 224),
                augment=False
            )

            dataloader = create_dataloader(dataset, batch_size=32, shuffle=False, num_workers=0)

            # 提取特征并计算相似度
            model.eval()
            all_features = []
            all_labels = []

            with torch.no_grad():
                for batch in dataloader:
                    images = batch['image'].to(model.device)
                    features = model(images)
                    features = F.normalize(features, p=2, dim=1)
                    all_features.append(features.cpu().numpy())
                    all_labels.extend(batch['user_id'].tolist())

            features = np.vstack(all_features)
            labels = np.array(all_labels)

            # 计算准确率：使用平均相似度比较
            # 对于每个样本，计算与所有同类样本的平均相似度，与异类样本的平均相似度比较
            similarity = features @ features.T
            correct = 0

            for i in range(len(labels)):
                query_label = labels[i]

                # 找出所有同类样本的索引
                same_user_indices = [j for j in range(len(labels)) if labels[j] == query_label and j != i]

                if len(same_user_indices) == 0:
                    continue

                # 计算与同类样本的平均相似度
                intra_sim = np.mean([similarity[i, j] for j in same_user_indices])

                # 计算与所有异类样本的平均相似度
                diff_user_indices = [j for j in range(len(labels)) if labels[j] != query_label]
                inter_sim = np.mean([similarity[i, j] for j in diff_user_indices])

                # 如果同类相似度 > 异类相似度，算正确
                if intra_sim > inter_sim:
                    correct += 1

            accuracy = correct / len(labels)
            logger.info(f"模型评估完成 - 准确率: {accuracy:.4f}")

            return accuracy

        except Exception as e:
            logger.warning(f"模型评估失败: {e}")
            return None

    def backup_model(self, model_state, round_num, accuracy=None):
        """备份模型"""
        backup_path = os.path.join(self.backup_dir, f'model_round_{round_num}.pth')

        torch.save({
            'model': model_state,
            'round': round_num,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat()
        }, backup_path)

        logger.info(f"模型已备份: {backup_path}")
        return backup_path

    def should_accept_update(self, old_accuracy, new_accuracy, round_num):
        """
        判断是否应该接受更新

        Returns:
            accept: 是否接受
            reason: 原因
        """
        if old_accuracy is None:
            # 首次评估，接受
            return True, "首次评估，接受更新"

        if new_accuracy is None:
            # 无法评估，保守起见不接受
            return False, "无法评估新模型，拒绝更新"

        if new_accuracy < MIN_ACCURACY_THRESHOLD:
            # 低于最低阈值
            return False, f"新模型准确率 {new_accuracy:.4f} 低于阈值 {MIN_ACCURACY_THRESHOLD}"

        if new_accuracy < old_accuracy * 0.98:
            # 准确率下降超过 2%，拒绝
            degradation = (old_accuracy - new_accuracy) / old_accuracy * 100
            return False, f"模型退化 {degradation:.2f}%，拒绝更新"

        return True, f"新模型准确率 {new_accuracy:.4f} >= 旧模型 {old_accuracy:.4f}，接受更新"

    def get_latest_good_model(self):
        """获取最新的好模型"""
        backups = sorted(Path(self.backup_dir).glob('model_round_*.pth'), reverse=True)

        for backup in backups:
            try:
                data = secure_torch_load(str(backup), map_location='cpu')
                if data.get('accuracy', 0) >= MIN_ACCURACY_THRESHOLD:
                    logger.info(f"找到可回滚模型: {backup}")
                    return backup, data
            except Exception as e:
                logger.warning(f"加载备份失败 {backup}: {e}")
                continue

        return None, None


# 创建保护管理器实例
protection_manager = ModelProtectionManager()


# ========== API 路由 ==========

@app.route('/')
def home():
    """系统首页"""
    return jsonify({
        'system': 'PalmSecureAI - Federated Learning Cloud Coordinator v2',
        'version': MODEL_VERSION,
        'status': 'running',
        'description': '云端训练的 ResNet18 掌纹识别模型',
        'model_loaded': global_model is not None and global_model.model is not None,
        'current_round': global_model.round if global_model else 0,
        'num_terminals': NUM_TERMINALS,
        'num_rounds': NUM_ROUNDS,
        'api_endpoints': {
            'model_info': '/model_info',
            'global_model': '/global_model',
            'upload_update': '/upload_update',
            'status': '/status',
            'reset': '/reset'
        }
    })


@app.route('/health')
def health():
    """健康检查"""
    try:
        return jsonify({
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'service': 'cloud_coordinator_v2',
            'current_round': global_model.round if global_model else 0,
            'pending_updates': len(pending_updates),
            'global_model_ready': global_model is not None and global_model.model is not None
        })
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/model_info')
def model_info():
    """获取模型信息"""
    if global_model is None:
        return jsonify({'status': 'error', 'message': '模型未初始化'})

    return jsonify({
        'status': 'success',
        'model': global_model.get_model_info()
    })


@app.route('/global_model', methods=['GET'])
def get_global_model():
    """终端下载全局模型"""
    global global_model

    try:
        if global_model is None or global_model.model is None:
            # 首次请求，初始化模型
            logger.info("首次请求，初始化全局模型...")

            # 演示模式：使用副本模型，不影响原始模型
            if DEMO_MODE:
                checkpoint_path = demo_manager.prepare_demo_model()
                logger.info(f"演示模式：使用演示模型 {checkpoint_path}")
                global_model = GlobalModel(checkpoint_path=checkpoint_path)
            else:
                global_model = GlobalModel(checkpoint_path=MODEL_CHECKPOINT)

            if global_model.model is None:
                return jsonify({
                    'status': 'error',
                    'message': '模型初始化失败'
                }), 500

        logger.info(f"分发全局模型 - 轮次: {global_model.round}")

        return jsonify({
            'status': 'success',
            'round': global_model.round,
            'model': global_model.get_state_dict_serializable(),
            'model_info': global_model.get_model_info()
        })

    except Exception as e:
        logger.error(f"获取全局模型失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/upload_update', methods=['POST'])
def upload_update():
    """终端上传本地训练的模型更新"""
    global pending_updates, pending_sample_counts, pending_update_rounds, updates_received, terminal_status, current_waiting_round

    try:
        data = request.get_json()

        if not data:
            return jsonify({'status': 'error', 'message': '请求数据为空'}), 400

        terminal_id = data.get('terminal_id')
        update = data.get('update')  # state_dict 的差异
        metrics = data.get('metrics', {})
        # sample_count 可能在顶层或 metrics 中
        sample_count = data.get('sample_count', metrics.get('sample_count', 100))
        signature = data.get('signature')  # HMAC signature for verification

        # Validate terminal_id
        if not terminal_id:
            return jsonify({'status': 'error', 'message': 'Terminal ID required'}), 400

        # Check if terminal is in allowed list (if security is configured)
        if os.getenv('FEDERATED_SHARED_SECRET'):
            # 获取允许的终端ID列表
            terminal_config = load_terminal_config()
            allowed_ids = [str(t['id']) for t in terminal_config.get('terminals', [])]
            if str(terminal_id) not in allowed_ids:
                logger.warning(f"Unauthorized terminal {terminal_id} attempted upload")
                return jsonify({'status': 'error', 'message': 'Terminal not authorized'}), 403

            # Verify signature if provided
            if signature:
                is_valid, sig_msg = verify_update_signature(update, signature, terminal_id)
                if not is_valid:
                    logger.warning(f"Invalid signature from terminal {terminal_id}: {sig_msg}")
                    return jsonify({'status': 'error', 'message': f'Invalid signature: {sig_msg}'}), 401
                logger.info(f"Signature verified for terminal {terminal_id}")
            else:
                logger.warning(f"No signature provided by terminal {terminal_id}, accepting anyway")

        logger.info(f"收到终端 {terminal_id} 的模型更新")
        logger.info(f"  样本数: {sample_count}")
        if metrics:
            logger.info(f"  指标: loss={metrics.get('loss', 'N/A'):.4f}")

        # 更新终端状态
        terminal_status[str(terminal_id)] = {
            'id': terminal_id,
            'name': get_terminal_name(terminal_id),
            'online': True,
            'last_seen': datetime.now().isoformat(),
            'sample_count': sample_count,
            'loss': metrics.get('loss', 0),
            'local_rounds': terminal_status.get(str(terminal_id), {}).get('local_rounds', 0) + 1
        }

        # 获取当前轮次（协调器模型当前轮次）
        current_round = global_model.round if global_model else 0
        update_round = metrics.get('round', current_round)  # 终端应该在请求中说明是哪一轮

        # 首次更新或未初始化时，设置当前等待的轮次
        # 我们正在等待的是将模型从 current_round 更新到 current_round + 1
        if current_waiting_round == 0:
            current_waiting_round = current_round + 1
            logger.info(f"初始化等待轮次: {current_waiting_round} (模型当前轮次: {current_round})")

        # 如果收到的更新轮次与当前等待轮次不匹配，说明是旧轮次的延迟更新或乱序更新
        if update_round < current_waiting_round:
            logger.warning(f"收到旧轮次更新: 终端{terminal_id} 发送 round {update_round}，但我们正在等待 round {current_waiting_round}，忽略")
            return jsonify({
                'status': 'stale',
                'message': f'收到旧轮次 {update_round} 的更新，当前等待 {current_waiting_round}'
            }), 400

        # 如果收到的更新轮次超前，说明是未来的更新
        if update_round > current_waiting_round:
            logger.warning(f"收到未来轮次更新: 终端{terminal_id} 发送 round {update_round}，但我们正在等待 round {current_waiting_round}，拒绝")
            return jsonify({
                'status': 'future',
                'message': f'收到未来轮次 {update_round} 的更新，当前等待 {current_waiting_round}'
            }), 400

        # 记录更新
        pending_updates.append(update)
        pending_sample_counts.append(sample_count)
        pending_update_rounds.append(update_round)
        updates_received += 1

        progress = len(pending_updates)
        logger.info(f"进度: {progress}/{NUM_TERMINALS} (round {update_round})")

        # 如果所有终端都上传了，进行聚合
        if len(pending_updates) >= NUM_TERMINALS:
            logger.info(f"收到所有 {NUM_TERMINALS} 个终端的更新，开始聚合...")

            # FedAvg 聚合
            aggregated = aggregator.aggregate_updates(pending_updates, pending_sample_counts)

            if aggregated is None:
                return jsonify({'status': 'error', 'message': '聚合失败'}), 500

            # 备份当前模型（在应用更新前）
            if ENABLE_PROTECTION:
                protection_manager.backup_model(
                    global_model.model.state_dict(),
                    global_model.round,
                    getattr(global_model, 'current_accuracy', None)
                )

            # 应用聚合后的更新（带保护检查）
            success, message = global_model.apply_aggregated_update(aggregated)

            if success:
                # 保存模型（演示模式保存到副本目录）
                save_path = demo_manager.get_save_path() if DEMO_MODE else None
                global_model.save_model(save_path)

                # 备份新模型
                if ENABLE_PROTECTION:
                    protection_manager.backup_model(
                        global_model.model.state_dict(),
                        global_model.round,
                        getattr(global_model, 'current_accuracy', None)
                    )

                # 更新等待轮次（递增）
                completed_round = current_waiting_round
                current_waiting_round = global_model.round + 1  # 下一轮更新

                # 清空待聚合的更新
                pending_updates.clear()
                pending_sample_counts.clear()
                pending_update_rounds.clear()
                updates_received = 0

                logger.info(f"第 {completed_round} 轮聚合完成，等待轮次更新为 {current_waiting_round}")

                return jsonify({
                    'status': 'aggregated',
                    'round': global_model.round,
                    'accuracy': getattr(global_model, 'current_accuracy', None),
                    'message': f'第 {global_model.round} 轮聚合完成 - {message}'
                })
            else:
                # 更新被拒绝，但不清空轮次追踪
                pending_updates.clear()
                pending_sample_counts.clear()
                pending_update_rounds.clear()
                updates_received = 0

                return jsonify({
                    'status': 'rejected',
                    'round': global_model.round,
                    'message': f'模型更新被拒绝: {message}'
                }), 400
        else:
            return jsonify({
                'status': 'received',
                'pending': len(pending_updates),
                'message': f'已收到 {len(pending_updates)} 个终端更新，等待更多终端...'
            })

    except Exception as e:
        logger.error(f"处理模型更新失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/status')
def status():
    """获取系统状态"""
    # 计算在线终端数
    online_count = sum(1 for t in terminal_status.values() if t.get('online', False))

    # 从配置获取终端列表，并与运行时状态合并
    config = load_terminal_config()
    terminals = []
    for t_config in config.get('terminals', []):
        tid = t_config['id']
        runtime_status = terminal_status.get(str(tid), {})
        terminals.append({
            'id': tid,
            'name': t_config.get('name', f'终端 {tid}'),
            'online': runtime_status.get('online', False),
            'training': runtime_status.get('training', False),
            'local_rounds': runtime_status.get('local_rounds', 0),
            'sample_count': runtime_status.get('sample_count', 0),
            'loss': runtime_status.get('loss', 0),
            'last_update': runtime_status.get('last_seen')
        })

    # 判断是否有终端正在训练
    any_training = any(t.get('training', False) for t in terminal_status.values())
    actual_is_training = is_training_active or any_training

    response_data = {
        'status': 'success',
        'data': {
            'model_loaded': global_model is not None and global_model.model is not None,
            'round': global_model.round if global_model else 0,
            'accuracy': getattr(global_model, 'current_accuracy', None) if global_model else None,
            'pending_updates': len(pending_updates),
            'num_terminals': len(config.get('terminals', [])),
            'online_terminals': online_count,
            'active_terminals': active_terminals,
            'protection_enabled': ENABLE_PROTECTION,
            'accuracy_history': protection_manager.accuracy_history[-10:] if protection_manager.accuracy_history else [],
            'terminals': terminals,
            'is_training': actual_is_training,
            'training_rounds': training_rounds
        }
    }
    return jsonify(response_data)


@app.route('/start_training', methods=['POST'])
def start_training():
    """启动联邦学习训练（通过 Web 触发）"""
    import subprocess
    import threading
    global is_training_active, training_rounds, active_terminals, terminal_processes

    data = request.get_json() or {}
    rounds = data.get('rounds', 1)

    # 获取实际终端数量
    config = load_terminal_config()
    terminals_list = config.get('terminals', [])
    num_terminals = len(terminals_list)

    if num_terminals == 0:
        return jsonify({'status': 'error', 'message': '没有配置终端'}), 400

    is_training_active = True
    training_rounds = rounds
    active_terminals = num_terminals
    terminal_processes = {}

    def run_terminal_clients():
        """在后台运行终端客户端"""
        global is_training_active, active_terminals, terminal_processes

        processes = []
        for t_config in terminals_list:
            terminal_id = t_config['id']
            terminal_name = t_config.get('name', f'终端 {terminal_id}')
            try:
                logger.info(f"启动终端 {terminal_name} 训练...")
                # 更新终端状态为训练中
                terminal_status[str(terminal_id)] = {
                    'id': terminal_id,
                    'name': terminal_name,
                    'online': True,
                    'training': True,
                    'last_seen': datetime.now().isoformat(),
                    'sample_count': 0,
                    'loss': 0,
                    'local_rounds': 0
                }

                # 使用 subprocess 启动终端客户端
                process = subprocess.Popen(
                    [sys.executable, 'terminal/client.py', str(terminal_id), '--rounds', str(rounds)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                processes.append((terminal_id, process))
                terminal_processes[terminal_id] = process
                logger.info(f"终端 {terminal_name} 已启动 (PID: {process.pid})")

            except Exception as e:
                logger.error(f"启动终端 {terminal_name} 失败: {e}")
                active_terminals -= 1

        # 等待所有进程完成
        for terminal_id, process in processes:
            try:
                process.wait()  # 等待进程完成
                logger.info(f"终端 {terminal_id} 训练完成")

                # 更新终端状态
                if str(terminal_id) in terminal_status:
                    terminal_status[str(terminal_id)]['training'] = False

            except Exception as e:
                logger.error(f"等待终端 {terminal_id} 完成时出错: {e}")

        # 所有训练完成后更新状态
        is_training_active = False
        active_terminals = 0
        terminal_processes = {}
        logger.info("所有终端训练完成")

    # 在后台线程中启动终端
    thread = threading.Thread(target=run_terminal_clients)
    thread.daemon = True
    thread.start()

    return jsonify({
        'status': 'success',
        'message': f'已启动 {num_terminals} 个终端进行 {rounds} 轮训练',
        'rounds': rounds,
        'terminals': num_terminals
    })


@app.route('/protection/status')
def protection_status():
    """获取保护机制状态"""
    backups = list(Path(MODEL_BACKUP_DIR).glob('model_round_*.pth'))

    return jsonify({
        'status': 'success',
        'data': {
            'enabled': ENABLE_PROTECTION,
            'min_accuracy_threshold': MIN_ACCURACY_THRESHOLD,
            'backup_count': len(backups),
            'accuracy_history': protection_manager.accuracy_history,
            'backups': [
                {'file': b.name, 'round': int(b.stem.split('_')[-1])}
                for b in sorted(backups, reverse=True)[:10]
            ]
        }
    })


@app.route('/reset', methods=['POST'])
def reset_system():
    """重置联邦学习系统"""
    global pending_updates, pending_sample_counts, pending_update_rounds, updates_received, terminal_status, is_training_active, current_waiting_round

    pending_updates = []
    pending_sample_counts = []
    pending_update_rounds = []
    updates_received = 0
    current_waiting_round = 0
    terminal_status = {}
    is_training_active = False

    logger.info("系统已重置")

    return jsonify({
        'status': 'success',
        'message': '系统已重置'
    })


# ========== 终端管理 API ==========

@app.route('/terminals', methods=['GET'])
def get_terminals():
    """获取所有终端配置"""
    config = load_terminal_config()
    return jsonify({
        'status': 'success',
        'data': config
    })


@app.route('/terminals', methods=['POST'])
def add_terminal():
    """添加新终端"""
    global NUM_TERMINALS
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': '请求数据为空'}), 400

        name = data.get('name', '').strip()

        if not name:
            return jsonify({'status': 'error', 'message': '终端名称不能为空'}), 400

        config = load_terminal_config()
        new_id = get_next_terminal_id()

        new_terminal = {
            'id': new_id,
            'name': name
        }

        config['terminals'].append(new_terminal)
        save_terminal_config(config)

        # 更新终端数量
        NUM_TERMINALS = len(config['terminals'])

        logger.info(f"添加新终端: {new_terminal}")

        return jsonify({
            'status': 'success',
            'message': '终端添加成功',
            'data': new_terminal
        })

    except Exception as e:
        logger.error(f"添加终端失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/terminals/<int:terminal_id>', methods=['PUT'])
def update_terminal(terminal_id):
    """更新终端信息"""
    global terminal_status
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': '请求数据为空'}), 400

        config = load_terminal_config()
        found = False

        for t in config['terminals']:
            if t['id'] == terminal_id:
                if 'name' in data:
                    t['name'] = data['name'].strip()
                found = True
                break

        if not found:
            return jsonify({'status': 'error', 'message': f'终端 {terminal_id} 不存在'}), 404

        save_terminal_config(config)

        # 更新运行时状态中的名称
        if str(terminal_id) in terminal_status:
            terminal_status[str(terminal_id)]['name'] = t['name']

        logger.info(f"更新终端 {terminal_id}: {t}")

        return jsonify({
            'status': 'success',
            'message': '终端更新成功',
            'data': t
        })

    except Exception as e:
        logger.error(f"更新终端失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/terminals/<int:terminal_id>', methods=['DELETE'])
def delete_terminal(terminal_id):
    """删除终端"""
    global NUM_TERMINALS, terminal_status
    try:
        config = load_terminal_config()

        # 查找并删除终端
        original_count = len(config['terminals'])
        config['terminals'] = [t for t in config['terminals'] if t['id'] != terminal_id]

        if len(config['terminals']) == original_count:
            return jsonify({'status': 'error', 'message': f'终端 {terminal_id} 不存在'}), 404

        save_terminal_config(config)

        # 更新终端数量
        NUM_TERMINALS = len(config['terminals'])

        # 从运行时状态中移除
        if str(terminal_id) in terminal_status:
            del terminal_status[str(terminal_id)]

        logger.info(f"删除终端: {terminal_id}")

        return jsonify({
            'status': 'success',
            'message': f'终端 {terminal_id} 已删除'
        })

    except Exception as e:
        logger.error(f"删除终端失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/terminals/<int:terminal_id>/start', methods=['POST'])
def start_single_terminal(terminal_id):
    """启动单个终端"""
    import subprocess
    try:
        terminal_name = get_terminal_name(terminal_id)
        logger.info(f"启动终端 {terminal_id} ({terminal_name})...")

        # 使用 subprocess 启动终端客户端
        process = subprocess.Popen(
            [sys.executable, 'terminal/client.py', str(terminal_id), '--rounds', '1'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        logger.info(f"终端 {terminal_name} 已启动 (PID: {process.pid})")

        # 更新终端状态为在线
        terminal_status[str(terminal_id)] = {
            'id': terminal_id,
            'name': terminal_name,
            'online': True,
            'last_seen': datetime.now().isoformat(),
            'sample_count': 0,
            'loss': 0,
            'local_rounds': 0
        }

        return jsonify({
            'status': 'success',
            'message': f'终端 {terminal_name} 已启动',
            'pid': process.pid
        })

    except Exception as e:
        logger.error(f"启动终端失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/terminals/<int:terminal_id>/stop', methods=['POST'])
def stop_single_terminal(terminal_id):
    """停止单个终端"""
    try:
        terminal_name = get_terminal_name(terminal_id)

        # 更新终端状态为离线
        if str(terminal_id) in terminal_status:
            terminal_status[str(terminal_id)]['online'] = False
            terminal_status[str(terminal_id)]['training'] = False

        # 尝试终止进程
        if terminal_id in terminal_processes:
            try:
                terminal_processes[terminal_id].terminate()
                logger.info(f"已终止终端 {terminal_name} 的进程")
            except:
                pass

        logger.info(f"终端 {terminal_name} 已停止")

        return jsonify({
            'status': 'success',
            'message': f'终端 {terminal_name} 已停止'
        })

    except Exception as e:
        logger.error(f"停止终端失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/terminals/<int:terminal_id>/upload', methods=['POST'])
def upload_single_terminal(terminal_id):
    """手动触发单个终端上传更新"""
    import subprocess
    try:
        terminal_name = get_terminal_name(terminal_id)

        # 检查终端是否存在
        config = load_terminal_config()
        terminal_exists = any(t['id'] == terminal_id for t in config.get('terminals', []))
        if not terminal_exists:
            return jsonify({'status': 'error', 'message': f'终端 {terminal_id} 不存在'}), 404

        logger.info(f"手动触发终端 {terminal_name} 上传更新...")

        # 更新终端状态
        terminal_status[str(terminal_id)] = {
            'id': terminal_id,
            'name': terminal_name,
            'online': True,
            'training': True,
            'last_seen': datetime.now().isoformat(),
            'sample_count': terminal_status.get(str(terminal_id), {}).get('sample_count', 0),
            'loss': terminal_status.get(str(terminal_id), {}).get('loss', 0),
            'local_rounds': terminal_status.get(str(terminal_id), {}).get('local_rounds', 0)
        }

        # 使用 subprocess 启动终端客户端进行上传
        process = subprocess.Popen(
            [sys.executable, 'terminal/client.py', str(terminal_id), '--rounds', '1'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        logger.info(f"终端 {terminal_name} 上传进程已启动 (PID: {process.pid})")

        return jsonify({
            'status': 'success',
            'message': f'终端 {terminal_name} 已开始上传更新',
            'pid': process.pid
        })

    except Exception as e:
        logger.error(f"触发上传失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/rollback/<int:target_round>', methods=['POST'])
def rollback_model(target_round):
    """回滚到指定轮次的模型"""
    global global_model

    try:
        backup_path = os.path.join(MODEL_BACKUP_DIR, f'model_round_{target_round}.pth')

        if not os.path.exists(backup_path):
            return jsonify({
                'status': 'error',
                'message': f'找不到轮次 {target_round} 的备份'
            }), 404

        # 加载备份模型 (安全加载)
        data = secure_torch_load(backup_path, map_location=global_model.device)
        global_model.model.load_state_dict(data['model'])
        global_model.round = data['round']
        global_model.current_accuracy = data.get('accuracy')

        # 保存回滚后的模型
        global_model.save_model()

        logger.info(f"模型已回滚到轮次 {target_round}")

        return jsonify({
            'status': 'success',
            'message': f'已回滚到轮次 {target_round}',
            'accuracy': data.get('accuracy')
        })

    except Exception as e:
        logger.error(f"回滚失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ========== 主程序 ==========

if __name__ == '__main__':
    # 设置编码
    if sys.platform == 'win32':
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')

    logger.info("=" * 70)
    logger.info("云端协调器 v2 启动中...")
    logger.info(f"版本: {MODEL_VERSION}")
    logger.info(f"模型检查点: {MODEL_CHECKPOINT}")
    logger.info(f"终端数量: {NUM_TERMINALS}")
    logger.info(f"训练轮数: {NUM_ROUNDS}")
    if DEMO_MODE:
        logger.info(f"演示模式: 已启用 (备份目录: {DEMO_BACKUP_DIR})")
        logger.info("提示: 训练将在副本上进行，原始模型不会被修改")
    logger.info("=" * 70)

    # 预加载模型（演示模式会使用副本）
    if DEMO_MODE:
        checkpoint = demo_manager.prepare_demo_model()
        logger.info(f"预加载演示模型: {checkpoint}")
        global_model = GlobalModel(checkpoint_path=checkpoint)
    else:
        global_model = GlobalModel(checkpoint_path=MODEL_CHECKPOINT)

    if global_model.model is None:
        logger.error("模型加载失败，请检查检查点文件")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info(f"监听端口: 5002")
    logger.info("=" * 70)

    # 运行Flask服务器
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=False,
        threaded=True,
        use_reloader=False
    )
