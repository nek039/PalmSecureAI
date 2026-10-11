# -*- coding: utf-8 -*-
"""
模型评估模块 - 联邦学习模型评估

功能：
1. 评估全局模型在测试集上的准确率
2. 计算关键指标（KNN准确率、Top-K准确率等）
3. 生成评估报告

使用方法：
    python federated/cloud/evaluator.py --model models/palm_recognizer.pth --test_dir data/polyu_test_users
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ========== 配置 ==========

BATCH_SIZE = 32


class FederatedEvaluator:
    """联邦学习模型评估器"""

    def __init__(self, model_path, test_dir=None, device=None):
        self.model_path = model_path
        self.test_dir = test_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.test_features = None
        self.test_labels = None

        print(f"评估器初始化 - 设备: {self.device}")

    def load_model(self):
        """加载模型"""
        from src.recognition.feature_extractor import CloudTrainedFeatureExtractor

        print(f"加载模型: {self.model_path}")

        self.model = CloudTrainedFeatureExtractor(
            checkpoint_path=self.model_path,
            device=self.device
        )
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型加载成功 - 参数数量: {total_params:,}")

        return True

    def extract_test_features(self):
        """提取测试集特征"""
        if self.test_dir is None:
            print("警告: 测试目录未指定")
            return False

        from data.dataset_loader import CSVLabelDataset, create_dataloader

        # 查找标签文件
        labels_file = None
        for possible_path in [
            os.path.join(self.test_dir, 'test_labels.csv'),
            os.path.join(os.path.dirname(self.test_dir), 'test_labels.csv'),
            os.path.join(os.path.dirname(self.test_dir), 'global_test_labels.csv')
        ]:
            if os.path.exists(possible_path):
                labels_file = possible_path
                break

        if labels_file is None:
            print(f"错误: 找不到标签文件")
            return False

        print(f"加载测试集: {self.test_dir}")
        print(f"标签文件: {labels_file}")

        dataset = CSVLabelDataset(
            csv_path=labels_file,
            image_dir=self.test_dir,
            image_size=(224, 224),
            augment=False
        )

        dataloader = create_dataloader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        print(f"测试集大小: {len(dataset)}")

        # 提取特征
        all_features = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="提取特征"):
                images = batch['image'].to(self.device)
                labels = batch['user_id']

                features = self.model(images)
                features = F.normalize(features, p=2, dim=1)

                all_features.append(features.cpu().numpy())
                all_labels.extend([str(l) for l in labels.tolist()])

        self.test_features = np.vstack(all_features)
        self.test_labels = np.array(all_labels)

        print(f"特征提取完成 - 形状: {self.test_features.shape}")
        return True

    def evaluate_knn_accuracy(self, k_values=[1, 3, 5]):
        """计算 KNN 准确率"""
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score

        results = {}

        # 编码标签
        le = LabelEncoder()
        labels_encoded = le.fit_transform(self.test_labels)

        # 使用自身作为训练集（验证特征质量）
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
            knn.fit(self.test_features, labels_encoded)
            predictions = knn.predict(self.test_features)
            acc = accuracy_score(labels_encoded, predictions)

            results[f'knn_{k}'] = acc
            print(f"KNN-{k} 准确率: {acc:.4f}")

        return results

    def evaluate_top_k_accuracy(self, k_values=[1, 5]):
        """计算 Top-K 准确率（余弦相似度）"""
        results = {}

        # 计算相似度矩阵
        similarity = self.test_features @ self.test_features.T

        for k in k_values:
            # 获取 Top-K 预测
            top_k_indices = np.argsort(-similarity, axis=1)[:, 1:k+1]  # 排除自身

            # 检查正确答案是否在 Top-K 中
            correct = 0
            for i, pred_indices in enumerate(top_k_indices):
                if self.test_labels[i] in self.test_labels[pred_indices]:
                    correct += 1

            acc = correct / len(self.test_labels)
            results[f'top_{k}'] = acc
            print(f"Top-{k} 准确率: {acc:.4f}")

        return results

    def evaluate_intra_inter_distance(self):
        """计算类内距离和类间距离"""
        unique_labels = np.unique(self.test_labels)

        intra_distances = []
        inter_distances = []

        for label in unique_labels:
            indices = np.where(self.test_labels == label)[0]

            # 类内距离
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    dist = 1 - np.dot(
                        self.test_features[indices[i]],
                        self.test_features[indices[j]]
                    )
                    intra_distances.append(dist)

        # 采样计算类间距离
        np.random.seed(42)
        for _ in range(min(500, len(unique_labels) * 10)):
            i, j = np.random.choice(len(self.test_features), 2, replace=False)
            if self.test_labels[i] != self.test_labels[j]:
                dist = 1 - np.dot(self.test_features[i], self.test_features[j])
                inter_distances.append(dist)

        intra_mean = np.mean(intra_distances) if intra_distances else 0
        intra_std = np.std(intra_distances) if intra_distances else 0
        inter_mean = np.mean(inter_distances) if inter_distances else 0
        inter_std = np.std(inter_distances) if inter_distances else 0

        separation = inter_mean / intra_mean if intra_mean > 0 else 0

        results = {
            'intra_mean': float(intra_mean),
            'intra_std': float(intra_std),
            'inter_mean': float(inter_mean),
            'inter_std': float(inter_std),
            'separation_ratio': float(separation)
        }

        print(f"类内距离: {intra_mean:.4f} ± {intra_std:.4f}")
        print(f"类间距离: {inter_mean:.4f} ± {inter_std:.4f}")
        print(f"分离比例: {separation:.2f}")

        return results

    def run_full_evaluation(self):
        """运行完整评估"""
        print("=" * 60)
        print("开始模型评估")
        print("=" * 60)

        # 加载模型
        if not self.load_model():
            return None

        # 提取特征
        if not self.extract_test_features():
            return None

        print("\n" + "=" * 60)
        print("评估指标")
        print("=" * 60)

        # 评估
        results = {
            'model_path': self.model_path,
            'test_dir': self.test_dir,
            'num_samples': len(self.test_labels),
            'num_classes': len(np.unique(self.test_labels)),
            'timestamp': datetime.now().isoformat()
        }

        print("\n1. KNN 准确率:")
        results.update(self.evaluate_knn_accuracy())

        print("\n2. Top-K 准确率:")
        results.update(self.evaluate_top_k_accuracy())

        print("\n3. 特征距离分析:")
        results.update(self.evaluate_intra_inter_distance())

        print("\n" + "=" * 60)
        print("评估完成")
        print("=" * 60)

        return results

    def save_results(self, results, output_path):
        """保存评估结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='联邦学习模型评估')
    parser.add_argument('--model', type=str, default='models/palm_recognizer.pth',
                        help='模型路径')
    parser.add_argument('--test_dir', type=str, default='data/federated/global_test',
                        help='测试集目录')
    parser.add_argument('--output', type=str, default='logs/evaluation_results.json',
                        help='输出文件')
    args = parser.parse_args()

    # 创建评估器
    evaluator = FederatedEvaluator(
        model_path=args.model,
        test_dir=args.test_dir
    )

    # 运行评估
    results = evaluator.run_full_evaluation()

    if results:
        # 保存结果
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        evaluator.save_results(results, args.output)


if __name__ == '__main__':
    main()
