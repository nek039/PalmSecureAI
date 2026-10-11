# PalmSecure

> 这是用于代码展示与项目交流的轻量源码版。为保护数据并保持仓库易于下载，训练数据、用户模板数据库、模型权重、实验输出和运行日志均未包含在此仓库。
>
> 如需在本地完整运行，请自行准备符合 `MODEL_CHECKPOINT` 配置的模型文件，以及本地模板数据库；不要将它们提交到 GitHub。

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt

# 2. 启动 Web 服务（项目根目录下）
.venv/bin/python -m apps.web.app

# 3. 访问系统
# 浏览器打开 http://localhost:8000
```

## 仓库范围

- 包含：Web 界面、API、掌纹识别核心逻辑、联邦学习协调代码、配置和依赖清单。
- 不包含：原始掌纹数据、训练权重、用户模板数据库、实验输出、日志与 IDE 配置。

### 启动选项

| 服务 | 命令 | 端口 |
|------|------|------|
| Web 应用 | `.venv/bin/python -m apps.web.app` | 8000 |
| RESTful API | `.venv/bin/python -m apps.api.server` | 8000 |

---

## 项目概述

### 核心特点

- **隐私保护**：数据不离开终端，联邦学习训练
- **边缘部署**：本地识别，无需网络，响应时间 <100ms
- **跨设备协同**：多终端通过联邦学习共享模型知识
- **离线可用**：网络中断不影响使用

### 应用场景

适用于海关通关、边境检查、机场安检等需要高安全性生物识别的场景。

---

## 项目结构

```
PalmSecureAI/
├── config/                # 配置管理
├── src/                   # 核心源码
│   └── recognition/       # 掌纹识别核心模块
│       ├── palm_recognizer.py    # 主识别器（Mixed+PLE模型）
│       ├── feature_extractor.py  # ResNet18特征提取（512维）
│       ├── palm_line_enhancement.py  # 掌纹线增强（PLE）
│       └── database.py           # SQLite模板数据库
│
├── apps/                  # 应用层
│   ├── web/               # Web应用
│   │   ├── app.py         # Flask服务器
│   │   ├── auth.py        # 用户认证
│   │   ├── middleware.py   # 安全中间件
│   │   └── templates/     # 前端模板
│   └── api/               # RESTful API
│       ├── server.py      # API服务
│       └── models.py      # API数据模型
│
├── archive/               # 历史代码存档
├── data/                  # 数据目录
│   ├── palm_templates.db  # 模板数据库
│   └── models/            # 模型文件
│       └── mixed_ple_final/
│           └── palm_recognizer.pth
│
├── scripts/               # 脚本工具
├── docs/                  # 文档
├── requirements.txt       # 项目依赖
└── .venv/                # Python虚拟环境
```

---

## 核心技术

### 掌纹识别流程

```
输入图像 → ROI提取 → PLE增强 → 特征提取 → 相似度匹配 → 识别结果
```

### 模型对比

| 指标 | 旧模型 | 新模型 (Mixed+PLE) | 提升 |
|------|--------|-------------------|------|
| **CASIA AUC** | 0.922 | 0.995 | +7.9% |
| **CASIA EER** | 14.2% | 3.9% | -10.3% |
| **PolyU AUC** | 0.956 | 0.986 | +3.1% |
| **PolyU EER** | 11.9% | 6.7% | -5.2% |

### 技术栈

- **深度学习**: PyTorch, ResNet18, Triplet Loss
- **图像处理**: OpenCV, PIL, PLE增强
- **联邦学习**: FedAvg聚合, HMAC安全验证
- **边缘部署**: TFLite, ONNX
- **Web服务**: Flask

---

## 使用说明

### 1. 掌纹识别

```python
from src.recognition.palm_recognizer import PalmRecognizer

# 初始化识别器
recognizer = PalmRecognizer(
    model_checkpoint="models/mixed_ple_final/palm_recognizer.pth",
    match_threshold=0.60,
    skip_roi=True,
    use_ple_preprocessing=False,
    feat_dim=512
)

# 注册用户
recognizer.enroll(user_id="user_001", image=palm_image)

# 识别用户
result = recognizer.recognize(palm_image)
print(f"用户: {result['user_id']}, 相似度: {result['similarity']:.2f}")
```

### 2. 联邦学习模拟

```bash
# 运行联邦学习模拟
.venv/bin/python scripts/run_federated_simulation.py
```

### 3. 模型训练

```bash
# 使用PLE增强训练
.venv/bin/python -m train.train_with_ple

# 混合数据训练
.venv/bin/python -m train.train_mixed_loss
```

---

## API 接口

### Web API (Flask - 端口 8000)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 主页 |
| `/enroll` | GET/POST | 用户注册 |
| `/recognize` | GET/POST | 掌纹识别 |
| `/users` | GET | 用户列表 |
| `/federated` | GET | 联邦学习状态 |

### RESTful API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/enroll` | POST | 用户注册 |
| `/api/recognize` | POST | 掌纹识别 |
| `/api/users` | GET | 用户列表 |
| `/api/stats` | GET | 识别统计 |

---

## 配置说明

主要配置在 `apps/web/app.py` 和 `apps/api/server.py` 中的 `APIConfig` 类：

```python
# API 配置
HOST = "0.0.0.0"          # 监听地址
PORT = 8000               # 端口
DEBUG = False             # 调试模式
MODEL_CHECKPOINT = "models/mixed_ple_final/palm_recognizer.pth"
DB_PATH = "data/palm_templates.db"
```

---

## 数据集

项目使用以下公开数据集：

- **PolyU Palmprint Database**: 香港理工大学掌纹数据库
- **CASIA Palmprint Database**: 中科院自动化所掌纹数据库

---

## 开发环境

- Python 3.10+
- PyTorch 2.5+
- CUDA 11.8+ (推荐)

---

## License

MIT License
