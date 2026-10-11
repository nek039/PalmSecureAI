# -*- coding: utf-8 -*-
"""
PalmSecureAI RESTful API 服务器

提供掌纹识别和系统管理的 HTTP API
"""

import os
import sys
import io
import base64
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from PIL import Image
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.recognition.palm_recognizer import PalmRecognizer
from src.recognition.database import PalmDatabase, get_database


# 配置
class APIConfig:
    """API 配置"""
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("API_DEBUG", "False").lower() == "true"
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
    API_KEY = os.getenv("API_KEY", None)  # 可选的 API Key 认证
    # 使用最优模型 (Mixed+PLE, 统一评估结果)
    MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "models/mixed_ple_final/palm_recognizer.pth")
    DB_PATH = os.getenv("DB_PATH", "data/palm_templates.db")


# 创建 Flask 应用
app = Flask(__name__)
CORS(app, origins=APIConfig.ALLOWED_ORIGINS)

# 初始化识别器和数据库
recognizer: Optional[PalmRecognizer] = None
database: Optional[PalmDatabase] = None


def init_services():
    """初始化识别器和数据库"""
    global recognizer, database

    # 初始化数据库
    database = get_database(APIConfig.DB_PATH)

    # 初始化识别器
    model_path = APIConfig.MODEL_CHECKPOINT
    if os.path.isfile(model_path):
        recognizer = PalmRecognizer(
            model_checkpoint=model_path,
            match_threshold=0.60,  # 最优阈值 (统一评估结果)
            skip_roi=True,
            use_ple_preprocessing=False,  # 禁用 PLE (评估显示推理时不用 PLE 效果更好)
            feat_dim=512,  # 模型特征维度
        )
    else:
        print(f"警告: 模型文件不存在: {model_path}")
        print("使用随机初始化模型（仅用于测试）")
        recognizer = PalmRecognizer(
            model_checkpoint=None,
            match_threshold=0.60,  # 最优阈值 (统一评估结果)
            skip_roi=True,
            use_ple_preprocessing=False,  # 禁用 PLE
            feat_dim=512,  # 模型特征维度
        )


# 路由：认证中间件
@app.before_request
def authenticate():
    """API Key 认证（如果配置了）"""
    if APIConfig.API_KEY:
        auth_header = request.headers.get("Authorization", "")
        expected = f"Bearer {APIConfig.API_KEY}"
        if auth_header != expected:
            return jsonify({
                "success": False,
                "error": "Unauthorized",
                "message": "Invalid or missing API key"
            }), 401


# 辅助函数：解析图像
def parse_image(image_data: str) -> Optional[np.ndarray]:
    """
    解析 base64 编码的图像

    Args:
        image_data: base64 编码的图像字符串（可选 data URL 前缀）

    Returns:
        OpenCV 图像数组 (BGR)
    """
    try:
        # 移除 data URL 前缀（如果有）
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        # 解码 base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # 转换为 OpenCV 格式 (BGR)
        if image.mode != "RGB":
            image = image.convert("RGB")
        opencv_image = np.array(image)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

        return opencv_image
    except Exception as e:
        print(f"图像解析失败: {e}")
        return None


# 路由：健康检查
@app.route("/api/health", methods=["GET"])
def health_check():
    """健康检查"""
    return jsonify({
        "success": True,
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "PalmSecure API"
    })


# 路由：系统状态
@app.route("/api/status", methods=["GET"])
def get_status():
    """获取系统状态"""
    user_count = recognizer.get_user_count() if recognizer else 0

    db_stats = {}
    if database:
        stats = database.get_statistics()
        db_stats = {
            "db_path": str(stats.get("db_path", "")),
            "first_user": stats.get("first_user"),
            "last_user": stats.get("last_user")
        }

    model_info = {
        "checkpoint_path": APIConfig.MODEL_CHECKPOINT,
        "loaded": recognizer is not None and os.path.isfile(APIConfig.MODEL_CHECKPOINT)
    }

    return jsonify({
        "success": True,
        "data": {
            "user_count": user_count,
            "database": db_stats,
            "model": model_info,
            "timestamp": datetime.now().isoformat()
        }
    })


# 路由：用户注册
@app.route("/api/enroll", methods=["POST"])
def enroll_user():
    """
    注册新用户

    请求体:
    {
        "user_id": "user_001",
        "user_name": "张三",
        "image": "base64_encoded_image"
    }

    响应:
    {
        "success": true,
        "data": {
            "user_id": "user_001",
            "message": "注册成功"
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "请求体不能为空"
            }), 400

        user_id = data.get("user_id")
        user_name = data.get("user_name")
        image_data = data.get("image")

        if not user_id:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "缺少 user_id 参数"
            }), 400

        if not image_data:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "缺少 image 参数"
            }), 400

        # 解析图像
        image = parse_image(image_data)
        if image is None:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "图像解析失败"
            }), 400

        # 注册用户
        if recognizer.enroll(image, user_id):
            # 如果提供了用户名，更新数据库
            if user_name and database:
                database.update_user(user_id, user_name=user_name)

            return jsonify({
                "success": True,
                "data": {
                    "user_id": user_id,
                    "user_name": user_name,
                    "message": "注册成功"
                }
            })
        else:
            return jsonify({
                "success": False,
                "error": "EnrollFailed",
                "message": "注册失败，用户可能已存在或特征提取失败"
            }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "InternalServerError",
            "message": str(e)
        }), 500


# 路由：掌纹识别
@app.route("/api/recognize", methods=["POST"])
def recognize():
    """
    掌纹识别

    请求体:
    {
        "image": "base64_encoded_image",
        "top_k": 3  // 可选，返回前K个匹配结果
    }

    响应:
    {
        "success": true,
        "data": {
            "matched": true,
            "user_id": "user_001",
            "user_name": "张三",
            "similarity": 0.95,
            "message": "识别成功"
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "请求体不能为空"
            }), 400

        image_data = data.get("image")
        if not image_data:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "缺少 image 参数"
            }), 400

        # 解析图像
        image = parse_image(image_data)
        if image is None:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "图像解析失败"
            }), 400

        # 识别
        result = recognizer.recognize(image)

        # 如果匹配，获取用户详细信息
        user_info = None
        if result.get("matched") and result.get("user_id") and database:
            user_record = database.get_user(result["user_id"])
            if user_record:
                user_info = {
                    "user_id": user_record["user_id"],
                    "user_name": user_record["user_name"],
                    "created_at": user_record["created_at"]
                }

        return jsonify({
            "success": True,
            "data": {
                "matched": result["matched"],
                "user_id": result["user_id"],
                "similarity": result["similarity"],
                "message": result["message"],
                "user_info": user_info
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "InternalServerError",
            "message": str(e)
        }), 500


# 路由：获取用户列表
@app.route("/api/users", methods=["GET"])
def list_users():
    """
    获取用户列表

    查询参数:
    - limit: 返回数量限制
    - offset: 偏移量

    响应:
    {
        "success": true,
        "data": {
            "total": 100,
            "users": [...]
        }
    }
    """
    try:
        limit = request.args.get("limit", type=int)
        offset = request.args.get("offset", type=int, default=0)

        users = database.get_all_users() if database else []

        # 过滤敏感信息（不返回特征向量）
        users_data = []
        for user in users:
            users_data.append({
                "user_id": user["user_id"],
                "user_name": user["user_name"],
                "created_at": user["created_at"],
                "updated_at": user["updated_at"]
            })

        # 应用分页
        total = len(users_data)
        if limit:
            users_data = users_data[offset:offset + limit]

        return jsonify({
            "success": True,
            "data": {
                "total": total,
                "users": users_data,
                "offset": offset,
                "limit": limit
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "InternalServerError",
            "message": str(e)
        }), 500


# 路由：获取用户详情
@app.route("/api/users/<user_id>", methods=["GET"])
def get_user(user_id: str):
    """
    获取用户详情

    响应:
    {
        "success": true,
        "data": {...}
    }
    """
    try:
        user = database.get_user(user_id) if database else None

        if not user:
            return jsonify({
                "success": False,
                "error": "NotFound",
                "message": f"用户不存在: {user_id}"
            }), 404

        return jsonify({
            "success": True,
            "data": {
                "user_id": user["user_id"],
                "user_name": user["user_name"],
                "created_at": user["created_at"],
                "updated_at": user["updated_at"],
                "feature_dim": user["feature_dim"]
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "InternalServerError",
            "message": str(e)
        }), 500


# 路由：删除用户
@app.route("/api/users/<user_id>", methods=["DELETE"])
def delete_user_api(user_id: str):
    """
    删除用户

    响应:
    {
        "success": true,
        "message": "用户已删除"
    }
    """
    try:
        success = recognizer.delete_user(user_id)

        if success:
            return jsonify({
                "success": True,
                "message": f"用户已删除: {user_id}"
            })
        else:
            return jsonify({
                "success": False,
                "error": "NotFound",
                "message": f"用户不存在: {user_id}"
            }), 404

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "InternalServerError",
            "message": str(e)
        }), 500


# 路由：批量删除用户
@app.route("/api/users/batch-delete", methods=["POST"])
def batch_delete_users():
    """
    批量删除用户

    请求体:
    {
        "user_ids": ["user_001", "user_002"]
    }

    响应:
    {
        "success": true,
        "data": {
            "deleted": 2,
            "failed": 0
        }
    }
    """
    try:
        data = request.get_json()
        user_ids = data.get("user_ids", [])

        if not user_ids:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "缺少 user_ids 参数"
            }), 400

        deleted = 0
        failed = 0

        for user_id in user_ids:
            if recognizer.delete_user(user_id):
                deleted += 1
            else:
                failed += 1

        return jsonify({
            "success": True,
            "data": {
                "deleted": deleted,
                "failed": failed
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "InternalServerError",
            "message": str(e)
        }), 500


# 路由：清空所有用户
@app.route("/api/users/clear", methods=["POST"])
def clear_users():
    """
    清空所有用户（危险操作）

    响应:
    {
        "success": true,
        "data": {
            "deleted": 100
        }
    }
    """
    try:
        deleted = recognizer.clear_users()

        return jsonify({
            "success": True,
            "data": {
                "deleted": deleted
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "InternalServerError",
            "message": str(e)
        }), 500


# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "NotFound",
        "message": "请求的资源不存在"
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "success": False,
        "error": "MethodNotAllowed",
        "message": "不支持的 HTTP 方法"
    }), 405


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "InternalServerError",
        "message": "服务器内部错误"
    }), 500


# 主函数
def main():
    """启动 API 服务器"""
    print("=" * 50)
    print("PalmSecure API 服务器")
    print("=" * 50)
    print(f"主机: {APIConfig.HOST}")
    print(f"端口: {APIConfig.PORT}")
    print(f"调试模式: {APIConfig.DEBUG}")
    print(f"模型文件: {APIConfig.MODEL_CHECKPOINT}")
    print(f"数据库路径: {APIConfig.DB_PATH}")
    print("=" * 50)

    # 初始化服务
    init_services()
    print("服务初始化完成\n")

    # 启动服务器
    app.run(
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        debug=APIConfig.DEBUG
    )


if __name__ == "__main__":
    # 需要导入 cv2 用于图像处理
    import cv2
    main()
