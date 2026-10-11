# -*- coding: utf-8 -*-
"""
PalmSecureAI Web Application Server

Security Enhanced Version - Implements passive security protections
(XSS prevention, SQL injection protection, command injection fix, security headers)
without breaking existing functionality for thesis project.
"""

import os
import sys
import io
import base64
import subprocess
import threading
import requests
import re
import html
import psutil
import mimetypes
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, send_from_directory, jsonify, request, session, redirect, url_for
from flask_cors import CORS

# 修复 Windows 上 JS/CSS 文件的 MIME 类型问题（必须在 Flask 启动前）
# Windows 注册表可能将 .js 映射为 text/plain，需要强制覆盖
mimetypes.types_map['.js'] = 'application/javascript'
mimetypes.types_map['.mjs'] = 'application/javascript'
mimetypes.types_map['.css'] = 'text/css'
mimetypes.types_map['.svg'] = 'image/svg+xml'
from PIL import Image
import numpy as np
import cv2
import random

# 全局变量：云协调器进程
coordinator_process: Optional[subprocess.Popen] = None
COORDINATOR_PORT = 5002

# 全局变量：终端模拟器进程
simulator_process: Optional[subprocess.Popen] = None

# 登录限流：记录每个 IP 的失败次数和时间
_login_rate_limit: Dict[str, list] = {}
_LOGIN_MAX_ATTEMPTS = 5
_LOGIN_LOCKOUT_SECONDS = 300  # 5分钟

def _cleanup_rate_limit():
    """定期清理过期的 rate limit 记录，防止内存泄漏"""
    import time
    now = time.time()
    expired = [ip for ip, attempts in _login_rate_limit.items()
               if all(now - t >= _LOGIN_LOCKOUT_SECONDS for t in attempts)]
    for ip in expired:
        del _login_rate_limit[ip]

def _check_login_rate_limit(ip: str) -> tuple[bool, str]:
    """检查登录频率限制，返回 (是否允许, 错误消息)"""
    import time
    now = time.time()
    attempts = _login_rate_limit.get(ip, [])

    # 清理超过锁定时间的记录
    attempts = [t for t in attempts if now - t < _LOGIN_LOCKOUT_SECONDS]
    _login_rate_limit[ip] = attempts

    if len(attempts) >= _LOGIN_MAX_ATTEMPTS:
        remaining = int(_LOGIN_LOCKOUT_SECONDS - (now - attempts[0]))
        return False, f"登录过于频繁，请在 {remaining} 秒后重试"
    return True, ""

def _record_failed_login(ip: str):
    """记录失败的登录尝试"""
    import time
    if ip not in _login_rate_limit:
        _login_rate_limit[ip] = []
    _login_rate_limit[ip].append(time.time())
    # 每 10 次记录触发一次清理
    if len(_login_rate_limit) > 100:
        _cleanup_rate_limit()

def _safe_error_message(e: Exception) -> str:
    """生产环境下返回通用错误消息，不暴露内部细节"""
    if os.getenv('FLASK_ENV') == 'production':
        return "服务器内部错误，请联系管理员"
    return f"服务器错误: {str(e)}"

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.recognition.palm_recognizer import PalmRecognizer
from src.recognition.database import PalmDatabase, get_database
from apps.web.middleware import (
    RequestValidator, setup_security_middleware
)
from apps.web.auth import auth_manager


# Configuration
class APIConfig:
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", "8000"))
    DEBUG = os.getenv("API_DEBUG", "False").lower() == "true"
    # 使用新模型 (Mixed Data + PLE)
    MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "models/mixed_ple_final/palm_recognizer.pth")
    DB_PATH = os.getenv("DB_PATH", "data/palm_templates.db")


def login_required(f):
    """登录保护装饰器"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            # 如果是 AJAX 请求，返回 JSON 错误
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    "success": False,
                    "error": "Unauthorized",
                    "message": "请先登录"
                }), 401
            # 否则重定向到登录页
            return _redirect_to_login()
        return f(*args, **kwargs)
    return decorated_function


def _redirect_to_login():
    """重定向到登录页"""
    return redirect(url_for('login'))


# Create Flask app (禁用默认静态文件，使用自定义路由处理)
app = Flask(__name__,
            template_folder=str(project_root / 'apps' / 'web' / 'templates'),
            static_folder=None,
            static_url_path=None)

# Configure CORS - 根据环境限制来源
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*')
if ALLOWED_ORIGINS != '*':
    ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS.split(',') if o.strip()]
CORS(app, origins=ALLOWED_ORIGINS)

# SECRET_KEY 配置 - 生产环境必须从环境变量读取
_is_production = os.getenv('FLASK_ENV') == 'production'
try:
    from config.security_config import SecurityConfig
    app.config['SECRET_KEY'] = SecurityConfig.get_secret_key()
except Exception as e:
    # 生产环境：SECRET_KEY 未设置必须退出
    if _is_production:
        raise RuntimeError(
            f"FATAL: SECRET_KEY environment variable is required in production. "
            f"Set it with: export SECRET_KEY=$(openssl rand -hex 32). Error: {e}"
        )
    # 开发模式：生成临时 key（仅警告，不阻止启动）
    import warnings
    warnings.warn(f"SECRET_KEY not set, using temporary key. Error: {e}")
    import secrets
    app.config['SECRET_KEY'] = secrets.token_hex(32)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# HTTPS：生产环境必须启用
app.config['SESSION_COOKIE_SECURE'] = _is_production
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# 启用安全中间件
setup_security_middleware(app)


# 自定义静态文件路由 - 修复 Windows 上的 MIME 类型问题
@app.route('/static/<path:filename>', endpoint='static_custom')
def serve_static_custom(filename):
    """提供静态文件，修复 Windows 上 MIME 类型问题"""
    from flask import send_file, Response
    import os
    static_folder = str(project_root / 'apps' / 'web' / 'static')
    file_path = os.path.join(static_folder, filename)

    # 安全检查：确保文件在 static 目录内
    real_path = os.path.realpath(file_path)
    real_static = os.path.realpath(static_folder)
    if not real_path.startswith(real_static):
        return "Forbidden", 403

    if not os.path.exists(real_path):
        return "Not Found", 404

    # 根据扩展名确定 MIME 类型
    mime_types = {
        '.js': 'application/javascript',
        '.mjs': 'application/javascript',
        '.css': 'text/css',
        '.svg': 'image/svg+xml',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.woff': 'font/woff',
        '.woff2': 'font/woff2',
        '.ttf': 'font/ttf',
        '.otf': 'font/otf',
        '.eot': 'application/vnd.ms-fontobject'
    }

    ext = os.path.splitext(filename)[1].lower()
    mimetype = mime_types.get(ext)

    response = send_file(real_path, mimetype=mimetype)
    # 强制覆盖 Content-Type
    if mimetype:
        response.content_type = mimetype
        response.headers['Content-Type'] = mimetype
    return response


# 禁用缓存（开发环境）
@app.after_request
def add_cache_control_headers(response):
    """添加缓存控制头，防止浏览器缓存问题，并修复 MIME 类型"""
    from flask import request

    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'

    # 注意：静态文件的 MIME 类型在 serve_static_custom 中设置

    return response


# Global error handler
@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    error_trace = traceback.format_exc()
    print(f"[GLOBAL ERROR] {error_trace}")
    with open("global_error.log", "a") as f:
        f.write(f"=== {datetime.now()} ===\n{error_trace}\n\n")

    # 生产环境不返回内部错误细节
    is_prod = os.getenv('FLASK_ENV') == 'production'
    if is_prod:
        return jsonify({
            "success": False,
            "error": "InternalServerError",
            "message": "服务器内部错误，请联系管理员"
        }), 500
    return jsonify({
        "success": False,
        "error": "InternalServerError",
        "message": _safe_error_message(e)
    }), 500


# Global variables
recognizer: Optional[PalmRecognizer] = None
database: Optional[PalmDatabase] = None


def init_services():
    global recognizer, database

    # Initialize database
    database = get_database(APIConfig.DB_PATH)

    # Initialize recognizer
    model_path = APIConfig.MODEL_CHECKPOINT

    if not os.path.isfile(model_path):
        raise RuntimeError(
            f"FATAL: Model file not found: {model_path}. "
            "Cannot start in production mode. Please set MODEL_CHECKPOINT env var or place model file."
        )

    try:
        recognizer = PalmRecognizer(
            model_checkpoint=model_path,
            match_threshold=0.80,
            use_enhanced_preprocessing=False,
            use_database=True,
            db_path=APIConfig.DB_PATH,
            use_mediapipe_roi=False,
            skip_roi=True,
            use_ple_preprocessing=True,
            feat_dim=512,
        )
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to initialize PalmRecognizer: {e}") from e


def parse_image(image_data: str) -> Optional[np.ndarray]:
    try:
        if not image_data:
            print("Error: image_data is empty")
            return None
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        opencv_image = np.array(image)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        print(f"Image parsed successfully: {opencv_image.shape}")
        return opencv_image
    except Exception as e:
        print(f"Image parsing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ==================== API Routes ====================

@app.route('/api/health', methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "PalmSecure API"
    })


@app.route('/api/status', methods=["GET"])
def get_status():
    # 优先从数据库获取用户数
    print("[API] /api/status 被调用", flush=True)

    # 获取当前管理员的站点信息
    current_station = session.get('station', '')
    is_admin = session.get('role') == 'admin'

    # admin角色看全部数据，station角色只看本站数据
    station_filter = None if is_admin else current_station

    if database:
        user_count = database.get_user_count(station=station_filter)
        total_templates = database.get_total_templates(station=station_filter)
        print(f"[API] database 存在: user_count={user_count}, total_templates={total_templates}", flush=True)
    else:
        user_count = recognizer.get_user_count() if recognizer else 0
        total_templates = recognizer.get_total_templates() if recognizer else 0
        print(f"[API] database 不存在, 使用 recognizer: user_count={user_count}, total_templates={total_templates}", flush=True)

    db_stats = {}
    if database:
        stats = database.get_statistics(station=station_filter)
        db_stats = {
            "db_path": str(stats.get("db_path", "")),
            "first_user": stats.get("first_user"),
            "last_user": stats.get("last_user"),
            "total_templates": total_templates
        }

    model_info = {
        "checkpoint_path": APIConfig.MODEL_CHECKPOINT,
        "loaded": recognizer is not None and os.path.isfile(APIConfig.MODEL_CHECKPOINT)
    }

    # 获取今日识别统计
    recognition_stats = {"total": 0, "success": 0, "fail": 0}
    if database:
        recognition_stats = database.get_recognition_stats(days=1, station=station_filter)

    response_data = {
        "success": True,
        "data": {
            "user_count": user_count,
            "sample_count": total_templates,
            "database": db_stats,
            "model": model_info,
            "recognition": recognition_stats,
            "timestamp": datetime.now().isoformat()
        }
    }
    print(f"[API] 返回数据: user_count={user_count}, sample_count={total_templates}, recognition={recognition_stats}", flush=True)
    return jsonify(response_data)


@app.route('/api/trend', methods=["GET"])
def get_trend():
    """获取识别趋势数据"""
    range_type = request.args.get('range', '7d')  # today, 7d, 30d
    data_type = request.args.get('type', 'count')  # count or success

    print(f"[API] /api/trend 被调用: range={range_type}, type={data_type}", flush=True)

    # 获取当前管理员的站点信息
    current_station = session.get('station', '')
    is_admin = session.get('role') == 'admin'

    # admin角色看全部数据，station角色只看本站数据
    station_filter = None if is_admin else current_station

    try:
        # 从数据库获取真实趋势数据
        if database:
            trend_data = database.get_recognition_trend(range_type, data_type, station=station_filter)
            labels = trend_data.get('labels', [])
            values = trend_data.get('values', [])
        else:
            labels = []
            values = []

        response_data = {
            "success": True,
            "data": {
                "labels": labels,
                "values": values,
                "type": data_type,
                "range": range_type
            }
        }

        print(f"[API] /api/trend 返回: {len(labels)} 个数据点, values={values}", flush=True)
        return jsonify(response_data)

    except Exception as e:
        print(f"[API] /api/trend 错误: {e}", flush=True)
        return jsonify({
            "success": False,
            "error": type(e).__name__,
            "data": None
        }), 500


@app.route('/api/quality', methods=["GET"])
def get_quality():
    """获取样本质量分布数据"""
    print("[API] /api/quality 被调用", flush=True)

    # 获取当前管理员的站点信息
    current_station = session.get('station', '')
    is_admin = session.get('role') == 'admin'

    # admin角色看全部数据，station角色只看本站数据
    station_filter = None if is_admin else current_station

    try:
        if database:
            quality_data = database.get_quality_distribution(station=station_filter)
            print(f"[API] /api/quality 返回: {quality_data}", flush=True)
        else:
            quality_data = {"excellent": 0, "good": 0, "average": 0, "poor": 0}

        return jsonify({
            "success": True,
            "data": quality_data
        })

    except Exception as e:
        print(f"[API] /api/quality 错误: {e}", flush=True)
        return jsonify({
            "success": False,
            "error": type(e).__name__,
            "data": {"excellent": 0, "good": 0, "average": 0, "poor": 0}
        }), 500


@app.route('/api/confidence-distribution', methods=["GET"])
def get_confidence_distribution():
    """获取置信度分数分布数据"""
    print("[API] /api/confidence-distribution 被调用", flush=True)

    try:
        if database and hasattr(database, 'get_confidence_distribution'):
            dist_data = database.get_confidence_distribution()
        else:
            # 返回模拟数据结构 (6个区间)
            dist_data = {
                "0.0-0.5": 0,
                "0.5-0.7": 0,
                "0.7-0.85": 0,
                "0.85-0.90": 0,
                "0.90-0.95": 0,
                "0.95-1.0": 0
            }

        print(f"[API] /api/confidence-distribution 返回: {dist_data}", flush=True)

        return jsonify({
            "success": True,
            "data": dist_data
        })

    except Exception as e:
        print(f"[API] /api/confidence-distribution 错误: {e}", flush=True)
        return jsonify({
            "success": False,
            "error": type(e).__name__,
            "data": {
                "0.0-0.5": 0,
                "0.5-0.7": 0,
                "0.7-0.85": 0,
                "0.85-0.90": 0,
                "0.90-0.95": 0,
                "0.95-1.0": 0
            }
        }), 500


@app.route('/api/enroll', methods=["POST"])
def enroll_user():
    import sys
    print("[DEBUG] enroll_user() 开始", flush=True)
    try:
        # 尝试获取 JSON 数据，捕获解析错误
        try:
            data = request.get_json(force=True, silent=False)
        except Exception as json_err:
            print(f"[ERROR] JSON parse error: {json_err}", flush=True)
            return jsonify({
                "success": False,
                "error": "JsonParseError",
                "message": f"JSON解析失败: {str(json_err)}"
            }), 400

        print(f"[DEBUG] 收到请求数据: user_id={data.get('user_id') if data else 'N/A'}, has_image={data.get('image') is not None}, has_infrared={data.get('infrared_image') is not None}, has_visible={data.get('visible_image') is not None}", flush=True)
        if not data:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "请求数据为空"
            }), 400

        user_id = data.get("user_id")
        user_name = data.get("user_name")
        image_data = data.get("image")
        infrared_data = data.get("infrared_image")  # 可选的红外图像
        visible_data = data.get("visible_image")    # 可选的可见光图像

        # Validate user_id
        is_valid, sanitized_id = RequestValidator.sanitize_user_id(user_id)
        if not is_valid:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": f"用户ID无效: {sanitized_id}"
            }), 400
        user_id = sanitized_id

        if not user_id:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "用户ID不能为空"
            }), 400
        if not image_data:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "图像数据不能为空"
            }), 400

        image = parse_image(image_data)
        print(f"[DEBUG] parse_image result: {image.shape if image is not None else 'None'}", flush=True)
        if image is None:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "图像解析失败"
            }), 400

        # 双目模式：活体检测（暂时完全禁用）
        liveness_passed = True  # 默认通过
        liveness_score = None
        print(f"[DEBUG] 跳过活体检测，直接进行注册", flush=True)

        # 检查用户是否已存在
        existing_count = recognizer._get_user_sample_count(user_id)
        if existing_count > 0 and not data.get('force'):
            print(f"[DEBUG] 用户 {user_id} 已存在，现有样本数: {existing_count}", flush=True)
            return jsonify({
                "success": False,
                "error": "UserExists",
                "message": f"用户 {user_id} 已存在（当前 {existing_count} 个模板），是否追加？",
                "data": {
                    "user_id": user_id,
                    "existing_count": existing_count
                }
            }), 200

        # 尝试注册
        print(f"[DEBUG] 准备调用 recognizer.enroll, image shape: {image.shape}", flush=True)
        current_station = session.get('station', '总站')
        try:
            enroll_result = recognizer.enroll(user_id, image, force=data.get('force', False), station=current_station)
            print(f"[DEBUG] enroll 返回: {enroll_result}, type: {type(enroll_result)}")
        except Exception as enroll_error:
            import traceback
            print(f"[ERROR] enroll 异常: {enroll_error}")
            traceback.print_exc()
            raise

        if enroll_result:
            if user_name and database:
                database.update_user(user_id, user_name=user_name)

            response_data = {
                "user_id": user_id,
                "user_name": user_name,
                "message": "注册成功"
            }

            # 如果有活体检测结果，添加到响应中
            if liveness_score is not None:
                response_data["liveness_passed"] = liveness_passed
                response_data["liveness_score"] = liveness_score

            return jsonify({
                "success": True,
                "data": response_data
            })
        else:
            # 返回详细的错误信息
            error_msg = "注册失败：未知错误"

            # 检查用户是否已达到最大样本数
            if database:
                existing_count = database.get_user_sample_count(user_id)
                max_samples = 10  # 默认最大样本数
                if existing_count >= max_samples:
                    error_msg = f"用户 {user_id} 已达到最大样本数 ({max_samples}张)"
                    return jsonify({
                        "success": False,
                        "error": "MaxSamplesReached",
                        "message": error_msg
                    }), 400

            # 检查特征提取是否失败
            feat, quality = recognizer.extract_feature(image)
            if feat is None:  # 提取特征失败
                error_msg = "特征提取失败：未检测到手掌，请确保手掌完整显示在图像中"
            elif quality < 0.10:  # 质量过低
                error_msg = f"图像质量过低 ({quality:.0%})，请使用更清晰的图像"
            elif len(image.shape) < 3:  # 图像尺寸问题
                error_msg = f"图像尺寸错误: {image.shape}"
            elif image.dtype != np.uint8:
                error_msg = f"图像格式错误: {image.dtype}"
            else:
                error_msg = "注册失败：数据库操作失败，请重试"

            return jsonify({
                "success": False,
                "error": "EnrollFailed",
                "message": error_msg
            }), 400

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] enroll_user exception:\n{error_trace}")
        with open("enroll_error.log", "a") as f:
            f.write(f"=== {datetime.now()} ===\n{error_trace}\n\n")
        return jsonify({
            "success": False,
            "error": "InternalServerError",
            "message": _safe_error_message(e)
        }), 500


@app.route('/api/enroll/batch-upload', methods=["POST"])
def batch_enroll_upload():
    """
    批量注册多张图片到同一用户
    请求体: { "user_id": "xxx", "user_name": "xxx", "images": [base64...] }
    """
    import sys
    print("[DEBUG] batch_enroll_upload() 开始", flush=True)

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "请求数据为空"
            }), 400

        user_id = data.get("user_id")
        user_name = data.get("user_name")
        images_data = data.get("images", [])

        if not user_id:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "用户ID不能为空"
            }), 400

        if not images_data or len(images_data) == 0:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "图片数据不能为空"
            }), 400

        # 检查用户是否已存在
        existing_count = recognizer._get_user_sample_count(user_id)
        if existing_count > 0 and not data.get('force'):
            print(f"[DEBUG] 用户 {user_id} 已存在，现有样本数: {existing_count}", flush=True)
            return jsonify({
                "success": False,
                "error": "UserExists",
                "message": f"用户 {user_id} 已存在（当前 {existing_count} 个模板），是否追加？",
                "data": {
                    "user_id": user_id,
                    "existing_count": existing_count
                }
            }), 200

        success_count = 0
        failed_count = 0
        results = []

        for i, image_data in enumerate(images_data):
            try:
                image = parse_image(image_data)
                if image is None:
                    failed_count += 1
                    results.append({"index": i, "success": False, "error": "图像解析失败"})
                    continue

                enroll_result = recognizer.enroll(user_id, image)

                if enroll_result:
                    success_count += 1
                    results.append({"index": i, "success": True})
                else:
                    failed_count += 1
                    results.append({"index": i, "success": False, "error": "特征提取失败"})
            except Exception as e:
                failed_count += 1
                results.append({"index": i, "success": False, "error": str(e)})

        # 更新用户名称
        if user_name and database and success_count > 0:
            database.update_user(user_id, user_name=user_name)

        return jsonify({
            "success": True,
            "data": {
                "user_id": user_id,
                "user_name": user_name,
                "total": len(images_data),
                "success_count": success_count,
                "failed_count": failed_count,
                "results": results,
                "message": f"批量注册完成: 成功 {success_count} 张, 失败 {failed_count} 张"
            }
        })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] batch_enroll_upload exception:\n{error_trace}")
        return jsonify({
            "success": False,
            "error": "InternalServerError",
            "message": _safe_error_message(e)
        }), 500


@app.route('/api/recognize', methods=["POST"])
def api_recognize():
    import time
    start_time = time.time()  # 开始计时
    try:
        data = request.get_json()
        print(f"Recognize request data keys: {data.keys() if data else 'None'}")

        if not data:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "请求数据为空"
            }), 400

        image_data = data.get("image")
        infrared_data = data.get("infrared_image")  # 可选的红外图像

        if not image_data:
            print("Error: image data is empty")
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "图像数据不能为空"
            }), 400

        print(f"Image data length: {len(image_data)}")

        image = parse_image(image_data)
        if image is None:
            print("Error: parse_image returned None")
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "图像解析失败"
            }), 400

        result = recognizer.recognize(image)
        print(f"Recognition result: {result}")

        # 记录详细日志用于诊断
        similarity = result.get("similarity", 0)
        matched = result.get("matched", False)
        user_id = result.get("user_id")
        quality = result.get("quality", 0)
        confidence = result.get("confidence", "unknown")
        rejection_reason = result.get("rejection_reason", "")

        # 详细日志
        print(f"[RECOGNIZE] similarity={similarity:.4f}, matched={matched}, "
              f"user_id={user_id}, quality={quality:.2f}, confidence={confidence}")

        # 低置信度警告
        if matched and similarity < 0.95:
            print(f"[LOW_CONFIDENCE] Warning: similarity={similarity:.4f} is below 0.95")

        # 拒绝原因
        if rejection_reason:
            print(f"[REJECTED] {rejection_reason}")

        # 活体检测已禁用，此处保留注释供参考
        # if liveness_score is not None and not liveness_passed:
        #     result['matched'] = False
        #     result['message'] = f'活体检测未通过 (分数: {liveness_score:.0%})'

        user_info = None
        if result.get("matched") and result.get("user_id") and database:
            user_record = database.get_user(result["user_id"])
            if user_record:
                user_info = {
                    "user_id": user_record["user_id"],
                    "user_name": user_record["user_name"],
                    "created_at": user_record["created_at"]
                }

        # V2: 添加质量分数
        quality = result.get("quality", 0.8)

        # 记录识别日志到数据库
        if database:
            try:
                elapsed_ms = int((time.time() - start_time) * 1000)  # 计算响应时间（毫秒）
                current_station = session.get('station', '总站')
                database.log_recognition(
                    user_id=result.get("user_id"),
                    success=result["matched"],
                    confidence=result.get("similarity", 0),
                    response_time_ms=elapsed_ms,
                    station=current_station
                )
                print(f"[API] 已记录识别日志: user_id={result.get('user_id')}, success={result['matched']}")
            except Exception as log_err:
                print(f"[API] 记录识别日志失败: {log_err}")

        response_data = {
            "matched": result["matched"],
            "user_id": result["user_id"],
            "similarity": result["similarity"],
            "quality": quality,
            "message": result["message"],
            "user_info": user_info
        }

        return jsonify({
            "success": True,
            "data": response_data
        })
    except Exception as e:
        print(f"Exception in recognize: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "InternalServerError",
            "message": _safe_error_message(e)
        }), 500


@app.route('/api/users', methods=["GET"])
def list_users():
    try:
        limit = request.args.get("limit", type=int)
        offset = request.args.get("offset", type=int, default=0)

        # 获取当前管理员的站点信息
        current_station = session.get('station', '')
        is_admin = session.get('role') == 'admin'

        # admin角色看全部数据，station角色只看本站数据
        station_filter = None if is_admin else current_station

        # 获取去重后的用户列表
        if database:
            unique_user_ids = database.get_user_ids(station=station_filter)
            # 去重（因为数据库返回的是所有记录的 user_id）
            unique_user_ids = list(dict.fromkeys(unique_user_ids))
            users_data = []

            for user_id in unique_user_ids:
                # 获取用户信息（取最新的一条记录）
                user_record = database.get_user(user_id)
                if not user_record:
                    continue

                # 再次检查站点权限（非admin只能看本站用户）
                if not is_admin and user_record.get("station") != current_station:
                    continue

                # 获取样本数量和平均质量
                sample_count = database.get_user_sample_count(user_id)
                templates = database.get_user_templates(user_id)
                avg_quality = 0.8  # 默认质量
                if templates and len(templates) > 0:
                    qualities = []
                    for t in templates:
                        meta = t.get("metadata")
                        if isinstance(meta, dict):
                            qualities.append(meta.get("quality", 0.8))
                        else:
                            qualities.append(0.8)
                    avg_quality = sum(qualities) / len(qualities) if qualities else 0.8

                users_data.append({
                    "user_id": user_id,
                    "user_name": user_record.get("user_name"),
                    "station": user_record.get("station"),
                    "created_at": user_record.get("created_at"),
                    "updated_at": user_record.get("updated_at"),
                    "sample_count": sample_count,
                    "avg_quality": avg_quality
                })
        else:
            users_data = []

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
            "message": _safe_error_message(e)
        }), 500


@app.route('/api/users/<user_id>', methods=["GET"])
def get_user(user_id: str):
    try:
        # Validate user_id
        is_valid, sanitized_id = RequestValidator.sanitize_user_id(user_id)
        if not is_valid:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": f"用户ID无效: {html.escape(sanitized_id)}"
            }), 400
        user_id = sanitized_id

        user = database.get_user(user_id) if database else None
        if not user:
            return jsonify({
                "success": False,
                "error": "NotFound",
                "message": "User not found: " + html.escape(user_id)
            }), 404

        # V2: 添加样本数量和平均质量
        sample_count = database.get_user_sample_count(user_id) if database else 1
        templates = database.get_user_templates(user_id) if database else []
        avg_quality = 0.8  # 默认质量
        if templates and len(templates) > 0:
            qualities = [t.get("quality", 0.8) for t in templates]
            avg_quality = sum(qualities) / len(qualities) if qualities else 0.8

        return jsonify({
            "success": True,
            "data": {
                "user_id": user["user_id"],
                "user_name": user["user_name"],
                "created_at": user["created_at"],
                "updated_at": user["updated_at"],
                "feature_dim": user["feature_dim"],
                "sample_count": sample_count,
                "avg_quality": avg_quality
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "InternalServerError",
            "message": _safe_error_message(e)
        }), 500


@app.route('/api/users/<user_id>', methods=["DELETE"])
def delete_user_api(user_id: str):
    try:
        # Validate user_id
        is_valid, sanitized_id = RequestValidator.sanitize_user_id(user_id)
        if not is_valid:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": f"用户ID无效: {html.escape(sanitized_id)}"
            }), 400
        user_id = sanitized_id

        # Station 管理员只能删除自己站点的用户
        current_role = session.get('role')
        current_station = session.get('station')
        if current_role == 'station':
            user_station = database.get_user_station(user_id) if database else None
            if user_station != current_station:
                return jsonify({
                    "success": False,
                    "error": "Forbidden",
                    "message": "无权删除其他站点的用户"
                }), 403

        success = recognizer.delete_user(user_id)
        if success:
            database.add_admin_log(
                session.get('username'), 'delete_user', user_id,
                f"station={current_station}", request.remote_addr, current_station
            )
            return jsonify({
                "success": True,
                "message": "User deleted: " + html.escape(user_id)
            })
        else:
            return jsonify({
                "success": False,
                "error": "NotFound",
                "message": "User not found: " + html.escape(user_id)
            }), 404
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "InternalServerError",
            "message": _safe_error_message(e)
        }), 500


@app.route('/api/users/<user_id>/station', methods=['PUT'])
def update_user_station(user_id):
    """修改用户所属站点"""
    try:
        is_valid, sanitized_id = RequestValidator.sanitize_user_id(user_id)
        if not is_valid:
            return jsonify({"success": False, "message": "用户ID无效"}), 400
        user_id = sanitized_id

        # 权限检查
        current_role = session.get('role')
        current_station = session.get('station')
        if current_role == 'station':
            user_station = database.get_user_station(user_id) if database else None
            if user_station != current_station:
                return jsonify({
                    "success": False,
                    "error": "Forbidden",
                    "message": "无权修改其他站点的用户"
                }), 403

        data = request.get_json()
        new_station = (data.get('station') or '').strip() if data else ''
        if not new_station:
            return jsonify({"success": False, "message": "站点名称不能为空"}), 400

        if database:
            user_station_before = database.get_user_station(user_id)
            success = database.update_user(user_id, station=new_station)
            if success:
                database.add_admin_log(
                    session.get('username'), 'update_user_station', user_id,
                    f"from={user_station_before}, to={new_station}", request.remote_addr, current_station
                )
                return jsonify({"success": True, "message": f"站点已修改为 {new_station}"})
            return jsonify({"success": False, "message": "用户不存在"}), 404
        return jsonify({"success": False, "message": "数据库未初始化"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": "InternalServerError", "message": _safe_error_message(e)}), 500


@app.route('/api/stations', methods=['GET'])
def list_stations():
    """获取所有站点"""
    if not session.get('logged_in'):
        return jsonify({"success": False, "message": "请先登录"}), 401
    try:
        stations = database.get_all_stations() if database else []
        return jsonify({"success": True, "data": stations})
    except Exception as e:
        return jsonify({"success": False, "error": "InternalServerError", "message": _safe_error_message(e)}), 500


@app.route('/api/stations', methods=['POST'])
def create_station():
    """创建站点（仅 admin）"""
    if not session.get('logged_in'):
        return jsonify({"success": False, "message": "请先登录"}), 401
    if session.get('role') != 'admin':
        return jsonify({"success": False, "message": "需要管理员权限"}), 403
    try:
        data = request.get_json()
        name = (data.get('name') or '').strip() if data else ''
        color = (data.get('color') or 'slate').strip()
        if not name:
            return jsonify({"success": False, "message": "站点名称不能为空"}), 400
        if database:
            ok = database.add_station(name, color)
            if ok:
                database.add_admin_log(
                    session.get('username'), 'create_station', name,
                    f"color={color}", request.remote_addr, current_station
                )
                return jsonify({"success": True, "message": f"站点 {name} 创建成功"})
            return jsonify({"success": False, "message": "站点已存在"}), 400
        return jsonify({"success": False, "message": "数据库未初始化"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": "InternalServerError", "message": _safe_error_message(e)}), 500


@app.route('/api/stations/<name>', methods=['PUT'])
def update_station(name):
    """更新站点（仅 admin）"""
    if not session.get('logged_in'):
        return jsonify({"success": False, "message": "请先登录"}), 401
    if session.get('role') != 'admin':
        return jsonify({"success": False, "message": "需要管理员权限"}), 403
    try:
        data = request.get_json()
        new_name = (data.get('name') or '').strip() if data else None
        color = (data.get('color') or '').strip() or None
        if database:
            ok = database.update_station(name, color=color, new_name=new_name)
            if ok:
                database.add_admin_log(
                    session.get('username'), 'update_station', name,
                    f"new_name={new_name}, color={color}", request.remote_addr, current_station
                )
                return jsonify({"success": True, "message": "站点已更新"})
            return jsonify({"success": False, "message": "站点不存在"}), 404
        return jsonify({"success": False, "message": "数据库未初始化"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": "InternalServerError", "message": _safe_error_message(e)}), 500


@app.route('/api/stations/<name>', methods=['DELETE'])
def delete_station(name):
    """删除站点（仅 admin）"""
    if not session.get('logged_in'):
        return jsonify({"success": False, "message": "请先登录"}), 401
    if session.get('role') != 'admin':
        return jsonify({"success": False, "message": "需要管理员权限"}), 403
    try:
        if database:
            ok = database.delete_station(name)
            if ok:
                database.add_admin_log(
                    session.get('username'), 'delete_station', name,
                    None, request.remote_addr, current_station
                )
                return jsonify({"success": True, "message": f"站点 {name} 已删除"})
            return jsonify({"success": False, "message": "站点不存在"}), 404
        return jsonify({"success": False, "message": "数据库未初始化"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": "InternalServerError", "message": _safe_error_message(e)}), 500


@app.route('/api/users/batch-delete', methods=["POST"])
def batch_delete_users():
    try:
        data = request.get_json()
        user_ids = data.get("user_ids", [])
        if not user_ids:
            return jsonify({
                "success": False,
                "error": "BadRequest",
                "message": "Missing user_ids"
            }), 400

        # Station 管理员只能删除自己站点的用户
        current_role = session.get('role')
        current_station = session.get('station')

        deleted = 0
        failed = 0
        for user_id in user_ids:
            # 权限检查
            if current_role == 'station' and database:
                user_station = database.get_user_station(user_id)
                if user_station != current_station:
                    failed += 1
                    continue
            if recognizer.delete_user(user_id):
                deleted += 1
                database.add_admin_log(
                    session.get('username'), 'delete_user', user_id,
                    f"batch_delete, station={current_station}", request.remote_addr, current_station
                )
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
            "message": _safe_error_message(e)
        }), 500


@app.route('/api/users/clear', methods=["POST"])
def clear_users():
    """清空所有用户（仅 admin）"""
    if not session.get('logged_in'):
        return jsonify({"success": False, "message": "请先登录"}), 401
    if session.get('role') != 'admin':
        return jsonify({"success": False, "message": "需要管理员权限"}), 403
    try:
        deleted = recognizer.clear_users()
        current_station = session.get('station', '总站')
        database.add_admin_log(
            session.get('username'), 'clear_users', None,
            f"deleted={deleted}", request.remote_addr, current_station
        )
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
            "message": _safe_error_message(e)
        }), 500


# ==================== Admin Management API ====================

@app.route('/api/admins', methods=['GET'])
def list_admins():
    """获取所有管理员（仅 admin）"""
    if not session.get('logged_in'):
        return jsonify({"success": False, "message": "请先登录"}), 401
    if session.get('role') != 'admin':
        return jsonify({"success": False, "message": "需要管理员权限"}), 403
    try:
        admins = auth_manager.list_admins()
        return jsonify({"success": True, "data": admins})
    except Exception as e:
        return jsonify({"success": False, "error": "InternalServerError", "message": _safe_error_message(e)}), 500


@app.route('/api/admins', methods=['POST'])
def create_admin():
    """创建管理员（仅 admin）"""
    if not session.get('logged_in'):
        return jsonify({"success": False, "message": "请先登录"}), 401
    if session.get('role') != 'admin':
        return jsonify({"success": False, "message": "需要管理员权限"}), 403
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "请求数据为空"}), 400
        username = (data.get('username') or '').strip()
        password = data.get('password', '')
        station = (data.get('station') or '').strip()
        role = data.get('role', 'station')
        if not username or not password or not station:
            return jsonify({"success": False, "message": "用户名、密码、站点不能为空"}), 400
        ok, msg = auth_manager.create_admin(username, password, station, role)
        if ok:
            current_station = session.get('station', '总站')
            database.add_admin_log(
                session.get('username'), 'create_admin', username,
                f"station={station}, role={role}", request.remote_addr, current_station
            )
            return jsonify({"success": True, "message": msg})
        else:
            return jsonify({"success": False, "message": msg}), 400
    except Exception as e:
        return jsonify({"success": False, "error": "InternalServerError", "message": _safe_error_message(e)}), 500


@app.route('/api/admins/<username>', methods=['DELETE'])
def delete_admin(username):
    """删除管理员（仅 admin）"""
    if not session.get('logged_in'):
        return jsonify({"success": False, "message": "请先登录"}), 401
    if session.get('role') != 'admin':
        return jsonify({"success": False, "message": "需要管理员权限"}), 403
    try:
        ok, msg = auth_manager.delete_admin(username)
        if ok:
            current_station = session.get('station', '总站')
            database.add_admin_log(
                session.get('username'), 'delete_admin', username,
                None, request.remote_addr, current_station
            )
            return jsonify({"success": True, "message": msg})
        else:
            return jsonify({"success": False, "message": msg}), 400
    except Exception as e:
        return jsonify({"success": False, "error": "InternalServerError", "message": _safe_error_message(e)}), 500


@app.route('/api/admins/<username>/password', methods=['PUT'])
def update_admin_password(username):
    """修改管理员密码（仅 admin）"""
    if not session.get('logged_in'):
        return jsonify({"success": False, "message": "请先登录"}), 401
    if session.get('role') != 'admin':
        return jsonify({"success": False, "message": "需要管理员权限"}), 403
    try:
        data = request.get_json()
        new_password = data.get('new_password', '') if data else ''
        ok, msg = auth_manager.update_admin_password(username, new_password)
        if ok:
            current_station = session.get('station', '总站')
            database.add_admin_log(
                session.get('username'), 'update_admin_password', username,
                None, request.remote_addr, current_station
            )
            return jsonify({"success": True, "message": msg})
        else:
            return jsonify({"success": False, "message": msg}), 400
    except Exception as e:
        return jsonify({"success": False, "error": "InternalServerError", "message": _safe_error_message(e)}), 500


@app.route('/api/admins/<username>/station', methods=['PUT'])
def update_admin_station(username):
    """修改管理员所属站点（仅 admin）"""
    if not session.get('logged_in'):
        return jsonify({"success": False, "message": "请先登录"}), 401
    if session.get('role') != 'admin':
        return jsonify({"success": False, "message": "需要管理员权限"}), 403
    try:
        data = request.get_json()
        new_station = (data.get('station') or '').strip() if data else ''
        if not new_station:
            return jsonify({"success": False, "message": "站点名称不能为空"}), 400
        ok, msg = auth_manager.update_admin_station(username, new_station)
        if ok:
            current_station = session.get('station', '总站')
            database.add_admin_log(
                session.get('username'), 'update_admin_station', username,
                f"new_station={new_station}", request.remote_addr, current_station
            )
            return jsonify({"success": True, "message": msg})
        else:
            return jsonify({"success": False, "message": msg}), 400
    except Exception as e:
        return jsonify({"success": False, "error": "InternalServerError", "message": _safe_error_message(e)}), 500


@app.route('/api/admins/<username>/username', methods=['PUT'])
def update_admin_username(username):
    """修改管理员用户名（仅 admin）"""
    if not session.get('logged_in'):
        return jsonify({"success": False, "message": "请先登录"}), 401
    if session.get('role') != 'admin':
        return jsonify({"success": False, "message": "需要管理员权限"}), 403
    try:
        data = request.get_json()
        new_username = (data.get('username') or '').strip() if data else ''
        if not new_username:
            return jsonify({"success": False, "message": "用户名不能为空"}), 400
        if len(new_username) < 2:
            return jsonify({"success": False, "message": "用户名长度不能少于2位"}), 400
        ok, msg = auth_manager.update_admin_username(username, new_username)
        if ok:
            current_station = session.get('station', '总站')
            database.add_admin_log(
                session.get('username'), 'update_admin_username', username,
                f"new_username={new_username}", request.remote_addr, current_station
            )
            return jsonify({"success": True, "message": msg})
        else:
            return jsonify({"success": False, "message": msg}), 400
    except Exception as e:
        return jsonify({"success": False, "error": "InternalServerError", "message": _safe_error_message(e)}), 500


@app.route('/api/admins/<username>/role', methods=['PUT'])
def update_admin_role(username):
    """修改管理员角色（仅 admin）"""
    if not session.get('logged_in'):
        return jsonify({"success": False, "message": "请先登录"}), 401
    if session.get('role') != 'admin':
        return jsonify({"success": False, "message": "需要管理员权限"}), 403
    try:
        data = request.get_json()
        new_role = (data.get('role') or '').strip() if data else ''
        if new_role not in ('admin', 'station'):
            return jsonify({"success": False, "message": "角色只能是 admin 或 station"}), 400
        ok, msg = auth_manager.update_admin_role(username, new_role)
        if ok:
            current_station = session.get('station', '总站')
            database.add_admin_log(
                session.get('username'), 'update_admin_role', username,
                f"new_role={new_role}", request.remote_addr, current_station
            )
            return jsonify({"success": True, "message": msg})
        else:
            return jsonify({"success": False, "message": msg}), 400
    except Exception as e:
        return jsonify({"success": False, "error": "InternalServerError", "message": _safe_error_message(e)}), 500


@app.route('/api/admin_logs', methods=['GET'])
def get_admin_logs():
    """获取审计日志（仅 admin）"""
    if not session.get('logged_in'):
        return jsonify({"success": False, "message": "请先登录"}), 401
    if session.get('role') != 'admin':
        return jsonify({"success": False, "message": "需要管理员权限"}), 403
    try:
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        action = request.args.get('action')
        admin_username = request.args.get('admin_username')

        if not database:
            return jsonify({"success": False, "message": "数据库未初始化"}), 500

        logs = database.get_admin_logs(
            limit=limit,
            offset=offset,
            action=action,
            admin_name=admin_username
        )
        return jsonify({"success": True, "data": logs})
    except Exception as e:
        return jsonify({"success": False, "error": "InternalServerError", "message": _safe_error_message(e)}), 500


# ==================== Login/Logout Routes ====================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """管理员登录页面"""
    # 如果已经登录，直接跳转
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))

    if request.method == 'GET':
        return render_template('login.html')

    # POST - 处理登录
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')

    # 登录频率限制
    client_ip = request.remote_addr or '127.0.0.1'
    allowed, error_msg = _check_login_rate_limit(client_ip)
    if not allowed:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"success": False, "message": error_msg}), 429
        return jsonify({"success": False, "message": error_msg}), 429

    # 验证凭证
    user_info = auth_manager.verify_password(username, password)
    if user_info:
        session['logged_in'] = True
        session['username'] = user_info['username']
        session['station'] = user_info['station']
        session['role'] = user_info['role']
        session.permanent = True

        print(f"[LOGIN] Admin logged in: {username}, station={user_info['station']}, role={user_info['role']}")

        # 审计日志（登录成功）
        if database:
            database.add_admin_log(username, 'login', None, f"station={user_info['station']}, role={user_info['role']}", client_ip, user_info['station'])

        # AJAX 请求返回 JSON
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                "success": True,
                "message": "登录成功",
                "redirect": url_for('dashboard')
            })

        # 普通表单提交也返回 JSON，让 JS 处理跳转
        return jsonify({
            "success": True,
            "message": "登录成功",
            "redirect": url_for('dashboard')
        })
    else:
        _record_failed_login(client_ip)
        if database:
            database.add_admin_log(username, 'login_failed', None, None, client_ip, '未知')
        print(f"[LOGIN] Failed login attempt: username={username}")

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                "success": False,
                "message": "用户名或密码错误"
            }), 401

        # 普通表单提交也返回 JSON 错误
        return jsonify({
            "success": False,
            "message": "用户名或密码错误"
        }), 401


@app.route('/logout')
def logout():
    """退出登录"""
    username = session.get('username', 'unknown')
    current_station = session.get('station', '总站')
    if database:
        database.add_admin_log(username, 'logout', None, None, request.remote_addr, current_station)
    session.clear()
    print(f"[LOGOUT] Admin logged out: {username}")
    return redirect(url_for('login'))


# ==================== Page Routes ====================

@app.route('/')
@login_required
def index():
    return render_template('dashboard.html', active_page='dashboard',
                           current_role=session.get('role', 'station'))


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', active_page='dashboard',
                           current_role=session.get('role', 'station'))


@app.route('/enroll')
@login_required
def enroll():
    return render_template('enroll.html', active_page='enroll',
                           current_role=session.get('role', 'station'))


@app.route('/recognize')
@login_required
def recognize():
    return render_template('recognize.html', active_page='recognize',
                           current_role=session.get('role', 'station'))


@app.route('/users')
@login_required
def users():
    return render_template('users.html', active_page='users',
                           current_role=session.get('role', 'station'),
                           current_station=session.get('station', ''))


@app.route('/federated')
@login_required
def federated():
    return render_template('federated.html', active_page='federated',
                           current_role=session.get('role', 'station'))


# ==================== Coordinator Management API ====================

def check_coordinator_status() -> Dict[str, Any]:
    """检查云协调器状态"""
    try:
        response = requests.get(f'http://localhost:{COORDINATOR_PORT}/status', timeout=2)
        if response.status_code == 200:
            return {"running": True, "data": response.json()}
    except:
        pass
    return {"running": False, "data": None}


@app.route('/api/coordinator/status', methods=['GET'])
def api_coordinator_status():
    """获取云协调器状态"""
    status = check_coordinator_status()
    return jsonify({
        "success": True,
        "running": status["running"],
        "data": status["data"]
    })


@app.route('/coordinator-proxy/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
@login_required
def coordinator_proxy(path):
    """代理协调器请求，解决跨域问题"""
    import urllib.request
    import urllib.error

    # 安全检查：防止路径遍历和非法字符
    if '..' in path or path.startswith('/') or '\\' in path:
        return jsonify({'error': 'Invalid path'}), 400

    # 只允许特定路径前缀，防止 SSRF
    allowed_prefixes = ('train', 'start_', 'status', 'metrics', 'config', 'clients', 'model', 'terminals')
    if not any(path.startswith(p) for p in allowed_prefixes):
        return jsonify({'error': 'Endpoint not allowed via proxy'}), 403

    coordinator_url = f'http://localhost:{COORDINATOR_PORT}/{path}'

    try:
        if request.method == 'GET':
            req = urllib.request.Request(coordinator_url)
        elif request.method == 'POST':
            req = urllib.request.Request(
                coordinator_url,
                data=request.get_data(),
                headers={'Content-Type': 'application/json'}
            )
        elif request.method in ('PUT', 'DELETE'):
            req = urllib.request.Request(
                coordinator_url,
                data=request.get_data(),
                method=request.method,
                headers={'Content-Type': 'application/json'}
            )
        else:
            req = urllib.request.Request(coordinator_url, method=request.method)

        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
            return data, 200, {'Content-Type': 'application/json'}
    except urllib.error.HTTPError as e:
        return jsonify({'error': str(e)}), e.code
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/coordinator/start', methods=['POST'])
@login_required
def api_coordinator_start():
    """启动云协调器"""
    global coordinator_process

    # 先检查是否已经运行
    status = check_coordinator_status()
    if status["running"]:
        return jsonify({
            "success": True,
            "message": "云协调器已经在运行中",
            "port": COORDINATOR_PORT
        })

    try:
        # 启动云协调器进程
        python_exe = sys.executable
        coordinator_path = project_root / 'federated' / 'cloud' / 'coordinator.py'

        # 使用 DETACHED_PROCESS 在 Windows 上创建独立进程，避免管道阻塞
        if sys.platform == 'win32':
            # 创建新进程组，完全分离
            coordinator_process = subprocess.Popen(
                [python_exe, str(coordinator_path)],
                cwd=str(project_root),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                close_fds=True
            )
        else:
            coordinator_process = subprocess.Popen(
                [python_exe, str(coordinator_path)],
                cwd=str(project_root),
                start_new_session=True,
                close_fds=True
            )

        # 等待启动 (增加到15秒以适应模型加载)
        import time
        for i in range(15):  # 最多等待15秒
            time.sleep(1)
            status = check_coordinator_status()
            if status["running"]:
                return jsonify({
                    "success": True,
                    "message": "云协调器启动成功",
                    "port": COORDINATOR_PORT
                })
            # 检查进程是否意外终止
            if coordinator_process and coordinator_process.poll() is not None:
                return jsonify({
                    "success": False,
                    "message": f"云协调器进程意外退出，返回码: {coordinator_process.returncode}"
                }), 500

        return jsonify({
            "success": False,
            "message": "云协调器启动超时，请检查端口5002是否被占用"
        }), 500

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"启动失败: {str(e)}"
        }), 500


@app.route('/api/coordinator/stop', methods=['POST'])
@login_required
def api_coordinator_stop():
    """停止云协调器"""
    global coordinator_process

    try:
        if coordinator_process:
            coordinator_process.terminate()
            coordinator_process.wait(timeout=5)
            coordinator_process = None
            return jsonify({
                "success": True,
                "message": "云协调器已停止"
            })
        else:
            # 尝试通过API停止
            try:
                requests.post(f'http://localhost:{COORDINATOR_PORT}/shutdown', timeout=2)
            except:
                pass
            return jsonify({
                "success": True,
                "message": "已发送停止信号"
            })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"停止失败: {str(e)}"
        }), 500


# ==================== Terminal Simulator Routes ====================

@app.route('/api/simulator/start', methods=['POST'])
@login_required
def api_simulator_start():
    """启动终端模拟器"""
    global simulator_process

    # 先检查云协调器是否运行
    status = check_coordinator_status()
    if not status["running"]:
        return jsonify({
            "success": False,
            "message": "请先启动云协调器"
        }), 400

    # 检查模拟器是否已经运行
    if simulator_process and simulator_process.poll() is None:
        return jsonify({
            "success": True,
            "message": "终端模拟器已经在运行中"
        })

    try:
        python_exe = sys.executable
        simulator_path = project_root / 'scripts' / 'terminal_simulator.py'

        # 使用分离进程启动模拟器
        if sys.platform == 'win32':
            simulator_process = subprocess.Popen(
                [python_exe, str(simulator_path), '--terminals', '3', '--rounds', '10', '--coordinator-url', f'http://localhost:{COORDINATOR_PORT}'],
                cwd=str(project_root),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                close_fds=True
            )
        else:
            simulator_process = subprocess.Popen(
                [python_exe, str(simulator_path), '--terminals', '3', '--rounds', '10', '--coordinator-url', f'http://localhost:{COORDINATOR_PORT}'],
                cwd=str(project_root),
                start_new_session=True,
                close_fds=True
            )

        return jsonify({
            "success": True,
            "message": "终端模拟器启动成功"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"启动失败: {str(e)}"
        }), 500


@app.route('/api/simulator/stop', methods=['POST'])
@login_required
def api_simulator_stop():
    """停止终端模拟器"""
    global simulator_process

    try:
        # 1. 先尝试停止通过 Web API 启动的进程
        if simulator_process and simulator_process.poll() is None:
            simulator_process.terminate()
            try:
                simulator_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                simulator_process.kill()
            simulator_process = None

        # 2. 使用 wmic 终止所有 terminal_simulator 进程
        import time
        time.sleep(0.5)

        # 使用 psutil 安全终止进程（避免命令注入）
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'terminal_simulator' in cmdline and proc.info['pid'] != os.getpid():
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return jsonify({
            "success": True,
            "message": "终端模拟器已停止"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"停止失败: {str(e)}"
        }), 500


@app.route('/api/simulator/status', methods=['GET'])
def api_simulator_status():
    """获取终端模拟器状态"""
    global simulator_process
    running = simulator_process is not None and simulator_process.poll() is None
    return jsonify({
        "success": True,
        "running": running
    })


# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(error):
    return render_template('dashboard.html', active_page='dashboard'), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('dashboard.html', active_page='dashboard'), 500


# ==================== Main ====================

def main():
    print("=" * 60)
    print("PalmSecure Web Application")
    print("=" * 60)
    print("Running on: http://" + APIConfig.HOST + ":" + str(APIConfig.PORT))
    print("Debug mode: " + str(APIConfig.DEBUG))
    print("Model: " + APIConfig.MODEL_CHECKPOINT)
    print("Database: " + APIConfig.DB_PATH)
    print("=" * 60)
    print("\nAvailable pages:")
    print("  - Dashboard: http://localhost:8000/dashboard")
    print("  - Enroll:    http://localhost:8000/enroll")
    print("  - Recognize: http://localhost:8000/recognize")
    print("  - Users:     http://localhost:8000/users")
    print("  - Federated: http://localhost:8000/federated")
    print("=" * 60)
    print("\nPress Ctrl+C to stop server\n")

    init_services()
    app.run(
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        debug=APIConfig.DEBUG,
        threaded=True
    )


if __name__ == "__main__":
    main()
