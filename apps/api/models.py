# -*- coding: utf-8 -*-
"""
API 请求和响应模型

定义 API 的请求体和响应体结构
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
from datetime import datetime


# ------------------------------------------------------------------
# 请求模型
# ------------------------------------------------------------------

@dataclass
class EnrollRequest:
    """用户注册请求"""
    user_id: str
    user_name: Optional[str] = None
    image: Optional[str] = None  # base64 编码的图像

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "image": self.image
        }


@dataclass
class RecognizeRequest:
    """掌纹识别请求"""
    image: str  # base64 编码的图像
    top_k: Optional[int] = 3  # 返回前K个匹配结果

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image": self.image,
            "top_k": self.top_k
        }


@dataclass
class BatchDeleteRequest:
    """批量删除用户请求"""
    user_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_ids": self.user_ids
        }


@dataclass
class UpdateUserRequest:
    """更新用户请求"""
    user_name: Optional[str] = None
    image: Optional[str] = None  # 用于更新特征

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.user_name is not None:
            result["user_name"] = self.user_name
        if self.image is not None:
            result["image"] = self.image
        return result


# ------------------------------------------------------------------
# 响应模型
# ------------------------------------------------------------------

@dataclass
class ApiResponse:
    """基础 API 响应"""
    success: bool
    error: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"success": self.success}
        if self.error:
            result["error"] = self.error
        if self.message:
            result["message"] = self.message
        return result


@dataclass
class EnrollResponse:
    """用户注册响应"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"success": self.success}
        if self.data:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        if self.message:
            result["message"] = self.message
        return result


@dataclass
class RecognizeResponse:
    """掌纹识别响应"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"success": self.success}
        if self.data:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class UserInfo:
    """用户信息"""
    user_id: str
    user_name: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


@dataclass
class UsersListResponse:
    """用户列表响应"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"success": self.success}
        if self.data:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class StatusResponse:
    """系统状态响应"""
    success: bool
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data
        }


@dataclass
class HealthResponse:
    """健康检查响应"""
    success: bool
    status: str
    timestamp: str
    service: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "timestamp": self.timestamp,
            "service": self.service
        }


# ------------------------------------------------------------------
# 验证函数
# ------------------------------------------------------------------

def validate_enroll_request(data: Dict[str, Any]) -> tuple[bool, Optional[str], Optional[EnrollRequest]]:
    """
    验证用户注册请求

    Returns:
        (是否有效, 错误消息, 解析后的请求)
    """
    if not data:
        return False, "请求体不能为空", None

    user_id = data.get("user_id")
    if not user_id:
        return False, "缺少 user_id 参数", None

    if not isinstance(user_id, str) or len(user_id) == 0:
        return False, "user_id 必须是非空字符串", None

    image = data.get("image")
    if not image:
        return False, "缺少 image 参数", None

    request = EnrollRequest(
        user_id=user_id,
        user_name=data.get("user_name"),
        image=image
    )

    return True, None, request


def validate_recognize_request(data: Dict[str, Any]) -> tuple[bool, Optional[str], Optional[RecognizeRequest]]:
    """
    验证识别请求

    Returns:
        (是否有效, 错误消息, 解析后的请求)
    """
    if not data:
        return False, "请求体不能为空", None

    image = data.get("image")
    if not image:
        return False, "缺少 image 参数", None

    top_k = data.get("top_k", 3)
    if top_k is not None:
        try:
            top_k = int(top_k)
            if top_k < 1 or top_k > 100:
                return False, "top_k 必须在 1-100 之间", None
        except (ValueError, TypeError):
            return False, "top_k 必须是整数", None

    request = RecognizeRequest(
        image=image,
        top_k=top_k
    )

    return True, None, request


# ------------------------------------------------------------------
# 示例请求
# ------------------------------------------------------------------

EXAMPLE_REQUESTS = {
    "enroll": EnrollRequest(
        user_id="user_001",
        user_name="张三",
        image="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    ),
    "recognize": RecognizeRequest(
        image="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
        top_k=3
    )
}


def get_example_request(endpoint: str) -> Optional[Dict[str, Any]]:
    """
    获取示例请求

    Args:
        endpoint: 端点名称 (enroll, recognize)

    Returns:
        示例请求字典
    """
    if endpoint in EXAMPLE_REQUESTS:
        return EXAMPLE_REQUESTS[endpoint].to_dict()
    return None


if __name__ == "__main__":
    # 测试代码
    print("=" * 50)
    print("API 模型测试")
    print("=" * 50)

    # 测试注册请求验证
    print("\n测试 1: 注册请求验证")
    valid, error, request = validate_enroll_request({
        "user_id": "user_001",
        "user_name": "张三",
        "image": "base64_data"
    })
    print(f"  有效: {valid}")
    print(f"  请求: {request.to_dict() if request else None}")

    # 测试识别请求验证
    print("\n测试 2: 识别请求验证")
    valid, error, request = validate_recognize_request({
        "image": "base64_data",
        "top_k": 5
    })
    print(f"  有效: {valid}")
    print(f"  请求: {request.to_dict() if request else None}")

    # 测试示例请求
    print("\n测试 3: 示例请求")
    for endpoint, example in EXAMPLE_REQUESTS.items():
        print(f"  {endpoint}: {example.to_dict()}")
