# -*- coding: utf-8 -*-
"""
PalmSecureAI Web Application - Authentication Module

Provides authentication against admins table, role-based access control (RBAC),
and security utilities for the web application.
"""

import os
import functools
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
from flask import request, session, current_app
from werkzeug.security import check_password_hash, generate_password_hash


class AuthManager:
    """Authentication manager using admins table"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_path = os.path.join(project_root, 'data', 'palm_templates.db')
        self.db_path = db_path

    def _get_db(self):
        """获取数据库连接"""
        return sqlite3.connect(self.db_path)

    def verify_password(self, username: str, password: str) -> Optional[Dict[str, str]]:
        """
        验证用户凭证，从 admins 表查询

        Returns:
            用户信息 dict（包含 username, station, role）验证成功
            None - 验证失败
        """
        conn = self._get_db()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT username, password_hash, station, role FROM admins WHERE username = ?",
                (username,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            username, password_hash, station, role = row
            if check_password_hash(password_hash, password):
                return {
                    'username': username,
                    'station': station,
                    'role': role,
                }
            return None
        finally:
            conn.close()

    def get_user_info(self, username: str) -> Optional[Dict[str, str]]:
        """获取用户信息"""
        conn = self._get_db()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT username, station, role FROM admins WHERE username = ?",
                (username,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return {'username': row[0], 'station': row[1], 'role': row[2]}
        finally:
            conn.close()

    def list_admins(self) -> list:
        """获取所有管理员"""
        conn = self._get_db()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id, username, station, role, created_at FROM admins ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [
                {'id': r[0], 'username': r[1], 'station': r[2], 'role': r[3], 'created_at': r[4]}
                for r in rows
            ]
        finally:
            conn.close()

    def create_admin(self, username: str, password: str, station: str, role: str = 'station') -> tuple[bool, str]:
        """创建管理员"""
        if role not in ('admin', 'station'):
            return False, '角色只能是 admin 或 station'
        if len(password) < 8:
            return False, '密码长度不能少于8位'
        conn = self._get_db()
        cursor = conn.cursor()
        try:
            # 检查是否已存在
            cursor.execute("SELECT id FROM admins WHERE username = ?", (username,))
            if cursor.fetchone():
                return False, '用户名已存在'
            password_hash = generate_password_hash(password)
            cursor.execute(
                "INSERT INTO admins (username, password_hash, station, role) VALUES (?, ?, ?, ?)",
                (username, password_hash, station, role)
            )
            conn.commit()
            return True, '创建成功'
        except Exception as e:
            return False, str(e)
        finally:
            conn.close()

    def delete_admin(self, username: str) -> tuple[bool, str]:
        """删除管理员"""
        conn = self._get_db()
        cursor = conn.cursor()
        try:
            # 不允许删除自己
            current = session.get('username') if session.get('logged_in') else None
            if username == current:
                return False, '不能删除自己'
            cursor.execute("DELETE FROM admins WHERE username = ?", (username,))
            conn.commit()
            if cursor.rowcount == 0:
                return False, '用户不存在'
            return True, '删除成功'
        finally:
            conn.close()

    def update_admin_password(self, username: str, new_password: str) -> tuple[bool, str]:
        """修改管理员密码"""
        if len(new_password) < 8:
            return False, '密码长度不能少于8位'
        conn = self._get_db()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM admins WHERE username = ?", (username,))
            if not cursor.fetchone():
                return False, '用户不存在'
            password_hash = generate_password_hash(new_password)
            cursor.execute("UPDATE admins SET password_hash = ? WHERE username = ?", (password_hash, username))
            conn.commit()
            return True, '密码修改成功'
        finally:
            conn.close()

    def update_admin_station(self, username: str, new_station: str) -> tuple[bool, str]:
        """修改管理员所属站点"""
        conn = self._get_db()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM admins WHERE username = ?", (username,))
            if not cursor.fetchone():
                return False, '用户不存在'
            cursor.execute("UPDATE admins SET station = ? WHERE username = ?", (new_station, username))
            conn.commit()
            return True, '站点修改成功'
        finally:
            conn.close()

    def update_admin_username(self, username: str, new_username: str) -> tuple[bool, str]:
        """修改管理员用户名"""
        conn = self._get_db()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM admins WHERE username = ?", (username,))
            if not cursor.fetchone():
                return False, '用户不存在'
            cursor.execute("SELECT id FROM admins WHERE username = ? AND username != ?", (new_username, username))
            if cursor.fetchone():
                return False, '新用户名已存在'
            cursor.execute("UPDATE admins SET username = ? WHERE username = ?", (new_username, username))
            conn.commit()
            return True, '用户名修改成功'
        finally:
            conn.close()

    def update_admin_role(self, username: str, new_role: str) -> tuple[bool, str]:
        """修改管理员角色"""
        conn = self._get_db()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM admins WHERE username = ?", (username,))
            if not cursor.fetchone():
                return False, '用户不存在'
            cursor.execute("UPDATE admins SET role = ? WHERE username = ?", (new_role, username))
            conn.commit()
            return True, '角色修改成功'
        finally:
            conn.close()


# Global auth manager instance
auth_manager = AuthManager()


def require_login(f: Callable) -> Callable:
    """装饰器：要求用户已登录"""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return {"success": False, "message": "请先登录"}, 401
        return f(*args, **kwargs)
    return decorated


def require_admin(f: Callable) -> Callable:
    """装饰器：要求 admin 角色"""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return {"success": False, "message": "请先登录"}, 401
        if session.get('role') != 'admin':
            return {"success": False, "message": "需要管理员权限"}, 403
        return f(*args, **kwargs)
    return decorated


def get_current_user() -> Optional[Dict[str, str]]:
    """获取当前登录用户信息"""
    if not session.get('logged_in'):
        return None
    return {
        'username': session.get('username'),
        'station': session.get('station'),
        'role': session.get('role'),
    }
