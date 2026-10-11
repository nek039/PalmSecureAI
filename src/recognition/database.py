# -*- coding: utf-8 -*-
"""
掌纹识别数据库模块

使用 SQLite 实现用户模板的持久化存储
替换原有的 pickle 文件存储方式
"""

import sqlite3
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConfig:
    """数据库配置"""

    def __init__(
        self,
        db_path: str = "data/palm_templates.db",
        table_name: str = "users"
    ):
        self.db_path = db_path
        self.table_name = table_name


class PalmDatabase:
    """
    掌纹识别数据库

    功能：
        - 用户模板的 CRUD 操作
        - 特征向量存储（JSON 序列化）
        - 批量查询和相似度搜索
        - 数据库初始化和迁移
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.db_path = self.config.db_path
        self.table_name = self.config.table_name
        self._ensure_data_dir()
        self._init_database()

    def _ensure_data_dir(self) -> None:
        """确保数据目录存在"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # 允许的表名白名单（防止SQL注入）
    ALLOWED_TABLE_NAMES = {'users', 'templates', 'palm_users', 'palm_templates'}

    def _validate_table_name(self, table_name: str) -> bool:
        """验证表名是否在白名单中"""
        return table_name in self.ALLOWED_TABLE_NAMES

    def _init_database(self) -> None:
        """初始化数据库表结构"""
        # 验证表名
        if not self._validate_table_name(self.table_name):
            raise ValueError(f"Invalid table name: {self.table_name}")

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 先检查表是否存在，避免重复创建索引
            # 使用参数化查询防止SQL注入，但表名无法参数化，已通过白名单验证
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (self.table_name,)
            )
            table_exists = cursor.fetchone() is not None

            # 创建表（支持同一用户多个模板）
            # 表名已通过白名单验证，可以安全使用
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    user_name TEXT,
                    feature_vector TEXT NOT NULL,
                    feature_dim INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    hand_type TEXT,
                    station TEXT DEFAULT '总站'
                )
            """)

            # 创建索引（仅在首次创建表时）
            if not table_exists:
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_user_id
                    ON {self.table_name}(user_id)
                """)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_created_at
                    ON {self.table_name}(created_at)
                """)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_hand_type
                    ON {self.table_name}(hand_type)
                """)

            # 创建识别日志表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    success INTEGER NOT NULL,
                    confidence REAL,
                    response_time INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 创建识别日志索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_recognition_created_at
                ON recognition_logs(created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_recognition_success
                ON recognition_logs(success)
            """)

            # 创建管理员操作审计日志表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS admin_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target TEXT,
                    details TEXT,
                    ip_address TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_admin_logs_timestamp
                ON admin_logs(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_admin_logs_admin
                ON admin_logs(admin_name)
            """)

            # 创建管理员表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS admins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    station TEXT NOT NULL DEFAULT '总站',
                    role TEXT NOT NULL DEFAULT 'station',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

            # 创建站点表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    color TEXT NOT NULL DEFAULT 'slate',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 插入默认站点（如果不存在）
            default_stations = [
                ('总站', 'slate'),
                ('分站1', 'blue'),
                ('分站2', 'green'),
                ('分站3', 'orange'),
            ]
            for name, color in default_stations:
                cursor.execute("SELECT id FROM stations WHERE name = ?", (name,))
                if not cursor.fetchone():
                    cursor.execute("INSERT INTO stations (name, color) VALUES (?, ?)", (name, color))
            conn.commit()

            logger.info(f"数据库初始化完成: {self.db_path}")

    def add_admin_log(self, admin_name: str, action: str, target: str = None,
                      details: str = None, ip_address: str = None, station: str = None) -> bool:
        """记录管理员操作日志"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO admin_logs (admin_name, action, target, details, ip_address, station) VALUES (?, ?, ?, ?, ?, ?)",
                    (admin_name, action, target, details, ip_address, station)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"添加管理员日志失败: {e}")
            return False

    def get_admin_logs(self, limit: int = 100, offset: int = 0,
                       admin_name: str = None, action: str = None, station: str = None) -> List[Dict[str, Any]]:
        """获取管理员操作日志"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                query = "SELECT * FROM admin_logs WHERE 1=1"
                params = []

                if admin_name:
                    query += " AND admin_name = ?"
                    params.append(admin_name)
                if action:
                    query += " AND action = ?"
                    params.append(action)
                if station:
                    query += " AND station = ?"
                    params.append(station)

                query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [
                    {
                        'id': r[0],
                        'admin_name': r[1],
                        'action': r[2],
                        'target': r[3],
                        'details': r[4],
                        'ip_address': r[5],
                        'timestamp': r[6],
                        'station': r[7] if len(r) > 7 else None
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"获取管理员日志失败: {e}")
            return []

    # ------------------------------------------------------------------
    # 基础 CRUD 操作
    # ------------------------------------------------------------------

    def add_user(
        self,
        user_id: str,
        feature_vector: np.ndarray,
        user_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        hand_type: Optional[str] = None,
        station: Optional[str] = None
    ) -> bool:
        """
        添加用户模板（支持同一用户多个模板）

        Args:
            user_id: 用户 ID
            feature_vector: 特征向量 (numpy 数组)
            user_name: 用户名称（可选）
            metadata: 额外的元数据（可选）
            hand_type: 手型 ('left' 或 'right')，可选
            station: 所属站点（可选）

        Returns:
            是否添加成功
        """
        try:
            # 将特征向量序列化为 JSON 字符串
            feature_json = json.dumps(feature_vector.tolist())
            feature_dim = len(feature_vector)

            # 序列化元数据（如果包含手型则合并）
            if metadata is None:
                metadata = {}
            metadata["hand_type"] = hand_type if hand_type else "unknown"
            metadata_json = json.dumps(metadata)

            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 使用 INSERT 而非 INSERT OR REPLACE，允许同一用户多个模板
                cursor.execute(f"""
                    INSERT INTO {self.table_name}
                    (user_id, user_name, feature_vector, feature_dim, metadata, hand_type, station, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (user_id, user_name, feature_json, feature_dim, metadata_json, hand_type, station))
                conn.commit()
            hand_info = f" ({hand_type})" if hand_type else ""
            station_info = f" @ {station}" if station else ""
            logger.info(f"添加用户模板: {user_id}{hand_info}{station_info}")
            return True
        except Exception as e:
            logger.error(f"添加用户失败: {e}")
            return False

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定用户信息

        Args:
            user_id: 用户 ID

        Returns:
            用户信息字典，不存在时返回 None
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT * FROM {self.table_name} WHERE user_id = ?
                """, (user_id,))
                row = cursor.fetchone()
                return self._row_to_dict(row) if row else None
        except Exception as e:
            logger.error(f"获取用户失败: {e}")
            return None

    def get_user_station(self, user_id: str) -> Optional[str]:
        """
        获取用户所属站点

        Args:
            user_id: 用户 ID

        Returns:
            站点名称，不存在时返回 None
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT station FROM {self.table_name} WHERE user_id = ? LIMIT 1", (user_id,))
                row = cursor.fetchone()
                return row["station"] if row else None
        except Exception as e:
            logger.error(f"获取用户站点失败: {e}")
            return None

    def get_all_users(self) -> List[Dict[str, Any]]:
        """
        获取所有用户信息

        Returns:
            用户信息列表
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT * FROM {self.table_name} ORDER BY created_at DESC
                """)
                rows = cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
        except Exception as e:
            logger.error(f"获取所有用户失败: {e}")
            return []

    def get_user_ids(self, station: Optional[str] = None) -> List[str]:
        """
        获取所有用户 ID

        Args:
            station: 可选，按站点过滤

        Returns:
            用户 ID 列表
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if station:
                    cursor.execute(f"SELECT DISTINCT user_id FROM {self.table_name} WHERE station = ?", (station,))
                else:
                    cursor.execute(f"SELECT DISTINCT user_id FROM {self.table_name}")
                rows = cursor.fetchall()
                return [row["user_id"] for row in rows]
        except Exception as e:
            logger.error(f"获取用户 ID 列表失败: {e}")
            return []

    def get_user_count(self, station: str = None) -> int:
        """
        获取唯一用户数量

        Args:
            station: 站点名称过滤（可选），None表示全部

        Returns:
            用户总数（去重后）
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if station:
                    cursor.execute(f"SELECT COUNT(DISTINCT user_id) as count FROM {self.table_name} WHERE station = ?", (station,))
                else:
                    cursor.execute(f"SELECT COUNT(DISTINCT user_id) as count FROM {self.table_name}")
                return cursor.fetchone()["count"]
        except Exception as e:
            logger.error(f"获取用户数量失败: {e}")
            return 0

    def get_total_templates(self, station: str = None) -> int:
        """
        获取总模板数量

        Args:
            station: 站点名称过滤（可选），None表示全部

        Returns:
            模板总数
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if station:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {self.table_name} WHERE station = ?", (station,))
                else:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {self.table_name}")
                return cursor.fetchone()["count"]
        except Exception as e:
            logger.error(f"获取模板数量失败: {e}")
            return 0

    def get_user_sample_count(self, user_id: str) -> int:
        """
        获取指定用户的样本数量

        Args:
            user_id: 用户ID

        Returns:
            样本数量
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT COUNT(*) as count FROM {self.table_name}
                    WHERE user_id = ?
                """, (user_id,))
                return cursor.fetchone()["count"]
        except Exception as e:
            logger.error(f"获取用户样本数量失败: {e}")
            return 0

    def get_user_templates(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取用户的所有模板

        Args:
            user_id: 用户ID

        Returns:
            模板列表
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT * FROM {self.table_name}
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                """, (user_id,))
                rows = cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
        except Exception as e:
            logger.error(f"获取用户模板失败: {e}")
            return []

    def add_template(self, template: Dict[str, Any], station: Optional[str] = None) -> bool:
        """
        添加单个模板（用于多模板注册）

        Args:
            template: 模板字典，包含 'user_id', 'feature', 'quality' 等
            station: 所属站点（可选）

        Returns:
            是否添加成功
        """
        try:
            user_id = template.get('user_id')
            feature = template.get('feature')
            quality = template.get('quality', 0.5)

            if not user_id or feature is None:
                logger.error("模板缺少必要字段")
                return False

            # 创建元数据
            metadata = {
                'quality': quality,
                'timestamp': template.get('timestamp'),
                'hand_type': template.get('hand_type', 'unknown')
            }

            return self.add_user(
                user_id=user_id,
                feature_vector=feature,
                metadata=metadata,
                station=station
            )
        except Exception as e:
            logger.error(f"添加模板失败: {e}")
            return False

    def update_user(
        self,
        user_id: str,
        user_name: Optional[str] = None,
        feature_vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        station: Optional[str] = None
    ) -> bool:
        """
        更新用户信息

        Args:
            user_id: 用户 ID
            user_name: 新的用户名称（可选）
            feature_vector: 新的特征向量（可选）
            metadata: 新的元数据（可选）
            station: 新的站点（可选）

        Returns:
            是否更新成功
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # 构建更新语句
                updates = []
                params = []

                if user_name is not None:
                    updates.append("user_name = ?")
                    params.append(user_name)

                if feature_vector is not None:
                    feature_json = json.dumps(feature_vector.tolist())
                    feature_dim = len(feature_vector)
                    updates.append("feature_vector = ?")
                    updates.append("feature_dim = ?")
                    params.extend([feature_json, feature_dim])

                if metadata is not None:
                    metadata_json = json.dumps(metadata)
                    updates.append("metadata = ?")
                    params.append(metadata_json)

                if station is not None:
                    updates.append("station = ?")
                    params.append(station)

                updates.append("updated_at = CURRENT_TIMESTAMP")
                params.append(user_id)

                if not updates:
                    return False

                sql = f"""
                    UPDATE {self.table_name}
                    SET {', '.join(updates)}
                    WHERE user_id = ?
                """
                cursor.execute(sql, params)
                conn.commit()

                if cursor.rowcount > 0:
                    logger.info(f"更新用户: {user_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"更新用户失败: {e}")
            return False

    def delete_user(self, user_id: str) -> bool:
        """
        删除用户

        Args:
            user_id: 用户 ID

        Returns:
            是否删除成功
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    DELETE FROM {self.table_name} WHERE user_id = ?
                """, (user_id,))
                conn.commit()
                if cursor.rowcount > 0:
                    logger.info(f"删除用户: {user_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"删除用户失败: {e}")
            return False

    def delete_all_users(self) -> int:
        """
        删除所有用户

        Returns:
            删除的用户数量
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"DELETE FROM {self.table_name}")
                conn.commit()
                count = cursor.rowcount
                logger.info(f"删除所有用户: {count} 个")
                return count
        except Exception as e:
            logger.error(f"删除所有用户失败: {e}")
            return 0

    # ------------------------------------------------------------------
    # 站点管理
    # ------------------------------------------------------------------

    def get_all_stations(self) -> List[Dict[str, Any]]:
        """获取所有站点"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name, color, created_at FROM stations ORDER BY id ASC")
                rows = cursor.fetchall()
                return [{'id': r[0], 'name': r[1], 'color': r[2], 'created_at': r[3]} for r in rows]
        except Exception as e:
            logger.error(f"获取站点失败: {e}")
            return []

    def add_station(self, name: str, color: str = 'slate') -> bool:
        """添加站点"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM stations WHERE name = ?", (name,))
                if cursor.fetchone():
                    return False
                cursor.execute("INSERT INTO stations (name, color) VALUES (?, ?)", (name, color))
                conn.commit()
                logger.info(f"添加站点: {name} ({color})")
                return True
        except Exception as e:
            logger.error(f"添加站点失败: {e}")
            return False

    def update_station(self, name: str, color: Optional[str] = None, new_name: Optional[str] = None) -> bool:
        """更新站点名称或颜色"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM stations WHERE name = ?", (name,))
                if not cursor.fetchone():
                    return False
                updates = []
                params = []
                if new_name:
                    updates.append("name = ?")
                    params.append(new_name)
                if color:
                    updates.append("color = ?")
                    params.append(color)
                if updates:
                    params.append(name)
                    cursor.execute(f"UPDATE stations SET {', '.join(updates)} WHERE name = ?", params)
                    conn.commit()
                # 如果改了站名，同步更新 users 表
                if new_name and new_name != name:
                    cursor.execute("UPDATE users SET station = ? WHERE station = ?", (new_name, name))
                    conn.commit()
                logger.info(f"更新站点: {name} -> {new_name or name} ({color or 'unchanged'})")
                return True
        except Exception as e:
            logger.error(f"更新站点失败: {e}")
            return False

    def delete_station(self, name: str) -> bool:
        """删除站点（将属于该站点的用户转移到默认站点）"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM stations WHERE name = ?", (name,))
                if not cursor.fetchone():
                    return False

                # 将该站点的用户转移到默认站点（总站）
                cursor.execute("UPDATE users SET station = '总站' WHERE station = ?", (name,))
                conn.commit()

                cursor.execute("DELETE FROM stations WHERE name = ?", (name,))
                conn.commit()
                logger.info(f"删除站点: {name}（用户已转移到总站）")
                return True
        except Exception as e:
            logger.error(f"删除站点失败: {e}")
            return False

    def get_station_color(self, name: str) -> str:
        """获取站点颜色"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT color FROM stations WHERE name = ?", (name,))
                row = cursor.fetchone()
                return row[0] if row else 'slate'
        except Exception as e:
            logger.error(f"获取站点颜色失败: {e}")
            return 'slate'

    # ------------------------------------------------------------------
    # 特征向量相关操作
    # ------------------------------------------------------------------

    def get_all_features(self) -> List[Tuple[str, np.ndarray]]:
        """
        获取所有用户的特征向量

        Returns:
            (user_id, feature_vector) 元组列表
        """
        users = self.get_all_users()
        return [(user["user_id"], user["feature_vector"]) for user in users]

    def user_exists(self, user_id: str) -> bool:
        """
        检查用户是否存在

        Args:
            user_id: 用户 ID

        Returns:
            用户是否存在
        """
        return self.get_user(user_id) is not None

    # ------------------------------------------------------------------
    # 模板兼容性方法（与 pickle 模式兼容）
    # ------------------------------------------------------------------

    def load_templates(self) -> List[Dict[str, Any]]:
        """
        加载所有模板（兼容原有 pickle 接口，支持手型、质量分数）

        Returns:
            模板列表，格式: [{'user_id': str, 'feature': np.ndarray, 'hand_type': str, 'quality': float}, ...]
        """
        users = self.get_all_users()
        templates = []
        for user in users:
            metadata = user.get("metadata") or {}
            # 确保metadata是字典
            if not isinstance(metadata, dict):
                metadata = {}
            templates.append({
                "user_id": user["user_id"],
                "feature": user["feature_vector"],
                "hand_type": metadata.get("hand_type", "unknown"),
                "quality": metadata.get("quality", 0.8),  # 默认质量分数
                "timestamp": metadata.get("timestamp"),
            })
        return templates

    def save_templates(self, templates: List[Dict[str, Any]]) -> int:
        """
        保存模板列表（兼容原有 pickle 接口）

        Args:
            templates: 模板列表

        Returns:
            保存的模板数量
        """
        count = 0
        for template in templates:
            if self.add_user(
                user_id=template["user_id"],
                feature_vector=template["feature"]
            ):
                count += 1
        return count

    def import_from_pickle(self, pickle_path: str) -> int:
        """
        从 pickle 文件导入数据

        Args:
            pickle_path: pickle 文件路径

        Returns:
            导入的用户数量
        """
        try:
            import pickle
            with open(pickle_path, "rb") as f:
                templates = pickle.load(f)

            count = 0
            for template in templates:
                if self.add_user(
                    user_id=template["user_id"],
                    feature_vector=template["feature"]
                ):
                    count += 1
            logger.info(f"从 pickle 文件导入 {count} 个用户")
            return count
        except Exception as e:
            logger.error(f"从 pickle 文件导入失败: {e}")
            return 0

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """将数据库行转换为字典"""
        if not row:
            return {}

        # 解析元数据
        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except:
                pass

        # 手型优先从 metadata 获取，其次从 hand_type 列获取
        hand_type = metadata.get("hand_type", "unknown")

        # 尝试直接从 hand_type 列获取
        try:
            if row["hand_type"] is not None and row["hand_type"] != "":
                hand_type = row["hand_type"]
        except (KeyError, IndexError):
            pass

        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "user_name": row["user_name"],
            "feature_vector": np.array(json.loads(row["feature_vector"]), dtype=np.float32),
            "feature_dim": row["feature_dim"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": metadata if metadata else None,
            "hand_type": hand_type,
            "station": row["station"],
        }

    def get_statistics(self, station: str = None) -> Dict[str, Any]:
        """
        获取数据库统计信息

        Args:
            station: 站点名称过滤（可选），None表示全部

        Returns:
            统计信息字典
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if station:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {self.table_name} WHERE station = ?", (station,))
                    user_count = cursor.fetchone()["count"]
                    cursor.execute(f"""
                        SELECT MIN(created_at) as first_user,
                               MAX(created_at) as last_user
                        FROM {self.table_name}
                        WHERE station = ?
                    """, (station,))
                else:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {self.table_name}")
                    user_count = cursor.fetchone()["count"]
                    cursor.execute(f"""
                        SELECT MIN(created_at) as first_user,
                               MAX(created_at) as last_user
                        FROM {self.table_name}
                    """)
                row = cursor.fetchone()

                return {
                    "user_count": user_count,
                    "first_user": row["first_user"],
                    "last_user": row["last_user"],
                    "db_path": str(Path(self.db_path).absolute()),
                }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

    def backup(self, backup_path: Optional[str] = None) -> str:
        """
        备份数据库

        Args:
            backup_path: 备份文件路径（可选）

        Returns:
            备份文件路径
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"backups/palm_templates_{timestamp}.db"

        backup_file = Path(backup_path)
        backup_file.parent.mkdir(parents=True, exist_ok=True)

        import shutil
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"数据库已备份到: {backup_path}")
        return backup_path

    # ------------------------------------------------------------------
    # 识别统计方法
    # ------------------------------------------------------------------

    def log_recognition(
        self,
        success: bool,
        user_id: Optional[str] = None,
        confidence: Optional[float] = None,
        response_time_ms: Optional[float] = None,
        station: Optional[str] = None
    ) -> bool:
        """
        记录识别事件

        Args:
            success: 是否识别成功
            user_id: 识别出的用户ID（可选）
            confidence: 置信度（可选）
            response_time_ms: 响应时间毫秒（可选）
            station: 站点名称（可选）

        Returns:
            是否记录成功
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO recognition_logs
                    (success, user_id, confidence, response_time, created_at, station)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (1 if success else 0, user_id, confidence, response_time_ms, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), station))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"记录识别日志失败: {e}")
            return False

    def get_recognition_stats(self, days: int = 0, station: str = None) -> Dict[str, Any]:
        """
        获取识别统计

        Args:
            days: 统计最近几天的数据，0表示全部数据
            station: 站点名称过滤（可选），None表示全部

        Returns:
            统计信息字典
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # 构建查询条件
                where_clause = ""
                params = []
                if station:
                    where_clause = "WHERE rl.station = ?"
                    params.append(station)

                # 如果 days>0，添加日期条件
                if days > 0:
                    start_date = (datetime.now() - timedelta(days=days-1)).replace(hour=0, minute=0, second=0, microsecond=0).strftime('%Y-%m-%d 00:00:00')
                    if where_clause:
                        where_clause += " AND rl.created_at >= ?"
                    else:
                        where_clause = "WHERE rl.created_at >= ?"
                    params.append(start_date)

                # 如果 days=0，查询全部数据；否则查询指定天数
                if days == 0:
                    cursor.execute(f"""
                        SELECT
                            COUNT(*) as total,
                            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                            SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as fail_count,
                            AVG(response_time) as avg_response_time
                        FROM recognition_logs rl
                        {where_clause}
                    """, params)
                else:
                    cursor.execute(f"""
                        SELECT
                            COUNT(*) as total,
                            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                            SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as fail_count,
                            AVG(response_time) as avg_response_time
                        FROM recognition_logs rl
                        {where_clause}
                    """, params)
                row = cursor.fetchone()

                return {
                    "total": row["total"] or 0,
                    "success": row["success_count"] or 0,
                    "fail": row["fail_count"] or 0,
                    "success_rate": round((row["success_count"] or 0) / (row["total"] or 1) * 100, 1),
                    "avg_response_time": round(row["avg_response_time"] or 0, 1)
                }
        except Exception as e:
            import traceback
            logger.error(f"获取识别统计失败: {e}")
            logger.error(traceback.format_exc())
            return {"total": 0, "success": 0, "fail": 0, "success_rate": 0, "avg_response_time": 0}

    def get_quality_distribution(self, station: str = None) -> Dict[str, int]:
        """
        获取样本质量分布

        Args:
            station: 站点名称过滤（可选），None表示全部

        Returns:
            {'excellent': int, 'good': int, 'average': int, 'poor': int}
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if station:
                    cursor.execute(f"SELECT metadata FROM {self.table_name} WHERE station = ?", (station,))
                else:
                    cursor.execute(f"SELECT metadata FROM {self.table_name}")
                rows = cursor.fetchall()

                excellent = 0  # >= 0.8
                good = 0       # 0.6 - 0.8
                average = 0    # 0.4 - 0.6
                poor = 0       # < 0.4

                for row in rows:
                    if row["metadata"]:
                        try:
                            meta = json.loads(row["metadata"])
                            quality = meta.get("quality", 0.8)
                            if quality >= 0.8:
                                excellent += 1
                            elif quality >= 0.6:
                                good += 1
                            elif quality >= 0.4:
                                average += 1
                            else:
                                poor += 1
                        except:
                            pass

                return {
                    "excellent": excellent,
                    "good": good,
                    "average": average,
                    "poor": poor
                }
        except Exception as e:
            logger.error(f"获取质量分布失败: {e}")
            return {"excellent": 0, "good": 0, "average": 0, "poor": 0}

    def get_time_distribution(self) -> List[int]:
        """
        获取识别时段分布

        Returns:
            [0-4时, 4-8时, 8-12时, 12-16时, 16-20时, 20-24时] 各时段的识别次数
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # 检查识别日志表是否存在
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='recognition_logs'
                """)
                if not cursor.fetchone():
                    return [0, 0, 0, 0, 0, 0]

                # 获取所有识别记录，按时段分组（不再限制为今天）
                cursor.execute("""
                    SELECT
                        CAST(strftime('%H', created_at) AS INTEGER) as hour,
                        COUNT(*) as count
                    FROM recognition_logs
                    GROUP BY hour
                """)
                rows = cursor.fetchall()

                # 初始化6个时段
                time_slots = [0, 0, 0, 0, 0, 0]  # 0-4, 4-8, 8-12, 12-16, 16-20, 20-24

                for row in rows:
                    hour = row["hour"]
                    count = row["count"]
                    if 0 <= hour < 4:
                        time_slots[0] += count
                    elif 4 <= hour < 8:
                        time_slots[1] += count
                    elif 8 <= hour < 12:
                        time_slots[2] += count
                    elif 12 <= hour < 16:
                        time_slots[3] += count
                    elif 16 <= hour < 20:
                        time_slots[4] += count
                    else:  # 20-24
                        time_slots[5] += count

                return time_slots
        except Exception as e:
            logger.error(f"获取时段分布失败: {e}")
            return [0, 0, 0, 0, 0, 0]

    def get_confidence_distribution(self) -> Dict[str, int]:
        """
        获取置信度分数分布

        Returns:
            {'0.0-0.5': int, '0.5-0.7': int, '0.7-0.85': int, '0.85-0.90': int, '0.90-0.95': int, '0.95-1.0': int}
            各区间的识别次数
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # 检查识别日志表是否存在
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='recognition_logs'
                """)
                if not cursor.fetchone():
                    return {
                        "0.0-0.5": 0,
                        "0.5-0.7": 0,
                        "0.7-0.85": 0,
                        "0.85-0.90": 0,
                        "0.90-0.95": 0,
                        "0.95-1.0": 0
                    }

                # 获取所有识别记录的置信度分数（不再限制为今天）
                cursor.execute("""
                    SELECT confidence
                    FROM recognition_logs
                    WHERE confidence IS NOT NULL
                """)
                rows = cursor.fetchall()

                # 初始化各区间计数
                distribution = {
                    "0.0-0.5": 0,   # 极低（可疑）
                    "0.5-0.7": 0,    # 低
                    "0.7-0.85": 0,   # 中等
                    "0.85-0.90": 0,  # 边界
                    "0.90-0.95": 0,  # 高置信度
                    "0.95-1.0": 0    # 极高
                }

                for row in rows:
                    conf = row["confidence"]
                    if conf is None:
                        continue
                    if conf < 0.5:
                        distribution["0.0-0.5"] += 1
                    elif conf < 0.7:
                        distribution["0.5-0.7"] += 1
                    elif conf < 0.85:
                        distribution["0.7-0.85"] += 1
                    elif conf < 0.90:
                        distribution["0.85-0.90"] += 1
                    elif conf < 0.95:
                        distribution["0.90-0.95"] += 1
                    else:
                        distribution["0.95-1.0"] += 1

                return distribution
        except Exception as e:
            logger.error(f"获取置信度分布失败: {e}")
            return {
                "0.0-0.5": 0,
                "0.5-0.7": 0,
                "0.7-0.85": 0,
                "0.85-0.90": 0,
                "0.90-0.95": 0,
                "0.95-1.0": 0
            }

    def get_recognition_trend(self, range_type: str = '7d', data_type: str = 'count', station: str = None) -> Dict[str, Any]:
        """
        获取识别趋势数据

        Args:
            range_type: 'today', '7d', '30d'
            data_type: 'count' (识别次数) 或 'success' (成功率%)
            station: 站点名称过滤（可选），None表示全部

        Returns:
            趋势数据字典，包含 labels 和 values
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                labels = []
                values = []

                now = datetime.now()

                # 构建station过滤子查询
                station_subquery = ""
                if station:
                    station_subquery = "AND rl.station = ?"

                if range_type == 'today':
                    # 按小时分组，然后聚合到时段
                    today_str = now.strftime('%Y-%m-%d')

                    # 获取每小时的详细数据
                    cursor.execute(f"""
                        SELECT strftime('%H', rl.created_at) as hour,
                               COUNT(*) as total,
                               SUM(CASE WHEN rl.success = 1 THEN 1 ELSE 0 END) as success_count
                        FROM recognition_logs rl
                        WHERE date(rl.created_at) = ? {station_subquery}
                        GROUP BY hour
                        ORDER BY hour
                    """, ([today_str] if not station else [today_str, station]))
                    rows = cursor.fetchall()

                    # 构建小时数据字典
                    hour_totals = {}
                    hour_successes = {}
                    for r in rows:
                        hour = int(r["hour"])
                        hour_totals[hour] = r["total"] or 0
                        hour_successes[hour] = r["success_count"] or 0

                    # 定义时段（每3小时一个时段，从0点开始）
                    current_hour = now.hour
                    time_slots = []
                    for h in range(0, 24, 3):
                        if h <= current_hour:
                            time_slots.append(h)

                    # 聚合每个时段的数据
                    for slot_start in time_slots:
                        slot_end = min(slot_start + 3, 24)
                        labels.append(f"{slot_start:02d}:00")

                        # 计算时段内的总和
                        slot_total = sum(hour_totals.get(h, 0) for h in range(slot_start, slot_end))
                        slot_success = sum(hour_successes.get(h, 0) for h in range(slot_start, slot_end))

                        if data_type == 'success':
                            # 成功率模式
                            rate = round(slot_success / slot_total * 100) if slot_total > 0 else 0
                            values.append(rate)
                        else:
                            # 识别次数模式
                            values.append(slot_total)

                elif range_type == '7d':
                    # 按天分组
                    start_date = (now - timedelta(days=6)).strftime('%Y-%m-%d')
                    if data_type == 'success':
                        cursor.execute(f"""
                            SELECT date(rl.created_at) as day,
                                   COUNT(*) as total,
                                   SUM(CASE WHEN rl.success = 1 THEN 1 ELSE 0 END) as success_count
                            FROM recognition_logs rl
                            WHERE rl.created_at >= ? {station_subquery}
                            GROUP BY day
                            ORDER BY day
                        """, ([start_date] if not station else [start_date, station]))
                        rows = cursor.fetchall()
                        day_data = {}
                        for r in rows:
                            total = r["total"] or 0
                            success = r["success_count"] or 0
                            rate = round(success / total * 100) if total > 0 else 0
                            day_data[r["day"]] = rate
                    else:
                        cursor.execute(f"""
                            SELECT date(rl.created_at) as day,
                                   COUNT(*) as count
                            FROM recognition_logs rl
                            WHERE rl.created_at >= ? {station_subquery}
                            GROUP BY day
                            ORDER BY day
                        """, ([start_date] if not station else [start_date, station]))
                        rows = cursor.fetchall()
                        day_data = {r["day"]: r["count"] for r in rows}

                    # 填充缺失的日期
                    for i in range(6, -1, -1):
                        d = now - timedelta(days=i)
                        day_str = d.strftime('%Y-%m-%d')
                        labels.append(d.strftime('%m/%d'))
                        values.append(day_data.get(day_str, 0))

                elif range_type == '30d':
                    # 按5天分组
                    start_date = (now - timedelta(days=29)).strftime('%Y-%m-%d')
                    if data_type == 'success':
                        cursor.execute(f"""
                            SELECT date(rl.created_at) as day,
                                   COUNT(*) as total,
                                   SUM(CASE WHEN rl.success = 1 THEN 1 ELSE 0 END) as success_count
                            FROM recognition_logs rl
                            WHERE rl.created_at >= ? {station_subquery}
                            GROUP BY day
                            ORDER BY day
                        """, ([start_date] if not station else [start_date, station]))
                    else:
                        cursor.execute(f"""
                            SELECT date(rl.created_at) as day,
                                   COUNT(*) as count
                            FROM recognition_logs rl
                            WHERE rl.created_at >= ? {station_subquery}
                            GROUP BY day
                            ORDER BY day
                        """, ([start_date] if not station else [start_date, station]))
                    rows = cursor.fetchall()

                    if data_type == 'success':
                        day_data = {}
                        for r in rows:
                            total = r["total"] or 0
                            success = r["success_count"] or 0
                            rate = round(success / total * 100) if total > 0 else 0
                            day_data[r["day"]] = rate
                    else:
                        day_data = {r["day"]: r["count"] for r in rows}

                    for i in range(25, -1, -5):
                        d = now - timedelta(days=i)
                        labels.append(d.strftime('%m/%d'))
                        # 汇总5天的数据
                        if data_type == 'success':
                            # 成功率：5天内有数据的天数对应的平均成功率（近似）
                            days_with_data = [day_data.get((now - timedelta(days=i-j)).strftime('%Y-%m-%d'), None) for j in range(5)]
                            valid_rates = [r for r in days_with_data if r is not None]
                            rate = round(sum(valid_rates) / len(valid_rates)) if valid_rates else 0
                            values.append(rate)
                        else:
                            total = 0
                            for j in range(5):
                                check_date = (now - timedelta(days=i-j)).strftime('%Y-%m-%d')
                                total += day_data.get(check_date, 0)
                            values.append(total)

                return {
                    "labels": labels,
                    "values": values
                }
        except Exception as e:
            logger.error(f"获取识别趋势失败: {e}")
            return {"labels": [], "values": []}


# 创建默认数据库实例
_default_db = None


def get_database(db_path: Optional[str] = None) -> PalmDatabase:
    """
    获取数据库实例（单例模式）

    Args:
        db_path: 数据库路径（可选）

    Returns:
        数据库实例
    """
    global _default_db
    # 只在第一次创建时使用传入的路径，之后忽略传入的路径参数
    if _default_db is None:
        config = DatabaseConfig(db_path=db_path) if db_path else DatabaseConfig()
        _default_db = PalmDatabase(config)
    return _default_db


if __name__ == "__main__":
    # 测试代码
    print("=" * 50)
    print("PalmDatabase 测试")
    print("=" * 50)

    db = PalmDatabase()

    # 测试 1：添加用户
    print("\n测试 1：添加用户")
    feature = np.random.rand(128)
    db.add_user("user_001", feature, user_name="测试用户1")
    db.add_user("user_002", feature * 0.9, user_name="测试用户2")

    # 测试 2：获取用户
    print("\n测试 2：获取用户")
    user = db.get_user("user_001")
    print(f"用户信息: {user}")

    # 测试 3：列出所有用户
    print("\n测试 3：列出所有用户")
    users = db.get_all_users()
    for u in users:
        print(f"  {u['user_id']}: {u['user_name']}")

    # 测试 4：获取统计信息
    print("\n测试 4：统计信息")
    stats = db.get_statistics()
    print(f"统计: {stats}")

    # 测试 5：删除用户
    print("\n测试 5：删除用户")
    db.delete_user("user_001")
    print(f"剩余用户数: {db.get_user_count()}")

    # 测试 6：清空所有
    print("\n测试 6：清空所有用户")
    db.delete_all_users()
    print(f"用户数: {db.get_user_count()}")
