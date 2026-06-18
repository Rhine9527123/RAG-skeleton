"""
多轮对话会话存储 - SQLite 后端
=============================

提供会话管理 + 上下文窗口 + 消息持久化。

使用方式：
    from session_store import SessionStore

    store = SessionStore()       # 自动读取 config 中的窗口大小
    sid = store.create_session()
    store.add_message(sid, "user", "你好")
    history = store.get_history(sid)

SQLite 表结构：
    sessions  — 会话元信息（id, title, 时间戳, 消息数）
    messages  — 消息记录（role, content, sources, turn_index）
"""

import sqlite3
import json
import time
import os
import uuid
import threading
from typing import Optional

# 线程本地连接（每个线程独立 connection，避免多线程冲突）
_local = threading.local()


class SessionStore:
    """多轮对话会话存储

    属性：
        max_turns: 上下文窗口大小（保留最近 N 个用户轮次）
        db_path: SQLite 数据库路径
    """

    def __init__(
        self,
        db_path: str = None,
        max_turns: int = 10,
        auto_init: bool = True,
    ):
        self.db_path = db_path or os.environ.get(
            "SESSION_DB_PATH",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions.db"),
        )
        self.max_turns = max_turns
        if auto_init:
            self._init_db()

    # ── 数据库连接（线程安全） ──

    def _conn(self):
        """获取当前线程的数据库连接"""
        if not hasattr(_local, "conn") or _local.conn is None:
            _local.conn = sqlite3.connect(self.db_path)
            _local.conn.row_factory = sqlite3.Row
            _local.conn.execute("PRAGMA journal_mode=WAL")  # 并发友好
        return _local.conn

    def _init_db(self):
        """初始化数据库表结构"""
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                message_count INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT DEFAULT '[]',
                timestamp REAL NOT NULL,
                turn_index INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_msgs_session
                ON messages(session_id, turn_index);
        """)
        conn.commit()

    # ── 会话 CRUD ──

    def create_session(self, title: str = "") -> str:
        """创建新会话，返回 session_id"""
        session_id = uuid.uuid4().hex[:8]
        now = time.time()
        conn = self._conn()
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
        conn.commit()
        return session_id

    def list_sessions(self, limit: int = 20) -> list:
        """列出最近会话（按更新时间倒序）"""
        conn = self._conn()
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at, message_count "
            "FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> Optional[dict]:
        """获取单个会话信息"""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def delete_session(self, session_id: str) -> bool:
        """删除会话及其所有消息"""
        conn = self._conn()
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        return True

    def update_title(self, session_id: str, title: str):
        """更新会话标题"""
        conn = self._conn()
        conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, time.time(), session_id),
        )
        conn.commit()

    # ── 消息管理 ──

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list = None,
    ) -> int:
        """添加一条消息，返回 turn_index"""
        now = time.time()
        conn = self._conn()

        # 获取下一个 turn_index（每次递增）
        row = conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_idx "
            "FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        turn_index = row["next_idx"]

        sources_json = json.dumps(sources or [], ensure_ascii=False)
        conn.execute(
            "INSERT INTO messages (session_id, role, content, sources, timestamp, turn_index) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, sources_json, now, turn_index),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ?, message_count = message_count + 1 "
            "WHERE id = ?",
            (now, session_id),
        )
        conn.commit()

        # 自动裁剪超出窗口的历史消息
        self._trim_history(session_id)

        return turn_index

    def get_history(
        self,
        session_id: str,
        max_turns: int = None,
    ) -> list[dict]:
        """获取最近 N 轮对话历史（按时间正序）"""
        max_turns = max_turns or self.max_turns
        conn = self._conn()
        # 取最近 max_turns*2 条（每轮用户+助手=2条）
        rows = conn.execute(
            "SELECT role, content, sources, turn_index, timestamp "
            "FROM messages WHERE session_id = ? "
            "ORDER BY turn_index DESC LIMIT ?",
            (session_id, max_turns * 2),
        ).fetchall()
        rows.reverse()
        result = []
        for r in rows:
            item = dict(r)
            item["sources"] = json.loads(item["sources"])
            result.append(item)
        return result

    def get_all_history(self, session_id: str) -> list[dict]:
        """获取全部历史（不裁剪，用于导出）"""
        conn = self._conn()
        rows = conn.execute(
            "SELECT role, content, sources, turn_index, timestamp "
            "FROM messages WHERE session_id = ? "
            "ORDER BY turn_index ASC",
            (session_id,),
        ).fetchall()
        result = []
        for r in rows:
            item = dict(r)
            item["sources"] = json.loads(item["sources"])
            result.append(item)
        return result

    def _trim_history(self, session_id: str):
        """裁剪超出上下文窗口的历史消息"""
        conn = self._conn()
        # 找到第 max_turns 个用户消息的 turn_index（从最新往前数）
        rows = conn.execute(
            "SELECT turn_index FROM messages "
            "WHERE session_id = ? AND role = 'user' "
            "ORDER BY turn_index DESC LIMIT 1 OFFSET ?",
            (session_id, self.max_turns - 1),
        ).fetchall()
        if rows:
            threshold = rows[0]["turn_index"]
            conn.execute(
                "DELETE FROM messages WHERE session_id = ? AND turn_index < ?",
                (session_id, threshold),
            )
            conn.commit()

    # ── 构建 LLM 对话上下文 ──

    def build_context(
        self,
        session_id: str,
        system_prompt: str,
        current_question: str,
        retrieved_context: str,
        max_turns: int = None,
    ) -> list[dict]:
        """构建 LLM messages 列表（System + 历史 + 当前）

        返回格式适配 llama_index / OpenAI 的 ChatMessage 格式：
            [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        """
        messages = [{"role": "system", "content": system_prompt}]

        # 插入对话历史
        history = self.get_history(session_id, max_turns)
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # 插入当前问题（含检索上下文）
        current_content = (
            f"参考资料：\n{retrieved_context}\n\n"
            f"用户问题：{current_question}\n\n"
            f"请回答："
        )
        messages.append({"role": "user", "content": current_content})

        return messages
