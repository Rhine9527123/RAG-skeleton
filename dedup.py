"""
文档去重模块 — URL + 内容哈希双重去重
======================================

设计思路：
  去重分两层 ——
  1. URL 去重：同一 URL 只入库一次（防止爬虫重复抓取）
  2. 内容去重：相同文本内容只保留一份（防止同一文档从不同来源重复入库）

存储：SQLite（dedup.db），零额外依赖，适合单机部署

表结构：
  dedup_records:
    id           INTEGER PRIMARY KEY
    url_hash     TEXT UNIQUE      — URL 的 SHA256（NULL 表示纯文件上传，无 URL）
    content_hash TEXT NOT NULL    — 文本内容的 SHA256
    filename     TEXT             — 源文件名
    source_type  TEXT             — 'url' | 'file'
    file_size    INTEGER          — 文件大小（字节）
    added_at     REAL             — Unix 时间戳

API：
  DedupManager(db_path)      — 初始化，自动建表
  .check(content, url, fn)   — 返回 (is_dup, reason, existing_record)
  .add(content, url, fn, sz) — 写入去重记录
  .remove_by_filename(fn)    — 按文件名移除（删除文件时同步）
  .stats()                   — 返回统计信息
  .rebuild_from_data_dir()   — 从 data/ 目录重建去重表（修复不一致）
  .list_all()                — 列出所有记录
"""

import hashlib
import sqlite3
import os
import time
from typing import Optional, Tuple, Dict, List


def _hash(text: str) -> str:
    """SHA256 哈希（统一入口）"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DedupManager:
    """文档去重管理器"""

    def __init__(self, db_path: str = "dedup.db"):
        self.db_path = db_path
        self._init_db()

    # ── 内部方法 ──────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # 更好的并发
        conn.execute("PRAGMA busy_timeout=5000")  # 5秒超时
        return conn

    def _init_db(self):
        """建表（幂等）"""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dedup_records (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    url_hash     TEXT UNIQUE,
                    content_hash TEXT NOT NULL,
                    filename     TEXT,
                    source_type  TEXT NOT NULL DEFAULT 'file',
                    file_size    INTEGER DEFAULT 0,
                    added_at     REAL NOT NULL
                )
            """)
            # content_hash 也建索引（查重时高效）
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_hash
                ON dedup_records(content_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_filename
                ON dedup_records(filename)
            """)

    # ── 公共 API ──────────────────────────────────────────

    def check(
        self,
        content: str,
        url: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        检查文档是否重复。

        参数：
          content  — 提取后的文本内容（已去噪）
          url      — 文档来源 URL（None 表示纯文件上传）
          filename — 文件名（用于人性化提示）

        返回：
          (is_duplicate, reason, existing_record)
          - is_duplicate: True 表示重复，应拒绝入库
          - reason: 人类可读的去重原因
          - existing_record: 已存在的记录（dict），不重复时为 None
        """
        content_hash = _hash(content)

        with self._connect() as conn:
            # 第一层：URL 去重
            if url:
                url_hash = _hash(url)
                row = conn.execute(
                    "SELECT * FROM dedup_records WHERE url_hash = ?",
                    (url_hash,),
                ).fetchone()
                if row:
                    return (
                        True,
                        f"URL 重复：{url} 已入库（文件: {row['filename']}, 时间: {_ts_str(row['added_at'])}）",
                        dict(row),
                    )

            # 第二层：内容去重
            row = conn.execute(
                "SELECT * FROM dedup_records WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            if row:
                source_info = f"文件: {row['filename']}"
                if row["url_hash"]:
                    source_info += "（来自 URL）"
                return (
                    True,
                    f"内容重复：与已入库文档「{row['filename']}」内容完全相同"
                    f"（入库时间: {_ts_str(row['added_at'])}）",
                    dict(row),
                )

        return False, "", None

    def add(
        self,
        content: str,
        url: Optional[str] = None,
        filename: Optional[str] = None,
        file_size: int = 0,
    ):
        """
        写入去重记录（调用前应先 check）。

        参数：
          content   — 文本内容
          url       — 来源 URL
          filename  — 文件名
          file_size — 文件大小（字节）
        """
        content_hash = _hash(content)
        url_hash = _hash(url) if url else None

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO dedup_records
                   (url_hash, content_hash, filename, source_type, file_size, added_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    url_hash,
                    content_hash,
                    filename,
                    "url" if url else "file",
                    file_size,
                    time.time(),
                ),
            )

    def remove_by_filename(self, filename: str) -> int:
        """按文件名移除去重记录，返回删除行数"""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM dedup_records WHERE filename = ?",
                (filename,),
            )
            return cur.rowcount

    def stats(self) -> Dict:
        """返回去重统计信息"""
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as n FROM dedup_records"
            ).fetchone()["n"]

            by_source = {}
            rows = conn.execute(
                "SELECT source_type, COUNT(*) as n FROM dedup_records GROUP BY source_type"
            ).fetchall()
            for r in rows:
                by_source[r["source_type"]] = r["n"]

            # 总文件大小
            total_size = conn.execute(
                "SELECT COALESCE(SUM(file_size), 0) as s FROM dedup_records"
            ).fetchone()["s"]

        return {
            "total_records": total,
            "by_source": by_source,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

    def list_all(self) -> List[Dict]:
        """列出所有去重记录"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM dedup_records ORDER BY added_at DESC, id DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def rebuild_from_data_dir(self, data_dir: str = "data") -> int:
        """
        从 data/ 目录重建去重表 —— 用于修复不一致（如手动删文件后去重表有残留）。

        返回重建后的记录数。
        """
        # 清空重建
        with self._connect() as conn:
            conn.execute("DELETE FROM dedup_records")

        count = 0
        if not os.path.isdir(data_dir):
            return 0

        for filename in sorted(os.listdir(data_dir)):
            filepath = os.path.join(data_dir, filename)
            if not os.path.isfile(filepath):
                continue

            ext = filename.lower()
            if not (ext.endswith(".txt") or ext.endswith(".pdf") or ext.endswith(".xlsx") or ext.endswith(".xls")):
                continue

            file_size = os.path.getsize(filepath)

            # 读取文本内容（简化版，不重复 server.py 的 extract 逻辑）
            try:
                if ext.endswith(".txt"):
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                elif ext.endswith(".pdf"):
                    # PDF 用简化的二进制哈希（重建表不重新解析）
                    with open(filepath, "rb") as f:
                        content = "pdf:" + hashlib.sha256(f.read()).hexdigest()
                elif ext.endswith((".xlsx", ".xls")):
                    with open(filepath, "rb") as f:
                        content = "xlsx:" + hashlib.sha256(f.read()).hexdigest()
                else:
                    continue
            except Exception:
                continue

            if not content:
                continue

            content_hash = _hash(content)
            with self._connect() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO dedup_records
                       (url_hash, content_hash, filename, source_type, file_size, added_at)
                       VALUES (NULL, ?, ?, 'file', ?, ?)""",
                    (content_hash, filename, file_size, time.time()),
                )
            count += 1

        return count


# ── 辅助函数 ──────────────────────────────────────────────

def _ts_str(ts: float) -> str:
    """时间戳 → 可读字符串"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


# ── 全局单例（server.py 用）───────────────────────────────
dedup: Optional[DedupManager] = None


def get_dedup(db_path: str = "dedup.db") -> DedupManager:
    """获取全局去重管理器（懒加载）"""
    global dedup
    if dedup is None:
        dedup = DedupManager(db_path)
    return dedup
