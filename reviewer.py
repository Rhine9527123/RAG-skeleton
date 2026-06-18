"""
reviewer.py — 人工审核分级系统
=================================

低温初筛 + 分级审核流水线：

  1. 低温 LLM 评分（temperature=0.0，保守/确定性）
  2. 按分数分三级：
     - auto_approved (≥7): AI 高置信度 → 直接入库
     - needs_review  (4-6): AI 不确定    → 进入审核队列，等用户人工审核
     - auto_rejected (≤3): AI 高置信度无关 → 丢弃

设计理念：
  - 低温推理不做发散猜测，评分更可靠
  - 用户只需关注「低分文档」，高分自动通过
  - 审核队列持久化（JSON），重启不丢

依赖：pip install requests（LLM 调用复用 cleaner.py 的 LLMClient）

用法：
  from reviewer import DocumentReviewer

  reviewer = DocumentReviewer()
  result = reviewer.submit("这是一篇关于降准的文章...", "央行降准", source="财联社")
  print(result)  # {"review_id": "...", "tier": "needs_review", "score": 5, ...}

  # 获取审核队列
  queue = reviewer.get_queue()

  # 审批
  reviewer.approve(review_id)

  # 拒绝
  reviewer.reject(review_id)
"""

import hashlib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

# 复用 cleaner.py 的 LLMClient
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cleaner import LLMClient, LLMConfig
from config import get_config

logger = logging.getLogger("reviewer")


# ============================================================
# 审核等级
# ============================================================

class ReviewTier(str, Enum):
    """审核等级"""
    auto_approved = "auto_approved"  # AI 高置信度相关 → 自动入库
    needs_review = "needs_review"    # AI 不确定 → 需人工审核
    auto_rejected = "auto_rejected"  # AI 高置信度无关 → 丢弃


# 分数阈值
AUTO_APPROVE_THRESHOLD = 7   # ≥7 分自动通过
NEEDS_REVIEW_MIN = 4         # 4-6 分待审核
# ≤3 分自动拒绝


# ============================================================
# 审核记录
# ============================================================

@dataclass
class ReviewItem:
    """一条审核记录"""
    review_id: str
    title: str
    content: str                    # 全文（用于人工审核时查看）
    content_preview: str            # 截断预览（200字）
    source: str = ""                # 来源（如 "财联社"、"用户上传"）
    filename: str = ""              # 原始文件名（如有）
    score: int = 0                  # AI 相关性评分 0-10
    tier: ReviewTier = ReviewTier.needs_review
    ai_reasoning: str = ""          # AI 评分理由
    status: str = "pending"         # pending / approved / rejected
    created_at: str = ""
    reviewed_at: str = ""
    reviewed_by: str = ""           # "auto" 或 "human"

    def to_dict(self) -> dict:
        return {
            "review_id": self.review_id,
            "title": self.title,
            "content_preview": self.content_preview,
            "source": self.source,
            "filename": self.filename,
            "score": self.score,
            "tier": self.tier.value,
            "ai_reasoning": self.ai_reasoning,
            "status": self.status,
            "created_at": self.created_at,
            "reviewed_at": self.reviewed_at,
            "reviewed_by": self.reviewed_by,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewItem":
        tier_raw = d.get("tier", "needs_review")
        return cls(
            review_id=d["review_id"],
            title=d.get("title", ""),
            content=d.get("content", ""),
            content_preview=d.get("content_preview", ""),
            source=d.get("source", ""),
            filename=d.get("filename", ""),
            score=d.get("score", 0),
            tier=ReviewTier(tier_raw) if tier_raw in [t.value for t in ReviewTier] else ReviewTier.needs_review,
            ai_reasoning=d.get("ai_reasoning", ""),
            status=d.get("status", "pending"),
            created_at=d.get("created_at", ""),
            reviewed_at=d.get("reviewed_at", ""),
            reviewed_by=d.get("reviewed_by", ""),
        )


# ============================================================
# 评分提示词（低温 T=0.0）
# ============================================================

SCORE_PROMPT = """你是一个内容审核员。请用低温、保守的方式评估以下文档。

评估维度：
1. **相关性**：文档与知识库主题的关联程度
2. **信息质量**：内容是否具体、有实质信息（而非空洞口号）
3. **可信度风险**：是否可能包含过时、错误或有误导性的信息

评分标准（0-10）：
- 0-2: 完全无关或低质量（娱乐八卦、广告、无意义内容）
- 3-4: 弱相关且信息量低
- 5-6: 有一定相关性但不够深入，或信息可信度存疑
- 7-8: 高度相关内容，信息有一定参考价值
- 9-10: 专业分析/政策解读/真实数据，高度可信

请回复 JSON 格式（不要任何其他文字）：
{"score": <0-10整数>, "reasoning": "<一句话说明为什么给这个分数>"}

文档标题: {title}
文档内容: {content}
"""


# ============================================================
# 审核管理器
# ============================================================

class DocumentReviewer:
    """
    文档审核分级器

    核心方法：
      - submit():    提交文档进行 AI 初筛 → 返回审核记录（含 tier + score）
      - get_queue(): 获取待审核队列
      - approve():   人工审批通过 → 返回文档内容供调用方入索引
      - reject():    人工拒绝 → 返回文档内容供调用方处理
      - get_stats(): 审核统计
    """

    def __init__(
        self,
        data_dir: str = None,
        review_dir: str = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        """
        Args:
            data_dir:   项目 data/ 目录（审批通过后文档存到这里）
            review_dir: 审核暂存目录（待审文档存在这里）
            llm_config: LLM 配置（默认自动检测 Ollama → DeepSeek）
        """
        base = data_dir or os.path.join(PROJECT_ROOT, "data")
        self.data_dir = base
        self.review_dir = review_dir or os.path.join(PROJECT_ROOT, ".review")
        self.queue_file = os.path.join(self.review_dir, "queue.json")
        self.content_dir = os.path.join(self.review_dir, "content")

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.review_dir, exist_ok=True)
        os.makedirs(self.content_dir, exist_ok=True)

        self.llm = LLMClient(llm_config)

        # 加载队列
        self._queue: dict[str, ReviewItem] = {}
        self._load_queue()

    # ── 队列持久化 ─────────────────────────────────

    def _load_queue(self):
        """从 JSON 文件加载审核队列"""
        if os.path.exists(self.queue_file):
            try:
                with open(self.queue_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item_data in data.get("items", []):
                    item = ReviewItem.from_dict(item_data)
                    # 只加载 pending 状态的（已处理的留在文件里做历史记录，但内存中不保留）
                    if item.status == "pending":
                        self._queue[item.review_id] = item
                logger.info(f"[审核] 加载队列: {len(self._queue)} 条待审核")
            except Exception as e:
                logger.warning(f"[审核] 加载队列失败: {e}")

    def _save_queue(self):
        """保存审核队列到 JSON 文件（保留所有记录作为历史）"""
        os.makedirs(self.review_dir, exist_ok=True)
        # 读取已有记录，合并更新
        existing = {}
        if os.path.exists(self.queue_file):
            try:
                with open(self.queue_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item_data in data.get("items", []):
                    existing[item_data["review_id"]] = item_data
            except Exception:
                pass

        # 合并：内存中的覆盖已有的
        for rid, item in self._queue.items():
            existing[rid] = item.to_dict()

        output = {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_pending": sum(1 for i in existing.values() if i["status"] == "pending"),
            "items": list(existing.values()),
        }
        with open(self.queue_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    # ── AI 评分 ────────────────────────────────────

    def _score_document(self, title: str, content: str) -> tuple[int, str]:
        """
        低温 LLM 评分（T=0.0，确定性输出）

        Returns:
            (score: int 0-10, reasoning: str)
        """
        # 截断过长内容（节省 token）
        content_snippet = content[:1200]

        prompt = SCORE_PROMPT.format(title=title, content=content_snippet)
        messages = [{"role": "user", "content": prompt}]

        try:
            reply = self.llm.chat(messages, temperature=0.0, max_tokens=200)

            # 尝试解析 JSON
            # 先提取 JSON 对象
            json_match = re.search(r'\{[^{}]*"score"\s*:\s*\d+[^{}]*\}', reply, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                score = max(0, min(10, int(data.get("score", 0))))
                reasoning = data.get("reasoning", "")
                return score, reasoning

            # JSON 解析失败，尝试直接提取数字
            match = re.search(r'\d+', reply)
            if match:
                score = max(0, min(10, int(match.group())))
                return score, reply[:100]

            logger.warning(f"[审核] LLM 返回无法解析: {reply[:100]}")
            return 0, "评分解析失败"

        except Exception as e:
            logger.error(f"[审核] LLM 评分失败: {e}")
            # 关键词兜底
            return self._keyword_score(title + " " + content[:500]), "LLM 不可用，使用关键词兜底"

    def _keyword_score(self, text: str) -> int:
        """关键词兜底评分"""
        cfg = get_config()
        keywords = cfg.domain_keywords or []
        hits = sum(1 for kw in keywords if kw in text)
        return min(8, hits // 2)

    def _assign_tier(self, score: int) -> ReviewTier:
        """根据分数分配审核等级"""
        if score >= AUTO_APPROVE_THRESHOLD:
            return ReviewTier.auto_approved
        elif score >= NEEDS_REVIEW_MIN:
            return ReviewTier.needs_review
        else:
            return ReviewTier.auto_rejected

    # ── 核心 API ───────────────────────────────────

    def submit(
        self,
        content: str,
        title: str = "",
        source: str = "",
        filename: str = "",
        force_review: bool = False,
    ) -> dict:
        """
        提交文档进行低温 AI 初筛。

        Args:
            content:      文档全文
            title:        文档标题
            source:       来源（如 "财联社"、"用户上传"）
            filename:     原始文件名
            force_review: 强制进入审核队列（忽略 AI 评分，直接标记 needs_review）

        Returns:
            {
                "review_id": str,
                "tier": "auto_approved" | "needs_review" | "auto_rejected",
                "score": 0-10,
                "ai_reasoning": str,
                "action": "approved" | "queued" | "rejected",
            }
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        review_id = f"rev_{uuid.uuid4().hex[:12]}"

        # 1. 低温评分
        if force_review:
            score = 0
            tier = ReviewTier.needs_review
            reasoning = "用户强制进入审核"
        else:
            score, reasoning = self._score_document(title, content)
            tier = self._assign_tier(score)

        # 2. 构建审核记录
        item = ReviewItem(
            review_id=review_id,
            title=title or content[:50].replace("\n", " "),
            content=content,
            content_preview=content[:200].replace("\n", " "),
            source=source,
            filename=filename,
            score=score,
            tier=tier,
            ai_reasoning=reasoning,
            status="pending" if tier == ReviewTier.needs_review else tier.value.replace("auto_", ""),
            created_at=now,
            reviewed_at=now if tier != ReviewTier.needs_review else "",
            reviewed_by="auto" if tier != ReviewTier.needs_review else "",
        )

        # 3. 保存全文到磁盘（用于后续审批时恢复）
        content_path = os.path.join(self.content_dir, f"{review_id}.txt")
        with open(content_path, "w", encoding="utf-8") as f:
            f.write(f"标题: {item.title}\n")
            f.write(f"来源: {source}\n" if source else "")
            f.write(f"文件名: {filename}\n" if filename else "")
            f.write(f"评分: {score}/10\n")
            f.write(f"等级: {tier.value}\n")
            f.write(f"理由: {reasoning}\n")
            f.write(f"{'─' * 50}\n")
            f.write(content)

        # 4. 入队 + 持久化
        if tier == ReviewTier.needs_review:
            self._queue[review_id] = item
        self._save_queue()

        # 5. 自动审批的：直接把文件保存到 data/ 目录
        filepath = ""
        if tier == ReviewTier.auto_approved and filename:
            # 复制到 data/ 目录
            import shutil
            dest = os.path.join(self.data_dir, filename)
            counter = 1
            while os.path.exists(dest):
                name, ext = os.path.splitext(filename)
                dest = os.path.join(self.data_dir, f"{name}_{counter}{ext}")
                counter += 1
            shutil.copy(content_path, dest)
            filepath = dest

        logger.info(
            f"[审核] {review_id}: {tier.value} (score={score}) "
            f"— {item.title[:40]}"
        )

        return {
            "review_id": review_id,
            "tier": tier.value,
            "score": score,
            "ai_reasoning": reasoning,
            "action": {
                ReviewTier.auto_approved: "approved",
                ReviewTier.needs_review: "queued",
                ReviewTier.auto_rejected: "rejected",
            }[tier],
            "filepath": filepath,
        }

    def get_queue(self, limit: int = 50) -> list[dict]:
        """
        获取待审核队列。

        Returns:
            审核项列表（不含全文，含 preview），按创建时间倒序
        """
        items = sorted(
            self._queue.values(),
            key=lambda x: x.created_at,
            reverse=True,
        )
        return [item.to_dict() for item in items[:limit]]

    def get_item(self, review_id: str) -> Optional[dict]:
        """
        获取审核项详情（含全文）。

        Returns:
            审核项完整数据，或 None
        """
        item = self._queue.get(review_id)
        if item:
            result = item.to_dict()
            result["content"] = item.content
            return result

        # 可能已被处理但还在历史中，从文件读取
        content_path = os.path.join(self.content_dir, f"{review_id}.txt")
        if os.path.exists(content_path):
            with open(content_path, "r", encoding="utf-8") as f:
                full = f.read()
            # 提取正文（分隔线之后的内容）
            parts = full.split("─" * 50, 1)
            content = parts[1].strip() if len(parts) > 1 else full

            # 从队列文件读取元数据
            if os.path.exists(self.queue_file):
                with open(self.queue_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item_data in data.get("items", []):
                    if item_data["review_id"] == review_id:
                        result = dict(item_data)
                        result["content"] = content
                        return result

            return {"review_id": review_id, "content": content, "status": "unknown"}

        return None

    def approve(self, review_id: str) -> dict:
        """
        人工审批通过 → 文档可入索引。

        Returns:
            {"status": "ok", "content": str, "title": str, "filename": str}
        """
        item = self._queue.get(review_id)
        if not item:
            return {"status": "error", "message": f"审核项不存在: {review_id}"}

        if item.status != "pending":
            return {"status": "error", "message": f"审核项状态不是 pending: {item.status}"}

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        item.status = "approved"
        item.reviewed_at = now
        item.reviewed_by = "human"

        # 保存到 data/ 目录
        import shutil
        content_path = os.path.join(self.content_dir, f"{review_id}.txt")
        dest_filename = item.filename or f"{review_id}.txt"
        dest = os.path.join(self.data_dir, dest_filename)
        counter = 1
        while os.path.exists(dest):
            name, ext = os.path.splitext(dest_filename)
            dest = os.path.join(self.data_dir, f"{name}_{counter}{ext}")
            counter += 1

        if os.path.exists(content_path):
            shutil.copy(content_path, dest)
        else:
            # 内容丢了，从内存恢复
            text = f"标题: {item.title}\n{item.content}"
            with open(dest, "w", encoding="utf-8") as f:
                f.write(text)

        # 从队列移除
        del self._queue[review_id]
        self._save_queue()

        logger.info(f"[审核] {review_id}: 人工审批通过 → {dest}")

        return {
            "status": "ok",
            "review_id": review_id,
            "action": "approved",
            "content": item.content,
            "title": item.title,
            "filename": os.path.basename(dest),
            "filepath": dest,
        }

    def reject(self, review_id: str) -> dict:
        """
        人工拒绝 → 移入垃圾桶或丢弃。

        Returns:
            {"status": "ok", "review_id": str, "action": "rejected"}
        """
        item = self._queue.get(review_id)
        if not item:
            return {"status": "error", "message": f"审核项不存在: {review_id}"}

        if item.status != "pending":
            return {"status": "error", "message": f"审核项状态不是 pending: {item.status}"}

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        item.status = "rejected"
        item.reviewed_at = now
        item.reviewed_by = "human"

        # 移入垃圾桶（复用 server.py 的垃圾桶逻辑）
        trash_dir = os.path.join(PROJECT_ROOT, ".trash")
        os.makedirs(trash_dir, exist_ok=True)
        content_path = os.path.join(self.content_dir, f"{review_id}.txt")
        trash_path = os.path.join(trash_dir, f"rejected_{review_id}.txt")
        if os.path.exists(content_path):
            import shutil
            shutil.move(content_path, trash_path)

        # 从队列移除
        del self._queue[review_id]
        self._save_queue()

        logger.info(f"[审核] {review_id}: 人工拒绝")

        return {
            "status": "ok",
            "review_id": review_id,
            "action": "rejected",
        }

    def get_stats(self) -> dict:
        """获取审核统计"""
        pending = len(self._queue)

        # 从历史计算
        total = 0
        approved = 0
        rejected = 0
        auto_approved = 0
        auto_rejected = 0

        if os.path.exists(self.queue_file):
            with open(self.queue_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item_data in data.get("items", []):
                total += 1
                status = item_data.get("status", "")
                tier = item_data.get("tier", "")
                if status == "approved" or tier == "auto_approved":
                    approved += 1
                    if tier == "auto_approved":
                        auto_approved += 1
                elif status == "rejected" or tier == "auto_rejected":
                    rejected += 1
                    if tier == "auto_rejected":
                        auto_rejected += 1

        return {
            "pending": pending,
            "total_processed": total,
            "approved": approved,
            "rejected": rejected,
            "auto_approved": auto_approved,
            "auto_rejected": auto_rejected,
            "human_approved": approved - auto_approved,
            "human_rejected": rejected - auto_rejected,
        }


# ============================================================
# 便捷函数
# ============================================================

def create_reviewer() -> DocumentReviewer:
    """创建默认的审核器实例"""
    return DocumentReviewer()


# ============================================================
# 命令行自检
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("=" * 60)
    print("人工审核分级系统 — 自检")
    print("=" * 60)

    reviewer = DocumentReviewer()

    # 测试1：提交高相关文档
    print("\n[测试1] 高相关文档（应自动通过）...")
    r1 = reviewer.submit(
        title="央行宣布降准0.5个百分点",
        content="中国人民银行决定于2026年6月15日下调金融机构存款准备金率0.5个百分点，"
                "预计释放长期资金约1万亿元。此举旨在支持实体经济发展，降低社会融资成本。",
        source="央行官网",
    )
    print(f"  结果: tier={r1['tier']}, score={r1['score']}, action={r1['action']}")
    print(f"  AI理由: {r1['ai_reasoning']}")

    # 测试2：提交低相关文档
    print("\n[测试2] 低相关文档（应自动拒绝）...")
    r2 = reviewer.submit(
        title="某明星参加综艺节目",
        content="近日，著名影星张某参加了某卫视的综艺节目录制，现场气氛热烈，"
                "张某表示新剧将于下月播出。",
        source="娱乐新闻",
    )
    print(f"  结果: tier={r2['tier']}, score={r2['score']}, action={r2['action']}")
    print(f"  AI理由: {r2['ai_reasoning']}")

    # 测试3：查看审核队列和统计
    print("\n[测试3] 审核队列和统计...")
    queue = reviewer.get_queue()
    stats = reviewer.get_stats()
    print(f"  队列长度: {len(queue)}")
    print(f"  统计: pending={stats['pending']}, approved={stats['approved']}, rejected={stats['rejected']}")
    print(f"  自动通过: {stats['auto_approved']}, 自动拒绝: {stats['auto_rejected']}")

    print("\n" + "=" * 60)
    print("自检完成")
