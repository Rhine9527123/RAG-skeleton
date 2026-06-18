"""
锚点集路由管理器（LSM-Tree 风格）
================================

核心思路：
  - 离线：扫描知识库所有文档 → 提取高频字符n-gram → 保存为 anchor_set.json
  - 在线：用户提问 → 提取n-gram → 命中 ≥ threshold 个 → 走快速 RAG
                                  命中 < threshold 个 → 走 Agentic RAG（多轮改写）
  
LSM-Tree 映射：
  - MemTable  → pending_buffer（新文档的词先进内存缓冲）
  - 刷盘阈值  → rebuild_threshold（默认20篇）
  - SSTable   → anchor_set.json（合并后的冻结锚点集）

无外部依赖：纯 Python 标准库 + re，不依赖 jieba。
"""

import os
import re
import json
import logging
from collections import Counter
from typing import Set, Dict, List, Tuple, Optional

logger = logging.getLogger("anchor_manager")

# -------------------------------------------------------
# Token 提取器（字符 n-gram）
# -------------------------------------------------------

# 中文字符 + 英文/数字视为有效内容
_RE_CJK = re.compile(r"[\u4e00-\u9fff]+")
_RE_ALNUM = re.compile(r"[a-zA-Z0-9]+")

def _extract_ngrams(text: str, n_range: Tuple[int, int] = (2, 4)) -> List[str]:
    """
    从文本中提取 2~4 字 n-gram（不依赖分词器）。
    
    策略：
      1. 提取所有连续中文段 → 生成 2/3/4-gram
      2. 提取所有英文/数字段 → 作为完整 token（如 "2025"、"Q1"）
      3. 返回所有 token 列表（含重复，用于频次统计）
    """
    tokens = []
    
    # 中文 n-gram
    for match in _RE_CJK.finditer(text):
        seq = match.group()
        for n in range(n_range[0], n_range[1] + 1):
            if len(seq) >= n:
                tokens.extend(seq[i:i+n] for i in range(len(seq) - n + 1))
    
    # 英文/数字 token（完整保留）
    for match in _RE_ALNUM.finditer(text):
        token = match.group()
        if len(token) >= 2:  # 太短没信息量
            tokens.append(token.lower())
    
    return tokens


# -------------------------------------------------------
# 停用词（高频但无检索意义的中文n-gram）
# -------------------------------------------------------
_STOP_NGRAMS = {
    # 2-gram 虚词
    "一个", "这个", "那个", "我们", "他们", "什么", "怎么", "可以",
    "没有", "不是", "已经", "还是", "因为", "所以", "但是", "如果",
    "而且", "或者", "不过", "虽然", "然后", "之后", "之前", "之后",
    "一些", "一下", "不会", "不能", "不要", "应该", "可能", "需要",
    "问题", "回答", "请问", "帮我", "我想", "我要", "关于",
    # 3-gram 虚词
    "有没有", "是不是", "能不能", "会不会", "可不可以",
    "什么是", "怎么样", "如何做", "怎么做", "做什么",
    # 4-gram 虚词
    "是什么意思", "有没有什么", "我该怎么办", "你能帮我",
    "我不知道", "我想了解", "我想知道",
    # 特定停用
    "这是", "那是", "这些", "那些", "这样", "那样",
    "的话", "的吗", "了呢", "了吗", "啊是", "啊啊",
}

def _filter_stop(tokens: List[str]) -> List[str]:
    """过滤停用 n-gram"""
    return [t for t in tokens if t not in _STOP_NGRAMS and len(t) >= 2]


# -------------------------------------------------------
# AnchorSetManager
# -------------------------------------------------------

class AnchorSetManager:
    """
    LSM-Tree 风格的锚点集管理器。
    
    用法：
        mgr = AnchorSetManager("anchor_set.json")
        
        # 启动时：自动加载或从文档构建
        mgr.initialize(documents_texts)
        
        # 上传时：新文档进 pending buffer
        mgr.add_document(text)
        
        # 提问时：路由判断
        route = mgr.route("个体户2025年报税怎么操作")
        # → "fast"（命中 3 个锚点）或 "agentic"（命中不足）
    """
    
    def __init__(
        self,
        anchor_file: str,
        rebuild_threshold: int = 20,
        route_threshold: int = 2,
        pending_top_n: int = 50,
        anchor_top_n: int = 300,
        ngram_min: int = 2,
        ngram_max: int = 4,
    ):
        """
        参数：
            anchor_file:      锚点集 JSON 文件路径
            rebuild_threshold: 攒够多少篇新文档触发重建（默认20）
            route_threshold:   命中多少个锚点走快速通道（默认2）
            pending_top_n:     pending buffer 保留 Top-N 高频词（默认50）
            anchor_top_n:      锚点集保留 Top-N 高频词（默认300）
            ngram_min/max:     n-gram 长度范围（默认2~4）
        """
        self.anchor_file = anchor_file
        self.rebuild_threshold = rebuild_threshold
        self.route_threshold = route_threshold
        self.pending_top_n = pending_top_n
        self.anchor_top_n = anchor_top_n
        self.ngram_range = (ngram_min, ngram_max)
        
        # 主锚点集（磁盘持久化）
        self.anchor_set: Set[str] = set()
        
        # pending buffer：新文档词汇频次
        self.pending_counter: Counter = Counter()
        self.pending_doc_count: int = 0
        
        # 统计
        self.total_docs_scanned: int = 0
    
    # ── 初始化 ──
    
    def initialize(self, documents_texts: List[str] = None) -> "AnchorSetManager":
        """
        初始化：尝试加载已有锚点集，不存在则从文档构建。
        
        参数：
            documents_texts: 知识库所有文档的文本列表（用于首次构建）
        """
        if os.path.exists(self.anchor_file):
            loaded = self._load()
            if loaded:
                logger.info(f"[AnchorSet] 加载已有锚点集: {len(self.anchor_set)} 个锚点")
                return self
        
        # 首次构建
        if documents_texts:
            logger.info("[AnchorSet] 首次构建锚点集...")
            self._rebuild_from_texts(documents_texts)
        else:
            logger.warning("[AnchorSet] 无锚点集文件且无文档，使用空集")
        
        return self
    
    # ── 新增文档（进 pending buffer）──
    
    def add_document(self, text: str):
        """
        新文档进 pending buffer。
        
        达到 rebuild_threshold 后自动触发合并重建。
        """
        if not text or not text.strip():
            return
        
        tokens = _filter_stop(_extract_ngrams(text, self.ngram_range))
        self.pending_counter.update(tokens)
        self.pending_doc_count += 1
        self.total_docs_scanned += 1
        
        logger.debug(
            f"[AnchorSet] pending +1 文档 ({len(tokens)} tokens), "
            f"累计 {self.pending_doc_count}/{self.rebuild_threshold}"
        )
        
        # 达到阈值 → 合并重建
        if self.pending_doc_count >= self.rebuild_threshold:
            self._merge_and_rebuild()
    
    # ── 路由判断 ──
    
    def route(self, question: str) -> Tuple[str, int, List[str]]:
        """
        根据问题中的锚点命中数决定路由。
        
        返回：(route, hit_count, hit_tokens)
          - route: "fast"（普通 RAG）或 "agentic"（Agentic RAG）
          - hit_count: 命中的锚点数
          - hit_tokens: 命中的锚点词列表（调试用）
        """
        if not self.anchor_set and not self.pending_counter:
            # 锚点集为空 → 通用 RAG，走普通路径
            return ("fast", 0, [])
        
        tokens = _filter_stop(_extract_ngrams(question, self.ngram_range))
        unique_tokens = set(tokens)
        
        # 同时查主集 + pending buffer 的 Top-N
        pending_top = {w for w, _ in self.pending_counter.most_common(self.pending_top_n)}
        combined_set = self.anchor_set | pending_top
        
        hits = unique_tokens & combined_set
        hit_count = len(hits)
        
        route = "fast" if hit_count >= self.route_threshold else "agentic"
        
        logger.debug(
            f"[AnchorSet] route={route} hits={hit_count}/{self.route_threshold} "
            f"tokens={list(hits)[:5]} question_tokens={list(unique_tokens)[:8]}"
        )
        
        return (route, hit_count, list(hits))
    
    def get_topic_hints(self, top_n: int = 8) -> List[str]:
        """
        返回知识库的主题提示词（用于 Agentic RAG 追问）。
        优先返回包含中文的锚点（更有语义意义），按长度降序排列。
        """
        combined = self.anchor_set | {
            w for w, _ in self.pending_counter.most_common(self.pending_top_n)
        }
        # 优先中文锚点（含 CJK 字符），按长度降序
        def _is_cjk(s):
            return any('\u4e00' <= ch <= '\u9fff' for ch in s)

        cjk_words = [w for w in combined if _is_cjk(w)]
        other_words = [w for w in combined if not _is_cjk(w)]

        cjk_words.sort(key=len, reverse=True)
        other_words.sort(key=len, reverse=True)

        # 中文优先，英文补充
        result = cjk_words[:top_n]
        if len(result) < top_n:
            result.extend(other_words[:top_n - len(result)])
        return result
    
    # ── 获取统计信息 ──
    
    def stats(self) -> dict:
        """返回管理器状态"""
        return {
            "anchor_count": len(self.anchor_set),
            "pending_doc_count": self.pending_doc_count,
            "pending_word_count": len(self.pending_counter),
            "total_docs_scanned": self.total_docs_scanned,
            "rebuild_threshold": self.rebuild_threshold,
            "route_threshold": self.route_threshold,
        }
    
    # ── 强制重建（调试用）──
    
    def force_rebuild(self, documents_texts: List[str]):
        """强制立即重建锚点集（忽略 pending 计数）"""
        self._rebuild_from_texts(documents_texts)
    
    # ═══════════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════════
    
    def _merge_and_rebuild(self):
        """合并 pending buffer 到主集并重建"""
        logger.info(
            f"[AnchorSet] 达到阈值 ({self.pending_doc_count} 篇), "
            f"合并 {len(self.pending_counter)} 个 pending 词 → 重建锚点集"
        )
        
        # 合并 pending 词频到全量计数器（无法回退，但这是 LSM-Tree 的设计）
        # 注意：这里我们只能从 pending_counter 推断，无法知道旧词的精确频次
        # 所以采用简化策略：
        #   1. 将 pending_counter 的词加入主集
        #   2. 如果主集超过 anchor_top_n，按 pending 频次 + 保留旧词
        
        # 取 pending 的 top-k 加入主集
        pending_top_words = {
            w for w, _ in self.pending_counter.most_common(self.pending_top_n)
        }
        self.anchor_set.update(pending_top_words)
        
        # 如果主集过大，裁剪到 anchor_top_n（保留最近添加的）
        if len(self.anchor_set) > self.anchor_top_n:
            # 优先保留 pending 词 + 随机保留旧词（简化处理）
            old_words = list(self.anchor_set - pending_top_words)
            keep_old = self.anchor_top_n - len(pending_top_words)
            if keep_old > 0 and old_words:
                import random
                self.anchor_set = pending_top_words | set(random.sample(
                    old_words, min(keep_old, len(old_words))
                ))
            else:
                self.anchor_set = pending_top_words
        
        # 重置 buffer
        self.pending_counter.clear()
        self.pending_doc_count = 0
        
        # 持久化
        self._save()
    
    def _rebuild_from_texts(self, documents_texts: List[str]):
        """从文档全文重建锚点集"""
        counter = Counter()
        for text in documents_texts:
            if text and text.strip():
                tokens = _filter_stop(_extract_ngrams(text, self.ngram_range))
                counter.update(tokens)
                self.total_docs_scanned += 1
        
        # 取 Top-N
        top_words = counter.most_common(self.anchor_top_n)
        self.anchor_set = {w for w, _ in top_words}
        
        # 清理 pending buffer
        self.pending_counter.clear()
        self.pending_doc_count = 0
        
        self._save()
        
        logger.info(
            f"[AnchorSet] 重建完成: {len(self.anchor_set)} 个锚点 "
            f"(来自 {self.total_docs_scanned} 篇文档, "
            f"从 {len(counter)} 个候选词中选取 Top-{self.anchor_top_n})"
        )
    
    def _save(self):
        """保存锚点集到磁盘"""
        data = {
            "anchor_set": sorted(self.anchor_set),
            "total_docs_scanned": self.total_docs_scanned,
            "pending_doc_count": self.pending_doc_count,
            "pending_top_words": [
                w for w, _ in self.pending_counter.most_common(self.pending_top_n)
            ],
            "version": "1.0",
        }
        with open(self.anchor_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"[AnchorSet] 保存到 {self.anchor_file}")
    
    def _load(self) -> bool:
        """从磁盘加载锚点集"""
        try:
            with open(self.anchor_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.anchor_set = set(data.get("anchor_set", []))
            self.total_docs_scanned = data.get("total_docs_scanned", 0)
            
            # 恢复 pending 状态（服务器重启后 pending 计数归零但词频保留在文件里）
            pending_words = data.get("pending_top_words", [])
            if pending_words:
                self.pending_counter.update({w: 1 for w in pending_words})
            
            logger.info(
                f"[AnchorSet] 加载: {len(self.anchor_set)} 锚点, "
                f"{self.total_docs_scanned} 篇文档已扫描"
            )
            return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[AnchorSet] 加载失败: {e}")
            return False


# ============================================================
# 便捷工厂函数
# ============================================================

def create_anchor_manager(
    data_dir: str = "data",
    anchor_file: str = "anchor_set.json",
    rebuild_threshold: int = 20,
    route_threshold: int = 2,
    documents_texts: List[str] = None,
) -> AnchorSetManager:
    """
    创建并初始化 AnchorSetManager。
    
    参数：
        data_dir:          文档目录
        anchor_file:       锚点集文件路径
        rebuild_threshold: 合并阈值
        route_threshold:   路由命中阈值
        documents_texts:   文档文本列表（首次构建用，None=自动从 data_dir 加载）
    """
    mgr = AnchorSetManager(
        anchor_file=anchor_file,
        rebuild_threshold=rebuild_threshold,
        route_threshold=route_threshold,
    )
    
    # 自动从 data_dir 加载文档
    if documents_texts is None and os.path.isdir(data_dir):
        documents_texts = _load_texts_from_dir(data_dir)
    
    mgr.initialize(documents_texts)
    return mgr


def _load_texts_from_dir(data_dir: str) -> List[str]:
    """从目录加载所有支持格式的文档文本"""
    texts = []
    if not os.path.isdir(data_dir):
        return texts
    
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        try:
            if filename.endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            elif filename.endswith(".pdf"):
                # PDF 文本提取（轻量回退——尝试直接读文本层）
                try:
                    import fitz
                    doc = fitz.open(filepath)
                    pdf_text = ""
                    for page in doc:
                        pdf_text += page.get_text()
                    doc.close()
                    if pdf_text.strip():
                        texts.append(pdf_text)
                except Exception:
                    pass  # PDF 没有文本层，跳过
            # .xlsx 跳过——结构化数据不做 n-gram 锚点提取
        except Exception:
            pass
    
    return texts
