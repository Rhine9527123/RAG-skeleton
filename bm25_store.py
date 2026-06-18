"""
bm25_store.py — LSM-Tree 风格的 BM25 索引管理器
================================================

设计思路：
  - **Base 层**：主 BM25 索引（覆盖所有已合并文档），磁盘持久化
  - **Delta 层**：近期新增的节点（内存中），轻量 BM25 索引
  - **查询时**：同时搜索 Base + Delta，合并去重返回
  - **合并时**：当 Delta 节点数超过阈值，触发全量重建（merge into base）

优点：
  - 新增少量文档时不需要全量重建 BM25（O(1) 追加 vs O(N) 重建）
  - 大数据量场景下效果显著
  - 查询延迟仅 linear 增加（Base top_k + Delta top_k）

用法：
  store = BM25IndexStore(persist_dir="bm25_index", merge_threshold=100)
  store.initialize(all_nodes)              # 首次初始化
  store.add_delta(new_nodes)              # 增量添加（上传时）
  results = store.retrieve(query, top_k)  # 查询
  store.force_merge()                     # 手动合并
"""

import os
import pickle
import logging
from typing import List, Optional

from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.retrievers.bm25 import BM25Retriever

logger = logging.getLogger("bm25_store")

# 持久化文件名
_BASE_NODES_FILE = "base_nodes.pkl"


class BM25StoreRetrieverAdapter:
    """
    将 BM25IndexStore 适配为 LlamaIndex BaseRetriever 兼容接口，
    使其能放入 QueryFusionRetriever 中与向量检索器配合。

    仅暴露 .retrieve(query_str) -> List[NodeWithScore] 方法。
    """

    def __init__(self, store: "BM25IndexStore"):
        self._store = store

    def retrieve(self, query: str):
        return self._store.retrieve(query)


class BM25IndexStore:
    """
    LSM-Tree 风格的 BM25 索引管理器。

    Base（磁盘持久化）+ Delta（内存）双层架构。
    """

    def __init__(
        self,
        persist_dir: str = "bm25_index",
        merge_threshold: int = 100,
        similarity_top_k: int = 10,
    ):
        """
        Args:
            persist_dir:      持久化目录（存放 base_nodes.pkl）
            merge_threshold:  当 delta 节点数 >= 此值，自动触发合并
            similarity_top_k: BM25 检索返回的最大片段数
        """
        self.persist_dir = persist_dir
        self.merge_threshold = merge_threshold
        self.similarity_top_k = similarity_top_k

        # Base 层
        self._base_nodes: List[TextNode] = []
        self._base_retriever: Optional[BM25Retriever] = None

        # Delta 层
        self._delta_nodes: List[TextNode] = []
        self._delta_retriever: Optional[BM25Retriever] = None

        # 统计
        self.total_merges = 0
        self.total_delta_adds = 0

    # ── 初始化 / 持久化 ──────────────────────────────

    def initialize(self, nodes: List[TextNode]) -> "BM25IndexStore":
        """
        首次初始化：从磁盘加载或从给定 nodes 构建 Base 索引。

        Args:
            nodes: 所有文档的节点列表（首次启动时从 _load_documents 获取）
        """
        os.makedirs(self.persist_dir, exist_ok=True)
        persist_path = os.path.join(self.persist_dir, _BASE_NODES_FILE)

        if os.path.exists(persist_path):
            # 从磁盘恢复
            try:
                with open(persist_path, "rb") as f:
                    self._base_nodes = pickle.load(f)
                if self._base_nodes:
                    self._base_retriever = BM25Retriever.from_defaults(
                        nodes=self._base_nodes,
                        similarity_top_k=self.similarity_top_k,
                    )
                logger.info(
                    f"[BM25Store] 从磁盘恢复 Base 索引: {len(self._base_nodes)} 个节点"
                )
            except Exception as e:
                logger.warning(f"[BM25Store] 磁盘恢复失败，将重建: {e}")
                self._build_base(nodes)
        else:
            # 首次启动，构建
            self._build_base(nodes)

        # Delta 从零开始（未合并的新增在上次运行后丢失，但下次启动会通过
        # _load_documents 全量加载，所以不影响一致性）
        self._delta_nodes = []
        self._delta_retriever = None

        return self

    def _build_base(self, nodes: List[TextNode]):
        """全量构建 Base 索引并持久化"""
        self._base_nodes = list(nodes)
        if self._base_nodes:
            self._base_retriever = BM25Retriever.from_defaults(
                nodes=self._base_nodes,
                similarity_top_k=self.similarity_top_k,
            )
        self._persist_base()
        logger.info(f"[BM25Store] Base 索引构建完成: {len(self._base_nodes)} 个节点")

    def _persist_base(self):
        """将 Base 节点列表写入磁盘"""
        persist_path = os.path.join(self.persist_dir, _BASE_NODES_FILE)
        with open(persist_path, "wb") as f:
            pickle.dump(self._base_nodes, f)

    # ── 增量操作 ────────────────────────────────────

    def add_delta(self, new_nodes: List[TextNode]) -> bool:
        """
        将新节点加入 Delta 层。如果 Delta 超过阈值，自动触发合并。

        Returns:
            True 如果触发了合并（调用方可用于日志）
        """
        if not new_nodes:
            return False

        self._delta_nodes.extend(new_nodes)
        self.total_delta_adds += len(new_nodes)

        # 重建 Delta 检索器（轻量操作，Delta 通常很小）
        self._delta_retriever = BM25Retriever.from_defaults(
            nodes=self._delta_nodes,
            similarity_top_k=self.similarity_top_k,
        ) if self._delta_nodes else None

        merged = False
        if len(self._delta_nodes) >= self.merge_threshold:
            self._merge_delta_to_base()
            merged = True

        return merged

    def remove_nodes(self, node_ids_to_remove: set) -> int:
        """
        从 Base 或 Delta 中移除指定节点。

        需要全量重建 Base（BM25 不支持高效的单节点删除）。
        所以这里直接触发全量重建。

        Args:
            node_ids_to_remove: 要移除的 node_id 集合

        Returns:
            实际移除的节点数
        """
        removed = 0

        # 从 Base 移除
        new_base = [n for n in self._base_nodes if n.node_id not in node_ids_to_remove]
        removed += len(self._base_nodes) - len(new_base)
        self._base_nodes = new_base
        self._base_retriever = BM25Retriever.from_defaults(
            nodes=self._base_nodes,
            similarity_top_k=self.similarity_top_k,
        ) if self._base_nodes else None

        # 从 Delta 移除
        new_delta = [n for n in self._delta_nodes if n.node_id not in node_ids_to_remove]
        removed += len(self._delta_nodes) - len(new_delta)
        self._delta_nodes = new_delta
        self._delta_retriever = BM25Retriever.from_defaults(
            nodes=self._delta_nodes,
            similarity_top_k=self.similarity_top_k,
        ) if self._delta_nodes else None

        self._persist_base()
        return removed

    def _merge_delta_to_base(self):
        """将 Delta 合并入 Base（全量重建 Base）"""
        if not self._delta_nodes:
            return

        self._base_nodes.extend(self._delta_nodes)
        self._base_retriever = BM25Retriever.from_defaults(
            nodes=self._base_nodes,
            similarity_top_k=self.similarity_top_k,
        )
        self._persist_base()

        logger.info(
            f"[BM25Store] Delta→Base 合并: "
            f"+{len(self._delta_nodes)} 节点, "
            f"Base 总计 {len(self._base_nodes)} 节点"
        )

        self._delta_nodes = []
        self._delta_retriever = None
        self.total_merges += 1

    def force_merge(self):
        """手动触发 Delta → Base 合并（用于 /rebuild API）"""
        if self._delta_nodes:
            self._merge_delta_to_base()

    # ── 检索 ────────────────────────────────────────

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        混合检索：同时查询 Base + Delta，合并去重返回。

        Args:
            query: 用户查询文本
            top_k: 返回的最大节点数（覆盖初始化时设置的 similarity_top_k）

        Returns:
            NodeWithScore 列表，按分数降序排列
        """
        k = top_k or self.similarity_top_k

        # 同时查询 Base 和 Delta
        base_results = []
        delta_results = []

        if self._base_retriever:
            base_results = self._base_retriever.retrieve(query)

        if self._delta_retriever:
            delta_results = self._delta_retriever.retrieve(query)

        # 合并去重：Base 优先，Delta 补充不重复的节点
        seen_ids = set()
        merged = []

        for node in base_results:
            seen_ids.add(node.node_id)
            merged.append(node)

        for node in delta_results:
            if node.node_id not in seen_ids:
                seen_ids.add(node.node_id)
                merged.append(node)

        # 按分数降序排列
        merged.sort(key=lambda n: n.score or 0.0, reverse=True)

        return merged[:k]

    # ── 全量重建 ────────────────────────────────────

    def rebuild(self, all_nodes: List[TextNode]):
        """
        从零重建整个 BM25 索引（用于 /dedup/rebuild 或文件大量变更后）。

        Args:
            all_nodes: 所有文档的完整节点列表
        """
        self._base_nodes = list(all_nodes)
        self._delta_nodes = []
        self._delta_retriever = None
        self._build_base(all_nodes)
        logger.info(
            f"[BM25Store] 全量重建完成: {len(self._base_nodes)} 个节点"
        )

    # ── 统计 ────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "base_nodes": len(self._base_nodes),
            "delta_nodes": len(self._delta_nodes),
            "total_merges": self.total_merges,
            "total_delta_adds": self.total_delta_adds,
            "merge_threshold": self.merge_threshold,
        }

    @property
    def all_nodes(self) -> List[TextNode]:
        """返回 Base + Delta 的所有节点"""
        return self._base_nodes + self._delta_nodes
