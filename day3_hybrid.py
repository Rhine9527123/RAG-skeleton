"""
Day 3: 混合检索（Hybrid Search）= 向量检索 + BM25

学习目标：
1. 理解为什么需要 BM25（向量检索对精确数字/关键词不敏感）
2. 学会同时构建向量索引和 BM25 索引
3. 学会用 QueryFusionRetriever 合并两路检索结果
4. 对比：纯向量 vs 纯BM25 vs 混合检索 的效果差异

核心原理：
- 向量检索：用 Embedding 模型算语义相似度，擅长理解"意思"
  但对精确数字（3%、10万）、专有名词不敏感
- BM25：用词频统计做精确匹配，极快，擅长数字/关键词
  但不理解"多赚点钱" ≈ "提升利润"这种语义
- 混合检索：两路同时跑，合并去重，各取所长
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from llama_index.core import VectorStoreIndex, Settings, Document, StorageContext, load_index_from_storage
from llama_index.core.embeddings import resolve_embed_model
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.llms.openai_like import OpenAILike

# ============================================================
# 第1步：准备知识库（复用已有的税务政策和天气销售文档）
# ============================================================
print("=" * 60)
print("[第1步] 加载知识库文档")
print("=" * 60)

# 读取已有文档内容
with open("data/tax_policy.txt", "r", encoding="utf-8") as f:
    tax_text = f.read()
with open("data/weather_sales.txt", "r", encoding="utf-8") as f:
    weather_text = f.read()

# 用 Document 对象手动创建（跟 day2 一样，方便贴元数据标签）
documents = [
    Document(
        text=tax_text,
        metadata={"category": "税务政策", "source": "国家税务总局"}
    ),
    Document(
        text=weather_text,
        metadata={"category": "经营策略", "source": "行业分析报告"}
    ),
    # 补充一条经营类文档，让知识库更丰富
    Document(
        text="个体工商户经营建议：建议每日记录收支流水，月底汇总分析。对于餐饮行业，食材成本应控制在营业额的30%-35%以内。人工成本建议不超过营业额的20%。房租成本建议控制在15%以内。三项成本合计控制在65%-70%为健康水平。",
        metadata={"category": "经营策略", "topic": "成本控制"}
    ),
    Document(
        text="2026年小规模纳税人最新政策：增值税征收率从3%减按1%执行。月销售额不超过10万元的免征增值税。个体工商户年应纳税所得额不超过200万元的部分，减半征收个人所得税。上述优惠政策执行期限至2027年12月31日。",
        metadata={"category": "税务政策", "year": "2026"}
    ),
]

print(f"共加载 {len(documents)} 个文档：")
for i, doc in enumerate(documents):
    print(f"  文档{i+1}: [{doc.metadata.get('category', '未知')}] {doc.text[:50]}...")

# ============================================================
# 第2步：构建两套索引 —— 向量索引 + BM25索引
# ============================================================
print("\n" + "=" * 60)
print("[第2步] 构建向量索引 + BM25索引")
print("=" * 60)

# --- 2a. 向量索引（跟之前一样）---
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-zh-v1.5")

VECTOR_INDEX_DIR = "chroma_data_hybrid"
if os.path.exists(VECTOR_INDEX_DIR):
    import shutil
    shutil.rmtree(VECTOR_INDEX_DIR)

print("[构建] 正在构建向量索引（Embedding + ChromaDB）...")
vector_index = VectorStoreIndex.from_documents(documents)
vector_index.storage_context.persist(persist_dir=VECTOR_INDEX_DIR)
print(f"[构建] 向量索引已保存到 {VECTOR_INDEX_DIR}/")

# 创建向量检索器
vector_retriever = vector_index.as_retriever(similarity_top_k=3)
print("[完成] 向量检索器创建成功（top_k=3）")

# --- 2b. BM25索引（新东西！）---
# BM25 不需要 Embedding 模型，直接基于词频统计
# 它需要把文档拆成小段才能检索（跟 chunking 一样）
print("\n[构建] 正在构建 BM25 索引（纯词频统计，极快）...")
from llama_index.core.node_parser import SentenceSplitter

# 先把文档切成节点（BM25 在节点级别检索）
splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents)
print(f"[切分] {len(documents)} 个文档 → {len(nodes)} 个片段")

# 创建 BM25 检索器
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=3,  # 返回 top 3
)
print("[完成] BM25 检索器创建成功（top_k=3）")

# ============================================================
# 第3步：对比实验 —— 同一个问题，三种检索方式
# ============================================================
print("\n" + "=" * 60)
print("[第3步] 对比实验：纯向量 vs 纯BM25 vs 混合检索")
print("=" * 60)

# 接入 Kimi（用于最终生成回答）
kimi = OpenAILike(
    model="moonshot-v1-8k",
    api_key=os.environ.get("KIMI_API_KEY", ""),
    api_base="https://api.moonshot.cn/v1",
    is_chat_model=True,
    max_tokens=4096,
    context_window=8192,
)
Settings.llm = kimi

# 选一个能体现混合检索优势的问题
test_questions = [
    "小规模纳税人征收率是1%还是3%？",           # 精确数字 → BM25 应该更强
    "我是个开小餐馆的，下雨天怎么备货？",        # 语义理解 → 向量应该更强
    "2026年个体户有什么税收优惠政策？",          # 两者都需要（精确数字+语义）→ 混合最强
]

for q_idx, question in enumerate(test_questions):
    print(f"\n{'─' * 60}")
    print(f"  问题 {q_idx+1}：{question}")
    print(f"{'─' * 60}")

    # --- A. 纯向量检索 ---
    print("\n  [A] 纯向量检索结果：")
    vector_nodes = vector_retriever.retrieve(question)
    for i, node in enumerate(vector_nodes):
        print(f"    片段{i+1}: {node.text[:80]}...")
        print(f"           来源: {node.metadata.get('category', '未知')}")

    # --- B. 纯 BM25 检索 ---
    print("\n  [B] 纯 BM25 检索结果：")
    bm25_nodes = bm25_retriever.retrieve(question)
    for i, node in enumerate(bm25_nodes):
        print(f"    片段{i+1}: {node.text[:80]}...")
        print(f"           来源: {node.metadata.get('category', '未知')}")

    # --- C. 混合检索（QueryFusionRetriever）---
    # 自动合并两路结果，去重，按综合得分排序
    print("\n  [C] 混合检索结果（向量 + BM25 合并）：")
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        # num_queries=1 表示只用原始问题检索（不做查询扩展）
        num_queries=1,
        # use_async=False 同步执行（教程用，生产环境建议 True）
        use_async=False,
        similarity_top_k=3,
    )
    hybrid_nodes = hybrid_retriever.retrieve(question)
    for i, node in enumerate(hybrid_nodes):
        print(f"    片段{i+1}: {node.text[:80]}...")
        print(f"           来源: {node.metadata.get('category', '未知')}")

# ============================================================
# 第4步：用混合检索 + 大模型生成最终回答
# ============================================================
print("\n" + "=" * 60)
print("[第4步] 混合检索 + Kimi 生成回答")
print("=" * 60)

# 用 LlamaIndex 的 RetrieverQueryEngine 把检索器和 LLM 串起来
from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    llm=kimi,
)

final_questions = [
    "小规模纳税人征收率是1%还是3%？",
    "2026年个体户有什么税收优惠政策？",
]

for question in final_questions:
    print(f"\n{'─' * 60}")
    print(f"  问：{question}")
    print(f"{'─' * 60}")
    response = query_engine.query(question)
    print(f"  答：{response}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("[总结] Day 3 核心要点")
print("=" * 60)
print("""
1. BM25 检索器：BM25Retriever.from_defaults(nodes=nodes)
   - 不需要 Embedding 模型，纯词频统计，极快
   - 擅长精确关键词和数字匹配

2. 向量检索器：vector_index.as_retriever()
   - 需要 Embedding 模型，理解语义
   - 擅长同义词、口语化表达

3. 混合检索：QueryFusionRetriever(retrievers=[向量, BM25])
   - 两路同时跑，自动合并去重
   - 兼具精确匹配 + 语义理解

4. 什么时候用混合检索？
   - 用户问题同时包含精确信息（数字、专有名词）和语义信息
   - 知识库文档类型多样，需要兼顾关键词和语义
   - 简单来说：几乎所有正式项目都应该用混合检索
""")
