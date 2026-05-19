"""
Day 2: 元数据过滤（Metadata Filtering）

学习目标：
1. 理解什么是元数据（Metadata）——"关于数据的数据"，就是标签
2. 学会给文档贴标签（手动创建 Document 对象）
3. 学会在查询时用元数据过滤，只搜特定类别的文档

核心原理：
- 之前所有文档混在一起检索，可能搜出不相关的内容
- 元数据过滤 = 先按标签筛选，再做向量检索
- 相当于"先缩小范围，再精准搜索"
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.llms.openai_like import OpenAILike

# ============================================================
# 第1步：用 Document 对象手动创建带元数据的文档
# ============================================================
# 之前用 SimpleDirectoryReader("data").load_data() 是批量读文件
# 现在用 Document() 一个个手动创建，可以给每个文档贴标签

documents = [
    Document(
        text="小规模纳税人增值税征收率为3%。\n小规模纳税人月销售额不超过10万元免征增值税。\n个体工商户需要按季度申报增值税。",
        metadata={"category": "税务", "source": "国家税务总局"}
    ),
    Document(
        text="冬季雨天，大排档应增加雨具如雨伞、雨衣的进货量。连续降雨可能导致生鲜蔬菜供应紧张和价格上涨，建议提前囤积耐储干货。火锅底料和速冻食品在低温天气销量会增长。",
        metadata={"category": "经营策略", "topic": "天气与进货"}
    ),
    Document(
        text="个体工商户个人所得税按经营所得计算，适用5%-35%的超额累进税率。年应纳税所得额不超过200万元的部分，减半征收个人所得税。",
        metadata={"category": "税务", "topic": "个人所得税"}
    ),
    Document(
        text="夏季高温天气，冷饮和啤酒销量会显著增长，建议增加冷饮类库存。同时高温会导致肉类保鲜时间缩短，应减少单次肉类进货量，增加进货频率。",
        metadata={"category": "经营策略", "topic": "天气与进货"}
    ),
]

# 看看元数据长什么样
for i, doc in enumerate(documents):
    print(f"文档{i+1}: {doc.metadata}")

# ============================================================
# 第2步：构建索引（元数据会自动跟着存进去）
# ============================================================
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-zh-v1.5")

# 元数据改变了 → 需要重新构建索引（删掉旧缓存）
INDEX_DIR = "chroma_data_metadata"
if os.path.exists(INDEX_DIR):
    import shutil
    shutil.rmtree(INDEX_DIR)

print("\n[构建] 正在构建带元数据的向量索引...")
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir=INDEX_DIR)
print(f"[构建] 索引已保存到 {INDEX_DIR}/")

# ============================================================
# 第3步：接入 Kimi
# ============================================================
kimi = OpenAILike(
    model="moonshot-v1-8k",
    api_key=os.environ.get("KIMI_API_KEY", ""),
    api_base="https://api.moonshot.cn/v1",
    is_chat_model=True,
    max_tokens=4096,
    context_window=8192,
)
Settings.llm = kimi

# ============================================================
# 第4步：对比 —— 不过滤 vs 用元数据过滤
# ============================================================

print("\n" + "=" * 60)
print("[对比实验] 问：'下雨天进货要注意什么？'")
print("=" * 60)

# --- A. 不过滤：在所有文档里搜（之前的做法）---
print("\n--- A. 不过滤（搜全部文档）---")
query_engine_all = index.as_query_engine(similarity_top_k=2)
response_a = query_engine_all.query("下雨天进货要注意什么？")
print(f"回答：{response_a}\n")

# --- B. 元数据过滤：只搜"经营策略"类的文档 ---
print("--- B. 元数据过滤（只搜 category=经营策略）---")
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="category", value="经营策略", operator=FilterOperator.EQ)
    ]
)
query_engine_filtered = index.as_query_engine(
    similarity_top_k=2,
    filters=filters
)
response_b = query_engine_filtered.query("下雨天进货要注意什么？")
print(f"回答：{response_b}\n")

# --- C. 再来一个：只搜税务类 ---
print("--- C. 元数据过滤（只搜 category=税务）---")
filters_tax = MetadataFilters(
    filters=[
        MetadataFilter(key="category", value="税务", operator=FilterOperator.EQ)
    ]
)
query_engine_tax = index.as_query_engine(
    similarity_top_k=10,
    filters=filters_tax
)
response_c = query_engine_tax.query("下雨天进货要注意什么？")
print(f"回答：{response_c}\n")

print("=" * 60)
print("[总结]")
print("A（不过滤）：所有文档都参与检索，可能混入不相关内容")
print("B（过滤经营策略）：只搜经营策略文档，回答更精准")
print("C（过滤税务）：只搜税务文档，应该搜不到相关内容或回答质量差")
print("=" * 60)
