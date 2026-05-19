import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd

# ============================================================
# 自己实现一个简单的 PandasQueryEngine（替代废弃的 llama-index-experimental）
# 原理：让 LLM 根据自然语言生成 pandas 代码，执行后返回结果
# ============================================================
def query_excel(llm, df, question, verbose=True):
    """
    用自然语言查询 DataFrame
    1. 把 DataFrame 的列名和前几行数据发给 LLM
    2. LLM 生成 python/pandas 代码
    3. 在本地执行代码并返回结果
    """
    # 构建 prompt：告诉 LLM DataFrame 的结构
    prompt = f"""你是一个数据分析助手。用户会用自然语言提问，你需要生成 Python pandas 代码来回答问题。

DataFrame 信息：
- 列名：{list(df.columns)}
- 列的数据类型：{dict(df.dtypes)}
- 前5行数据：
{df.head().to_string(index=False)}
- 共 {len(df)} 行数据

规则：
1. 只输出一行 python 代码，不要输出任何其他文字
2. 代码必须是对 df 执行操作并返回结果，例如：df[df['天气'].str.contains('雨')]['营业额'].sum()
3. 如果需要中文匹配，用 str.contains() 方法
4. 如果用户问的是统计值，用 sum()、mean()、count() 等聚合函数
5. 如果用户问的是具体数据，用 df[条件] 筛选

用户问题：{question}

请只输出一行 pandas 代码："""

    # 调用 LLM 生成代码
    response = llm.complete(prompt)
    code = str(response).strip()

    # 去掉可能的 markdown 包裹
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()

    if verbose:
        print(f"[生成代码] {code}")

    # 安全检查：只允许 pandas 操作
    allowed = ["df[", "df.", "pd.", "sum(", "mean(", "count(", "max(", "min(", "str.",
                "contains(", "head(", "groupby(", "describe(", "values", "index",
                "int", "float", "round(", "len(", "sort_values(", "unique("]
    if not any(code.startswith(a) or code.strip().startswith(a) for a in allowed):
        print(f"[安全检查] 代码看起来不太对，跳过执行：{code}")
        return "无法生成安全的查询代码"

    try:
        # 预处理：处理可能的 NaN 值
        df_clean = df.fillna("")
        result = eval(code, {"df": df_clean, "pd": pd})
        if verbose:
            print(f"[查询结果] {result}")
        return result
    except Exception as e:
        if verbose:
            print(f"[执行错误] {e}")
        return f"代码执行出错：{e}"


# ============================================================
# 下面是原有的 RAG 代码（文本知识库部分）
# ============================================================

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.embeddings import resolve_embed_model

# 1. 创建测试文档
os.makedirs("data", exist_ok=True)
with open("data/test.txt", "w", encoding="utf-8") as f:
    f.write("小规模纳税人增值税征收率为3%。\n")
    f.write("小规模纳税人月销售额不超过10万元免征增值税。\n")
    f.write("个体工商户需要按季度申报增值税。\n")

# 2. 加载文档
documents = SimpleDirectoryReader("data").load_data()

# 3. 用本地 Embedding 模型
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-zh-v1.5")

# 4. 构建索引（持久化：第一次构建后保存，以后直接加载）
INDEX_DIR = "chroma_data"
if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
    print("[缓存] 发现已有索引，直接加载（跳过构建）")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)
else:
    print("[构建] 第一次运行，正在构建向量索引...")
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print(f"[构建] 索引已保存到 {INDEX_DIR}/")

# 5. 接入 Kimi
from llama_index.llms.openai_like import OpenAILike
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
# 6. 用自然语言查 Excel（我们自己写的 query_excel）
# ============================================================
print("=" * 50)
print("[Excel] 查询 Excel 销售数据")
print("=" * 50)
df = pd.read_excel("./data/sales_data.xlsx", sheet_name="sales_detail")
result = query_excel(kimi, df, "下雨天的总营业额是多少？", verbose=True)

# ============================================================
# 7. 文本 RAG 检索（税务政策等知识问答）
# ============================================================
print("\n" + "=" * 50)
print("[RAG] 查询税务政策知识库")
print("=" * 50)
text_query_engine = index.as_query_engine()
response = text_query_engine.query("冬天，明天下雨，我是做烧烤大排挡的，我的蔬菜和肉类该怎么进货，营业额趋势如何？")
print(response)
