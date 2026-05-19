"""
第7天：Excel 解析 + RAG 问答

目标：让 RAG 能"看懂" Excel 账本数据
  - 用 pandas 读取 Excel（支持多 Sheet）
  - 把结构化表格数据转成 RAG 能检索的 Document
  - 用"行摘要 + 整表概要"两种策略，保证既能回答具体数据问题，也能回答整体趋势问题

和之前 day6_pdf.py 的思路一致：
  PDF → extract_pdf() → Document() → VectorStoreIndex
  Excel → extract_excel() → Document() → VectorStoreIndex
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
from llama_index.core import VectorStoreIndex, Settings, Document, StorageContext, load_index_from_storage
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike

# ============================================================
# 1. Excel 解析：把表格变成 RAG 文档
# ============================================================

def extract_excel(filepath: str) -> list[Document]:
    """
    读取 Excel 文件，返回 LlamaIndex Document 列表

    核心思路：结构化数据不能直接丢给 RAG（一行"2026-04-02|晴天|57|1604..."
    检索效果很差），需要"翻译"成人类能读懂的自然语言描述。

    策略（两份文档，互补）：
      ① 行摘要文档：每一行变成一句自然语言描述
         → 适合回答"4月3号营业额多少""哪天下雨了"等具体问题
      ② 概要文档：统计汇总 + 趋势分析
         → 适合回答"平均毛利率是多少""天气对营业额有什么影响"等总结性问题
    """
    # ---- 读取 Excel ----
    xls = pd.ExcelFile(filepath)
    all_docs = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # ---- 数据清洗 ----
        df = df.fillna("")       # NaN 替换成空字符串（避免 JSON 序列化报错）
        df = df.dropna(how="all")  # 删除全空行

        if df.empty:
            continue

        # ---- ① 行摘要文档 ----
        row_texts = []
        columns = df.columns.tolist()

        for _, row in df.iterrows():
            # 把每一行转成 "字段名: 值" 的自然语言
            parts = []
            for col in columns:
                val = row[col]
                if val != "":
                    parts.append(f"{col}: {val}")
            if parts:
                row_texts.append("，".join(parts))

        row_doc_text = "\n".join(row_texts)

        row_doc = Document(
            text=row_doc_text,
            metadata={
                "filename": os.path.basename(filepath),
                "sheet": sheet_name,
                "type": "excel_row_detail",
                "rows": len(df),
                "category": "经营数据",
            },
        )
        all_docs.append(row_doc)

        # ---- ② 概要文档 ----
        # 数字列自动计算统计值，文本列统计分类
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()

        summary_parts = [f"表格《{sheet_name}》共有 {len(df)} 条记录，{len(columns)} 个字段。"]

        # 数值列统计
        for col in numeric_cols:
            col_mean = df[col].mean()
            col_max = df[col].max()
            col_min = df[col].min()
            col_sum = df[col].sum()
            summary_parts.append(
                f"{col}：平均值 {col_mean:.1f}，最大值 {col_max}，最小值 {col_min}，总计 {col_sum:.1f}。"
            )

        # 文本列统计（取唯一值前5个）
        for col in text_cols:
            unique_vals = df[col].unique().tolist()
            unique_vals = [v for v in unique_vals if v != ""]
            if unique_vals:
                display = unique_vals[:5]
                if len(unique_vals) > 5:
                    display.append(f"等{len(unique_vals)}种")
                summary_parts.append(f"{col}包含：{'、'.join(str(v) for v in display)}。")

        summary_text = "\n".join(summary_parts)

        summary_doc = Document(
            text=summary_text,
            metadata={
                "filename": os.path.basename(filepath),
                "sheet": sheet_name,
                "type": "excel_summary",
                "rows": len(df),
                "category": "经营数据概要",
            },
        )
        all_docs.append(summary_doc)

    xls.close()
    return all_docs


# ============================================================
# 2. 主流程：解析 Excel → 构建索引 → RAG 问答
# ============================================================

def main():
    print("=" * 60)
    print("第7天：Excel 解析 + RAG 问答")
    print("=" * 60)

    # ---- Step 1: 加载 Embedding 模型 ----
    print("\n[1/5] 加载 Embedding 模型...")
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-zh-v1.5")
    print("       OK - bge-small-zh-v1.5")

    # ---- Step 2: 接入 Kimi LLM ----
    print("[2/5] 接入 Kimi LLM...")
    kimi = OpenAILike(
        model="moonshot-v1-8k",
        api_key=os.environ.get("KIMI_API_KEY", ""),
        api_base="https://api.moonshot.cn/v1",
        is_chat_model=True,
        max_tokens=4096,
        context_window=8192,
    )
    Settings.llm = kimi
    print("       OK - moonshot-v1-8k")

    # ---- Step 3: 解析 Excel ----
    print("[3/5] 解析 Excel 文件...")
    filepath = "data/sales_data.xlsx"
    docs = extract_excel(filepath)

    print(f"       解析出 {len(docs)} 个文档：")
    for doc in docs:
        print(f"       - [{doc.metadata['type']}] {doc.metadata['sheet']}"
              f" ({len(doc.text)} 字)")

    # ---- Step 4: 构建向量索引 ----
    print("\n[4/5] 构建向量索引...")
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    index_dir = "chroma_data_excel"

    if os.path.exists(index_dir) and os.listdir(index_dir):
        print(f"       发现已有索引，直接加载（{index_dir}/）")
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
    else:
        print("       首次运行，构建向量索引...")
        index = VectorStoreIndex.from_documents(docs, transformations=[splitter])
        index.storage_context.persist(persist_dir=index_dir)
        print(f"       索引已保存到 {index_dir}/")

    query_engine = index.as_query_engine(similarity_top_k=5, llm=kimi)
    print("       OK - 查询引擎就绪")

    # ---- Step 5: RAG 问答演示 ----
    print("\n[5/5] RAG 问答演示")
    print("-" * 60)

    questions = [
        "4月3号的营业额是多少？",
        "哪天的客流最多？",
        "平均毛利率大概是多少？",
        "天气对营业额有什么影响？",
        "哪些天是下雨天？营业额分别是多少？",
    ]

    for q in questions:
        print(f"\n问：{q}")
        response = query_engine.query(q)
        print(f"答：{response}")

        # 打印来源
        if hasattr(response, "source_nodes") and response.source_nodes:
            print("  来源：")
            for i, node in enumerate(response.source_nodes[:2], 1):
                print(f"    [{i}] {node.text[:100]}...")

    print("\n" + "=" * 60)
    print("第7天完成！Excel 结构化数据已成功接入 RAG")
    print("=" * 60)


if __name__ == "__main__":
    main()
