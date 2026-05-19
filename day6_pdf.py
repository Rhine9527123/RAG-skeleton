"""
Day 6 — PDF 解析 + RAG 问答（完整版）

学习目标：
1. 用 PyMuPDF 提取 PDF 表格数据
2. 清洗脏数据，转成 RAG 友好的文本
3. 将 PDF 内容加入 RAG 知识库
4. 实现 PDF 文档问答

流程：
PDF文件 → PyMuPDF提取表格 → 清洗 → Document对象 → RAG索引 → 问答
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import fitz  # PyMuPDF
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.embeddings import resolve_embed_model
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike


# ============================================================
# 第1步：PDF 解析（今天新学的）
# ============================================================
print("=" * 60)
print("[第1步] 解析 PDF 文件")
print("=" * 60)

def extract_pdf(pdf_path):
    """
    从 PDF 提取干净的文本
    返回：字符串（所有表格拼接后的文本）
    """
    doc = fitz.open(pdf_path)
    all_text_parts = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        tables = page.find_tables()
        
        for table in tables.tables:
            raw_data = table.extract()
            cleaned = []
            
            for row in raw_data:
                # None → 空字符串，去掉换行和多余空白
                clean_row = []
                for cell in row:
                    if cell is None or str(cell).strip() == "":
                        clean_row.append("")
                    else:
                        clean_row.append(str(cell).replace("\n", " ").strip())
                
                # 跳过全空行
                if all(cell == "" for cell in clean_row):
                    continue
                cleaned.append(clean_row)
            
            # 跳过太小的表格（1-2行大概率是页眉页脚）
            if len(cleaned) <= 1:
                continue
            
            # 转成文本：每行用 | 分隔
            lines = []
            for row in cleaned:
                meaningful = [cell for cell in row if cell != ""]
                if meaningful:
                    lines.append(" | ".join(meaningful))
            
            if lines:
                all_text_parts.append("\n".join(lines))
    
    doc.close()
    return "\n\n".join(all_text_parts)


# 解析成绩 PDF
pdf_path = r"./data/成绩.pdf"
pdf_text = extract_pdf(pdf_path)
print(f"[提取] PDF 文本共 {len(pdf_text)} 字符")
print(f"[预览] {pdf_text[:200]}...")


# ============================================================
# 第2步：构建知识库（原有txt + 新增PDF）
# ============================================================
print("\n" + "=" * 60)
print("[第2步] 构建知识库（txt + PDF）")
print("=" * 60)

# 加载原有文档
with open("data/tax_policy.txt", "r", encoding="utf-8") as f:
    tax_text = f.read()
with open("data/weather_sales.txt", "r", encoding="utf-8") as f:
    weather_text = f.read()

documents = [
    Document(
        text=tax_text,
        metadata={"category": "税务政策", "source": "国家税务总局"}
    ),
    Document(
        text=weather_text,
        metadata={"category": "经营策略", "source": "行业分析报告"}
    ),
    Document(
        text="个体工商户经营建议：建议每日记录收支流水，月底汇总分析。对于餐饮行业，食材成本应控制在营业额的30%-35%以内。人工成本建议不超过营业额的20%。房租成本建议控制在15%以内。三项成本合计控制在65%-70%为健康水平。",
        metadata={"category": "经营策略", "topic": "成本控制"}
    ),
    Document(
        text="2026年小规模纳税人最新政策：增值税征收率从3%减按1%执行。月销售额不超过10万元的免征增值税。个体工商户年应纳税所得额不超过200万元的部分，减半征收个人所得税。上述优惠政策执行期限至2027年12月31日。",
        metadata={"category": "税务政策", "year": "2026"}
    ),
    # ★ 新增：PDF 解析出的成绩数据 ★
    Document(
        text=pdf_text,
        metadata={
            "category": "学生成绩",
            "source": "成绩.pdf",
            "type": "pdf_extract"
        }
    ),
]

print(f"共加载 {len(documents)} 个文档：")
for i, doc in enumerate(documents):
    tag = "[PDF]" if doc.metadata.get("type") == "pdf_extract" else "[TXT]"
    print(f"  {tag} 文档{i+1}: [{doc.metadata.get('category')}] {doc.text[:50]}...")


# ============================================================
# 第3步：构建索引（和 day4 一样）
# ============================================================
print("\n" + "=" * 60)
print("[第3步] 构建向量索引 + BM25索引")
print("=" * 60)

Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-zh-v1.5")

VECTOR_INDEX_DIR = "chroma_data_pdf"
if os.path.exists(VECTOR_INDEX_DIR):
    import shutil
    shutil.rmtree(VECTOR_INDEX_DIR)

print("[构建] 向量索引（含 PDF 数据）...")
vector_index = VectorStoreIndex.from_documents(documents)
vector_index.storage_context.persist(persist_dir=VECTOR_INDEX_DIR)
print(f"[完成] 保存到 {VECTOR_INDEX_DIR}/")

vector_retriever = vector_index.as_retriever(similarity_top_k=3)

print("[构建] BM25 索引...")
splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents)
print(f"[切分] {len(documents)} 个文档 → {len(nodes)} 个片段")

bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=3)
print("[完成] BM25 检索器就绪")


# ============================================================
# 第4步：混合检索 + Reranker + 问答
# ============================================================
print("\n" + "=" * 60)
print("[第4步] 混合检索 + Reranker + 问答")
print("=" * 60)

kimi = OpenAILike(
    model="moonshot-v1-8k",
    api_key=os.environ.get("KIMI_API_KEY", ""),
    api_base="https://api.moonshot.cn/v1",
    is_chat_model=True,
    max_tokens=4096,
    context_window=8192,
)
Settings.llm = kimi

# 混合检索器
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    num_queries=1,
    use_async=False,
    similarity_top_k=10,
)

# Reranker
print("[加载] Reranker 模型...")
rerank = SentenceTransformerRerank(
    model="./models/bge-reranker-v2-m3",
    top_n=3,
)

# 查询引擎
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    llm=kimi,
    node_postprocessors=[rerank],
)
print("[完成] 查询引擎就绪\n")

# 测试问题 —— 针对成绩 PDF
test_questions = [
    "刘俊贤的平均绩点是多少？",
    "C语言程序设计考了多少分？",
    "哪些科目成绩在90分以上？",
    "支撑课和基础课分别有哪些？",
]

for question in test_questions:
    print("-" * 60)
    print(f"  Q: {question}")
    print("-" * 60)
    response = query_engine.query(question)
    print(f"  A: {response}\n")

# ============================================================
# 总结
# ============================================================
print("=" * 60)
print("[总结] Day 6 核心要点")
print("=" * 60)
print("""
1. PDF 解析三步走：
   - page.get_text()          → 提取纯文本
   - page.find_tables()       → 识别表格（结构化）
   - 清洗（去None、去空行）    → 干净文本

2. PDF 分两种：
   - 文本型：PyMuPDF 直接提取文字
   - 扫描型：需要 OCR（今天没涉及，后续可接 Umi-OCR）

3. 接入 RAG 的方式：
   PDF → extract_pdf() → Document(text=pdf_text, metadata=...) 
   → 跟普通文档一样喂给 VectorStoreIndex

4. 关键教训：
   - 表格数据不清洗直接喂 RAG → 噪音多、幻觉严重
   - 知识库有什么才能答什么，没有的内容大模型会瞎编
""")
