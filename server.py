"""
RAG 财务知识库服务 - FastAPI 版

启动方式：python server.py
访问地址：http://localhost:8000
API 文档：http://localhost:8000/docs

架构：
  启动时（只做一次）：
    1. 从 config.json（优先）或 .env（Docker兼容）读取配置
    2. 加载 Embedding 模型 (bge-small-zh-v1.5)
    3. 加载/构建 向量索引 + BM25 索引
    4. 加载 Reranker 模型 (bge-reranker-v2-m3)
    5. 组装 query_engine（混合检索 + 精排 + LLM）
    6. 启动 HTTP 服务

  收到 /chat 请求时：
    1. 接收用户问题
    2. query_engine.query(question)  ← 内部走：混合检索→精排→LLM
    3. 返回答案 + 来源片段
"""
import os
import sys
import json
import io
import shutil
from contextlib import asynccontextmanager
from typing import Optional

# Windows 兼容：Git Bash / Docker 环境可能没有 USERNAME，torch 初始化会崩
if sys.platform == "win32" and not os.environ.get("USERNAME"):
    os.environ["USERNAME"] = os.environ.get("USER", "default")

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


def load_config():
    """
    加载配置，优先级：
      1. config.json（安装包模式，与 server.py 同目录）
      2. 环境变量（Docker 模式，从 .env 读取）
    """
    # 获取可执行文件/脚本所在目录
    if getattr(sys, "frozen", False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(base_dir, "config.json")
    config = {}

    # 先从 config.json 读取
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config.update(json.load(f))
        except Exception:
            pass

    # 环境变量覆盖（优先级更高，Docker 模式用）
    env_map = {
        "kimi_api_key": "KIMI_API_KEY",
        "kimi_api_base": "KIMI_API_BASE",
        "kimi_model": "KIMI_MODEL",
        "reranker_model_path": "RERANKER_MODEL_PATH",
        "use_ollama": "USE_OLLAMA",
        "ollama_base_url": "OLLAMA_BASE_URL",
        "ollama_model": "OLLAMA_MODEL",
    }
    for key, env_var in env_map.items():
        val = os.environ.get(env_var)
        if val:
            config[key] = val

    # Reranker 模型路径默认推断
    if not config.get("reranker_model_path"):
        default_path = os.path.join(base_dir, "models", "BAAI", "bge-reranker-v2-m3")
        if os.path.exists(default_path):
            config["reranker_model_path"] = default_path

    return config


_CONFIG = load_config()

# 导出给 server.py 内部使用（兼容 Docker 模式）
os.environ.setdefault("DEEPSEEK_API_KEY", _CONFIG.get("deepseek_api_key", ""))
if _CONFIG.get("reranker_model_path"):
    os.environ.setdefault("RERANKER_MODEL_PATH", _CONFIG["reranker_model_path"])



# ============================================================
# LlamaIndex 核心依赖
# ============================================================
from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever

# ============================================================
# PDF 解析依赖
# ============================================================
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# ============================================================
# Excel 解析依赖
# ============================================================
import pandas as pd

# ============================================================
# 全局变量（启动时初始化，请求时复用）
# ============================================================
query_engine = None  # RAG 查询引擎
bm25_retriever = None  # BM25 检索器（需要单独持有，用于增量添加文档）
splitter = None  # 文本切分器
index = None  # 向量索引

# 配置
DATA_DIR = "data"
VECTOR_INDEX_DIR = "chroma_data_server"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
TOP_K = 10  # 粗筛数量
TOP_N = 3  # 精排数量


def extract_pdf(pdf_bytes):
    """
    从 PDF 字节流提取干净文本（和 app_single.py 同款逻辑）
    三步走：表格识别 → 纯文本提取 → OCR 扫描件回退
    返回：字符串
    """
    all_text_parts = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # 第一步：优先提取表格
    for page_num in range(len(doc)):
        page = doc[page_num]
        tables = page.find_tables()

        for table in tables.tables:
            raw_data = table.extract()
            cleaned = []

            for row in raw_data:
                clean_row = []
                for cell in row:
                    if cell is None or str(cell).strip() == "":
                        clean_row.append("")
                    else:
                        clean_row.append(str(cell).replace("\n", " ").strip())
                if all(cell == "" for cell in clean_row):
                    continue
                cleaned.append(clean_row)

            if len(cleaned) <= 1:
                continue

            lines = []
            for row in cleaned:
                meaningful = [cell for cell in row if cell != ""]
                if meaningful:
                    lines.append(" | ".join(meaningful))
            if lines:
                all_text_parts.append("\n".join(lines))

    # 第二步：如果没识别到表格，回退到纯文本提取
    if not all_text_parts:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            if text:
                all_text_parts.append(text)

    doc.close()

    # 第三步：如果纯文本提取也为空，说明是扫描件，走 OCR
    combined_text = "\n\n".join(all_text_parts)
    if not combined_text.strip():
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        ocr_texts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang="chi_sim+eng").strip()
            if text:
                ocr_texts.append(text)
        doc.close()
        if ocr_texts:
            all_text_parts = ocr_texts

    return "\n\n".join(all_text_parts)


def extract_excel(xlsx_bytes) -> list[Document]:
    """
    从 Excel 字节流提取结构化数据，返回 LlamaIndex Document 列表

    核心思路：Excel 是结构化数据（行列+类型），RAG 看不懂原始数字表格。
    需要翻译成自然语言描述，Embedding 模型才能做有意义的向量化。

    策略（每个 Sheet 生成两份文档，互补）：
      ① 行摘要文档：每一行变成 "字段名: 值" 的自然语言
         → 适合回答"4月3号营业额多少""哪天下雨了"等具体问题
      ② 概要文档：数值列自动统计（均值/最大/最小/总计），文本列列出来有几种值
         → 适合回答"平均营业额是多少""天气对营业额有什么影响"等总结性问题

    参数：xlsx_bytes — Excel 文件的字节流（io.BytesIO）
    返回：list[Document] — 每个 Sheet 两个 Document
    """
    import io

    all_docs = []
    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # 数据清洗：NaN 替换为空字符串（避免 JSON 序列化报错 + 行摘要出现 "nan" 垃圾文本）
        df = df.fillna("")
        # 删除全空行（Excel 底部常见）
        df = df.dropna(how="all")

        if df.empty:
            continue

        columns = df.columns.tolist()

        # ---- ① 行摘要文档 ----
        row_texts = []
        for _, row in df.iterrows():
            parts = []
            for col in columns:
                val = row[col]
                if val != "":
                    parts.append(f"{col}: {val}")
            if parts:
                row_texts.append("，".join(parts))

        row_doc = Document(
            text="\n".join(row_texts),
            metadata={
                "sheet": sheet_name,
                "type": "excel_row_detail",
                "rows": len(df),
                "category": "经营数据",
            },
        )
        all_docs.append(row_doc)

        # ---- ② 概要文档 ----
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

        # 文本列统计（取唯一值前5个，避免文本过长）
        for col in text_cols:
            unique_vals = df[col].unique().tolist()
            unique_vals = [v for v in unique_vals if v != ""]
            if unique_vals:
                display = unique_vals[:5]
                if len(unique_vals) > 5:
                    display.append(f"等{len(unique_vals)}种")
                summary_parts.append(f"{col}包含：{'、'.join(str(v) for v in display)}。")

        summary_doc = Document(
            text="\n".join(summary_parts),
            metadata={
                "sheet": sheet_name,
                "type": "excel_summary",
                "rows": len(df),
                "category": "经营数据概要",
            },
        )
        all_docs.append(summary_doc)

    xls.close()
    return all_docs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理：
    - 启动时：加载所有模型和索引（只做一次）
    - 关闭时：清理资源
    """
    global query_engine, bm25_retriever, splitter, index

    print("=" * 60)
    print("[启动] RAG 财务知识库服务")
    print("=" * 60)

    # ---- 1. 加载 Embedding 模型 ----
    print("[1/6] 加载 Embedding 模型...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
    print("       [OK] Embedding 模型加载完成")

    # ---- 2. 接入 LLM（自动检测：Ollama 可用就用 Ollama，否则用 Kimi）----
    use_ollama = os.environ.get("USE_OLLAMA", "false").lower() in ("true", "1", "yes")
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

    # 智能检测 Ollama 是否可用（即使没显式设置 USE_OLLAMA）
    if not use_ollama:
        try:
            import urllib.request
            req = urllib.request.Request(f"{ollama_base_url}/api/tags", method="GET")
            urllib.request.urlopen(req, timeout=3)
            print(f"       [自动检测] 发现 Ollama 可用（{ollama_base_url}）")
            use_ollama = True
        except Exception:
            use_ollama = False

    if use_ollama:
        print(f"[2/6] 接入本地 Ollama（{ollama_model}）...")
        print(f"       [提示] 跳过 Reranker 以节省内存（~2.1GB），低温度抑制幻觉")
        llm = OpenAILike(
            model=ollama_model,
            api_key="ollama",  # Ollama 不校验 key
            api_base=f"{ollama_base_url}/v1",
            is_chat_model=True,
            max_tokens=4096,
            context_window=8192,
            temperature=0.1,
        )
        Settings.llm = llm
        print(f"       [OK] Ollama LLM 配置完成（{ollama_base_url}）")
    else:
        print("[2/6] 接入 DeepSeek LLM...")
        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        deepseek = OpenAILike(
            model="deepseek-chat",
            api_key=deepseek_api_key,
            api_base="https://api.deepseek.com/v1",
            is_chat_model=True,
            max_tokens=4096,
            context_window=8192,
        )
        Settings.llm = deepseek
        print("       [OK] DeepSeek LLM 配置完成")

    # ---- 3. 加载/构建 向量索引 ----
    print("[3/6] 加载向量索引...")
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # 加载知识库文档
    documents = _load_documents()

    if os.path.exists(VECTOR_INDEX_DIR) and os.listdir(VECTOR_INDEX_DIR):
        print(f"       发现已有索引，直接加载（{VECTOR_INDEX_DIR}/）")
        storage_context = StorageContext.from_defaults(persist_dir=VECTOR_INDEX_DIR)
        index = load_index_from_storage(storage_context)
    else:
        print("       首次运行，构建向量索引...")
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=VECTOR_INDEX_DIR)
        print(f"       索引已保存到 {VECTOR_INDEX_DIR}/")
    print("       [OK] 向量索引就绪")

    # ---- 4. 构建 BM25 索引 ----
    print("[4/6] 构建 BM25 索引...")
    nodes = splitter.get_nodes_from_documents(documents)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=TOP_K)
    print(f"       [OK] BM25 索引就绪（{len(nodes)} 个片段）")

    # ---- 5. 加载 Reranker + 组装查询引擎 ----
    vector_retriever = index.as_retriever(similarity_top_k=TOP_K)
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        num_queries=1,
        use_async=False,
        similarity_top_k=TOP_K,
    )

    if use_ollama:
        # Ollama 离线模式：跳过 Reranker，省 ~2.1GB 内存（适配 8GB 小主机）
        print("[5/6] Ollama 模式：跳过 Reranker...")
        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            llm=llm,
        )
        print("       [OK] 查询引擎组装完成（混合检索 + Ollama LLM，无 Reranker）")
    else:
        print("[5/6] 加载 Reranker 模型...")
        rerank_model_path = os.environ.get("RERANKER_MODEL_PATH", "BAAI/bge-reranker-v2-m3")
        rerank = SentenceTransformerRerank(
            model=rerank_model_path,
            top_n=TOP_N,
        )
        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            node_postprocessors=[rerank],
            llm=Settings.llm,
        )
        print("       [OK] Reranker 加载完成")
        print("       [OK] 查询引擎组装完成（混合检索 + 精排 + LLM）")

    # ---- 6. 启动完成 ----
    print("[6/6] 服务启动完成！")
    print("=" * 60)
    print("  接口文档: http://localhost:8000/docs")
    print("  聊天接口: POST http://localhost:8000/chat")
    print("=" * 60)

    # 等待请求...
    yield

    # 关闭时清理
    print("[关闭] 服务正在停止...")


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="RAG 财务知识库服务",
    description="""
    ## 🧾 RAG 财务知识库服务

    面向个体工商户的 AI 财务助手，基于**检索增强生成（RAG）**技术。

    ### 工作流程
    1. 用户提问
    2. 混合检索（向量 + BM25）→ 粗筛 TOP_K 条
    3. Reranker 精排 → 保留 TOP_N 条
    4. LLM（Kimi）基于精排结果生成答案

    ### 技术栈
    - **Embedding**: BAAI/bge-small-zh-v1.5
    - **Reranker**: BAAI/bge-reranker-v2-m3（本地）
    - **LLM**: Kimi moonshot-v1-8k
    - **检索**: 向量检索 + BM25 混合检索
    """,
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================================
# API 接口
# ============================================================

class ChatRequest(BaseModel):
    """聊天请求"""
    question: str = "用户的问题，例如：小规模纳税人增值税税率是多少？"
    top_k: Optional[int] = None  # 可选：临时覆盖粗筛数量（默认10）
    top_n: Optional[int] = None  # 可选：临时覆盖精排数量（默认3）

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "小规模纳税人增值税税率是多少？",
                    "top_k": 10,
                    "top_n": 3,
                },
                {
                    "question": "个体工商户怎么建账？",
                },
            ]
        }
    }


class ChatResponse(BaseModel):
    """聊天响应"""
    question: str = "用户的问题"
    answer: str = "AI 生成的回答"
    sources: list = "来源片段列表（包含原文、相关度分数、元数据）"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "小规模纳税人增值税税率是多少？",
                    "answer": "根据现行税收政策，小规模纳税人适用...",
                    "sources": [
                        {
                            "text": "小规模纳税人增值税征收率为3%...",
                            "score": 0.8921,
                            "metadata": {"category": "税务政策", "source": "国家税务总局"},
                        }
                    ],
                }
            ]
        }
    }


class UploadResponse(BaseModel):
    """上传响应"""
    status: str = "上传结果：ok / error"
    filename: str = "文件名"
    chunks: int = "切分后的文本片段数量"
    file_type: str = "文件类型：txt / pdf"
    message: str = "附加信息（如 PDF 解析方式）"


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "服务状态：running / initializing"
    model_loaded: bool = "模型是否加载完成"


@app.post("/chat", response_model=ChatResponse, summary="💰 RAG 问答", tags=["核心接口"])
async def chat(request: ChatRequest):
    """
    核心接口：用户提问，AI 基于知识库回答。

    **处理流程**：用户问题 → 混合检索(向量+BM25) → Reranker精排 → LLM生成答案

    **返回**：答案 + 来源片段（方便验证信息可靠性）
    """
    try:
        response = query_engine.query(request.question)

        # 提取来源片段
        sources = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                score = node.score
                # numpy.float32 无法 JSON 序列化，需转为 Python float
                if score is not None:
                    score = float(score)
                sources.append({
                    "text": node.text[:200],  # 截取前200字
                    "score": round(score, 4) if score else None,
                    "metadata": dict(node.metadata) if node.metadata else {},
                })

        return ChatResponse(
            question=request.question,
            answer=str(response),
            sources=sources,
        )
    except Exception as e:
        return ChatResponse(
            question=request.question,
            answer=f"服务内部错误：{str(e)}",
            sources=[],
        )


@app.post("/upload", response_model=UploadResponse, summary="📤 上传知识文件", tags=["管理接口"])
async def upload_file(
    file: UploadFile = File(..., description="知识文件（支持 .txt、.pdf 和 .xlsx）"),
    category: str = Form("未知", description="分类标签，例如 '税务政策'、'训练方案'"),
):
    """
    上传新的知识文件到知识库（支持 txt、pdf 和 xlsx 格式）。

    **处理流程**：
    - txt 文件：直接读取文本 → 切片 → 写入索引
    - pdf 文件：PyMuPDF 解析（表格优先 → 纯文本回退 → OCR 扫描件兜底）→ 切片 → 写入索引
    - xlsx 文件：pandas 解析（行摘要 + 概要，每个 Sheet 生成两份文档）→ 写入索引

    **参数说明**：
    - `file`: 文件（multipart/form-data）
    - `category`: 分类标签
    """
    global index, bm25_retriever, query_engine

    try:
        filename = file.filename
        file_bytes = await file.read()

        # 保存到 data/ 目录
        os.makedirs(DATA_DIR, exist_ok=True)
        filepath = os.path.join(DATA_DIR, filename)

        # 防止同名文件覆盖
        save_path = filepath
        counter = 1
        while os.path.exists(save_path):
            name, ext = os.path.splitext(filename)
            save_path = os.path.join(DATA_DIR, f"{name}_{counter}{ext}")
            counter += 1

        with open(save_path, "wb") as f:
            f.write(file_bytes)

        # 根据文件类型提取文本
        message = ""
        file_type = "unknown"
        if filename.lower().endswith(".pdf"):
            text = extract_pdf(file_bytes)
            if not text.strip():
                os.remove(save_path)
                return UploadResponse(status="error", filename=filename, chunks=0, file_type="pdf", message="PDF 提取为空（可能是空文件或无法解析）")
            message = "PDF 解析成功（表格提取/纯文本/OCR）"
            file_type = "pdf"
            # 单个 Document
            actual_filename = os.path.basename(save_path)
            doc = Document(
                text=text,
                metadata={"filename": actual_filename, "category": category, "type": "file_upload"},
            )
            # 暂存，后面统一走 _load_documents() 重建索引
        elif filename.lower().endswith(".xlsx") or filename.lower().endswith(".xls"):
            try:
                new_docs = extract_excel(file_bytes)
            except Exception as e:
                os.remove(save_path)
                return UploadResponse(status="error", filename=filename, chunks=0, file_type="xlsx", message=f"Excel 解析失败: {str(e)}")
            if not new_docs:
                os.remove(save_path)
                return UploadResponse(status="error", filename=filename, chunks=0, file_type="xlsx", message="Excel 解析为空（可能是空文件）")
            for doc in new_docs:
                doc.metadata["filename"] = os.path.basename(save_path)
                doc.metadata["category"] = category
            message = f"Excel 解析成功（{len(new_docs)} 个文档：行摘要+概要）"
            file_type = "xlsx"
        elif filename.lower().endswith(".txt"):
            text = file_bytes.decode("utf-8")
            message = "TXT 文本上传成功"
            file_type = "txt"
            if not text.strip():
                os.remove(save_path)
                return UploadResponse(status="error", filename=filename, chunks=0, file_type="txt", message="文件内容为空")
        else:
            os.remove(save_path)
            return UploadResponse(status="error", filename=filename, chunks=0, file_type="unknown", message="不支持的文件格式，仅支持 .txt、.pdf 和 .xlsx")

        # 文件已保存，从零重建所有索引（保证向量+BM25 一致）
        all_documents = _load_documents()
        all_nodes = splitter.get_nodes_from_documents(all_documents)

        # 重建向量索引（用 from_documents 一次性构建，避免 insert_nodes 兼容问题）
        index = VectorStoreIndex(all_nodes)
        index.storage_context.persist(persist_dir=VECTOR_INDEX_DIR)

        # 重建 BM25 索引
        bm25_retriever = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=TOP_K)

        # 重建 query_engine（因为索引更新了）
        use_ollama = os.environ.get("USE_OLLAMA", "false").lower() in ("true", "1", "yes")
        vector_retriever = index.as_retriever(similarity_top_k=TOP_K)
        hybrid_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            num_queries=1,
            use_async=False,
            similarity_top_k=TOP_K,
        )

        if use_ollama:
            # Ollama 模式：不加载 Reranker
            query_engine = RetrieverQueryEngine.from_args(
                retriever=hybrid_retriever,
                llm=Settings.llm,
            )
        else:
            rerank_model_path = os.environ.get("RERANKER_MODEL_PATH", "BAAI/bge-reranker-v2-m3")
            rerank = SentenceTransformerRerank(
                model=rerank_model_path,
                top_n=TOP_N,
            )
            query_engine = RetrieverQueryEngine.from_args(
                retriever=hybrid_retriever,
                node_postprocessors=[rerank],
                llm=Settings.llm,
            )

        return UploadResponse(
            status="ok",
            filename=os.path.basename(save_path),
            chunks=len(all_nodes),
            file_type=file_type,
            message=message,
        )
    except Exception as e:
        return UploadResponse(status="error", filename=file.filename if file else "unknown", chunks=0, file_type="unknown", message=str(e))


@app.get("/health", response_model=HealthResponse, summary="💚 健康检查", tags=["运维接口"])
async def health():
    """
    检查服务是否正常运行，模型是否加载完成。

    **用途**：部署后用于监控探针，确认服务可用。
    """
    return HealthResponse(
        status="running" if query_engine else "initializing",
        model_loaded=query_engine is not None,
    )


@app.get("/files", summary="📂 查看知识库文件列表", tags=["管理接口"])
async def list_files():
    """
    查看当前知识库中所有文件。

    **返回**：文件名列表，包含文件类型、大小、修改时间。
    """
    import time

    os.makedirs(DATA_DIR, exist_ok=True)
    files = []

    for filename in sorted(os.listdir(DATA_DIR)):
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(filepath):
            continue
        if not filename.lower().endswith((".txt", ".pdf", ".xlsx", ".xls")):
            continue

        stat = os.stat(filepath)
        if filename.lower().endswith((".xlsx", ".xls")):
            file_type = "xlsx"
        else:
            file_type = "pdf" if filename.lower().endswith(".pdf") else "txt"
        files.append({
            "filename": filename,
            "file_type": file_type,
            "size_bytes": stat.st_size,
            "size_human": f"{stat.st_size:,} 字节",
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        })

    return {"total": len(files), "files": files}


@app.delete("/files/{filename}", summary="🗑️ 删除知识库文件", tags=["管理接口"])
async def delete_file(filename: str):
    """
    删除知识库中的指定文件，并从零重建索引。

    **处理流程**：
    1. 删除 data/ 中的源文件
    2. 清除向量索引目录（从零重建，避免残留脏数据）
    3. 重新加载剩余文档 → 重建所有索引和查询引擎

    **注意**：重建索引需要约 30 秒，期间 /chat 接口暂不可用。
    """
    global index, bm25_retriever, query_engine, splitter

    try:
        # 安全校验：防止路径穿越
        safe_name = os.path.basename(filename)
        filepath = os.path.join(DATA_DIR, safe_name)

        if not os.path.exists(filepath):
            return JSONResponse(status_code=404, content={
                "status": "error",
                "message": f"文件不存在: {safe_name}",
            })

        # 1. 删除源文件
        os.remove(filepath)

        # 2. 清除向量索引
        if os.path.exists(VECTOR_INDEX_DIR):
            shutil.rmtree(VECTOR_INDEX_DIR)

        # 3. 重新加载剩余文档并重建索引
        documents = _load_documents()

        if documents:
            # 重建向量索引
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=VECTOR_INDEX_DIR)

            # 重建 BM25 索引
            nodes = splitter.get_nodes_from_documents(documents)
            bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=TOP_K)

            # 重建查询引擎
            use_ollama = os.environ.get("USE_OLLAMA", "false").lower() in ("true", "1", "yes")
            vector_retriever = index.as_retriever(similarity_top_k=TOP_K)
            hybrid_retriever = QueryFusionRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                num_queries=1,
                use_async=False,
                similarity_top_k=TOP_K,
            )

            if use_ollama:
                query_engine = RetrieverQueryEngine.from_args(
                    retriever=hybrid_retriever,
                    llm=Settings.llm,
                )
            else:
                rerank = SentenceTransformerRerank(
                    model=os.environ.get("RERANKER_MODEL_PATH", "BAAI/bge-reranker-v2-m3"),
                    top_n=TOP_N,
                )
                query_engine = RetrieverQueryEngine.from_args(
                    retriever=hybrid_retriever,
                    node_postprocessors=[rerank],
                    llm=Settings.llm,
                )

            remaining = len(documents)
            return {"status": "ok", "message": f"已删除「{safe_name}」，索引已重建（剩余 {remaining} 个文档）"}
        else:
            # 知识库已空，重置引擎
            index = None
            bm25_retriever = None
            query_engine = None
            return {"status": "ok", "message": f"已删除「{safe_name}」，知识库已清空"}

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"删除失败: {str(e)}",
        })


# ============================================================
# 辅助函数
# ============================================================

def _load_documents():
    """从 data/ 目录加载所有 .txt 和 .pdf 文件为 Document 对象"""
    documents = []
    os.makedirs(DATA_DIR, exist_ok=True)

    # 文件名 → 元数据标签的映射
    metadata_map = {
        "tax_policy.txt": {"category": "税务政策", "source": "国家税务总局"},
        "weather_sales.txt": {"category": "经营策略", "source": "行业分析报告"},
        "test.txt": {"category": "测试", "source": "本地"},
    }

    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)

        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            # Excel 文件：走 extract_excel 解析
            with open(filepath, "rb") as f:
                xlsx_bytes = f.read()
            try:
                excel_docs = extract_excel(xlsx_bytes)
            except Exception as e:
                print(f"       跳过 Excel {filename}（解析失败: {e}）")
                continue
            if not excel_docs:
                print(f"       跳过 Excel {filename}（解析为空）")
                continue
            for doc in excel_docs:
                doc.metadata["filename"] = filename
                doc.metadata["source"] = "本地Excel"
            documents.extend(excel_docs)
            print(f"       加载文档: {filename} → {len(excel_docs)} 个文档（行摘要+概要）")
        elif filename.endswith(".pdf"):
            # PDF 文件：走 extract_pdf 解析
            with open(filepath, "rb") as f:
                pdf_bytes = f.read()
            try:
                text = extract_pdf(pdf_bytes)
            except Exception as e:
                print(f"       跳过 PDF {filename}（解析失败: {e}）")
                continue
            if not text.strip():
                print(f"       跳过 PDF {filename}（提取为空）")
                continue
            documents.append(Document(
                text=text,
                metadata={"category": "未知", "source": "本地PDF", "filename": filename, "type": "pdf_extract"},
            ))
            print(f"       加载文档: {filename} ({len(text)} 字) → [PDF提取]")
        elif filename.endswith(".txt"):
            # TXT 文件：原有逻辑
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                continue
            meta = metadata_map.get(filename, {"category": "未知", "source": "本地"})
            documents.append(Document(text=text, metadata=meta))
            print(f"       加载文档: {filename} ({len(text)} 字) → [{meta['category']}]")

    return documents


# ============================================================
# 启动服务
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
