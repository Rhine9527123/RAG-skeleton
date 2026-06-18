"""
RAG-Skeleton 知识库服务 - FastAPI 版
====================================

通用 AI 知识库骨架。换一个知识库，就是一个新应用。

启动方式：python server.py
访问地址：http://localhost:8000
API 文档：http://localhost:8000/docs

架构：
  启动时（只做一次）：
    1. 从 config.py（优先）或 .env（环境变量覆盖）读取配置
    2. 加载 Embedding 模型 (bge-small-zh-v1.5)
    3. 加载/构建 向量索引 + BM25 索引
    4. 加载 Reranker 模型 (bge-reranker-v2-m3)
    5. 组装 query_engine（混合检索 + 精排 + LLM）
    6. 启动 HTTP 服务

  收到 /chat 请求时：
    1. 接收用户问题
    2. query_engine.query(question)  ← 内部走：混合检索→精排→LLM
    3. 返回答案 + 来源片段

领域切换：设置环境变量 RAG_DOMAIN=xxx 或直接修改 config.py 预设。
"""
import os
import sys
import json
import io
import requests
import shutil
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

# 加载 .env 文件（API Key 等环境变量）
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("server")

# 导入中心化配置
from config import get_config
from session_store import SessionStore

# Windows 兼容：Git Bash / Docker 环境可能没有 USERNAME，torch 初始化会崩
if sys.platform == "win32" and not os.environ.get("USERNAME"):
    os.environ["USERNAME"] = os.environ.get("USER", "default")

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# 加载配置
_CFG = get_config()

# ── 去重模块 ──
from dedup import get_dedup

# ── BM25 增量索引（LSM-Tree 风格）──
from bm25_store import BM25IndexStore

# ── APScheduler 定时任务 ──
import scheduler as _scheduler_mod

# ── 锚点集路由（LSM-Tree 风格）──
from anchor_manager import AnchorSetManager, create_anchor_manager


def load_config():
    """
    加载配置，优先级：
      1. config.py（中心化配置，自动从环境变量 RAG_DOMAIN 检测领域）
      2. 环境变量（兼容 Docker 模式）
    
    如需切换领域，设置环境变量 RAG_DOMAIN=finance/medical/legal/...
    或直接编辑 config.py 中的 DOMAIN_PRESETS。
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
from llama_index.core.llms import ChatMessage

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
# 多模态依赖 — OCR + STT
# ============================================================
from multimodal import (
    ocr_image,
    transcribe_audio,
    describe_image,
    detect_file_type,
    SUPPORTED_IMAGE_EXTS,
    SUPPORTED_AUDIO_EXTS,
)


# ── 辅助函数 ──
def _assemble_query_engine(idx, bm25_adapter):
    """
    组装混合检索引擎（DRY）。
    从 index + bm25_adapter 构建 QueryFusionRetriever → RetrieverQueryEngine。
    """
    global query_engine

    use_ollama = os.environ.get("USE_OLLAMA", "false").lower() in ("true", "1", "yes")
    vector_retriever = idx.as_retriever(similarity_top_k=TOP_K)
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_adapter],
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

# ============================================================
# 全局变量（启动时初始化，请求时复用）
# ============================================================
query_engine = None  # RAG 查询引擎
bm25_store: "BM25IndexStore" = None  # BM25 索引管理器（LSM-Tree 双层架构）
session_store: "SessionStore" = None  # 多轮对话会话存储
splitter = None  # 文本切分器
index = None  # 向量索引
anchor_mgr: "AnchorSetManager" = None  # 锚点集路由管理器

# 配置
DATA_DIR = "data"
VECTOR_INDEX_DIR = "chroma_data_server"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
TOP_K = 10  # 粗筛数量
TOP_N = 3  # 精排数量
BM25_PERSIST_DIR = "bm25_index"  # BM25 增量索引持久化目录
BM25_MERGE_THRESHOLD = 100  # Delta 节点数达到此值自动合并

# 垃圾桶配置
TRASH_DIR = ".trash"
TRASH_META_FILE = ".trash_meta.json"
TRASH_DAYS = 30  # 过期文档保留天数


def extract_pdf(pdf_bytes, use_paddle_ocr=True):
    """
    从 PDF 字节流提取干净文本（增强版）
    四步走：表格识别(含列头) → 纯文本提取 → 嵌入图片OCR → OCR扫描件回退

    参数：
        pdf_bytes: PDF 文件字节流
        use_paddle_ocr: True=优先 PaddleOCR（更好中文识别），False=只用 tesseract

    返回：字符串
    """
    all_text_parts = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # 第一步：优先提取表格（含列头标注，方便 RAG 理解表格结构）
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

            # 如果第一行可能是列头（包含多个非空单元格），标记列头名
            lines = []
            header_done = False
            for row in cleaned:
                meaningful = [cell for cell in row if cell != ""]
                if meaningful:
                    line = " | ".join(meaningful)
                    # 第一行作为列头标注
                    if not header_done and len(meaningful) >= 2:
                        line = f"[表格列头] {line}"
                        header_done = True
                    lines.append(line)

            if lines:
                all_text_parts.append("\n".join(lines))

    # 第二步：如果没识别到表格，回退到纯文本提取
    if not all_text_parts:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            if text:
                all_text_parts.append(text)

    # 第三步：提取嵌入图片并 OCR（PDF 中直接嵌入的图片）
    embedded_image_texts = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                # 只处理常见图片格式，跳过太小的图（可能是图标/装饰）
                if len(image_bytes) < 4096:
                    continue
                if image_ext not in ("png", "jpeg", "jpg", "bmp"):
                    continue

                # OCR 图片
                try:
                    import io as _io
                    img = Image.open(_io.BytesIO(image_bytes))
                    w, h = img.size
                    if w < 100 or h < 100:
                        continue  # 太小的图跳过

                    # 尝试 PaddleOCR 或 tesseract
                    if use_paddle_ocr:
                        try:
                            from multimodal import ocr_image
                            img_text = ocr_image(image_bytes)
                        except Exception:
                            img_text = pytesseract.image_to_string(img, lang="chi_sim+eng").strip()
                    else:
                        img_text = pytesseract.image_to_string(img, lang="chi_sim+eng").strip()

                    if img_text:
                        embedded_image_texts.append(
                            f"[PDF第{page_num+1}页嵌入图片] {img_text}"
                        )
                except Exception:
                    pass  # 单张图片 OCR 失败不中断整体
            except Exception:
                pass  # 图片提取失败跳过

    if embedded_image_texts:
        all_text_parts.append("[PDF嵌入图片内容]")
        all_text_parts.extend(embedded_image_texts)

    doc.close()

    # 第四步：如果纯文本提取也为空，说明是扫描件，走整页 OCR
    combined_text = "\n\n".join(all_text_parts)
    if not combined_text.strip():
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        ocr_texts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # PaddleOCR 优先
            if use_paddle_ocr:
                try:
                    import io as _io
                    buf = _io.BytesIO()
                    img.save(buf, format="PNG")
                    img_bytes_png = buf.getvalue()
                    from multimodal import ocr_image
                    text = ocr_image(img_bytes_png)
                except Exception:
                    text = pytesseract.image_to_string(img, lang="chi_sim+eng").strip()
            else:
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
                "category": _CFG.excel_category,
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
                "category": _CFG.excel_summary_category,
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
    global query_engine, bm25_store, splitter, index

    print("=" * 60)
    print(f"[启动] {_CFG.app_name}")
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
            temperature=0.1,
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

    # ---- 4. 构建 BM25 索引（LSM-Tree 双层架构）----
    print("[4/7] 构建 BM25 索引（增量模式）...")
    nodes = splitter.get_nodes_from_documents(documents)
    bm25_store = BM25IndexStore(
        persist_dir=BM25_PERSIST_DIR,
        merge_threshold=BM25_MERGE_THRESHOLD,
        similarity_top_k=TOP_K,
    ).initialize(nodes)
    print(f"       [OK] BM25 索引就绪（{len(bm25_store.all_nodes)} 个片段, "
          f"Delta={len(bm25_store._delta_nodes)}）")

    # ---- 5. 锚点集路由管理器 ----
    global anchor_mgr
    print("[5/7] 初始化锚点集路由管理器...")
    # 从已加载的文档中提取文本用于构建锚点集
    doc_texts = [doc.text for doc in documents if doc.text and doc.text.strip()]
    anchor_mgr = AnchorSetManager(
        anchor_file=os.path.join(DATA_DIR, "anchor_set.json"),
        rebuild_threshold=20,
        route_threshold=2,
    )
    anchor_mgr.initialize(doc_texts)
    mgr_stats = anchor_mgr.stats()
    print(f"       [OK] 锚点集就绪（{mgr_stats['anchor_count']} 个锚点, "
          f"来自 {mgr_stats['total_docs_scanned']} 篇文档）")

    # ---- 6. 组装查询引擎 ----
    print("[6/7] 组装查询引擎...")
    from bm25_store import BM25StoreRetrieverAdapter
    bm25_adapter = BM25StoreRetrieverAdapter(bm25_store)
    _assemble_query_engine(index, bm25_adapter)

    # ---- 6b. 初始化多轮对话引擎 ----
    global session_store
    if _CFG.session_enabled:
        print("[6b/7] 初始化多轮对话引擎...")
        session_store = SessionStore(
            db_path=_CFG.session_db_path or None,
            max_turns=_CFG.session_max_turns,
        )
        print(f"       [OK] 会话存储就绪（窗口大小={_CFG.session_max_turns}轮）")
    else:
        session_store = None
        print("       [跳过] 多轮对话已禁用")

    # ---- 7. 启动完成 ----
    print("[7/7] 服务启动完成！")
    print("=" * 60)
    print("  接口文档: http://localhost:8000/docs")
    print("  聊天接口: POST http://localhost:8000/chat")
    print("=" * 60)

    # ---- 启动后自动清理过期垃圾桶文件 ----
    import time as _time
    meta = _load_trash_meta()
    if meta:
        now = _time.time()
        expired = [f for f, i in meta.items() if (now - i["deleted_at"]) / 86400 >= TRASH_DAYS]
        if expired:
            print(f"\n🧹 发现 {len(expired)} 个过期垃圾桶文件，正在清理...")
            for filename in expired:
                trash_path = os.path.join(TRASH_DIR, filename)
                if os.path.exists(trash_path):
                    os.remove(trash_path)
                meta.pop(filename, None)
                print(f"   🗑️  已永久删除: {filename}")
            _save_trash_meta(meta)
            remaining = len(meta)
            print(f"   ✅ 清理完成，垃圾桶剩余 {remaining} 个文件（未过期）\n")
        else:
            print(f"\n📦 垃圾桶中有 {len(meta)} 个文件，均未过期\n")

    # ── 启动 APScheduler 定时任务 ──
    print("\n[定时任务] 启动 APScheduler...")
    try:
        started = _scheduler_mod.start_scheduler()
        if started:
            print("         [OK] APScheduler 已启动（爬虫调度 + 旧知识清理）")
        else:
            print("         [跳过] APScheduler 未启用")
    except Exception as e:
        print(f"         [警告] APScheduler 启动失败: {e}")

    # 等待请求...
    yield

    # 关闭时清理
    print("[关闭] 服务正在停止...")
    _scheduler_mod.stop_scheduler()


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title=_CFG.app_title,
    description=f"""
    ## 📚 {_CFG.app_title}

    {_CFG.app_description}

    ### 工作流程
    1. 用户提问
    2. 混合检索（向量 + BM25）→ 粗筛 TOP_K 条
    3. Reranker 精排（可选）→ 保留 TOP_N 条
    4. LLM 基于精排结果生成答案

    ### 技术栈
    - **Embedding**: BAAI/bge-small-zh-v1.5
    - **Reranker**: BAAI/bge-reranker-v2-m3（本地，可选）
    - **LLM**: DeepSeek（在线）/ Ollama（本地离线）
    - **检索**: 向量检索 + BM25 混合检索
    
    ### 领域配置
    当前领域预设：{os.environ.get('RAG_DOMAIN', '通用')}
    切换领域：设置环境变量 RAG_DOMAIN=finance/medical/legal/...
    """,
    version="0.2.0",
    lifespan=lifespan,
)


# ============================================================
# API 接口
# ============================================================

class ChatRequest(BaseModel):
    """聊天请求"""
    question: str = "用户的问题"
    session_id: Optional[str] = None  # 多轮对话会话ID（空=新建会话）
    top_k: Optional[int] = None  # 可选：临时覆盖粗筛数量（默认10）
    top_n: Optional[int] = None  # 可选：临时覆盖精排数量（默认3）
    verify: bool = False  # 是否启用 FactGuard 事实核查（RAG 回答自动过审）
    verify_scenario: str = "balanced"  # 核查严格度: conservative | balanced | relaxed

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "请根据知识库回答我的问题",
                    "session_id": "abc12345",
                    "top_k": 10,
                    "top_n": 3,
                    "verify": False,
                },
            ]
        }
    }


class ChatResponse(BaseModel):
    """聊天响应"""
    question: str = "用户的问题"
    answer: str = "AI 生成的回答"
    sources: list = "来源片段列表（包含原文、相关度分数、元数据）"
    session_id: Optional[str] = None  # 多轮对话会话ID
    history_count: int = 0  # 历史对话轮数
    route_info: Optional[dict] = None  # 锚点路由信息 {route, hits, threshold, tokens}
    factcheck: Optional[dict] = None  # FactGuard 核查结果（仅 verify=true 时有值）

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "这个文件讲了什么？",
                    "answer": "根据知识库中的资料...",
                    "sources": [
                        {
                            "text": "相关内容片段...",
                            "score": 0.8921,
                            "metadata": {"category": "资料", "filename": "example.txt"},
                        }
                    ],
                    "session_id": "abc12345",
                    "history_count": 3,
                    "factcheck": None,
                }
            ]
        }
    }


class UploadResponse(BaseModel):
    """上传响应"""
    status: str = "上传结果：ok / error"
    filename: str = "文件名"
    chunks: int = "切分后的文本片段数量"
    file_type: str = "文件类型：txt / pdf / xlsx / image / audio"
    message: str = "附加信息（如 OCR 识别结果预览、音频转录结果等）"
    preview: Optional[str] = None  # 文本预览（OCR 文字 / 转写文字前200字）


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "服务状态：running / initializing"
    model_loaded: bool = "模型是否加载完成"


# ─── FactGuard 核查辅助函数 ─────────────────────
FACTGUARD_URL = "http://localhost:8001"

async def _factguard_verify(text: str, scenario: str = "balanced") -> Optional[dict]:
    """调用 FactGuard 对 RAG 回答做事实核查，返回精简结果"""
    try:
        resp = await asyncio.to_thread(
            requests.post,
            f"{FACTGUARD_URL}/full",
            json={"document": text, "scenario": scenario, "hybrid": True, "segment_size": 1},
            timeout=60,
        )
        if resp.status_code != 200:
            return {"error": f"FactGuard 返回 {resp.status_code}", "available": False}
        data = resp.json()
        summary = data.get("summary", {})
        reviews = data.get("reviews", [])

        # 精简返回：只保留关键信息
        issues = []
        for r in reviews:
            if r.get("final_verdict") != "factual":
                issues.append({
                    "anchor_id": r.get("anchor_id"),
                    "verdict": r.get("final_verdict"),
                    "encoder": r.get("encoder_verdict"),
                    "decoder": r.get("decoder_verdict"),
                    "reason": r.get("encoder_reason") or r.get("decoder_reason"),
                })

        return {
            "available": True,
            "total_anchors": summary.get("total", 0),
            "factual": summary.get("factual", 0),
            "unfactual": summary.get("unfactual", 0),
            "uncertain": summary.get("uncertain", 0),
            "accuracy": summary.get("accuracy", 0),
            "risk_level": summary.get("risk_level", "未知"),
            "issues": issues,
            "verdict": "pass" if summary.get("unfactual", 0) == 0 else "warn",
        }
    except requests.exceptions.ConnectionError:
        return {"error": "FactGuard 服务未启动 (端口8001)", "available": False}
    except Exception as e:
        return {"error": f"核查异常: {str(e)}", "available": False}


@app.post("/chat", response_model=ChatResponse, summary="💬 RAG 问答（支持多轮对话）", tags=["核心接口"])
async def chat(request: ChatRequest):
    """
    核心接口：用户提问，AI 基于知识库回答。

    **支持多轮对话**：
    - 传 `session_id` 继续已有会话（自动加载历史上下文）
    - 不传 `session_id` 新建会话（首次提问）
    - 传 `session_id=""` 无状态模式（向后兼容）

    **处理流程**：用户问题 → 锚点路由判断 → 加载历史(可选) → 混合检索(向量+BM25) → Reranker精排 → LLM生成答案
    → (可选) FactGuard 事实核查 → 保存记录

    **锚点路由**：自动判断问题中是否包含知识库高频锚点，命中不足时走 Agentic RAG（多轮改写检索）。
    **事实核查**：传 `verify=true` 启用，会自动调用 FactGuard 对 RAG 回答做 Encoder+Decoder 双路审查
    **返回**：答案 + 来源片段 + 会话ID + 历史轮数 + route_info(路由信息) + factcheck(可选)
    """
    try:
        # ── 会话管理 ──
        sid = None
        history_count = 0
        if session_store and request.session_id is not None:
            sid = request.session_id
            if not sid:
                # session_id="" 表示无状态模式，不保存历史
                pass
            elif not session_store.get_session(sid):
                # session_id 不存在？新建（自动容错）
                sid = session_store.create_session()
            # 保存用户消息（等回答出来后一起更新）
        elif session_store and request.session_id is None:
            # 首次提问，自动创建新会话
            sid = session_store.create_session()
            # 自动生成标题（取问题前20字）
            title = request.question[:20].rstrip("吗？。！？，")
            if len(title) > 3:
                session_store.update_title(sid, title)

        # ── 锚点路由判断 ──
        route = "fast"
        route_hits = 0
        route_tokens = []
        if anchor_mgr is not None:
            route, route_hits, route_tokens = anchor_mgr.route(request.question)
            if route == "agentic":
                logger.info(f"[Route] → Agentic RAG (命中 {route_hits}/{anchor_mgr.route_threshold} 锚点)")
            else:
                logger.debug(f"[Route] → Fast RAG (命中 {route_hits} 锚点: {route_tokens[:5]})")

        # ── 执行 RAG 检索（手动流程，与流式端点一致）──
        loop = asyncio.get_event_loop()
        nodes = await loop.run_in_executor(
            None, query_engine.retriever.retrieve, request.question
        )

        # Reranker 精排
        postprocessors = getattr(query_engine, "_node_postprocessors", [])
        if postprocessors:
            from llama_index.core import QueryBundle
            qb = QueryBundle(request.question)
            for pp in postprocessors:
                nodes = await loop.run_in_executor(
                    None, pp.postprocess_nodes, nodes, qb
                )

        # 提取来源片段
        sources = []
        for node in nodes:
            score = float(node.score) if node.score is not None else None
            sources.append({
                "text": node.text[:200],
                "score": round(score, 4) if score else None,
                "metadata": dict(node.metadata) if node.metadata else {},
            })

        # ── Agentic RAG: 锚点命中不足 + 检索质量低 → 追问 ──
        top_score = sources[0]["score"] if sources and sources[0].get("score") else 0.0
        needs_clarification = False
        if route == "agentic" and top_score < 0.3:
            needs_clarification = True
            topic_hints = anchor_mgr.get_topic_hints(8) if anchor_mgr else []
            hints_str = "、".join(topic_hints[:6]) if topic_hints else "未知"
            answer = (
                f"🤔 您的问题「{request.question}」与知识库的匹配度较低"
                f"（锚点命中 {route_hits}/{anchor_mgr.route_threshold if anchor_mgr else 2}，"
                f"最高相关度 {top_score:.1%}）。\n\n"
                f"当前知识库主要涵盖以下主题：**{hints_str}**\n\n"
                f"请您尝试：\n"
                f"1. 换一个更具体的问法（例如包含上述关键词）\n"
                f"2. 或者直接告诉我您想了解哪个方面的内容"
            )
            logger.info(f"[Agentic] 触发追问: top_score={top_score:.4f}, hints={topic_hints[:4]}")
        # ── 检索结果为空 ──
        elif not sources:
            answer = "⚠️ 知识库中未找到与您问题相关的内容。请尝试换一种问法，或上传相关文档到知识库。"
        else:
            # ── 正常 RAG 回答 ──
            context_parts = []
            for i, node in enumerate(nodes, 1):
                context_parts.append(f"[资料{i}]\n{node.text[:1000]}")
            context = "\n\n".join(context_parts)

            # 构建 messages：System + 历史 + 当前
            messages = [ChatMessage(role="system", content=_CFG.system_prompt)]

            if session_store and sid:
                history = session_store.get_history(sid)
                for msg in history:
                    messages.append(ChatMessage(role=msg["role"], content=msg["content"]))

            messages.append(ChatMessage(
                role="user",
                content=f"参考资料：\n{context}\n\n用户问题：{request.question}\n\n请严格基于上述参考资料回答，不要发散或添加资料中没有的信息。回答要简洁直接。如果资料中未找到相关内容，请明确说明"资料中未找到相关内容"：",
            ))

            # 调用 LLM（非流式）
            llm_response = await loop.run_in_executor(
                None, lambda: Settings.llm.chat(messages)
            )
            answer = llm_response.message.content if hasattr(llm_response, "message") else str(llm_response)

        # ── FactGuard 事实核查（可选）──
        factcheck = None
        if request.verify:
            factcheck = await _factguard_verify(answer, request.verify_scenario)

        # ── 保存对话记录 ──
        if session_store and sid:
            session_store.add_message(sid, "user", request.question)
            session_store.add_message(sid, "assistant", answer, sources)
            session_info = session_store.get_session(sid)
            history_count = session_info["message_count"] // 2 if session_info else 0

        return ChatResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            session_id=sid or "",
            history_count=history_count,
            route_info={
                "route": route,
                "hits": route_hits,
                "threshold": anchor_mgr.route_threshold if anchor_mgr else 2,
                "tokens": route_tokens[:10],
                "needs_clarification": needs_clarification,
            },
            factcheck=factcheck,
        )
    except Exception as e:
        # 路由信息在异常前已完成，保留以便调试
        error_ri = None
        try:
            error_ri = {
                "route": route,
                "hits": route_hits,
                "threshold": anchor_mgr.route_threshold if anchor_mgr else 2,
                "tokens": (route_tokens or [])[:10],
            }
        except Exception:
            pass
        return ChatResponse(
            question=request.question,
            answer=f"服务内部错误：{str(e)}",
            sources=[],
            session_id=request.session_id or "",
            history_count=0,
            route_info=error_ri,
        )


@app.post("/chat/stream", summary="💬 RAG 流式问答 (SSE，支持多轮对话)", tags=["核心接口"])
async def chat_stream(request: ChatRequest):
    """
    流式问答接口：逐步返回检索各阶段的进度，再逐 token 推送 AI 生成答案。

    **支持多轮对话**：
    - 传 `session_id` 继续已有会话（自动加载历史上下文）
    - 不传则自动创建新会话
    - 传 `session_id=""` 无状态模式

    SSE 事件格式：
    - data: {"type": "session", "session_id": "..."}  — 会话ID
    - data: {"type": "route_info", "route": "fast"|"agentic", "hits": N, "threshold": N, "tokens": [...]}  — 锚点路由结果
    - data: {"type": "progress", "stage": "...", "message": "..."}
    - data: {"type": "sources", "sources": [...]}
    - data: {"type": "token", "content": "..."}
    - data: {"type": "history_count", "count": N}
    - data: {"type": "done"}
    - data: {"type": "error", "message": "..."}

    前端：Streamlit st.write_stream() 消费 token 流（需过滤非 token 事件）
    降级：流式失败时自动回退到非流式 /chat
    """

    async def generate():
        # ── 会话管理 ──
        sid = None
        try:
            if session_store and request.session_id is not None:
                sid = request.session_id
                if not sid:
                    pass  # 无状态模式
                elif not session_store.get_session(sid):
                    sid = session_store.create_session()
                    title = request.question[:20].rstrip("吗？。！？，")
                    if len(title) > 3:
                        session_store.update_title(sid, title)
            elif session_store and request.session_id is None:
                sid = session_store.create_session()
                title = request.question[:20].rstrip("吗？。！？，")
                if len(title) > 3:
                    session_store.update_title(sid, title)
        except Exception:
            pass

        # 发送 session_id（如果有）
        if sid:
            yield f"data: {json.dumps({'type': 'session', 'session_id': sid}, ensure_ascii=False)}\n\n"

        try:
            loop = asyncio.get_event_loop()

            # ── 锚点路由判断 ──
            route = "fast"
            route_hits = 0
            route_tokens = []
            if anchor_mgr is not None:
                route, route_hits, route_tokens = anchor_mgr.route(request.question)
            yield f"data: {json.dumps({'type': 'route_info', 'route': route, 'hits': route_hits, 'threshold': anchor_mgr.route_threshold if anchor_mgr else 2, 'tokens': route_tokens[:10]}, ensure_ascii=False)}\n\n"

            # ── 阶段 1: 向量检索 ──
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'vector', 'message': '🔍 向量检索中...'}, ensure_ascii=False)}\n\n"

            nodes = await loop.run_in_executor(
                None, query_engine.retriever.retrieve, request.question
            )

            # ── 阶段 2: Reranker 精排 ──
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'rerank', 'message': '⚙️ Reranker 精排中...'}, ensure_ascii=False)}\n\n"

            postprocessors = getattr(query_engine, "_node_postprocessors", [])
            if postprocessors:
                from llama_index.core import QueryBundle
                qb = QueryBundle(request.question)
                for pp in postprocessors:
                    nodes = await loop.run_in_executor(
                        None, pp.postprocess_nodes, nodes, qb
                    )

            # 构建 sources 数据
            sources_data = []
            for node in nodes:
                score = float(node.score) if node.score is not None else None
                sources_data.append({
                    "text": node.text[:200],
                    "score": round(score, 4) if score else None,
                    "metadata": dict(node.metadata) if node.metadata else {},
                })

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_data}, ensure_ascii=False)}\n\n"

            # ── Agentic RAG: 锚点命中不足 + 检索质量低 → 追问 ──
            top_score = sources_data[0]["score"] if sources_data and sources_data[0].get("score") else 0.0
            if route == "agentic" and top_score < 0.3:
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'agentic', 'message': '🤔 问题匹配度低，触发追问...'}, ensure_ascii=False)}\n\n"

                topic_hints = anchor_mgr.get_topic_hints(8) if anchor_mgr else []
                hints_str = "、".join(topic_hints[:6]) if topic_hints else "未知"
                clarification = (
                    f"🤔 您的问题「{request.question}」与知识库的匹配度较低"
                    f"（锚点命中 {route_hits}/{anchor_mgr.route_threshold if anchor_mgr else 2}，"
                    f"最高相关度 {top_score:.1%}）。\n\n"
                    f"当前知识库主要涵盖以下主题：**{hints_str}**\n\n"
                    f"请您尝试：\n"
                    f"1. 换一个更具体的问法（例如包含上述关键词）\n"
                    f"2. 或者直接告诉我您想了解哪个方面的内容"
                )

                # 逐字推送追问文本
                for char in clarification:
                    yield f"data: {json.dumps({'type': 'token', 'content': char}, ensure_ascii=False)}\n\n"

                # 保存对话记录
                history_count = 0
                if session_store and sid:
                    session_store.add_message(sid, "user", request.question)
                    session_store.add_message(sid, "assistant", clarification, sources_data)
                    session_info = session_store.get_session(sid)
                    if session_info:
                        history_count = session_info["message_count"] // 2

                if sid:
                    yield f"data: {json.dumps({'type': 'history_count', 'count': history_count}, ensure_ascii=False)}\n\n"

                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            # ── 阶段 3: 构建提示词（含历史上下文） ──
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'prompt', 'message': '🧠 构建提示词中...'}, ensure_ascii=False)}\n\n"

            context_parts = []
            for i, node in enumerate(nodes, 1):
                context_parts.append(f"[资料{i}]\n{node.text[:1000]}")
            context = "\n\n".join(context_parts)

            system_prompt = _CFG.system_prompt

            # 构建 messages：System + 历史 + 当前
            messages = [ChatMessage(role="system", content=system_prompt)]

            # 插入对话历史
            if session_store and sid:
                history = session_store.get_history(sid)
                for msg in history:
                    messages.append(ChatMessage(role=msg["role"], content=msg["content"]))

            # 当前问题
            messages.append(ChatMessage(
                role="user",
                content=f"参考资料：\n{context}\n\n用户问题：{request.question}\n\n请严格基于上述参考资料回答，不要发散或添加资料中没有的信息。回答要简洁直接。如果资料中未找到相关内容，请明确说明"资料中未找到相关内容"：",
            ))

            # ── 阶段 4: 流式调用 LLM ──
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'llm', 'message': '🤖 AI 生成中...'}, ensure_ascii=False)}\n\n"

            full_answer = ""
            response_gen = Settings.llm.stream_chat(messages)
            for chunk in response_gen:
                delta = getattr(chunk, "delta", None)
                if not delta:
                    msg = getattr(chunk, "message", None)
                    delta = getattr(msg, "content", None) if msg else None
                if delta:
                    full_answer += delta
                    yield f"data: {json.dumps({'type': 'token', 'content': delta}, ensure_ascii=False)}\n\n"

            # ── 保存对话记录 ──
            history_count = 0
            if session_store and sid:
                session_store.add_message(sid, "user", request.question)
                session_store.add_message(sid, "assistant", full_answer, sources_data)
                session_info = session_store.get_session(sid)
                if session_info:
                    history_count = session_info["message_count"] // 2

            # 发送历史计数
            if sid:
                yield f"data: {json.dumps({'type': 'history_count', 'count': history_count}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/upload", response_model=UploadResponse, summary="📤 上传知识文件（支持多模态）", tags=["管理接口"])
async def upload_file(
    file: UploadFile = File(..., description="知识文件（支持 .txt / .pdf / .xlsx / 图片 / 音频）"),
    category: str = Form(_CFG.default_category, description="分类标签"),
    force: bool = Form(False, description="强制上传，跳过去重检查"),
):
    """
    上传新的知识文件到知识库（支持 txt、pdf、xlsx、图片、音频）。

    **处理流程**：
    - txt 文件：直接读取文本 → 切片 → 写入索引
    - pdf 文件：PyMuPDF 解析（表格优先 → 纯文本回退 → OCR 扫描件兜底）→ 切片 → 写入索引
    - xlsx 文件：pandas 解析（行摘要 + 概要，每个 Sheet 生成两份文档）→ 写入索引
    - 图片文件：OCR 识别 → 提取文字 → 入索引
    - 音频文件：STT 语音转文字 → 入索引

    **参数说明**：
    - `file`: 文件（multipart/form-data）
    - `category`: 分类标签
    """
    global index, bm25_store, query_engine

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
            # 暂存，后面统一走增量索引更新
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
        elif detect_file_type(filename) == "image":
            # 图片文件：OCR 识别 → 提取文字 → 保存为 .txt 便于索引
            preview = None
            try:
                text = describe_image(file_bytes, lang=_CFG.ocr_language)
                if not text.strip():
                    os.remove(save_path)
                    return UploadResponse(status="error", filename=filename, chunks=0, file_type="image", message="OCR 未识别到文字")
                preview = text[:200]
                # 保存 OCR 文字为同名的 .txt 文件
                txt_filename = os.path.splitext(save_path)[0] + ".txt"
                with open(txt_filename, "w", encoding="utf-8") as f:
                    f.write(text)
                message = f"图片 OCR 识别成功（{len(text)} 字）"
                file_type = "image"
            except Exception as e:
                os.remove(save_path)
                return UploadResponse(status="error", filename=filename, chunks=0, file_type="image", message=f"图片 OCR 失败: {str(e)}")
        elif detect_file_type(filename) == "audio":
            # 音频文件：STT 语音转文字 → 保存为 .txt 便于索引
            preview = None
            try:
                text = transcribe_audio(file_bytes, model_size=_CFG.stt_model_size, language=_CFG.stt_language)
                if not text.strip():
                    os.remove(save_path)
                    return UploadResponse(status="error", filename=filename, chunks=0, file_type="audio", message="语音转录为空")
                preview = text[:200]
                # 保存转写文字为同名的 .txt 文件
                txt_filename = os.path.splitext(save_path)[0] + ".txt"
                with open(txt_filename, "w", encoding="utf-8") as f:
                    f.write(text)
                message = f"语音转录成功（{len(text)} 字）"
                file_type = "audio"
            except Exception as e:
                os.remove(save_path)
                return UploadResponse(status="error", filename=filename, chunks=0, file_type="audio", message=f"语音转录失败: {str(e)}")
        else:
            os.remove(save_path)
            return UploadResponse(status="error", filename=filename, chunks=0, file_type="unknown", message="不支持的文件格式，仅支持 .txt / .pdf / .xlsx / 图片(.png/.jpg) / 音频(.wav/.mp3)")

        # ── 去重检查 ──
        # 收集用于去重的文本（统一处理不同文件类型）
        if file_type == "xlsx":
            dedup_text = "\n".join(doc.text for doc in new_docs)
        else:
            dedup_text = text

        dedup_mgr = get_dedup("dedup.db")
        if not force:
            is_dup, reason, existing = dedup_mgr.check(
                content=dedup_text,
                filename=os.path.basename(save_path),
            )
            if is_dup:
                os.remove(save_path)
                # xlsx 还会产生额外的 txt 文件吗？当前代码不会
                return UploadResponse(
                    status="error",
                    filename=filename,
                    chunks=0,
                    file_type=file_type,
                    message=f"去重拦截：{reason}。如需强制上传，请设置 force=true",
                )

        # ── 增量索引更新 ──
        # 加载所有文档 → 切分 → 重建向量索引 + 增量更新 BM25
        all_documents = _load_documents()
        all_nodes = splitter.get_nodes_from_documents(all_documents)

        # 向量索引：仍用全量重建（保证索引与磁盘一致，后续可优化为 insert_nodes）
        global index, bm25_store, query_engine
        index = VectorStoreIndex(all_nodes)
        index.storage_context.persist(persist_dir=VECTOR_INDEX_DIR)

        # BM25：增量模式 —— 找到新增文件的节点，只加到 Delta
        # （不是全量重建！这就是 LSM-Tree 优化的核心）
        new_filename = os.path.basename(save_path)
        new_file_nodes = [n for n in all_nodes if n.metadata.get("filename") == new_filename]
        if new_file_nodes:
            merged = bm25_store.add_delta(new_file_nodes)
            if merged:
                print(f"[BM25] Delta 达到阈值，已自动合并到 Base（Base={len(bm25_store._base_nodes)} 节点）")
            else:
                print(f"[BM25] Delta +{len(new_file_nodes)} 节点（Delta={len(bm25_store._delta_nodes)}）")

        # 重建 query_engine（向量索引变了）
        from bm25_store import BM25StoreRetrieverAdapter
        bm25_adapter = BM25StoreRetrieverAdapter(bm25_store)
        _assemble_query_engine(index, bm25_adapter)

        # ── 写入去重记录 ──
        dedup_mgr = get_dedup("dedup.db")
        dedup_mgr.add(
            content=dedup_text,
            filename=os.path.basename(save_path),
            file_size=len(file_bytes),
        )

        # ── 喂锚点管理器（LSM-Tree：新文档进 pending buffer）──
        if anchor_mgr is not None and dedup_text:
            anchor_mgr.add_document(dedup_text)
            logger.debug(f"[AnchorSet] 新文档已进 pending buffer")

        return UploadResponse(
            status="ok",
            filename=os.path.basename(save_path),
            chunks=len(all_nodes),
            file_type=file_type,
            message=message,
            preview=preview if file_type in ("image", "audio") else None,
        )
    except Exception as e:
        return UploadResponse(status="error", filename=file.filename if file else "unknown", chunks=0, file_type="unknown", message=str(e))


# ============================================================
# 语音转写 + RAG 问答接口
# ============================================================

class TranscribeRequest(BaseModel):
    """语音转写请求"""
    question: Optional[str] = None  # 可选：转写后立即对文字提问


class TranscribeResponse(BaseModel):
    """语音转写响应"""
    status: str = "ok / error"
    text: str = "转写后的文字"
    language: Optional[str] = None  # 检测到的语言
    answer: Optional[str] = None    # 如果提供了 question，RAG 回答
    sources: Optional[list] = None  # 回答来源


@app.post("/transcribe", response_model=TranscribeResponse, summary="🎤 语音转文字 + RAG 问答", tags=["核心接口"])
async def transcribe(
    file: UploadFile = File(..., description="音频文件（.wav / .mp3 / .m4a / .ogg）"),
    question: Optional[str] = Form(None, description="可选：转写后对文本提的问题"),
):
    """
    上传音频，自动转写为文字，并可立即在知识库中检索回答。

    **处理流程**：
    1. 接收音频文件 → faster-whisper STT 转写
    2. 如果提供了 question → 用转写文字 + 知识库进行 RAG 问答
    3. 返回转写结果（+ 回答，如果有）

    **使用场景**：
    - 语音消息 → 自动转文字 → 直接提问
    - 会议录音 → 转文字 → 查知识库
    """
    try:
        file_bytes = await file.read()

        # 1. 语音转文字（从配置读取模型设置）
        text = transcribe_audio(file_bytes, model_size=_CFG.stt_model_size, language=_CFG.stt_language)

        if not text.strip():
            return TranscribeResponse(
                status="error",
                text="",
                message="语音转录为空（可能是静音或格式不兼容）",
            )

        # 2. 如果提供了 question，用转写文字 + RAG 回答
        answer = None
        sources = None
        if question and query_engine:
            try:
                # 拼接转写文字作为上下文 + 问题
                full_query = f"以下是一段语音转写的内容：\n\n{text}\n\n基于以上内容和知识库，回答问题：{question}"
                response = query_engine.query(full_query)
                answer = str(response)
                if hasattr(response, "source_nodes"):
                    sources = []
                    for node in response.source_nodes:
                        score = float(node.score) if node.score is not None else None
                        sources.append({
                            "text": node.text[:200],
                            "score": round(score, 4) if score else None,
                            "metadata": dict(node.metadata) if node.metadata else {},
                        })
            except Exception as e:
                answer = f"RAG 问答失败: {str(e)}"

        return TranscribeResponse(
            status="ok",
            text=text,
            answer=answer,
            sources=sources,
        )

    except Exception as e:
        return TranscribeResponse(
            status="error",
            text="",
            message=str(e),
        )


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


@app.get("/anchor/stats", summary="🎯 锚点路由统计", tags=["运维接口"])
async def anchor_stats():
    """
    查看锚点集路由管理器的运行状态。

    **返回**：锚点总数、pending 文档数、路由阈值等。
    """
    if anchor_mgr is None:
        return {"status": "not_initialized"}
    return {"status": "ok", **anchor_mgr.stats()}


@app.post("/anchor/rebuild", summary="🔄 强制重建锚点集", tags=["运维接口"])
async def anchor_rebuild():
    """
    强制立即重建锚点集（忽略 pending 阈值）。
    通常在知识库大量更新后手动触发。
    """
    global anchor_mgr
    if anchor_mgr is None:
        return {"status": "not_initialized"}
    documents = _load_documents()
    doc_texts = [doc.text for doc in documents if doc.text and doc.text.strip()]
    anchor_mgr.force_rebuild(doc_texts)
    return {"status": "ok", **anchor_mgr.stats()}


@app.get("/anchor/test", summary="🧪 测试锚点路由", tags=["运维接口"])
async def anchor_test(q: str = "请根据知识库回答我的问题"):
    """
    测试一句话的路由判断结果（调试用）。

    **参数**：q — 要测试的问题
    **返回**：路由决策 + 命中的锚点词
    """
    if anchor_mgr is None:
        return {"status": "not_initialized"}
    route, hits, tokens = anchor_mgr.route(q)
    return {
        "question": q,
        "route": route,
        "hits": hits,
        "threshold": anchor_mgr.route_threshold,
        "tokens": tokens,
        "total_anchors": len(anchor_mgr.anchor_set),
    }


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

        ext = filename.lower()
        if ext.endswith((".txt", ".pdf", ".xlsx", ".xls")):
            pass  # 已有类型
        elif detect_file_type(filename) == "image":
            pass  # 图片
        elif detect_file_type(filename) == "audio":
            pass  # 音频
        else:
            continue

        stat = os.stat(filepath)

        # 判断文件类型
        ft = detect_file_type(filename)
        if ft == "image":
            file_type = "image"
        elif ft == "audio":
            file_type = "audio"
        elif ext.endswith((".xlsx", ".xls")):
            file_type = "xlsx"
        else:
            file_type = "pdf" if ext.endswith(".pdf") else "txt"
        files.append({
            "filename": filename,
            "file_type": file_type,
            "size_bytes": stat.st_size,
            "size_human": f"{stat.st_size:,} 字节",
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        })

    return {"total": len(files), "files": files}


@app.delete("/files/{filename}", summary="🗑️ 移入垃圾桶（30天保留）", tags=["管理接口"])
async def delete_file(filename: str):
    """
    将文件移入垃圾桶（软删除），30天后自动永久删除。

    **处理流程**：
    1. 将文件从 data/ 移动到 .trash/
    2. 记录删除时间戳到 .trash_meta.json
    3. 重建索引（文件已从 data/ 移除）

    **恢复方法**：POST /trash/{filename}/restore
    **永久删除**：DELETE /trash/{filename}
    **自动清理**：POST /trash/auto-clean（或启动时自动执行）
    """
    try:
        safe_name = os.path.basename(filename)
        filepath = os.path.join(DATA_DIR, safe_name)

        if not os.path.exists(filepath):
            return JSONResponse(status_code=404, content={
                "status": "error",
                "message": f"文件不存在: {safe_name}",
            })

        # 1. 移动文件到垃圾桶
        trash_path = os.path.join(TRASH_DIR, safe_name)
        shutil.move(filepath, trash_path)

        # 2. 记录元数据
        import time
        stat = os.stat(trash_path)
        meta = _load_trash_meta()

        # 判断文件类型（支持多模态）
        ft = detect_file_type(safe_name)
        if ft == "image":
            file_type = "image"
        elif ft == "audio":
            file_type = "audio"
        elif safe_name.lower().endswith((".xlsx", ".xls")):
            file_type = "xlsx"
        elif safe_name.lower().endswith(".pdf"):
            file_type = "pdf"
        else:
            file_type = "txt"

        # 如果是图片/音频，同时清理生成的 .txt 文件
        if ft in ("image", "audio"):
            companion_txt = os.path.splitext(safe_name)[0] + ".txt"
            companion_path = os.path.join(DATA_DIR, companion_txt)
            if os.path.exists(companion_path):
                companion_trash = os.path.join(TRASH_DIR, companion_txt)
                shutil.move(companion_path, companion_trash)
                companion_stat = os.stat(companion_trash)
                meta[companion_txt] = {
                    "deleted_at": time.time(),
                    "deleted_at_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "expires_at_human": time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(time.time() + TRASH_DAYS * 86400),
                    ),
                    "size_bytes": companion_stat.st_size,
                    "size_human": f"{companion_stat.st_size:,} 字节",
                    "file_type": "txt",
                    "companion_of": safe_name,
                }
        meta[safe_name] = {
            "deleted_at": time.time(),
            "deleted_at_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "expires_at_human": time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(time.time() + TRASH_DAYS * 86400),
            ),
            "size_bytes": stat.st_size,
            "size_human": f"{stat.st_size:,} 字节",
            "file_type": file_type,
        }
        _save_trash_meta(meta)

        # 3. 同步去重表
        dedup_mgr = get_dedup("dedup.db")
        dedup_mgr.remove_by_filename(safe_name)

        # 4. 重建索引
        _, remaining = _rebuild_index()

        return {
            "status": "ok",
            "message": (
                f"已把「{safe_name}」移入垃圾桶，{TRASH_DAYS}天后自动永久删除。"
                f"索引已重建（剩余 {remaining} 个文档）"
            ),
            "trash": True,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"删除失败: {str(e)}",
        })


# ============================================================
# 垃圾桶管理接口
# ============================================================

@app.get("/trash", summary="🗑️ 查看垃圾桶", tags=["管理接口"])
async def list_trash():
    """
    查看垃圾桶中的所有文件及其元数据。

    **返回**：垃圾桶文件列表，包含删除时间、过期时间、文件大小等。
    """
    import time
    meta = _load_trash_meta()

    items = []
    now = time.time()
    for filename, info in sorted(meta.items()):
        trash_path = os.path.join(TRASH_DIR, filename)
        exists = os.path.exists(trash_path)
        days_left = max(0, int((info["deleted_at"] + TRASH_DAYS * 86400 - now) / 86400))
        items.append({
            "filename": filename,
            "file_type": info.get("file_type", "unknown"),
            "size_human": info.get("size_human", "未知"),
            "deleted_at": info.get("deleted_at_human", "未知"),
            "expires_at": info.get("expires_at_human", "未知"),
            "days_left": days_left,
            "expired": days_left <= 0,
            "file_exists": exists,
        })

    return {
        "total": len(items),
        "trash_days": TRASH_DAYS,
        "items": items,
    }


@app.post("/trash/{filename}/restore", summary="🔄 从垃圾桶恢复文件", tags=["管理接口"])
async def restore_file(filename: str):
    """
    从垃圾桶恢复指定文件到知识库，并重建索引。

    **处理流程**：
    1. 将文件从 .trash/ 移动回 data/
    2. 从垃圾桶元数据中移除记录
    3. 重建索引
    """
    try:
        safe_name = os.path.basename(filename)
        trash_path = os.path.join(TRASH_DIR, safe_name)

        if not os.path.exists(trash_path):
            return JSONResponse(status_code=404, content={
                "status": "error",
                "message": f"垃圾桶中无此文件: {safe_name}",
            })

        # 1. 移回 data/
        data_path = os.path.join(DATA_DIR, safe_name)
        shutil.move(trash_path, data_path)

        # 2. 清除元数据
        meta = _load_trash_meta()
        meta.pop(safe_name, None)
        _save_trash_meta(meta)

        # 3. 重建索引
        _, remaining = _rebuild_index()

        return {
            "status": "ok",
            "message": f"已恢复「{safe_name}」，索引已重建（共 {remaining} 个文档）",
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"恢复失败: {str(e)}",
        })


@app.delete("/trash/{filename}", summary="💀 永久删除垃圾桶文件", tags=["管理接口"])
async def permanent_delete(filename: str):
    """
    从垃圾桶中永久删除指定文件（不可恢复）。

    **说明**：文件已从 data/ 移除且索引已重建，此操作仅清理 .trash/ 中的副本。
    """
    try:
        safe_name = os.path.basename(filename)
        trash_path = os.path.join(TRASH_DIR, safe_name)

        if not os.path.exists(trash_path):
            return JSONResponse(status_code=404, content={
                "status": "error",
                "message": f"垃圾桶中无此文件: {safe_name}",
            })

        os.remove(trash_path)

        meta = _load_trash_meta()
        meta.pop(safe_name, None)
        _save_trash_meta(meta)

        return {
            "status": "ok",
            "message": f"已永久删除「{safe_name}」",
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"永久删除失败: {str(e)}",
        })


@app.post("/trash/auto-clean", summary="🧹 自动清理过期文件", tags=["管理接口"])
async def auto_clean_trash():
    """
    自动清理垃圾桶中超过 {TRASH_DAYS} 天的文件。

    **触发时机**：
    - 启动时自动执行一次
    - 手动调用来立即清理
    """
    import time
    meta = _load_trash_meta()
    now = time.time()
    cleaned = []
    errors = []

    for filename, info in list(meta.items()):
        age_days = (now - info["deleted_at"]) / 86400
        if age_days >= TRASH_DAYS:
            trash_path = os.path.join(TRASH_DIR, filename)
            try:
                if os.path.exists(trash_path):
                    os.remove(trash_path)
                meta.pop(filename, None)
                cleaned.append(filename)
                print(f"   🧹 自动清理: {filename}（已过期 {int(age_days)} 天）")
            except Exception as e:
                errors.append(f"{filename}: {e}")

    _save_trash_meta(meta)

    return {
        "status": "ok",
        "cleaned": len(cleaned),
        "cleaned_files": cleaned,
        "errors": errors,
        "remaining": len(meta),
        "message": f"清理完成：移除了 {len(cleaned)} 个过期文件，垃圾桶剩余 {len(meta)} 个",
    }


# ============================================================
# 人工审核分级接口
# ============================================================

from reviewer import DocumentReviewer

# 审核器实例（懒加载，避免启动时就调 LLM）
_reviewer: Optional["DocumentReviewer"] = None

def _get_reviewer() -> DocumentReviewer:
    """获取审核器实例（懒加载）"""
    global _reviewer
    if _reviewer is None:
        _reviewer = DocumentReviewer()
    return _reviewer


class ReviewSubmitRequest(BaseModel):
    """审核提交请求（纯文本）"""
    content: str = "文档全文"
    title: str = ""             # 可选标题
    source: str = ""            # 可选来源
    force_review: bool = False  # 强制进入人工审核

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "content": "中国人民银行决定于2026年6月15日下调...",
                "title": "央行降准通知",
                "source": "央行官网",
            }]
        }
    }


class ReviewSubmitResponse(BaseModel):
    """审核提交响应"""
    review_id: str
    tier: str            # auto_approved / needs_review / auto_rejected
    score: int           # 0-10
    ai_reasoning: str    # AI 评分理由
    action: str          # approved / queued / rejected
    filepath: str = ""   # 自动审批通过后的文件路径


class ReviewQueueItem(BaseModel):
    """审核队列项目"""
    review_id: str
    title: str
    content_preview: str
    source: str
    score: int
    tier: str
    ai_reasoning: str
    status: str
    created_at: str


class ReviewDetailResponse(BaseModel):
    """审核详情响应（含全文）"""
    review_id: str
    title: str
    content: str          # 全文
    content_preview: str
    source: str
    score: int
    tier: str
    ai_reasoning: str
    status: str
    created_at: str


class ReviewActionResponse(BaseModel):
    """审核操作响应（审批/拒绝）"""
    status: str
    review_id: str
    action: str
    message: str = ""


class ReviewStatsResponse(BaseModel):
    """审核统计"""
    pending: int
    total_processed: int
    approved: int
    rejected: int
    auto_approved: int
    auto_rejected: int
    human_approved: int
    human_rejected: int


@app.post(
    "/review/submit",
    response_model=ReviewSubmitResponse,
    summary="📝 提交文本进行 AI 低温初筛",
    tags=["审核接口"],
)
async def review_submit(request: ReviewSubmitRequest):
    """
    提交文档（纯文本）进行低温 AI 初筛，自动分级。

    **处理流程**：
    1. 低温 LLM（T=0.0）评分 → 确定性输出
    2. ≥7分 → auto_approved：自动入库
    3. 4-6分 → needs_review：进入审核队列
    4. ≤3分 → auto_rejected：自动丢弃

    **force_review=true**：跳过 AI 评分，直接进入审核队列。
    """
    try:
        reviewer = _get_reviewer()
        result = reviewer.submit(
            content=request.content,
            title=request.title,
            source=request.source,
            force_review=request.force_review,
        )
        return ReviewSubmitResponse(**result)

    except Exception as e:
        logger.error(f"[审核] 提交失败: {e}")
        raise HTTPException(status_code=500, detail=f"审核提交失败: {str(e)}")


@app.post(
    "/review/upload",
    summary="📤 上传文件进行 AI 低温初筛",
    tags=["审核接口"],
)
async def review_upload(
    file: UploadFile = File(..., description="知识文件（.txt / .pdf / .xlsx / 图片 / 音频）"),
    title: str = Form("", description="文档标题"),
    source: str = Form("", description="来源标签"),
    force_review: bool = Form(False, description="强制进入审核队列"),
):
    """
    上传文件，先 AI 低温初筛再决定是否入库。

    与 /upload 的区别：
    - /upload：直接入库（跳过审核）
    - /review/upload：AI 初筛 → 高分自动入库，低分进审核队列

    **处理流程**：
    1. 接收文件 → 提取文本（PDF/Excel/OCR/STT）
    2. 低温 LLM 评分 → 分级
    3. 高分自动入库，低分进审核队列
    """
    try:
        filename = file.filename
        file_bytes = await file.read()

        # ── 提取文本 ──
        content = ""
        file_type = "unknown"
        actual_filename = ""

        if filename.lower().endswith(".pdf"):
            content = extract_pdf(file_bytes)
            file_type = "pdf"
        elif filename.lower().endswith((".xlsx", ".xls")):
            docs = extract_excel(file_bytes)
            content = "\n\n".join(doc.text for doc in docs)
            file_type = "xlsx"
        elif filename.lower().endswith(".txt"):
            content = file_bytes.decode("utf-8")
            file_type = "txt"
        elif detect_file_type(filename) == "image":
            content = describe_image(file_bytes, lang=_CFG.ocr_language)
            file_type = "image"
        elif detect_file_type(filename) == "audio":
            content = transcribe_audio(file_bytes, model_size=_CFG.stt_model_size, language=_CFG.stt_language)
            file_type = "audio"

        if not content.strip():
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "文件内容提取为空"},
            )

        # ── 保存原始文件到审核暂存区 ──
        reviewer = _get_reviewer()
        content_path = os.path.join(reviewer.content_dir, f"upload_{filename}")
        counter = 1
        while os.path.exists(content_path):
            name, ext = os.path.splitext(filename)
            content_path = os.path.join(reviewer.content_dir, f"upload_{name}_{counter}{ext}")
            counter += 1

        with open(content_path, "wb") as f:
            f.write(file_bytes)
        actual_filename = os.path.basename(content_path)

        # ── 提交审核 ──
        result = reviewer.submit(
            content=content,
            title=title or filename,
            source=source or f"文件上传 ({file_type})",
            filename=actual_filename,
            force_review=force_review,
        )

        # 自动通过的：已经自动保存到 data/，触发索引重建
        if result["action"] == "approved" and result.get("filepath"):
            _, remaining = _rebuild_index()
            result["message"] = f"自动审批通过，已入库并重建索引（{remaining} 个文档）"
        elif result["action"] == "queued":
            result["message"] = f"已进入审核队列，请前往 GET /review/queue 查看"
        else:
            result["message"] = f"AI 判定不相关（{result['score']}/10），已自动拒绝"

        return result

    except Exception as e:
        logger.error(f"[审核] 上传审核失败: {e}")
        raise HTTPException(status_code=500, detail=f"审核上传失败: {str(e)}")


@app.get(
    "/review/queue",
    summary="📋 查看审核队列",
    tags=["审核接口"],
)
async def review_queue(limit: int = Query(50, description="返回条数上限")):
    """
    查看待人工审核的文档列表。

    **返回**：按创建时间倒序排列的审核项（含预览，不含全文）。
    """
    try:
        reviewer = _get_reviewer()
        items = reviewer.get_queue(limit=limit)
        return {
            "total": len(items),
            "items": items,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取审核队列失败: {str(e)}")


@app.get(
    "/review/queue/{review_id}",
    summary="🔍 查看审核项详情（含全文）",
    tags=["审核接口"],
)
async def review_detail(review_id: str):
    """
    查看某条审核项的详细信息，包含文档全文。

    用于人工审核时查看完整内容再做决定。
    """
    try:
        reviewer = _get_reviewer()
        item = reviewer.get_item(review_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"审核项不存在: {review_id}")
        return item
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取审核详情失败: {str(e)}")


@app.post(
    "/review/queue/{review_id}/approve",
    response_model=ReviewActionResponse,
    summary="✅ 审批通过",
    tags=["审核接口"],
)
async def review_approve(review_id: str):
    """
    人工审批通过一条审核项 → 文档入库并重建索引。

    **处理流程**：
    1. 审核项状态改为 approved
    2. 文档保存到 data/ 目录
    3. 重建 RAG 索引
    """
    try:
        reviewer = _get_reviewer()
        result = reviewer.approve(review_id)

        if result["status"] != "ok":
            raise HTTPException(status_code=400, detail=result.get("message", "审批失败"))

        # 触发索引重建
        if result.get("filepath"):
            _, remaining = _rebuild_index()
            result["message"] = f"审批通过，已入库并重建索引（{remaining} 个文档）"

        return ReviewActionResponse(
            status="ok",
            review_id=review_id,
            action="approved",
            message=result.get("message", "审批通过"),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"审批失败: {str(e)}")


@app.post(
    "/review/queue/{review_id}/reject",
    response_model=ReviewActionResponse,
    summary="❌ 审批拒绝",
    tags=["审核接口"],
)
async def review_reject(review_id: str):
    """
    人工拒绝一条审核项 → 文档移入垃圾桶。

    **说明**：
    - 拒绝后文档会移到 .trash/，30天后永久删除
    - 不会重建索引（因为文档未入库）
    """
    try:
        reviewer = _get_reviewer()
        result = reviewer.reject(review_id)

        if result["status"] != "ok":
            raise HTTPException(status_code=400, detail=result.get("message", "拒绝失败"))

        return ReviewActionResponse(
            status="ok",
            review_id=review_id,
            action="rejected",
            message="已拒绝，文档移入垃圾桶（30天后自动清理）",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"拒绝失败: {str(e)}")


@app.get(
    "/review/stats",
    response_model=ReviewStatsResponse,
    summary="📊 审核统计",
    tags=["审核接口"],
)
async def review_stats():
    """
    查看审核系统的统计数据。

    **统计维度**：
    - pending: 待审核数量
    - auto_approved / auto_rejected: AI 自动处理数量
    - human_approved / human_rejected: 人工处理数量
    """
    try:
        reviewer = _get_reviewer()
        stats = reviewer.get_stats()
        return ReviewStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")


# ============================================================
# 辅助函数
# ============================================================

def _load_trash_meta():
    """加载垃圾桶元数据文件"""
    os.makedirs(TRASH_DIR, exist_ok=True)
    meta_path = os.path.join(TRASH_DIR, TRASH_META_FILE)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_trash_meta(meta):
    """保存垃圾桶元数据文件"""
    os.makedirs(TRASH_DIR, exist_ok=True)
    meta_path = os.path.join(TRASH_DIR, TRASH_META_FILE)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _rebuild_index():
    """
    从 data/ 目录重新加载所有文档并重建全部索引。
    用于文件增删后的索引刷新。
    返回：(documents, remaining_count)
    """
    global index, bm25_store, query_engine, splitter

    # 清除旧的向量索引
    if os.path.exists(VECTOR_INDEX_DIR):
        shutil.rmtree(VECTOR_INDEX_DIR)

    # 重新加载文档
    documents = _load_documents()

    if documents:
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=VECTOR_INDEX_DIR)

        nodes = splitter.get_nodes_from_documents(documents)

        # BM25：全量重建（LSM-Tree 的 merge 操作）
        bm25_store.rebuild(nodes)

        from bm25_store import BM25StoreRetrieverAdapter
        bm25_adapter = BM25StoreRetrieverAdapter(bm25_store)
        _assemble_query_engine(index, bm25_adapter)

        return documents, len(documents)
    else:
        index = None
        bm25_store = None
        query_engine = None
        return [], 0


def _load_documents():
    """从 data/ 目录加载所有 .txt .pdf .xlsx 文件为 Document 对象"""
    documents = []
    os.makedirs(DATA_DIR, exist_ok=True)

    # 元数据映射 —— 可通过 metadata.json 自定义
    # 格式：{"filename.txt": {"category": "...", "source": "..."}}
    metadata_map = {}
    metadata_json_path = os.path.join(DATA_DIR, "metadata.json")
    if os.path.exists(metadata_json_path):
        try:
            with open(metadata_json_path, "r", encoding="utf-8") as f:
                metadata_map = json.load(f)
        except Exception:
            pass

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
                doc.metadata["source"] = _CFG.excel_source
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
                metadata={"category": _CFG.default_category, "source": "本地PDF", "filename": filename, "type": "pdf_extract"},
            ))
            print(f"       加载文档: {filename} ({len(text)} 字) → [PDF提取]")
        elif filename.endswith(".txt"):
            # TXT 文件：原有逻辑
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                continue
            meta = metadata_map.get(filename, {"category": _CFG.default_category, "source": "本地"})
            documents.append(Document(text=text, metadata=meta))
            print(f"       加载文档: {filename} ({len(text)} 字) → [{meta['category']}]")

    return documents


# ============================================================
# 去重管理接口
# ============================================================

@app.get("/dedup/stats", summary="📊 去重统计", tags=["管理接口"])
async def dedup_stats():
    """查看去重表的统计信息：总记录数、来源类型分布、总文件大小。"""
    dedup_mgr = get_dedup("dedup.db")
    return dedup_mgr.stats()


@app.get("/dedup/list", summary="📋 去重记录列表", tags=["管理接口"])
async def dedup_list():
    """列出所有去重记录（最近入库的在前面）。"""
    dedup_mgr = get_dedup("dedup.db")
    records = dedup_mgr.list_all()
    # 时间戳转可读字符串
    import time
    for r in records:
        r["added_at_human"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(r["added_at"])
        )
    return {"total": len(records), "records": records}


@app.post("/dedup/rebuild", summary="🔄 重建去重表", tags=["管理接口"])
async def dedup_rebuild():
    """从 data/ 目录重建去重表（用于修复不一致，如同步删文件后去重表有残留）。"""
    dedup_mgr = get_dedup("dedup.db")
    count = dedup_mgr.rebuild_from_data_dir(DATA_DIR)
    return {"status": "ok", "records_rebuilt": count}


# ============================================================
# 索引管理接口（BM25 LSM-Tree + 全量重建）
# ============================================================

@app.get("/index/stats", summary="📊 BM25 索引统计（LSM-Tree 双层架构）", tags=["管理接口"])
async def index_stats():
    """
    查看 BM25 增量索引的状态。

    **返回字段**：
    - base_nodes: Base 层节点数（已持久化到磁盘）
    - delta_nodes: Delta 层节点数（内存中，等待合并）
    - total_merges: 历史合并次数
    - merge_threshold: 合并阈值
    """
    if not bm25_store:
        return {"status": "error", "message": "BM25 索引尚未初始化"}
    return {"status": "ok", **bm25_store.stats}


@app.post("/index/rebuild", summary="🔄 手动重建全部索引（合并 BM25 Delta）", tags=["管理接口"])
async def index_rebuild():
    """
    手动触发全量索引重建。

    **用途**：
    - Delta 积压较多但未达到阈值时手动合并
    - 索引状态异常时修复
    - 服务部署后首次启动强制刷新

    **注意**：重建过程可能耗时（取决于文档数量），请求会等待完成。
    """
    global index, bm25_store, query_engine

    if not bm25_store:
        return {"status": "error", "message": "BM25 索引尚未初始化"}

    try:
        before_stats = bm25_store.stats
        documents, count = _rebuild_index()
        return {
            "status": "ok",
            "documents": count,
            "nodes": len(bm25_store.all_nodes) if bm25_store else 0,
            "before": before_stats,
            "after": bm25_store.stats if bm25_store else {},
        }
    except Exception as e:
        return {"status": "error", "message": f"重建失败: {str(e)}"}


# ============================================================
# 定时任务管理接口
# ============================================================

@app.get("/scheduler/status", summary="⏰ 定时任务状态", tags=["管理接口"])
async def scheduler_status():
    """
    查看 APScheduler 定时任务的运行状态。

    **返回**：调度器状态 + 各任务下次执行时间。
    """
    return _scheduler_mod.get_scheduler_status()


@app.post("/scheduler/trigger/crawler", summary="🕷️ 手动触发爬虫", tags=["管理接口"])
async def scheduler_trigger_crawler():
    """
    手动触发一次爬虫流水线（在后台线程中执行，不阻塞）。
    """
    return _scheduler_mod.trigger_crawler_now()


@app.post("/scheduler/trigger/cleanup", summary="🧹 手动触发清理", tags=["管理接口"])
async def scheduler_trigger_cleanup():
    """
    手动触发一次旧知识清理（在后台线程中执行，不阻塞）。
    """
    return _scheduler_mod.trigger_cleanup_now()


# ============================================================
# 多模态接口 — OCR + 状态检查
# ============================================================

class OCRResponse(BaseModel):
    """OCR 识别响应"""
    status: str = "ok / error"
    text: str = "识别出的文字"
    method: str = "paddleocr / tesseract / none"
    char_count: int = 0


@app.post("/ocr", response_model=OCRResponse, summary="📷 图片 OCR 文字识别", tags=["多模态接口"])
async def ocr_endpoint(
    file: UploadFile = File(..., description="图片文件（.png / .jpg / .bmp / .tiff / .webp）"),
):
    """
    上传图片，返回 OCR 识别出的文字。

    **处理流程**：
    1. 接收图片文件
    2. 自动检测文字区域 + 旋转校正
    3. PaddleOCR（首选）或 Tesseract（降级）识别
    4. 返回文字内容

    **与 /upload 的区别**：
    - /ocr：仅识别文字，不保存文件，不入库
    - /upload：识别后保存到知识库并重建索引
    """
    try:
        file_bytes = await file.read()

        # 检测文件类型
        ext = os.path.splitext(file.filename or "unknown.jpg")[1].lower()
        SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
        if ext not in SUPPORTED_IMAGE_EXTS:
            return OCRResponse(
                status="error",
                text="",
                method="none",
                message=f"不支持的图片格式: {ext}，支持: {', '.join(SUPPORTED_IMAGE_EXTS)}",
            )

        # 调用 OCR（从配置读取语言设置）
        from multimodal import ocr_image, _check_paddle
        text = ocr_image(file_bytes, lang=_CFG.ocr_language)

        if not text.strip():
            return OCRResponse(
                status="error",
                text="",
                method="paddleocr" if _check_paddle() else "tesseract",
                message="OCR 未识别到文字（图片可能没有文字或质量过低）",
            )

        method = "paddleocr" if _check_paddle() else "tesseract"
        return OCRResponse(
            status="ok",
            text=text,
            method=method,
            char_count=len(text),
        )

    except Exception as e:
        logger.error(f"[OCR] 识别失败: {e}")
        return OCRResponse(
            status="error",
            text="",
            method="none",
            message=str(e),
        )


class MultimodalStatusResponse(BaseModel):
    """多模态依赖状态响应"""
    paddleocr: bool = False
    tesseract: bool = False
    faster_whisper: bool = False
    message: str = ""


@app.get("/multimodal/status", response_model=MultimodalStatusResponse, summary="🔧 多模态依赖检查", tags=["多模态接口"])
async def multimodal_status():
    """
    检查各多模态依赖的可用性。

    **返回**：PaddleOCR、Tesseract、faster-whisper 的可用状态。
    用于调试和部署检测。
    """
    from multimodal import check_dependencies
    deps = check_dependencies()
    all_ok = all(deps.values())

    if all_ok:
        msg = "所有多模态依赖就绪 🎉"
    elif any(deps.values()):
        missing = [k for k, v in deps.items() if not v]
        msg = f"部分依赖缺失: {', '.join(missing)}"
    else:
        msg = "无多模态依赖可用。安装指南: pip install paddleocr faster-whisper"

    return MultimodalStatusResponse(
        paddleocr=deps.get("paddleocr", False),
        tesseract=deps.get("tesseract", False),
        faster_whisper=deps.get("faster_whisper", False),
        message=msg,
    )


# ============================================================
# 多轮对话管理接口
# ============================================================

@app.get("/sessions", summary="💬 会话列表", tags=["核心接口"])
async def list_sessions(limit: int = 20):
    """
    获取会话列表（按更新时间倒序）。

    **返回**：
    - sessions: 会话列表（含 id, title, message_count, updated_at）
    """
    if not session_store:
        return {"status": "error", "message": "多轮对话未启用"}
    sessions = session_store.list_sessions(limit=limit)
    return {"status": "ok", "sessions": sessions}


@app.post("/sessions", summary="💬 新建会话", tags=["核心接口"])
async def create_session(title: str = ""):
    """
    新建一个会话。

    **参数**：
    - title: 可选，会话标题（默认自动生成）

    **返回**：session_id
    """
    if not session_store:
        return {"status": "error", "message": "多轮对话未启用"}
    sid = session_store.create_session(title=title)
    return {"status": "ok", "session_id": sid}


@app.get("/sessions/{session_id}", summary="💬 获取会话详情", tags=["核心接口"])
async def get_session(session_id: str):
    """
    获取会话信息和消息历史。

    **返回**：
    - session: 会话元信息
    - messages: 全部消息历史（含 role, content, sources, timestamp）
    """
    if not session_store:
        return {"status": "error", "message": "多轮对话未启用"}
    sess = session_store.get_session(session_id)
    if not sess:
        return {"status": "error", "message": f"会话 {session_id} 不存在"}
    messages = session_store.get_all_history(session_id)
    return {"status": "ok", "session": sess, "messages": messages}


@app.delete("/sessions/{session_id}", summary="💬 删除会话", tags=["核心接口"])
async def delete_session(session_id: str):
    """删除一个会话及其所有消息。"""
    if not session_store:
        return {"status": "error", "message": "多轮对话未启用"}
    if not session_store.get_session(session_id):
        return {"status": "error", "message": f"会话 {session_id} 不存在"}
    session_store.delete_session(session_id)
    return {"status": "ok", "message": f"会话 {session_id} 已删除"}


@app.patch("/sessions/{session_id}", summary="💬 更新会话标题", tags=["核心接口"])
async def update_session_title(session_id: str, title: str):
    """更新会话标题。"""
    if not session_store:
        return {"status": "error", "message": "多轮对话未启用"}
    if not session_store.get_session(session_id):
        return {"status": "error", "message": f"会话 {session_id} 不存在"}
    session_store.update_title(session_id, title)
    return {"status": "ok", "message": f"会话标题已更新为: {title}"}


# ============================================================
# 启动服务
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
