"""
RAG 财务助手 - 单文件版
双击运行即可，不需要启动任何后端服务

启动：streamlit run app_single.py
或者：python -m streamlit run app_single.py
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, Document, StorageContext, load_index_from_storage
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.openai_like import OpenAILike
import fitz  # PyMuPDF（PDF 解析）

# ---- 配置 ----
DATA_DIR = "data"
VECTOR_INDEX_DIR = "chroma_data_server"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
TOP_K = 10
TOP_N = 3


# ---- OCR 配置 ----
import os
os.environ["TESSDATA_PREFIX"] = r"./tessdata"
os.environ["PATH"] = r"C:\Tesseract-OCR;" + os.environ.get("PATH", "")
import pytesseract
from PIL import Image


# ---- 页面配置 ----
st.set_page_config(
    page_title="财务助手",
    page_icon="🧾",
    layout="centered",
)


# ---- 辅助函数 ----
def extract_pdf(pdf_bytes):
    """
    从 PDF 字节流提取干净文本（和 day6_pdf.py 同款逻辑）
    返回：字符串
    """
    import io
    all_text_parts = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

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

    # 如果没识别到表格，回退到纯文本提取
    if not all_text_parts:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            if text:
                all_text_parts.append(text)

    doc.close()

    # 如果纯文本提取也为空，说明是扫描件，走 OCR
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


def load_documents():
    """从 data/ 目录加载所有 .txt 和 .pdf 文件"""
    documents = []
    os.makedirs(DATA_DIR, exist_ok=True)
    metadata_map = {
        "tax_policy.txt": {"category": "税务政策", "source": "国家税务总局"},
        "weather_sales.txt": {"category": "经营策略", "source": "行业分析报告"},
        "test.txt": {"category": "测试", "source": "本地"},
    }
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)

        if filename.endswith(".pdf"):
            with open(filepath, "rb") as f:
                pdf_bytes = f.read()
            try:
                text = extract_pdf(pdf_bytes)
            except Exception:
                continue
            if not text.strip():
                continue
            documents.append(Document(
                text=text,
                metadata={"category": "未知", "source": "本地PDF", "filename": filename, "type": "pdf_extract"},
            ))
        elif filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                continue
            meta = metadata_map.get(filename, {"category": "未知", "source": "本地"})
            documents.append(Document(text=text, metadata=meta))
    return documents


def build_engine(progress=None):
    """加载所有模型和索引，构建查询引擎"""
    def update(pct, text):
        if progress:
            progress.progress(pct, text=text)

    # 1. Embedding
    update(0, "🔍 [1/6] 加载 Embedding 模型...")
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-zh-v1.5")

    # 2. LLM
    update(17, "🤖 [2/6] 接入 Kimi 大模型...")
    kimi = OpenAILike(
        model="moonshot-v1-8k",
        api_key=os.environ.get("KIMI_API_KEY", ""),
        api_base="https://api.moonshot.cn/v1",
        is_chat_model=True,
        max_tokens=4096,
        context_window=8192,
    )
    Settings.llm = kimi

    # 3. 加载文档 + 向量索引
    update(33, "📚 [3/6] 加载知识库和向量索引...")
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    documents = load_documents()

    if os.path.exists(VECTOR_INDEX_DIR) and os.listdir(VECTOR_INDEX_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=VECTOR_INDEX_DIR)
        index = load_index_from_storage(storage_context)
    else:
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=VECTOR_INDEX_DIR)

    # 4. BM25 索引
    update(50, "📋 [4/6] 构建 BM25 索引...")
    nodes = splitter.get_nodes_from_documents(documents)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=TOP_K)

    # 5. 混合检索 + Reranker
    update(67, "🔄 [5/6] 加载 Reranker 精排模型...")
    vector_retriever = index.as_retriever(similarity_top_k=TOP_K)
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        num_queries=1,
        use_async=False,
        similarity_top_k=TOP_K,
    )
    rerank = SentenceTransformerRerank(
        model="./models/bge-reranker-v2-m3",
        top_n=TOP_N,
    )

    # 6. 组装
    update(83, "🔧 [6/6] 组装查询引擎...")
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[rerank],
        llm=kimi,
    )

    update(100, "✅ 加载完成！")

    # 返回所有组件，上传时需要重建
    return {
        "engine": query_engine,
        "index": index,
        "splitter": splitter,
        "kimi": kimi,
        "rerank": rerank,
    }


def rebuild_engine(components):
    """上传新文件后重建查询引擎（从零重建所有索引，保证向量+BM25节点ID一致）"""
    splitter = components["splitter"]
    kimi = components["kimi"]
    rerank = components["rerank"]

    # 重新加载所有文档
    documents = load_documents()
    all_nodes = splitter.get_nodes_from_documents(documents)

    # 从零重建向量索引
    index = VectorStoreIndex.from_documents([])
    for node in all_nodes:
        index.insert_nodes([node])
    index.storage_context.persist(persist_dir=VECTOR_INDEX_DIR)

    # 从零重建 BM25 索引
    bm25_retriever = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=TOP_K)

    # 重建混合检索 + 查询引擎
    vector_retriever = index.as_retriever(similarity_top_k=TOP_K)
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        num_queries=1,
        use_async=False,
        similarity_top_k=TOP_K,
    )
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[rerank],
        llm=kimi,
    )
    components["engine"] = query_engine
    components["index"] = index
    return components


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.header("📂 知识库管理")

    # 显示已有文件
    st.subheader("已有知识文件")
    os.makedirs(DATA_DIR, exist_ok=True)
    existing_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".pdf"))])
    if existing_files:
        for f in existing_files:
            filepath = os.path.join(DATA_DIR, f)
            size = os.path.getsize(filepath)
            icon = "📕" if f.endswith(".pdf") else "📄"
            col_name, col_del = st.columns([5, 1])
            with col_name:
                st.text(f"{icon} {f}（{size:,} 字节）")
            with col_del:
                if st.button("🗑️", key=f"del_{f}", help=f"删除 {f}"):
                    import shutil
                    try:
                        # 1. 删除源文件
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            file_deleted = True
                        else:
                            file_deleted = False
                        # 2. 清除向量索引（从零重建，避免残留脏数据）
                        if os.path.exists(VECTOR_INDEX_DIR):
                            shutil.rmtree(VECTOR_INDEX_DIR)
                            index_cleared = True
                        else:
                            index_cleared = False
                        # 3. 清除缓存，强制下次重建
                        if "components" in st.session_state:
                            del st.session_state.components
                        st.success(f"已删除「{f}」(文件: {'成功' if file_deleted else '不存在'}, 索引: {'已清除' if index_cleared else '无需清除'})")
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {e}")
    else:
        st.caption("暂无文件，请上传")

    st.divider()

    # 上传新文件
    st.subheader("📤 上传新文件")
    uploaded_file = st.file_uploader(
        "选择文件",
        type=["txt", "pdf"],
        help="支持 .txt 和 .pdf 格式",
    )
    category = st.text_input("分类标签", placeholder="例如：税务政策、经营策略", value="")

    if uploaded_file and st.button("确认导入", type="primary", use_container_width=True):
        filename = uploaded_file.name
        is_pdf = filename.lower().endswith(".pdf")

        if is_pdf:
            # PDF 文件：保存 + 解析
            save_path = os.path.join(DATA_DIR, filename)
            pdf_bytes = uploaded_file.read()
            with open(save_path, "wb") as f:
                f.write(pdf_bytes)

            with st.spinner("正在解析 PDF 文档..."):
                try:
                    text = extract_pdf(pdf_bytes)
                except Exception as e:
                    st.error(f"PDF 解析失败: {e}")
                    text = ""

            if not text.strip():
                st.error("PDF 内容为空或无法提取文字（可能是扫描件，需要 OCR）")
            else:
                # rebuild_engine 会从零重建所有索引，不需要手动 insert_nodes
                components = rebuild_engine(st.session_state.components)
                st.session_state.components = components

                st.success(f"✅ 已导入「{filename}」({len(text)} 字符)")
                st.rerun()
        else:
            # TXT 文件：原有逻辑
            text = uploaded_file.read().decode("utf-8")

            if not text.strip():
                st.error("文件内容为空，请检查")
            else:
                save_path = os.path.join(DATA_DIR, filename)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(text)

                with st.spinner("正在处理新文件..."):
                    # rebuild_engine 会从零重建所有索引
                    components = rebuild_engine(st.session_state.components)
                    st.session_state.components = components

                st.success(f"✅ 已导入「{filename}」")
                st.rerun()

    st.divider()
    st.caption("💡 提示：上传 txt/pdf 文件即可扩展知识库，AI 会基于这些内容回答问题")


# ============================================================
# 主页面
# ============================================================
st.title("🧾 个体工商户财务助手")
st.caption("基于 RAG 技术，为你提供专业的财务税务解答")

# ---- 初始化 ----
if "components" not in st.session_state:
    progress = st.progress(0, text="⏳ 准备加载...")
    st.session_state.components = build_engine(progress)
    import time
    time.sleep(0.5)
    progress.empty()
    st.toast("✅ 模型加载完成！")

engine = st.session_state.components["engine"]

# ---- 聊天记录 ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 查看来源"):
                for i, src in enumerate(msg["sources"], 1):
                    score = src.get("score", "")
                    text = src.get("text", "")
                    meta = src.get("metadata", {})
                    st.markdown(f"**来源 {i}**（相关度: {score}）")
                    st.markdown(f"> {text}")
                    if meta:
                        st.caption(f"分类: {meta.get('category', '未知')} | 来源: {meta.get('source', '未知')}")

# ---- 输入框 ----
if prompt := st.chat_input("在这里输入你的财务问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            response = engine.query(prompt)

            sources = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    score = node.score
                    if score is not None:
                        score = float(score)
                    sources.append({
                        "text": node.text[:200],
                        "score": round(score, 4) if score else None,
                        "metadata": dict(node.metadata) if node.metadata else {},
                    })

            answer = str(response)
            st.markdown(answer)

            if sources:
                with st.expander("📎 查看来源"):
                    for i, src in enumerate(sources, 1):
                        score = src.get("score", "")
                        text = src.get("text", "")
                        meta = src.get("metadata", {})
                        st.markdown(f"**来源 {i}**（相关度: {score}）")
                        st.markdown(f"> {text}")
                        if meta:
                            st.caption(f"分类: {meta.get('category', '未知')} | 来源: {meta.get('source', '未知')}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
