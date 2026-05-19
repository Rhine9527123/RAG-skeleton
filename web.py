"""
RAG 财务助手 - 前后端分离版（API 客户端）

启动前需要先运行：python server.py（后端服务）
然后运行：streamlit run web.py

架构：
  web.py（前端，本文件） → HTTP API → server.py（后端）
"""
import streamlit as st
import requests
import os

# ---- 后端 API 地址（自动适配 ngrok / 本地） ----
# 优先级：URL参数 > 环境变量 > 默认 localhost
_query_params = st.query_params
BACKEND_URL = (
    _query_params.get("backend", [None])[0]
    or os.environ.get("BACKEND_URL")
    or "http://localhost:8000"
)

# ---- 页面配置 ----
st.set_page_config(
    page_title="个人助手",
    page_icon="🧾",
    layout="centered",
)


def check_backend():
    """检查后端服务是否可用"""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def api_chat(question):
    """调用后端聊天接口"""
    resp = requests.post(f"{BACKEND_URL}/chat", json={"question": question}, timeout=60)
    return resp.json()


def api_upload(file_bytes, filename, category="未知"):
    """调用后端上传接口"""
    resp = requests.post(
        f"{BACKEND_URL}/upload",
        files={"file": (filename, file_bytes)},
        data={"category": category},
        timeout=120,
    )
    return resp.json()


def api_list_files():
    """调用后端获取文件列表"""
    resp = requests.get(f"{BACKEND_URL}/files", timeout=10)
    return resp.json()


def api_delete_file(filename):
    """调用后端删除文件"""
    resp = requests.delete(f"{BACKEND_URL}/files/{filename}", timeout=60)
    return resp.json()


# ============================================================
# 侧边栏 - 知识库管理
# ============================================================
with st.sidebar:
    st.header("📂 知识库管理")

    # 显示已有文件（从 API 获取）
    st.subheader("已有知识文件")
    try:
        files_data = api_list_files()
        file_list = files_data.get("files", [])
    except Exception:
        file_list = []

    if file_list:
        for f_info in file_list:
            fname = f_info["filename"]
            ftype = f_info["file_type"]
            fsize = f_info["size_human"]
            icon = "📕" if ftype == "pdf" else "📄"
            col_name, col_del = st.columns([5, 1])
            with col_name:
                st.text(f"{icon} {fname}（{fsize}）")
            with col_del:
                if st.button("🗑️", key=f"del_{fname}", help=f"删除 {fname}"):
                    with st.spinner("正在删除并重建索引..."):
                        result = api_delete_file(fname)
                    if result.get("status") == "ok":
                        st.success(result["message"])
                        st.rerun()
                    else:
                        st.error(result.get("message", "删除失败"))
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
        file_bytes = uploaded_file.read()

        with st.spinner("正在上传并处理..."):
            result = api_upload(file_bytes, filename, category or "未知")

        if result.get("status") == "ok":
            st.success(f"✅ 已导入「{result['filename']}」，切分为 {result['chunks']} 个片段")
            st.caption(result.get("message", ""))
            st.rerun()
        else:
            st.error(f"❌ 上传失败：{result.get('message', '未知错误')}")

    st.divider()
    st.caption("💡 提示：上传 txt/pdf 文件即可扩展知识库，AI 会基于这些内容回答问题")


# ============================================================
# 主页面
# ============================================================
st.title("🧾 个体工商户个人助手")
st.caption("基于 RAG 技术，为你提供专业的财务税务解答")

# ---- 检查后端连接 ----
if not check_backend():
    st.error("⚠️ 后端服务未启动！请先运行 `启动RAG服务.bat` 或 `python server.py`")
    st.stop()

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
            try:
                result = api_chat(prompt)
                answer = result.get("answer", "服务返回了空回答")
                sources = result.get("sources", [])
            except Exception as e:
                answer = f"请求失败，请检查后端服务是否正常运行：{e}"
                sources = []

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
