"""
RAG-Skeleton - Streamlit 前端（API 客户端）v3 — 多轮对话支持
=========================================================

通用 AI 知识库骨架的前端界面。换一个知识库，就是一个新应用。

启动前需要先运行：python server.py（后端服务）
然后运行：streamlit run web.py

架构：
  web.py（前端，本文件） → HTTP API → server.py（后端）

新功能：多轮对话会话管理、锚点判断路由、PDF/Excel 上传
"""
import streamlit as st
import requests
import os
import json as _json
import time as _time

# 导入中心化配置
from config import get_config
_cfg = get_config()

# ---- 主题配置（固定深色） ----
bg_primary = "#0a0a0a"
bg_secondary = "#161616"
bg_sidebar = "#0f0f0f"
text_primary = "#e8e8e8"
text_secondary = "#a0a0a0"
border_color = "#2a2a2a"
accent_color = "#6bb5ff"
input_bg = "#161616"
user_bubble = "#1a2a3a"
assistant_bubble = "#131313"
shadow = "none"

# ---- 后端 API 地址（自动适配 ngrok / 本地） ----
_query_params = st.query_params
BACKEND_URL = (
    _query_params.get("backend", [None])[0]
    or os.environ.get("BACKEND_URL")
    or "http://localhost:8000"
)

# ---- 页面配置 ----
st.set_page_config(
    page_title=_cfg.page_title,
    page_icon=_cfg.page_icon,
    layout="centered",
    initial_sidebar_state="expanded",
)

theme_css = f"""
<style>
/* ========== 主题变量 ========== */
.stApp {{
    background-color: {bg_primary};
    color: {text_primary};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}}

/* ========== 侧边栏 ========== */
[data-testid="stSidebar"] {{
    background-color: {bg_sidebar} !important;
    border-right: 1px solid {border_color};
}}

[data-testid="stSidebar"] * {{
    color: {text_primary} !important;
}}

/* ========== 主区域 ========== */
.main .block-container {{
    background-color: {bg_primary};
    padding-top: 1.5rem;
    max-width: 820px;
}}

/* ========== 标题 ========== */
h1 {{
    font-weight: 600;
    font-size: 1.6rem;
    letter-spacing: -0.02em;
    color: {text_primary};
    margin-bottom: 0.2rem;
}}

/* ========== 聊天消息 ========== */
[data-testid="stChatMessage"] {{
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
}}

.stChatMessage {{
    background: transparent !important;
}}

/* 用户消息 */
[data-testid="stChatMessage"][data-testid="user"] {{
    background: {user_bubble} !important;
    border-radius: 16px 16px 4px 16px !important;
    border: 1px solid {border_color} !important;
}}

/* 助手消息 */
[data-testid="stChatMessage"][data-testid="assistant"] {{
    background: {assistant_bubble} !important;
    border-radius: 16px 16px 16px 4px !important;
    border: 1px solid {border_color} !important;
}}

/* 消息头像 */
[data-testid="stChatMessageAvatar"] {{
    border: 2px solid {border_color} !important;
}}

/* ========== 输入框 ========== */
[data-testid="stChatInput"] textarea {{
    background-color: transparent !important;
    border: 1.5px solid {border_color} !important;
    border-radius: 8px !important;
    color: {text_primary} !important;
    box-shadow: none !important;
    outline: none !important;
    transition: border-color 0.2s;
}}

[data-testid="stChatInput"] textarea:focus {{
    border-color: {accent_color} !important;
    outline: none !important;
    box-shadow: none !important;
}}

[data-testid="stChatInput"] {{
    background-color: transparent !important;
}}

/* ========== 按钮（全局统一线条风格） ========== */
.stButton>button,
.stButton>button[kind="primary"],
.stButton>button[kind="secondary"] {{
    background-color: transparent !important;
    border: 1.5px solid {border_color} !important;
    color: {text_primary} !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
    box-shadow: none !important;
    outline: none !important;
    text-shadow: none !important;
}}

.stButton>button:hover,
.stButton>button[kind="primary"]:hover,
.stButton>button[kind="secondary"]:hover,
[data-testid="stSidebar"] .stButton>button:hover {{
    background-color: transparent !important;
    border-color: {accent_color} !important;
    color: {accent_color} !important;
    box-shadow: none !important;
}}

.stButton>button:focus,
.stButton>button:focus-visible,
.stButton>button:focus-within,
.stButton>button[kind="primary"]:focus,
.stButton>button[kind="secondary"]:focus {{
    outline: none !important;
    border-color: {accent_color} !important;
    box-shadow: none !important;
}}

.stButton>button:active {{
    opacity: 0.7 !important;
    transform: none !important;
    box-shadow: none !important;
}}

.stButton>button[kind="primary"] {{
    border-color: {accent_color} !important;
    color: {accent_color} !important;
}}

.stButton>button[kind="primary"]:hover {{
    background-color: transparent !important;
}}

/* ========== 选择器（统一线条风格） ========== */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] [role="combobox"] {{
    background-color: transparent !important;
    border: 1.5px solid {border_color} !important;
    border-radius: 8px !important;
    color: {text_primary} !important;
    box-shadow: none !important;
    outline: none !important;
}}

[data-testid="stSelectbox"] > div > div:focus,
[data-testid="stSelectbox"] [role="combobox"]:focus {{
    border-color: {accent_color} !important;
    outline: none !important;
    box-shadow: none !important;
}}

/* ========== 文本输入（统一线条风格） ========== */
[data-testid="stTextInput"] input,
[data-testid="stTextInput"] > div > div > div > div {{
    background-color: transparent !important;
    border: 1.5px solid {border_color} !important;
    border-radius: 8px !important;
    color: {text_primary} !important;
    box-shadow: none !important;
    outline: none !important;
}}

[data-testid="stTextInput"] input:focus {{
    border-color: {accent_color} !important;
    outline: none !important;
    box-shadow: none !important;
}}

/* ========== 文件上传（统一线条风格） ========== */
[data-testid="stFileUploader"] {{
    border: 1.5px dashed {border_color} !important;
    border-radius: 8px !important;
    background-color: transparent !important;
    box-shadow: none !important;
}}

[data-testid="stFileUploader"]:hover {{
    border-color: {accent_color} !important;
}}

[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {{
    color: {text_secondary} !important;
}}

/* ========== 分割线 ========== */
hr, [data-testid="stSidebar"] hr {{
    border-color: {border_color};
    opacity: 0.6;
}}

/* ========== 文本样式 ========== */
.stCaption, [data-testid="stCaption"] {{
    color: {text_secondary} !important;
    font-size: 0.8rem;
}}

p, span, div, label {{
    color: {text_primary};
}}

/* ========== 提示框 ========== */
[data-testid="stAlert"] {{
    border-radius: 10px;
    border: 1px solid {border_color};
    background-color: {bg_secondary};
}}

/* ========== Expander ========== */
[data-testid="stExpander"] {{
    border: 1px solid {border_color};
    border-radius: 10px;
    background-color: {bg_secondary};
}}

/* ========== Spinner ========== */
.stSpinner > div {{
    border-color: {accent_color} transparent transparent transparent;
}}

/* ========== 来源标签 ========== */
.source-tag {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: {bg_secondary};
    border: 1px solid {border_color};
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    color: {text_secondary};
}}

/* ========== 路由徽章 ========== */
.route-badge {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: {bg_secondary};
    border: 1px solid {border_color};
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: {text_secondary};
    letter-spacing: 0.02em;
}}

/* ========== 滚动条 ========== */
::-webkit-scrollbar {{
    width: 5px;
}}
::-webkit-scrollbar-track {{
    background: transparent;
}}
::-webkit-scrollbar-thumb {{
    background: {border_color};
    border-radius: 10px;
}}

/* ========== 响应式 ========== */
@media (max-width: 768px) {{
    .main .block-container {{
        padding: 1rem;
    }}
    h1 {{
        font-size: 1.3rem;
    }}
}}
</style>
"""

st.markdown(theme_css, unsafe_allow_html=True)

# ---- 支持的格式（线条风格图标） ----
LINE_ICONS = {
    "pdf": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
    "txt": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="12" x2="8" y2="12"/><line x1="16" y1="16" x2="8" y2="16"/></svg>',
    "xlsx": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg>',
    "image": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>',
    "audio": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>',
    "default": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>',
}


def check_backend():
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def api_chat(question, session_id=None):
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    resp = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=60)
    return resp.json()


def api_chat_stream(question, session_id=None):
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    resp = requests.post(
        f"{BACKEND_URL}/chat/stream",
        json=payload,
        stream=True,
        timeout=120,
    )
    # 用可变容器存储，避免 nonlocal 重绑定导致返回值过期
    ctx = {
        "sources": [],
        "returned_session_id": session_id,
        "history_count": 0,
        "route_info": None,
        "progress_messages": [],
    }

    def token_gen():
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data = _json.loads(line[6:])
                t = data["type"]
                if t == "sources":
                    ctx["sources"] = data.get("sources", [])
                elif t == "session":
                    ctx["returned_session_id"] = data["session_id"]
                elif t == "history_count":
                    ctx["history_count"] = data["count"]
                elif t == "route_info":
                    ctx["route_info"] = data
                elif t == "progress":
                    ctx["progress_messages"].append(data.get("message", ""))
                elif t == "token":
                    yield data["content"]
                elif t == "done":
                    break
                elif t == "error":
                    raise RuntimeError(data.get("message", "流式传输错误"))

    return token_gen, ctx


def api_upload(file_bytes, filename, category="未知"):
    resp = requests.post(
        f"{BACKEND_URL}/upload",
        files={"file": (filename, file_bytes)},
        data={"category": category},
        timeout=180,
    )
    return resp.json()


def api_list_files():
    resp = requests.get(f"{BACKEND_URL}/files", timeout=10)
    return resp.json()


def api_delete_file(filename):
    resp = requests.delete(f"{BACKEND_URL}/files/{filename}", timeout=60)
    return resp.json()


def api_list_sessions():
    """获取会话列表"""
    try:
        resp = requests.get(f"{BACKEND_URL}/sessions", timeout=5)
        data = resp.json()
        return data.get("sessions", [])
    except Exception:
        return []


def api_create_session(title=""):
    """新建会话"""
    try:
        resp = requests.post(f"{BACKEND_URL}/sessions", params={"title": title}, timeout=5)
        data = resp.json()
        return data.get("session_id", "")
    except Exception:
        return ""


def api_delete_session(session_id):
    """删除会话"""
    try:
        resp = requests.delete(f"{BACKEND_URL}/sessions/{session_id}", timeout=5)
        return resp.json()
    except Exception:
        return {"status": "error"}


# ============================================================
# 侧边栏 - 知识库管理 + 会话管理
# ============================================================
with st.sidebar:
    st.header("管理面板")

    # ---- 会话管理 ----
    st.subheader("💬 会话管理")

    # 初始化会话列表
    if "sessions" not in st.session_state:
        st.session_state.sessions = []
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None

    # 刷新会话列表
    st.session_state.sessions = api_list_sessions()

    # 新建会话
    col_new, col_refresh = st.columns([3, 1])
    with col_new:
        if st.button("➕ 新建会话", use_container_width=True):
            sid = api_create_session("新会话")
            if sid:
                st.session_state.current_session_id = sid
                st.session_state.messages = []
                st.rerun()
    with col_refresh:
        if st.button("🔄", help="刷新会话列表"):
            st.rerun()

    # 会话选择器
    sessions = st.session_state.sessions
    if sessions:
        session_options = {}
        for s in sessions:
            title = s["title"] or "新会话"
            count = s["message_count"]
            label = f"{title}（{count//2}轮）"
            session_options[s["id"]] = label

        current_id = st.session_state.current_session_id
        # 确保当前会话在选项中
        selected_label = session_options.get(current_id)
        if selected_label is None and current_id:
            session_options[current_id] = "当前会话"

        # 选择器
        selected = st.selectbox(
            "切换会话",
            options=list(session_options.keys()),
            format_func=lambda x: session_options.get(x, x[:8]),
            index=(
                list(session_options.keys()).index(current_id)
                if current_id in session_options
                else 0
            ),
            label_visibility="collapsed",
        )

        if selected and selected != st.session_state.current_session_id:
            st.session_state.current_session_id = selected
            st.session_state.messages = []
            st.rerun()

        # 删除按钮
        if st.session_state.current_session_id:
            if st.button("🗑️ 删除当前会话", type="secondary", use_container_width=True):
                api_delete_session(st.session_state.current_session_id)
                st.session_state.current_session_id = None
                st.session_state.messages = []
                st.rerun()
    else:
        st.caption("暂无会话，点击「新建会话」开始")

    st.divider()

    # ---- 知识库文件管理 ----
    st.subheader("📂 知识库管理")
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
            icon = LINE_ICONS.get(ftype, LINE_ICONS["default"])
            col_name, col_del = st.columns([5, 1])
            with col_name:
                st.markdown(f"{icon} {fname}（{fsize}）", unsafe_allow_html=True)
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

    # ---- 文件上传（支持多模态） ----
    st.subheader("📤 上传文件")
    uploaded_file = st.file_uploader(
        "拖拽或选择文件",
        type=["txt", "pdf", "xlsx", "png", "jpg", "jpeg", "bmp"],
        help="支持：文本/PDF/Excel/图片",
        label_visibility="collapsed",
    )

    if uploaded_file and uploaded_file.name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

    category = st.text_input("分类标签", placeholder=_cfg.category_placeholder, value="")

    if uploaded_file and st.button("确认导入", type="primary", use_container_width=True):
        filename = uploaded_file.name
        file_bytes = uploaded_file.read()

        ext = filename.lower()
        if any(ext.endswith(e) for e in (".png", ".jpg", ".jpeg", ".bmp")):
            spinner_text = "正在 OCR 识别图片..."
        elif ext.endswith(".pdf"):
            spinner_text = "正在解析 PDF..."
        elif ext.endswith((".xlsx", ".xls")):
            spinner_text = "正在解析 Excel..."
        else:
            spinner_text = "正在上传并处理..."

        with st.spinner(spinner_text):
            result = api_upload(file_bytes, filename, category or _cfg.default_category)

        if result.get("status") == "ok":
            msg = f"✅ 已导入「{result['filename']}」，切分为 {result['chunks']} 个片段"
            st.success(msg)
            st.caption(result.get("message", ""))
            preview = result.get("preview")
            if preview:
                with st.expander("📝 识别内容预览"):
                    st.text(preview)
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"❌ 上传失败：{result.get('message', '未知错误')}")

    st.divider()
    st.caption("💡 支持格式：txt / pdf / xlsx")


# ============================================================
# 主页面
# ============================================================
st.title(f"📚 {_cfg.app_name}")
st.caption(_cfg.app_description)

# 显示当前会话信息
if st.session_state.current_session_id:
    st.caption(f"💬 会话: {st.session_state.current_session_id[:8]}...")
else:
    st.caption("💬 无会话（提问将自动创建新会话）")

# ---- 检查后端连接 ----
if not check_backend():
    st.error("⚠️ 后端服务未启动！请先运行 `python server.py`")
    st.stop()

# ---- 聊天记录 ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # 历史消息也显示路由徽章
        if msg["role"] == "assistant" and msg.get("route_info"):
            ri = msg["route_info"]
            r = ri.get("route", "fast")
            hits = ri.get("hits", 0)
            threshold = ri.get("threshold", 2)
            needs_clar = ri.get("needs_clarification", False)
            if r == "fast":
                st.caption(f"🚀 Fast RAG · 命中 {hits}/{threshold} 锚点")
            elif needs_clar:
                st.caption(f"🤔 Agentic RAG 追问 · 命中 {hits}/{threshold} 锚点")
            else:
                st.caption(f"🧠 Agentic RAG · 命中 {hits}/{threshold} 锚点")

        st.markdown(msg["content"])

        # 历史消息也用小字展示引用资料
        if msg["role"] == "assistant" and msg.get("sources"):
            src_lines = []
            for i, src in enumerate(msg["sources"][:5], 1):
                s = src.get("score")
                pct = f"{s*100:.1f}%" if s is not None else "?"
                fname = src.get("metadata", {}).get("source") or src.get("metadata", {}).get("filename", "未知")
                src_lines.append(f"[{i}] {fname} ({pct})")
            st.caption("📚 引用资料: " + " | ".join(src_lines))

            with st.expander("🔍 查看详细片段", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    score = src.get("score")
                    pct_display = f"**{score*100:.1f}%**" if score is not None else "**?**"
                    text = src.get("text", "")
                    meta = src.get("metadata", {})
                    fname = meta.get("source", meta.get("filename", "未知文件"))
                    cat = meta.get("category", "未知")
                    st.markdown(f"**来源 {i}** - {fname} | 相关度: {pct_display} | 分类: {cat}")
                    st.markdown(f"> {text}")
                    st.divider()


def do_chat(prompt_text):
    """执行一次对话（流式），传入文字内容"""
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        sources = []
        answer = ""
        stream_failed = False
        returned_session_id = None
        route_info = None
        progress_messages = []

        try:
            token_gen, ctx = api_chat_stream(
                prompt_text,
                session_id=st.session_state.current_session_id,
            )
            gen = token_gen()

            # ── Peek 第一个 token，提前获取 route_info ──
            first_chunk = None
            try:
                first_chunk = next(gen)
            except StopIteration:
                pass

            route_info = ctx["route_info"]
            progress_messages = ctx.get("progress_messages", [])

            # ── 在答案前显示路由徽章 ──
            if route_info:
                r = route_info.get("route", "fast")
                hits = route_info.get("hits", 0)
                threshold = route_info.get("threshold", 2)
                tokens = route_info.get("tokens", [])
                is_clarification = any("追问" in p for p in progress_messages)
                if r == "fast":
                    st.caption(f"🚀 Fast RAG · 命中 {hits}/{threshold} 锚点: {', '.join(tokens[:5])}")
                elif is_clarification:
                    st.warning(f"🤔 Agentic RAG 追问 · 仅命中 {hits}/{threshold} 锚点，检索匹配度低，需要您换个问法")
                else:
                    st.info(f"🧠 Agentic RAG · 仅命中 {hits}/{threshold} 锚点，尝试多角度检索回答")

            # ── 流式输出答案 ──
            def combined_gen():
                if first_chunk:
                    yield first_chunk
                yield from gen

            answer = st.write_stream(combined_gen())
            sources = ctx["sources"]
            returned_session_id = ctx["returned_session_id"]

            # 更新会话 ID（首次提问时自动创建）
            if returned_session_id and returned_session_id != st.session_state.current_session_id:
                st.session_state.current_session_id = returned_session_id

        except Exception:
            stream_failed = True

        if stream_failed:
            with st.spinner("正在重试（非流式）..."):
                try:
                    result = api_chat(
                        prompt_text,
                        session_id=st.session_state.current_session_id,
                    )
                    answer = result.get("answer", "服务返回了空回答")
                    sources = result.get("sources", [])
                    route_info = result.get("route_info")
                    sid = result.get("session_id")
                    if sid and sid != st.session_state.current_session_id:
                        st.session_state.current_session_id = sid
                except Exception as e:
                    answer = f"请求失败：{e}"
                    sources = []
            st.markdown(answer)

            # 非流式回退也显示路由徽章
            if route_info:
                r = route_info.get("route", "fast")
                hits = route_info.get("hits", 0)
                threshold = route_info.get("threshold", 2)
                needs_clar = route_info.get("needs_clarification", False)
                if r == "fast":
                    st.caption(f"🚀 Fast RAG · 命中 {hits}/{threshold} 锚点")
                elif needs_clar:
                    st.warning(f"🤔 Agentic RAG 追问 · 仅命中 {hits}/{threshold} 锚点，需要澄清")
                else:
                    st.info(f"🧠 Agentic RAG · 仅命中 {hits}/{threshold} 锚点，尝试回答")

        # ── 答案后用小字展示引用资料及匹配度 ──
        if sources:
            src_lines = []
            for i, src in enumerate(sources[:5], 1):
                s = src.get("score")
                pct = f"{s*100:.1f}%" if s is not None else "?"
                fname = src.get("metadata", {}).get("source") or src.get("metadata", {}).get("filename", "未知")
                src_lines.append(f"[{i}] {fname} ({pct})")
            st.caption("📚 引用资料: " + " | ".join(src_lines))

            # 详细片段（默认折叠，小字摘要已够用）
            with st.expander("🔍 查看详细片段", expanded=False):
                for i, src in enumerate(sources, 1):
                    score = src.get("score")
                    pct_display = f"**{score*100:.1f}%**" if score is not None else "**?**"
                    text = src.get("text", "")
                    meta = src.get("metadata", {})
                    fname = meta.get("source", meta.get("filename", "未知文件"))
                    cat = meta.get("category", "未知")
                    st.markdown(f"**来源 {i}** - {fname} | 相关度: {pct_display} | 分类: {cat}")
                    st.markdown(f"> {text}")
                    st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "route_info": route_info,
        })


# ---- 输入框 ----
if prompt := st.chat_input("在这里输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    do_chat(prompt)
