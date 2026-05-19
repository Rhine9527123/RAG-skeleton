"""
RAG MCP Server - Hermes 的"财务知识库翻译官"

作用：把 Hermes 的 MCP 协议调用，翻译成对 RAG FastAPI 的 HTTP 请求

架构：
  用户 → Hermes → MCP(stdio) → 本文件(HTTP) → server.py:8000 → ChromaDB

启动方式：不需要手动启动，Hermes 会自动通过 stdio 启动这个脚本

暴露给 Hermes 的工具：
  1. rag_chat    - 财务知识库问答（核心）
  2. rag_files   - 查看知识库文件列表
  3. rag_delete  - 删除知识库文件

依赖：pip install mcp requests
"""

import json
import requests
from mcp.server.fastmcp import FastMCP

# ============================================================
# 配置：RAG FastAPI 服务的地址
# ============================================================
RAG_BASE_URL = "http://localhost:8000"

# ============================================================
# 创建 MCP Server 实例
# ============================================================
mcp = FastMCP(
    name="rag-finance",
    instructions=(
        "你连接了用户的专属知识库，里面可能包含各种业务文档："
        "税务政策、会计数据、经营分析、天气影响、竞品调研等。"
        "内容类型不固定，取决于用户上传了什么。"
        "\n\n判断逻辑："
        "\n- 涉及「用户自己的数据/文档/业务」→ 调用 rag_chat"
        "\n- 纯闲聊/通用常识/你确定能答对 → 不调用"
        "\n- 拿不准 → 宁可调用，查了再说"
        "\n\n重要：不要根据话题类型（如'天气''财务'）硬性判断，"
        "因为用户可能上传了任何主题的文档。"
        "如果需要确认知识库里有什么，先调用 rag_files 查看。"
    ),
)


# ============================================================
# 工具 1：财务知识库问答（核心）
# ============================================================
@mcp.tool()
def rag_chat(question: str) -> str:
    """
    从用户的知识库中检索相关文档并生成回答。

    知识库是用户专属的，内容不固定（可能含税务政策、经营数据、分析报告等任何文档）。
    只要问题涉及「用户的业务、数据或上传的资料」，就应该调用此工具。

    示例：
    - 「这个月利润为什么降了」→ 调用（涉及用户自己的经营数据）
    - 「天气对我生意有什么影响」→ 调用（用户可能上传过天气分析文档）
    - 「小规模纳税人怎么报税」→ 调用（涉及具体政策）
    - 「你好」「1+1等于几」→ 不调用（纯闲聊/通用常识）

    参数：
    - question: 用户的问题，直接传入原话即可
    """
    try:
        resp = requests.post(
            f"{RAG_BASE_URL}/chat",
            json={"question": question},
            timeout=60,
        )
        data = resp.json()

        answer = data.get("answer", "未获取到回答")
        sources = data.get("sources", [])

        # 把来源片段拼到回答后面，方便 Hermes 引用
        result = f"**回答：**\n{answer}"

        if sources:
            result += "\n\n**参考来源：**\n"
            for i, src in enumerate(sources, 1):
                text = src.get("text", "")[:100]  # 每条来源只取前100字
                meta = src.get("metadata", {})
                source_file = meta.get("filename", "未知")
                result += f"{i}. [{source_file}] {text}...\n"

        return result

    except requests.exceptions.ConnectionError:
        return "错误：RAG 服务未启动。请先运行 server.py（http://localhost:8000）"
    except requests.exceptions.Timeout:
        return "错误：RAG 服务响应超时（60秒），请稍后重试"
    except Exception as e:
        return f"错误：{str(e)}"


# ============================================================
# 工具 2：查看知识库文件列表
# ============================================================
@mcp.tool()
def rag_files() -> str:
    """
    查看当前财务知识库中有哪些文件。

    适用场景：用户想知道知识库里有什么文档、上传了哪些文件时调用。

    返回：文件列表，包含文件名、类型、大小、上传时间。
    """
    try:
        resp = requests.get(
            f"{RAG_BASE_URL}/files",
            timeout=10,
        )
        data = resp.json()

        files = data.get("files", [])
        total = data.get("total", 0)

        if total == 0:
            return "知识库当前为空，没有上传任何文件。"

        result = f"知识库共有 {total} 个文件：\n\n"
        for f in files:
            result += (
                f"- {f['filename']} "
                f"({f['file_type']}, {f['size_human']}, "
                f"更新于 {f['modified']})\n"
            )

        return result

    except requests.exceptions.ConnectionError:
        return "错误：RAG 服务未启动。请先运行 server.py（http://localhost:8000）"
    except Exception as e:
        return f"错误：{str(e)}"


# ============================================================
# 工具 3：删除知识库文件
# ============================================================
@mcp.tool()
def rag_delete(filename: str) -> str:
    """
    删除财务知识库中的指定文件。

    适用场景：用户想要移除某个过时或错误的文档时调用。

    参数：
    - filename: 要删除的文件名，例如 "tax_policy.txt"

    注意：删除后会自动重建索引，可能需要约30秒。

    返回：删除结果。
    """
    try:
        resp = requests.delete(
            f"{RAG_BASE_URL}/files/{filename}",
            timeout=60,
        )
        data = resp.json()

        status = data.get("status", "")
        message = data.get("message", "")

        if status == "ok":
            return f"操作成功：{message}"
        else:
            return f"操作失败：{message}"

    except requests.exceptions.ConnectionError:
        return "错误：RAG 服务未启动。请先运行 server.py（http://localhost:8000）"
    except Exception as e:
        return f"错误：{str(e)}"


# ============================================================
# 启动 MCP Server（通过 stdio 与 Hermes 通信）
# ============================================================
if __name__ == "__main__":
    mcp.run(transport="stdio")
