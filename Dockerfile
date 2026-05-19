# ============================================================
# RAG 财务知识库服务 - Docker 镜像构建文件
# ============================================================
# 构建命令：docker build -t rag-service .
# 运行命令：docker run -p 8000:8000 rag-service
# （实际用 docker compose 启动，不用手动 run）
# ============================================================

# ---- 基础镜像 ----
# python:3.13-slim 是精简版，只有 Python + 最小系统库，体积小
FROM python:3.13-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    DATA_DIR=/app/data \
    VECTOR_INDEX_DIR=/app/chroma_data_server \
    CHUNK_SIZE=256 \
    CHUNK_OVERLAP=50 \
    TOP_K=10 \
    TOP_N=3 \
    RERANKER_MODEL_PATH=/models/bge-reranker-v2-m3

# ---- 安装系统依赖 ----
# tesseract=OCR支持 pandas/openpyxl=Excel解析 需要的基础库
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

# ---- 安装 Python 依赖 ----
# requirements.txt 在同一目录下，先复制进来装依赖（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- 复制模型文件 ----
# 把 Reranker 模型打包进镜像（2.2GB），用户不用单独下载
COPY models/bge-reranker-v2-m3 /models/bge-reranker-v2-m3

# ---- 复制项目代码 ----
COPY server.py /app/server.py

# 创建数据目录（挂载 Volume 时宿主机的文件会映射到这里）
RUN mkdir -p /app/data /app/chroma_data_server

# 工作目录
WORKDIR /app

# 暴露端口（FastAPI 默认 8000）
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
