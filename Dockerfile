# ============================================================
# RAG_Memory API 服务 Dockerfile
# 基于精简的 Python 镜像，多阶段构建保持镜像体积最小
# ============================================================

FROM python:3.11-slim-bookworm AS base

# 设置工作目录
WORKDIR /app

# 设置环境变量，防止 Python 生成 .pyc 文件并关闭缓冲
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- 依赖安装阶段 ----
FROM base AS builder

# 安装系统级构建依赖
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#     && rm -rf /var/lib/apt/lists/*
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources \
    && sed -i 's/security.debian.org\/debian-security/mirrors.tuna.tsinghua.edu.cn\/debian-security/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ---- 运行阶段 ----
FROM base AS runner

# 从 builder 阶段复制安装好的依赖
COPY --from=builder /install /usr/local

# 创建运行时所需的数据/日志目录
RUN mkdir -p /app/data/users /app/logs/system /app/logs/conversations

# 复制应用源代码
COPY app/ ./app/

# 暴露 API 端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 启动命令：使用 uvicorn 运行 FastAPI 应用
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
