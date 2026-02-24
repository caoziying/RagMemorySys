"""
app/main.py
===========
程序入口：FastAPI 实例初始化、中间件注册、路由挂载与生命周期管理。

启动方式：
  开发环境：uvicorn app.main:app --reload
  生产环境：uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.api.endpoints import router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logger import get_logger, setup_logging

# 初始化日志系统（必须在所有其他模块之前）
setup_logging()
logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# 应用生命周期管理
# ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    FastAPI 应用生命周期钩子（替代已废弃的 on_event 方式）。

    startup：启动时执行初始化任务（目录创建、连通性检查等）。
    shutdown：关闭时执行清理任务（释放连接等）。
    """
    # ── Startup ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("RAG_Memory 服务启动中...")
    logger.info("  API Base URL : {}", settings.my_api_base)
    logger.info("  LLM Model    : {}", settings.my_model)
    logger.info("  Embed Model  : {}", settings.my_embedding_model)
    logger.info("  Milvus       : {}:{}", settings.milvus_host, settings.milvus_port)
    logger.info("  Reranker URL : {}", settings.reranker_url)
    logger.info("  Data Dir     : {}", settings.data_dir)
    logger.info("  Log Dir      : {}", settings.log_dir)
    logger.info("=" * 60)

    # 确保数据与日志目录存在
    import os
    for path in [
        settings.data_dir,
        f"{settings.data_dir}/users",
        settings.log_dir,
        f"{settings.log_dir}/system",
        f"{settings.log_dir}/conversations",
    ]:
        os.makedirs(path, exist_ok=True)

    logger.info("RAG_Memory 服务启动完成，准备接受请求。")

    yield  # 应用运行阶段

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("RAG_Memory 服务正在关闭，执行清理任务...")
    logger.info("RAG_Memory 服务已安全关闭。")


# ──────────────────────────────────────────────────────────────
# FastAPI 实例创建
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG_Memory - 多用户对话记忆管理系统",
    description=(
        "即插即用的 AI 对话记忆管理微服务。\n\n"
        "采用基础记忆与增强向量检索解耦的分层架构，\n"
        "支持多用户隔离、自动用户画像构建与对话历史压缩。\n\n"
        "**快速开始**：\n"
        "1. `POST /api/v1/chat/memory/query` - 查询记忆上下文\n"
        "2. `POST /api/v1/chat/memory/upload` - 上传对话历史"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────
# 中间件注册
# ──────────────────────────────────────────────────────────────

# CORS 跨域支持（生产环境请缩小 allow_origins 范围）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 开发阶段允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求日志中间件（记录每次请求的基础信息）
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import time


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """记录每个 HTTP 请求的方法、路径与响应时间。"""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "{} {} → {} | {:.1f}ms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response


app.add_middleware(RequestLoggingMiddleware)


# ──────────────────────────────────────────────────────────────
# 全局异常处理器注册
# ──────────────────────────────────────────────────────────────

register_exception_handlers(app)


# ──────────────────────────────────────────────────────────────
# 路由挂载
# ──────────────────────────────────────────────────────────────

# 业务路由：统一挂载到 /api/v1 前缀
app.include_router(router, prefix="/api/v1", tags=["记忆管理"])

# 健康检查路由（同时在根路径和 /api/v1/health 提供）
from fastapi.responses import JSONResponse

@app.get("/health", tags=["运维"], summary="根路径健康检查")
async def root_health() -> JSONResponse:
    """根路径健康检查，供 Docker HEALTHCHECK 使用。"""
    return JSONResponse({"status": "ok", "service": "RAG_Memory", "version": "1.0.0"})


@app.get("/", tags=["运维"], summary="服务根路径")
async def root() -> JSONResponse:
    """返回服务基础信息。"""
    return JSONResponse({
        "service": "RAG_Memory",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    })
