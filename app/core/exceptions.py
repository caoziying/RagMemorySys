"""
app/core/exceptions.py
======================
全局异常定义与 FastAPI 异常处理器注册。

设计原则：
  - 业务异常继承自 RAGMemoryException，携带 HTTP 状态码与结构化信息。
  - 所有未捕获的内部错误统一返回 500，不向客户端暴露内部细节。
  - 通过 register_exception_handlers(app) 统一挂载到 FastAPI 实例。
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger


# ──────────────────────────────────────────────────────────────
# 自定义业务异常体系
# ──────────────────────────────────────────────────────────────

class RAGMemoryException(Exception):
    """RAG_Memory 系统基础异常，所有业务异常的父类。"""

    def __init__(self, message: str, status_code: int = 500, detail: str | None = None):
        """
        Args:
            message:     对客户端展示的简洁错误描述。
            status_code: HTTP 状态码，默认 500。
            detail:      内部调试信息（不返回给客户端）。
        """
        self.message = message
        self.status_code = status_code
        self.detail = detail or message
        super().__init__(self.message)


class MilvusUnavailableError(RAGMemoryException):
    """Milvus 数据库连接失败或不可用。"""

    def __init__(self, detail: str = ""):
        super().__init__(
            message="向量数据库暂时不可用，已使用降级策略。",
            status_code=503,
            detail=detail,
        )


class EmbeddingError(RAGMemoryException):
    """Embedding 服务调用失败。"""

    def __init__(self, detail: str = ""):
        super().__init__(
            message="向量化服务调用失败，请稍后重试。",
            status_code=502,
            detail=detail,
        )


class RerankerError(RAGMemoryException):
    """Reranker 服务调用失败（可降级）。"""

    def __init__(self, detail: str = ""):
        super().__init__(
            message="重排服务暂时不可用，已返回原始召回结果。",
            status_code=503,
            detail=detail,
        )


class UserProfileError(RAGMemoryException):
    """用户画像文件读写失败。"""

    def __init__(self, detail: str = ""):
        super().__init__(
            message="用户画像数据读写异常。",
            status_code=500,
            detail=detail,
        )


class LLMClientError(RAGMemoryException):
    """LLM 客户端调用失败。"""

    def __init__(self, detail: str = ""):
        super().__init__(
            message="大语言模型调用失败，请检查 API 配置。",
            status_code=502,
            detail=detail,
        )


class InvalidRequestError(RAGMemoryException):
    """请求参数非法或缺失。"""

    def __init__(self, detail: str = ""):
        super().__init__(
            message=f"请求参数无效：{detail}",
            status_code=400,
            detail=detail,
        )


# ──────────────────────────────────────────────────────────────
# FastAPI 异常处理器
# ──────────────────────────────────────────────────────────────

def register_exception_handlers(app: FastAPI) -> None:
    """
    向 FastAPI 实例注册全局异常处理器。
    应在 app/main.py 中调用。

    Args:
        app: FastAPI 应用实例。
    """

    @app.exception_handler(RAGMemoryException)
    async def rag_exception_handler(request: Request, exc: RAGMemoryException) -> JSONResponse:
        """处理所有 RAGMemoryException 及其子类。"""
        logger.warning(
            "业务异常 [{}] {} | 路径: {} | 详情: {}",
            exc.status_code,
            exc.message,
            request.url.path,
            exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.message,
                "code": exc.status_code,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """兜底处理所有未捕获的内部异常，防止内部细节泄露给客户端。"""
        logger.exception("未预期的内部错误 | 路径: {} | 异常: {}", request.url.path, str(exc))
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "服务器内部错误，请联系管理员。",
                "code": 500,
            },
        )
