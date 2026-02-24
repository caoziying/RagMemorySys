# AI_Assistant

> 基于 RAG_Memory 的个性化 AI 助手，支持持久记忆、多用户、多会话管理。

---

## 架构说明

本项目与 RAG_Memory **完全解耦**，仅通过 HTTP 接口通信：

```
用户浏览器
    │
    ▼
前端 (Nginx :3000)
    │ REST API
    ▼
后端 (FastAPI :8001)
    │                    │
    │ POST /chat         │ POST /api/v1/chat/memory/query
    │ (DeepSeek LLM)     │ POST /api/v1/chat/memory/upload
    ▼                    ▼
  LLM API          RAG_Memory (:8000)
```

---

## 功能清单

- **用户注册 / 登录**：JWT 认证，多用户独立数据
- **新建 / 删除 / 重命名对话**：完整会话管理
- **流式 AI 回复**：SSE 实时输出，打字机效果
- **RAG 记忆增强**：每次对话前自动查询历史记忆，回答更个性化
- **自动存储对话**：每轮回答后自动上传至 RAG_Memory，记忆持续积累
- **Markdown 渲染**：支持代码块、加粗、列表等格式

---

## 快速开始

### 前置条件

RAG_Memory 项目需先启动（参考其 README），确保 `http://localhost:8000/health` 可访问。

### 启动步骤

```bash
# 1. 进入项目目录
cd AI_Assistant

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 MY_API_KEY 和正确的 RAG_MEMORY_URL

# 3. 一键启动
docker compose up -d

# 4. 访问前端
# 浏览器打开 http://localhost:3000
```

### 环境变量说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MY_API_KEY` | LLM API 密钥（与 RAG_Memory 共用） | — |
| `MY_API_BASE` | LLM API 地址 | `https://api.chat.csu.edu.cn/v1` |
| `MY_MODEL` | 使用的 LLM 模型 | `deepseek-v3-thinking` |
| `RAG_MEMORY_URL` | RAG_Memory 服务地址 | `http://localhost:8000` |
| `SECRET_KEY` | JWT 签名密钥（生产环境必须修改） | 内置默认值 |

---

## RAG_MEMORY_URL 配置指南

| 场景 | 配置值 |
|------|--------|
| RAG_Memory 在宿主机本地运行 | `http://host.docker.internal:8000` |
| RAG_Memory 也在 Docker 中，同一网络 | `http://rag-api:8000` |
| 远程服务器 | `http://<IP>:8000` |

若希望两个 Compose 项目共享同一 Docker 网络，在 `RAG_Memory/docker-compose.yml` 的网络配置中将 `rag-network` 设为 `external: true`，并在本项目 `docker-compose.yml` 中同样引用该网络名称。
