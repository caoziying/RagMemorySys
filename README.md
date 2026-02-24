# RAG Memory System

> 即插即用的多用户对话记忆管理系统，专为 AI 对话代理设计。

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [API 接口文档](#api-接口文档)
- [配置说明](#配置说明)
- [工作流详解](#工作流详解)
- [降级策略](#降级策略)
- [已知问题与修复记录](#已知问题与修复记录)

---

## 项目简介

RAG_Memory 是一个以 **FastAPI** 为核心的微服务，为外部 AI 对话代理提供持久化的多用户对话记忆能力。它采用**基础记忆与向量检索解耦**的分层架构：

- **基础记忆层**：本地文件（JSONL + Markdown），维护滑动窗口历史与 LLM 自动压缩摘要。
- **向量检索层**：Milvus 向量数据库，支持语义级别的历史记忆召回与重排。
- **用户画像层**：LLM 自动从对话中提取用户信息，持久化为 `user.md`。

外部 AI 代理只需在每轮对话前后调用两个标准 HTTP 接口，即可获得「记忆增强的上下文」与「自动更新的用户画像」，无需关心底层存储细节。

---

## 核心特性

- **多租户隔离**：所有数据以 `user_id` 为键严格隔离，向量检索带过滤条件，文件存储独立目录。
- **自动用户画像**：每次对话后异步触发 LLM 提取用户信息，增量合并至 `user.md`，无需手动维护。
- **滑动窗口 + 自动压缩**：本地历史保留最近 N 条，超阈值后 LLM 自动生成摘要压缩旧记忆。
- **三级重排降级链**：本地 Reranker HTTP → Embedding 余弦相似度 → 原始召回顺序，任意环节失败自动降级。
- **Milvus 懒加载重连**：服务启动时 Milvus 未就绪不影响 API 可用性，每次操作前自动重连。
- **完全解耦的 Prompt 管理**：所有 LLM Prompt 集中在 `app/prompts/`，业务代码零散落。
- **结构化日志**：系统运行日志 + 每用户每日对话明细双轨记录。

---

## 系统架构

```
外部 AI 代理
     │
     │  POST /api/v1/chat/memory/query   （对话前：获取记忆上下文）
     │  POST /api/v1/chat/memory/upload  （对话后：存储本轮记忆）
     ▼
┌─────────────────────────────────────────────┐
│               FastAPI (rag-api)              │
│                                             │
│  ┌──────────────┐    ┌─────────────────┐   │
│  │  基础记忆层   │    │   向量检索层     │   │
│  │ history.jsonl│    │  Milvus 向量库   │   │
│  │ compressed.md│    │  Embedding API  │   │
│  │   user.md    │    │  Reranker HTTP  │   │
│  └──────────────┘    └─────────────────┘   │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │     LLM（信息提取 / 对话压缩）        │  │
│  │     app/prompts/ （解耦的 Prompt）    │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
     │
     ├── Milvus Standalone（向量数据库）
     ├── Attu（Milvus 可视化管理，:8080）
     └── 本地 Reranker HTTP 服务（外部提供）
```

---

## 目录结构

```
RAG_Memory/
├── app/
│   ├── api/
│   │   ├── endpoints.py        # 路由定义（query / upload / health）
│   │   └── schemas.py          # Pydantic 请求/响应模型
│   ├── core/
│   │   ├── config.py           # 全局配置（读取 .env，单例）
│   │   ├── logger.py           # 双层日志（控制台 + 文件）
│   │   └── exceptions.py       # 全局异常体系与处理器
│   ├── prompts/
│   │   ├── extraction.py       # 用户信息提取 Prompt 模板
│   │   └── summarization.py    # 对话历史压缩 Prompt 模板
│   ├── memory/
│   │   ├── manager.py          # 滑动窗口管理、对话压缩逻辑
│   │   └── profile.py          # user.md 的提取与读写
│   ├── retrieval/
│   │   ├── chunking.py         # 滑动窗口文本分块
│   │   ├── embeddings.py       # 向量化服务（调用自定义 API）
│   │   ├── milvus_client.py    # Milvus 连接、插入、检索
│   │   ├── reranker.py         # 三级降级重排策略
│   │   └── retriever.py        # 核心调度器（串联完整 RAG 流程）
│   ├── llm/
│   │   └── client.py           # LLM 客户端（OpenAI SDK + LangChain）
│   └── main.py                 # FastAPI 实例、中间件、路由注册
├── data/
│   └── users/
│       └── {user_id}/
│           ├── user.md         # 用户画像（LLM 自动维护）
│           ├── history.jsonl   # 对话历史（滑动窗口）
│           └── compressed.md  # 历史摘要（超阈值后 LLM 压缩生成）
├── logs/
│   ├── system/
│   │   └── system.log          # 系统运行日志（按天轮转）
│   └── conversations/
│       └── {user_id}_YYYY-MM-DD.md  # 每用户每日对话明细
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── .env
```

---

## 快速开始

### 前置要求

- Docker & Docker Compose v2+
- 可用的 OpenAI 兼容 API（支持 Chat + Embedding 模型）
- 本地 Reranker HTTP 服务（可选，不可用时自动降级）

### 1. 克隆并配置环境变量

```bash
git clone <your-repo-url> RAG_Memory
cd RAG_Memory

cp .env.example .env
```

编辑 `.env`，至少填写以下必填项：

```env
MY_API_KEY=your_actual_api_key
MY_API_BASE=https://api.chat.csu.edu.cn/v1
MY_MODEL=deepseek-v3-thinking
MY_EMBEDDING_MODEL=bge-m3

# 若 Reranker 运行在宿主机，容器内需用 host.docker.internal
RERANKER_URL=http://host.docker.internal:8080/rerank
```

### 2. 启动服务

```bash
docker compose up -d
```

首次启动会拉取 Milvus 相关镜像（约 2~3 分钟），可通过以下命令观察进度：

```bash
docker compose logs -f rag-api
```

看到以下输出表示启动成功：

```
INFO  | RAG_Memory 服务启动完成，准备接受请求。
```

### 3. 验证服务

```bash
curl http://localhost:8000/health
# {"status":"ok","service":"RAG_Memory","version":"1.0.0"}
```

Milvus 可视化管理界面：http://localhost:8080

---

## API 接口文档

交互式文档（Swagger UI）：http://localhost:8000/docs

### POST `/api/v1/chat/memory/query`

对话前调用，返回与当前查询相关的历史记忆及用户画像，供 AI 代理注入系统 Prompt。

**请求体：**

```json
{
  "user_id": "user_12345",
  "query": "我上周说的那个项目怎么样了？",
  "time": "2024-10-27T10:00:00Z"
}
```

**响应体：**

```json
{
  "success": true,
  "message": "查询成功",
  "user_id": "user_12345",
  "user_profile": "## 职业\n- Python 后端工程师\n\n## 项目\n- 正在开发 RAG 系统",
  "retrieved_chunks": [
    {
      "content": "[user]: 向量检索部分已经完成了，下周准备做 Reranker 模块。",
      "score": 0.923,
      "source": "milvus",
      "metadata": { "timestamp": "2024-10-27T09:00:00Z" }
    }
  ],
  "augmented_context": "## 用户画像\n...\n\n## 相关历史记忆\n...",
  "query_time_ms": 412.5
}
```

`augmented_context` 字段可直接拼接到 AI 代理的系统 Prompt 中使用。

---

### POST `/api/v1/chat/memory/upload`

对话后调用，将本轮对话内容存入记忆系统，并异步触发用户画像更新。

**请求体（对话消息格式）：**

```json
{
  "user_id": "user_12345",
  "messages": [
    {"role": "user", "content": "我是张三，Python 工程师，正在做 RAG 项目。"},
    {"role": "assistant", "content": "好的，请问目前进展如何？"},
    {"role": "user", "content": "向量检索完成了，下周做 Reranker。"}
  ],
  "time": "2024-10-27T10:05:00Z"
}
```

**请求体（Base64 文件格式）：**

```json
{
  "user_id": "user_12345",
  "multifiles": ["base64编码的文件内容..."],
  "time": "2024-10-27T10:05:00Z"
}
```

**响应体：**

```json
{
  "success": true,
  "message": "上传成功，画像更新已在后台异步进行",
  "user_id": "user_12345",
  "chunks_stored": 3,
  "profile_updated": false,
  "process_time_ms": 3771.8
}
```

---

## 配置说明

所有配置通过 `.env` 文件管理，完整说明见 `.env.example`。

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `MY_API_KEY` | OpenAI 兼容 API 密钥 | — |
| `MY_API_BASE` | API 基础地址 | `https://api.chat.csu.edu.cn/v1` |
| `MY_MODEL` | 对话/提取用 LLM 模型名 | `deepseek-v3-thinking` |
| `MY_EMBEDDING_MODEL` | Embedding 模型名 | `bge-m3` |
| `MILVUS_HOST` | Milvus 服务地址 | `milvus-standalone` |
| `MILVUS_PORT` | Milvus gRPC 端口 | `19530` |
| `MILVUS_DIM` | 向量维度（需与 Embedding 模型匹配） | `1024` |
| `RERANKER_URL` | 本地 Reranker HTTP 接口地址 | `http://localhost:8080/rerank` |
| `MEMORY_WINDOW_SIZE` | 滑动窗口保留对话轮数 | `10` |
| `MEMORY_COMPRESS_THRESHOLD` | 触发压缩的历史条数阈值 | `20` |
| `RETRIEVAL_TOP_K` | 向量召回数量 | `10` |
| `RERANK_TOP_N` | 重排后返回数量 | `5` |

> ⚠️ `MILVUS_DIM` 必须与实际 Embedding 模型输出维度一致，否则插入时报维度不匹配错误。`bge-m3` 输出维度为 `1024`。

---

## 工作流详解

### Query 流程（对话前）

```
请求到达
  → 向量化 query（embeddings.py）
  → Milvus ANN 检索 Top-K（按 user_id 过滤）
  → 三级重排 → Top-N 结果
  → 读取 user.md 用户画像
  → 合并构造 augmented_context
  → 返回响应
  → [后台] 记录对话日志
```

### Upload 流程（对话后）

```
请求到达
  → 解析 messages / Base64 文件为文本
  → 文本分块（滑动窗口 chunking）
  → 批量向量化
  → 写入 Milvus
  → 更新本地滑动窗口历史（history.jsonl）
  → 若历史条数超阈值，LLM 自动压缩 → compressed.md
  → 返回响应
  → [后台异步] LLM 提取用户信息 → 合并更新 user.md
```

---

## 降级策略

系统任意组件故障均不会导致主请求失败，具体降级行为如下：

| 组件 | 故障时行为 |
|------|-----------|
| Milvus 不可用 | 每次操作前自动重连；重连失败时 insert 返回 0，search 返回空列表 |
| Reranker HTTP 超时/连接失败 | 自动切换为 Embedding 余弦相似度重排 |
| Embedding API 失败 | 切换为原始召回顺序直接返回 |
| LLM 提取/压缩失败 | 后台任务静默失败，仅记录日志，不影响主响应 |
| user.md 不存在 | 返回空字符串，不报错 |

---

## 已知问题与修复记录

| 版本 | 问题 | 根因 | 修复方案 |
|------|------|------|---------|
| v1 | 分块结果为空，跳过存储 | `MIN_CHUNK_SIZE=50`，对话短消息全部被过滤 | `min_chunk_size` 改为 `10` |
| v1 | `chunk_idx` 全局编号错误 | `chunk_texts()` 重编号逻辑有 bug | 改为顺序 append 赋值 |
| v2 | Milvus 已启动但报"未连接" | `Retriever` 单例初始化时 Milvus 未就绪，`_connected` 永远 `False` | `insert/search` 前调用 `_ensure_connected()` 自动重连 |
| v3 | 重连时报索引冲突 | `connect()` 预调用 `disconnect()` 导致服务端 release collection，再 `load()` 触发索引 ID 冲突 | 移除 `disconnect()` 预调用 |
| v4 | 重连仍报索引冲突 | Collection 已处于 Loaded 状态，重复调用 `load()` 触发校验报错 | 用 `utility.load_state()` 检查，已加载则跳过 `load()` |
| v5 | 检索时 `TypeError: Hit.get() takes 2 positional arguments` | pymilvus 2.4.x `Hit.get()` 不支持默认值参数 | 改为 `entity.get(field) or ""` |
