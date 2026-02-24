"""
main.py - AI_Assistant 后端主程序

接口清单：
  POST /auth/register          注册
  POST /auth/login             登录
  GET  /conversations          获取会话列表
  POST /conversations          新建会话
  PATCH /conversations/{id}    重命名会话
  DELETE /conversations/{id}   删除会话
  GET  /conversations/{id}/messages  获取消息历史
  POST /conversations/{id}/chat      流式对话（SSE）
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import create_access_token, decode_token, hash_password, verify_password
from app.chat import chat_stream
from app.database import get_db, init_db
from app.models import Conversation, Message, User


# ── 生命周期 ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    print("AI_Assistant 后端启动完成")
    yield


app = FastAPI(title="AI_Assistant Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 认证依赖 ──────────────────────────────────────────────────

async def get_current_user(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未提供认证令牌")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="令牌无效或已过期")
    result = await db.execute(select(User).where(User.id == payload["sub"]))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user


# ── Schemas ───────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(min_length=2, max_length=32)
    password: str = Field(min_length=6, max_length=64)

class LoginRequest(BaseModel):
    username: str
    password: str

class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=128)

class ChatRequest(BaseModel):
    content: str = Field(min_length=1)


# ── 认证接口 ──────────────────────────────────────────────────

@app.post("/auth/register")
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == req.username))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="用户名已存在")
    user = User(username=req.username, hashed_password=hash_password(req.password))
    db.add(user)
    await db.commit()
    await db.refresh(user)
    token = create_access_token(user.id, user.username)
    return {"token": token, "user_id": user.id, "username": user.username}


@app.post("/auth/login")
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == req.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(user.id, user.username)
    return {"token": token, "user_id": user.id, "username": user.username}


# ── 会话管理接口 ──────────────────────────────────────────────

@app.get("/conversations")
async def list_conversations(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    )
    convs = result.scalars().all()
    return [
        {"id": c.id, "title": c.title, "created_at": c.created_at.isoformat(),
         "updated_at": c.updated_at.isoformat()}
        for c in convs
    ]


@app.post("/conversations", status_code=201)
async def create_conversation(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    conv = Conversation(user_id=user.id, title="新对话")
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return {"id": conv.id, "title": conv.title, "created_at": conv.created_at.isoformat()}


@app.patch("/conversations/{conv_id}")
async def rename_conversation(
    conv_id: str,
    req: RenameRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation).where(Conversation.id == conv_id, Conversation.user_id == user.id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    conv.title = req.title
    await db.commit()
    return {"id": conv.id, "title": conv.title}


@app.delete("/conversations/{conv_id}", status_code=204)
async def delete_conversation(
    conv_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation).where(Conversation.id == conv_id, Conversation.user_id == user.id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    await db.delete(conv)
    await db.commit()


@app.get("/conversations/{conv_id}/messages")
async def get_messages(
    conv_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conv_id, Conversation.user_id == user.id)
        .options(selectinload(Conversation.messages))
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    return [
        {"id": m.id, "role": m.role, "content": m.content,
         "created_at": m.created_at.isoformat()}
        for m in conv.messages
    ]


# ── 流式对话接口 ──────────────────────────────────────────────

@app.post("/conversations/{conv_id}/chat")
async def chat(
    conv_id: str,
    req: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    SSE 流式对话。
    1. 保存用户消息到数据库
    2. 读取历史消息（供 LLM 使用）
    3. 流式生成回答
    4. 在流结束后保存 assistant 消息（由 stream 内部完成）
    """
    # 验证会话归属
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conv_id, Conversation.user_id == user.id)
        .options(selectinload(Conversation.messages))
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 保存用户消息
    user_msg = Message(conversation_id=conv_id, role="user", content=req.content)
    db.add(user_msg)

    # 若是新对话，自动以首条消息前20字作为标题
    if conv.title == "新对话" and not conv.messages:
        conv.title = req.content[:20] + ("..." if len(req.content) > 20 else "")

    await db.commit()

    # 准备历史消息（不含刚保存的 user_msg，chat_stream 内部会追加）
    history = [{"role": m.role, "content": m.content} for m in conv.messages]

    # 用于流结束后保存 assistant 消息的闭包
    async def stream_with_save():
        full_content = []
        async for chunk in chat_stream(
            user_id=user.id,   # 用 user.id 作为 RAG_Memory 的 user_id
            query=req.content,
            history=history,
        ):
            if chunk != "data: [DONE]\n\n":
                import json as _json
                try:
                    data = _json.loads(chunk[6:])  # strip "data: "
                    if "content" in data:
                        full_content.append(data["content"])
                except Exception:
                    pass
            yield chunk

        # 流结束后保存 assistant 消息
        if full_content:
            async with db.begin():
                assistant_msg = Message(
                    conversation_id=conv_id,
                    role="assistant",
                    content="".join(full_content),
                )
                db.add(assistant_msg)

    return StreamingResponse(
        stream_with_save(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "AI_Assistant"}
