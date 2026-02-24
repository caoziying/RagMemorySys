"""
database.py - 数据库初始化与会话管理
使用 SQLAlchemy 异步引擎 + SQLite
"""
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = "sqlite+aiosqlite:///./ai_assistant.db"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    from app.models import User, Conversation, Message  # noqa
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
