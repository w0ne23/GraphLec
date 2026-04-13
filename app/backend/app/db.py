import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.models import Base

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@db/graphlec")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    """DB 초기화: 모든 모델 테이블 생성"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("--- [DB] Database initialized via SQLAlchemy models. ---")

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
