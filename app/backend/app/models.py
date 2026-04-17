import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class Job(Base):
    """분석 작업 대기열 및 상태 관리 전용"""
    __tablename__ = "jobs"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status          = Column(String, nullable=False, default="pending")  # pending, running, done, error
    input_path      = Column(Text, nullable=False)
    current_stage   = Column(Text, nullable=True)
    error_message   = Column(Text, nullable=True)
    pipeline_stages = Column(JSONB, nullable=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    updated_at      = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class LectureContent(Base):
    """실제 강의 콘텐츠 데이터"""
    __tablename__ = "lectures_content"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id      = Column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=True)
    stem        = Column(String, unique=True, index=True, nullable=False)
    title       = Column(String, nullable=True)
    category    = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    video_path  = Column(Text, nullable=False)
    output_dir  = Column(Text, nullable=False)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("Job", backref="content")