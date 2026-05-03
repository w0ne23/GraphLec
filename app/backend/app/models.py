import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, UniqueConstraint
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


class Lecture(Base):
    """실제 강의 콘텐츠 데이터"""
    __tablename__ = "lectures"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id      = Column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=True)
    stem        = Column(String, unique=True, index=True, nullable=False)
    title       = Column(String, nullable=True)
    category    = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    video_path  = Column(Text, nullable=False)
    output_dir  = Column(Text, nullable=False)
    graphrag_workspace = Column(Text, nullable=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("Job", backref="content")


class GraphSession(Base):
    """강의별 Neo4j 로드 상태 관리를 위한 세션"""
    __tablename__ = "graph_sessions"
    __table_args__ = (
        UniqueConstraint("lecture_id", "session_id", name="uq_graph_sessions_lecture_session"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lecture_id = Column(UUID(as_uuid=True), ForeignKey("lectures.id", ondelete="CASCADE"), nullable=False, index=True)
    stem = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    last_heartbeat_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())