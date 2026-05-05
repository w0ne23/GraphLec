import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class ProcessingJob(Base):
    """파이프라인 실행 로그 — Lecture당 N개 (다대일)"""
    __tablename__ = "processing_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lecture_id = Column(
        UUID(as_uuid=True),
        ForeignKey("lectures.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status = Column(String, nullable=False, default="pending")  # pending, running, done, error
    current_stage = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    pipeline_stages = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    lecture = relationship("Lecture", back_populates="processing_jobs")


class Lecture(Base):
    """강의 영상의 영구 데이터 — 분석과 무관하게 항상 존재"""
    __tablename__ = "lectures"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=True)
    category = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    video_path = Column(Text, nullable=False)   # inputs/{lecture_id}/{filename}
    output_dir = Column(Text, nullable=False)   # results/{lecture_id}/
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    processing_jobs = relationship(
        "ProcessingJob",
        back_populates="lecture",
        order_by="ProcessingJob.created_at",
    )

    @property
    def active_job(self):
        """현재 실행 중인 job — 최대 1개"""
        return next(
            (j for j in self.processing_jobs if j.status in ("pending", "running")),
            None,
        )

    @property
    def last_job(self):
        return self.processing_jobs[-1] if self.processing_jobs else None


class GraphSession(Base):
    """강의별 Neo4j 로드 상태 관리 — lecture_id로 식별"""
    __tablename__ = "graph_sessions"
    __table_args__ = (
        UniqueConstraint("lecture_id", "session_id", name="uq_graph_sessions_lecture_session"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lecture_id = Column(
        UUID(as_uuid=True),
        ForeignKey("lectures.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stem = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    last_heartbeat_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())