import os
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.models import Base

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@db/graphlec")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    """DB 초기화: 모든 모델 테이블 생성"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("""
            DO $$
            BEGIN
                IF to_regclass('public.lectures_content') IS NOT NULL THEN
                    ALTER TABLE lectures_content
                    ADD COLUMN IF NOT EXISTS graphrag_workspace TEXT;
                END IF;
            END
            $$;
        """))
        await conn.execute(text("""
            DO $$
            BEGIN
                IF to_regclass('public.lectures_content') IS NOT NULL THEN
                    INSERT INTO lectures (
                        id, title, category, description, video_path, output_dir, created_at
                    )
                    SELECT
                        COALESCE(
                            CASE
                                WHEN lc.stem ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
                                THEN lc.stem::uuid
                            END,
                            lc.job_id,
                            lc.id
                        ) AS id,
                        lc.title,
                        lc.category,
                        lc.description,
                        lc.video_path,
                        lc.output_dir,
                        lc.created_at
                    FROM lectures_content lc
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        category = EXCLUDED.category,
                        description = EXCLUDED.description,
                        video_path = EXCLUDED.video_path,
                        output_dir = EXCLUDED.output_dir;
                END IF;
            END
            $$;
        """))
        await conn.execute(text("""
            DO $$
            BEGIN
                IF to_regclass('public.jobs') IS NOT NULL THEN
                    IF to_regclass('public.lectures_content') IS NOT NULL THEN
                        INSERT INTO lectures (
                            id, title, category, description, video_path, output_dir, created_at
                        )
                        SELECT
                            j.id,
                            COALESCE(NULLIF(regexp_replace(j.input_path, '^.*/', ''), ''), j.id::text),
                            '기타',
                            NULL,
                            j.input_path,
                            COALESCE(j.output_dir, regexp_replace(j.input_path, '/inputs/([^/]+)/.*$', '/results/\\1')),
                            j.created_at
                        FROM jobs j
                        WHERE NOT EXISTS (
                            SELECT 1 FROM lectures_content lc WHERE lc.job_id = j.id
                        )
                        ON CONFLICT (id) DO NOTHING;
                    ELSE
                        INSERT INTO lectures (
                            id, title, category, description, video_path, output_dir, created_at
                        )
                        SELECT
                            j.id,
                            COALESCE(NULLIF(regexp_replace(j.input_path, '^.*/', ''), ''), j.id::text),
                            '기타',
                            NULL,
                            j.input_path,
                            COALESCE(j.output_dir, regexp_replace(j.input_path, '/inputs/([^/]+)/.*$', '/results/\\1')),
                            j.created_at
                        FROM jobs j
                        ON CONFLICT (id) DO NOTHING;
                    END IF;
                END IF;
            END
            $$;
        """))
        await conn.execute(text("""
            DO $$
            BEGIN
                IF to_regclass('public.jobs') IS NOT NULL THEN
                    IF to_regclass('public.lectures_content') IS NOT NULL THEN
                        INSERT INTO processing_jobs (
                            id, lecture_id, status, current_stage, error_message, pipeline_stages, created_at
                        )
                        SELECT
                            j.id,
                            COALESCE(
                                CASE
                                    WHEN lc.stem ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
                                    THEN lc.stem::uuid
                                END,
                                lc.job_id,
                                j.id
                            ) AS lecture_id,
                            j.status,
                            j.current_stage,
                            j.error_message,
                            j.pipeline_stages,
                            j.created_at
                        FROM jobs j
                        LEFT JOIN lectures_content lc ON lc.job_id = j.id
                        ON CONFLICT (id) DO UPDATE SET
                            lecture_id = EXCLUDED.lecture_id,
                            status = EXCLUDED.status,
                            current_stage = EXCLUDED.current_stage,
                            error_message = EXCLUDED.error_message,
                            pipeline_stages = EXCLUDED.pipeline_stages;
                    ELSE
                        INSERT INTO processing_jobs (
                            id, lecture_id, status, current_stage, error_message, pipeline_stages, created_at
                        )
                        SELECT
                            j.id,
                            j.id,
                            j.status,
                            j.current_stage,
                            j.error_message,
                            j.pipeline_stages,
                            j.created_at
                        FROM jobs j
                        ON CONFLICT (id) DO UPDATE SET
                            lecture_id = EXCLUDED.lecture_id,
                            status = EXCLUDED.status,
                            current_stage = EXCLUDED.current_stage,
                            error_message = EXCLUDED.error_message,
                            pipeline_stages = EXCLUDED.pipeline_stages;
                    END IF;
                END IF;
            END
            $$;
        """))
        await conn.execute(text("""
            DO $$
            BEGIN
                IF to_regclass('public.graph_sessions') IS NOT NULL THEN
                    ALTER TABLE graph_sessions
                    DROP CONSTRAINT IF EXISTS graph_sessions_lecture_id_fkey;

                    UPDATE graph_sessions gs
                    SET lecture_id = gs.stem::uuid
                    WHERE gs.stem ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
                      AND EXISTS (
                          SELECT 1 FROM lectures l WHERE l.id = gs.stem::uuid
                      );

                    DELETE FROM graph_sessions gs
                    WHERE NOT EXISTS (
                        SELECT 1 FROM lectures l WHERE l.id = gs.lecture_id
                    );

                    ALTER TABLE graph_sessions
                    ADD CONSTRAINT graph_sessions_lecture_id_fkey
                    FOREIGN KEY (lecture_id) REFERENCES lectures(id) ON DELETE CASCADE;
                END IF;
            END
            $$;
        """))
    logger.info("--- [DB] Database initialized via SQLAlchemy models. ---")

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
