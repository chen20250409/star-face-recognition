# -*- coding: utf-8 -*-
"""MySQL 连接与表定义（SQLAlchemy 2.x + PyMySQL）"""
import os
import time
import logging
from datetime import datetime

from sqlalchemy import create_engine, text, String, Integer, Float, DateTime, LargeBinary, ForeignKey, Index
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

log = logging.getLogger("db")

DB_USER = os.getenv("DB_USER", "face")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "faces")


def db_url() -> URL:
    return URL.create(
        "mysql+pymysql",
        username=DB_USER,
        password=DB_PASS,       # URL.create 会自动转义密码里的 @ 等字符
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        query={"charset": "utf8mb4"},
    )


engine = create_engine(db_url(), pool_pre_ping=True, pool_recycle=280, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)

MYSQL_ARGS = {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"}


class Base(DeclarativeBase):
    pass


class Person(Base):
    __tablename__ = "persons"
    __table_args__ = MYSQL_ARGS
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


class FaceSample(Base):
    __tablename__ = "face_samples"
    __table_args__ = (Index("idx_fs_person", "person_id"), MYSQL_ARGS)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    person_id: Mapped[int] = mapped_column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    embedding: Mapped[bytes] = mapped_column(LargeBinary(512), nullable=False)
    det_score: Mapped[float] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


class RecogLog(Base):
    __tablename__ = "recognition_logs"
    __table_args__ = MYSQL_ARGS
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    original_path: Mapped[str] = mapped_column(String(512), nullable=False)
    result_path: Mapped[str] = mapped_column(String(512), nullable=True)
    elapsed_ms: Mapped[int] = mapped_column(Integer, nullable=True)
    face_count: Mapped[int] = mapped_column(Integer, nullable=True)


class RecogFace(Base):
    __tablename__ = "recognition_faces"
    __table_args__ = (Index("idx_rf_log", "log_id"), MYSQL_ARGS)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    log_id: Mapped[int] = mapped_column(Integer, ForeignKey("recognition_logs.id", ondelete="CASCADE"), nullable=False)
    person_name: Mapped[str] = mapped_column(String(64), nullable=True)
    score: Mapped[float] = mapped_column(Float, nullable=True)
    is_unknown: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    candidate_name: Mapped[str] = mapped_column(String(64), nullable=True)
    bbox: Mapped[str] = mapped_column(String(64), nullable=True)
    det_score: Mapped[float] = mapped_column(Float, nullable=True)


def wait_for_db(retries: int = 30, delay: float = 2.0) -> bool:
    """MySQL 可能比容器起得慢，这里做重连重试"""
    for i in range(1, retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            log.info("MySQL 连接成功 (%s:%s/%s)", DB_HOST, DB_PORT, DB_NAME)
            return True
        except Exception as e:
            log.warning("第 %d/%d 次连接 MySQL 失败: %s", i, retries, str(e)[:160])
            time.sleep(delay)
    return False


def init_db() -> None:
    Base.metadata.create_all(engine)
    log.info("数据表已就绪（如不存在则创建）")
