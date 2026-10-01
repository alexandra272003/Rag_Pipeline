from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.core.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"
    # The same file may be ingested more than once with DIFFERENT chunk
    # settings (that's the Day 33 experiment), but the exact same
    # file + settings twice is a duplicate.
    __table_args__ = (
        UniqueConstraint("content_sha256", "chunk_size", "chunk_overlap", name="uq_doc_content_params"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    source_type: Mapped[str] = mapped_column(String(10))  # pdf | txt | md
    content_sha256: Mapped[str] = mapped_column(String(64), index=True)
    page_count: Mapped[int]
    chunk_size: Mapped[int]
    chunk_overlap: Mapped[int]
    chunk_count: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Chunk.chunk_index"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index", name="uq_chunk_position"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    chunk_index: Mapped[int]  # position within the document, 0-based
    page_number: Mapped[int | None] = mapped_column(nullable=True)
    content: Mapped[str] = mapped_column(Text)
    # Offsets are into the CLEANED text of this chunk's page (or of the whole
    # document for txt/md), so content == cleaned_page_text[char_start:char_end].
    char_start: Mapped[int]
    char_end: Mapped[int]
    token_count: Mapped[int]
    # Day 29: one embedding per chunk, generated locally at ingestion time
    # (see core/embeddings.py -- no external provider or API key involved).
    # - Postgres: a real pgvector column, so similarity search can use the
    #   ivfflat index created in the Day 29 migration.
    # - SQLite (test suite only): pgvector has no SQLite backend, so this
    #   falls back to a JSON array there. Same data, different storage --
    #   the test suite never needs a real embedding provider or Postgres.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dimensions).with_variant(JSON(), "sqlite"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document: Mapped["Document"] = relationship(back_populates="chunks")
