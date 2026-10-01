from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import DuplicateDocumentError
from app.models import Chunk, Document


async def find_duplicate(
    session: AsyncSession, sha256: str, chunk_size: int, chunk_overlap: int
) -> Document | None:
    result = await session.execute(
        select(Document).where(
            Document.content_sha256 == sha256,
            Document.chunk_size == chunk_size,
            Document.chunk_overlap == chunk_overlap,
        )
    )
    return result.scalar_one_or_none()


async def create_document_with_chunks(
    session: AsyncSession, document: Document, chunks: list[Chunk]
) -> Document:
    """
    One commit for the document AND all its chunks: ingestion is atomic, so
    a failure halfway never leaves a document row with only some of its
    chunks. The unique constraint is the backstop for two identical uploads
    racing past the pre-check in the service.
    """
    document.chunks = chunks
    session.add(document)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise DuplicateDocumentError("This file was already ingested with these chunk settings")
    await session.refresh(document, attribute_names=["id", "created_at"])
    return document


async def list_documents(session: AsyncSession) -> list[Document]:
    result = await session.execute(select(Document).order_by(Document.id.desc()))
    return list(result.scalars().all())


async def get_document(session: AsyncSession, document_id: int) -> Document | None:
    return await session.get(Document, document_id)


async def list_chunks(
    session: AsyncSession, document_id: int, limit: int, offset: int
) -> list[Chunk]:
    result = await session.execute(
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all())


async def delete_document(session: AsyncSession, document: Document) -> None:
    await session.delete(document)
    await session.commit()
