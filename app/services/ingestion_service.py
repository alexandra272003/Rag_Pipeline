import hashlib

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.embeddings import embed_texts
from app.core.errors import (
    DuplicateDocumentError,
    InvalidChunkParamsError,
    InvalidDocumentError,
    NotFoundError,
    PayloadTooLargeError,
)
from app.ingestion.chunking import chunk_text, count_tokens
from app.ingestion.cleaning import clean_text
from app.ingestion.parsers import extract_pages, source_type_for
from app.models import Chunk, Document
from app.repositories import document_repository as repo
from app.schemas import ChunkPreviewResponse, PreviewChunk


def _chunk_pages(pages, chunk_size: int, chunk_overlap: int):
    """
    pages -> [(page_number, Chunk)] with cleaning applied per page.

    Chunks never span pages: that keeps each chunk's page number exact (so a
    citation can say "page 4"), at the cost of not merging a sentence that
    runs across a page break. A deliberate trade-off.
    """
    out = []
    for page in pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue  # blank page, or a scanned page with no text layer
        for chunk in chunk_text(cleaned, chunk_size, chunk_overlap):
            out.append((page.number, chunk))
    return out


def _validate_params(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_overlap >= chunk_size:
        raise InvalidChunkParamsError("chunk_overlap must be smaller than chunk_size")


async def ingest_document(
    session: AsyncSession, filename: str, data: bytes, chunk_size: int, chunk_overlap: int
) -> Document:
    if len(data) > settings.max_upload_bytes:
        raise PayloadTooLargeError(
            f"File exceeds the {settings.max_upload_bytes} byte limit"
        )
    if not data:
        raise InvalidDocumentError("File is empty")
    _validate_params(chunk_size, chunk_overlap)

    source_type = source_type_for(filename)  # rejects unsupported types early
    sha256 = hashlib.sha256(data).hexdigest()

    existing = await repo.find_duplicate(session, sha256, chunk_size, chunk_overlap)
    if existing is not None:
        raise DuplicateDocumentError(
            "This file was already ingested with these chunk settings",
            details={"existing_document_id": existing.id},
        )

    pages = extract_pages(filename, data)
    paged_chunks = _chunk_pages(pages, chunk_size, chunk_overlap)
    if not paged_chunks:
        raise InvalidDocumentError(
            "No extractable text found (scanned PDFs need OCR, which is not supported)"
        )

    # Day 29: embed every chunk's text before persisting, so the row is
    # written once with its vector already attached rather than in a
    # second pass. embed_texts batches internally and raises EmbeddingError
    # (mapped to a 502) if the provider fails after retries -- nothing is
    # saved for a document whose embeddings couldn't be generated.
    vectors = await embed_texts([c.text for _, c in paged_chunks])

    document = Document(
        filename=filename,
        source_type=source_type,
        content_sha256=sha256,
        page_count=len(pages),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_count=len(paged_chunks),
    )
    chunk_rows = [
        Chunk(
            chunk_index=i,
            page_number=page_number,
            content=c.text,
            char_start=c.char_start,
            char_end=c.char_end,
            token_count=c.token_count,
            embedding=vectors[i],
        )
        for i, (page_number, c) in enumerate(paged_chunks)
    ]
    return await repo.create_document_with_chunks(session, document, chunk_rows)


async def get_document_or_404(session: AsyncSession, document_id: int) -> Document:
    document = await repo.get_document(session, document_id)
    if document is None:
        raise NotFoundError(f"Document {document_id} not found")
    return document


def preview_chunks(text: str, chunk_size: int, chunk_overlap: int, clean: bool) -> ChunkPreviewResponse:
    """Chunk text without saving anything -- for experimenting with settings."""
    _validate_params(chunk_size, chunk_overlap)
    body = clean_text(text) if clean else text
    chunks = chunk_text(body, chunk_size, chunk_overlap)
    return ChunkPreviewResponse(
        cleaned_char_count=len(body),
        total_tokens=count_tokens(body),
        chunk_count=len(chunks),
        chunks=[
            PreviewChunk(
                chunk_index=i,
                content=c.text,
                char_start=c.char_start,
                char_end=c.char_end,
                token_count=c.token_count,
            )
            for i, c in enumerate(chunks)
        ],
    )
