from fastapi import APIRouter, Depends, File, Query, Response, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import get_session
from app.core.errors import PayloadTooLargeError
from app.repositories import document_repository as repo
from app.schemas import ChunkRead, DocumentRead
from app.services import ingestion_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Query(default=settings.default_chunk_size, ge=5, le=4000),
    chunk_overlap: int = Query(default=settings.default_chunk_overlap, ge=0),
    session: AsyncSession = Depends(get_session),
):
    # Read one byte past the limit: enough to detect "too big" without
    # pulling an arbitrarily large upload fully into memory.
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise PayloadTooLargeError(f"File exceeds the {settings.max_upload_bytes} byte limit")
    return await ingestion_service.ingest_document(
        session, file.filename or "upload", data, chunk_size, chunk_overlap
    )


@router.get("", response_model=list[DocumentRead])
async def list_documents(session: AsyncSession = Depends(get_session)):
    return await repo.list_documents(session)


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(document_id: int, session: AsyncSession = Depends(get_session)):
    return await ingestion_service.get_document_or_404(session, document_id)


@router.get("/{document_id}/chunks", response_model=list[ChunkRead])
async def list_chunks(
    document_id: int,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    await ingestion_service.get_document_or_404(session, document_id)
    return await repo.list_chunks(session, document_id, limit, offset)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: int, session: AsyncSession = Depends(get_session)):
    document = await ingestion_service.get_document_or_404(session, document_id)
    await repo.delete_document(session, document)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
