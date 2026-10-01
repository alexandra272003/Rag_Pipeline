from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    filename: str
    source_type: str
    page_count: int
    chunk_size: int
    chunk_overlap: int
    chunk_count: int
    created_at: datetime


class ChunkRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    chunk_index: int
    page_number: int | None
    content: str
    char_start: int
    char_end: int
    token_count: int


class ChunkPreviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)
    chunk_size: int = Field(default=500, ge=5, le=4000)
    chunk_overlap: int = Field(default=50, ge=0)
    clean: bool = True


class PreviewChunk(BaseModel):
    chunk_index: int
    page_number: int | None = None
    content: str
    char_start: int
    char_end: int
    token_count: int


class ChunkPreviewResponse(BaseModel):
    cleaned_char_count: int
    total_tokens: int
    chunk_count: int
    chunks: list[PreviewChunk]


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    # Metadata filters (Day 30): narrow the search before ranking, rather
    # than ranking the whole corpus and discarding results afterward.
    document_id: int | None = None
    source_type: str | None = None  # "pdf" | "txt" | "md"


class RetrievalResult(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    page_number: int | None
    content: str
    # Cosine distance: 0 = identical direction, 2 = opposite. Returned raw
    # (not converted to a 0-1 "similarity score") so it's clear this is a
    # distance -- smaller is more relevant -- rather than implying a
    # normalized confidence the metric doesn't actually provide.
    distance: float


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievalResult]


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: int | None = None
    source_type: str | None = None


class Citation(BaseModel):
    number: int  # the [N] the model cited, matching position in the retrieved list
    chunk_id: int
    document_id: int
    filename: str
    page_number: int | None
    distance: float


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    # False if the model cited a source number that was never actually
    # retrieved -- a fabricated citation, checked rather than trusted.
    all_citations_valid: bool
    retrieved_count: int
