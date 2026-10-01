import io
import os
import tempfile

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.db import Base, get_session
from app.main import app


@pytest_asyncio.fixture
async def engine():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    eng = create_async_engine(f"sqlite+aiosqlite:///{path}", future=True)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()
    os.remove(path)


@pytest_asyncio.fixture
async def client(engine):
    SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

    async def override_get_session():
        async with SessionLocal() as s:
            yield s

    app.dependency_overrides[get_session] = override_get_session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def session(engine):
    """
    A direct DB session for tests that need to inspect a column not
    exposed through the API (e.g. Chunk.embedding) -- the response schemas
    stay minimal on purpose, so this is the only way to assert on it.
    """
    SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with SessionLocal() as s:
        yield s


@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch):
    """
    Replaces the real embedding provider call with a deterministic fake --
    same "no network call needed to run the test suite" philosophy as every
    other provider call mocked in this sprint. Returns one fixed-length
    zero vector per input text (length matches settings.embedding_dimensions),
    enough to prove embeddings were generated and stored without a real key.
    """
    from app.core.config import settings

    async def fake_embed_texts(texts):
        return [[0.0] * settings.embedding_dimensions for _ in texts]

    monkeypatch.setattr("app.services.ingestion_service.embed_texts", fake_embed_texts)
    return fake_embed_texts


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    """
    Replaces the real generation (chat) provider call with a deterministic
    fake -- same "no network call needed to run the test suite" philosophy
    as mock_embeddings above. Returns a fixed answer citing source [1],
    which is enough for most tests; tests that need a specific answer text
    (e.g. to check citation parsing) override this with their own
    monkeypatch.setattr call.
    """
    async def fake_get_chat_completion(messages):
        return "This is a mocked answer, grounded in the context [1]."

    monkeypatch.setattr(
        "app.services.generation_service.get_chat_completion", fake_get_chat_completion
    )
    return fake_get_chat_completion


def make_pdf(pages: list[list[str]]) -> bytes:
    """Build a real multi-page PDF; each inner list is the lines of one page."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for lines in pages:
        y = 800
        for line in lines:
            c.drawString(50, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


LONG_TEXT = " ".join(
    f"Sentence number {i} talks about topic {i % 5} in some detail." for i in range(40)
)


def _topic_vector(text: str, dim: int) -> list[float]:
    """
    A tiny hand-built embedding: puts a 1.0 in one of three "topic slots"
    based on keywords, everything else 0. Two texts about the same topic
    land on the SAME vector (distance 0); texts about different topics are
    orthogonal (distance 1). Deliberately crude -- real semantic nuance
    isn't the point, having a fake with predictable, checkable distances is.
    """
    vec = [0.0] * dim
    lowered = text.lower()
    if any(w in lowered for w in ("cat", "kitten", "feline")):
        vec[0] = 1.0
    elif any(w in lowered for w in ("revenue", "quarter", "finance")):
        vec[1] = 1.0
    else:
        vec[2] = 1.0
    return vec


def install_topic_embeddings(monkeypatch):
    """
    Overrides the default all-zero embedding mock (see mock_embeddings
    above) with the keyword-based fake, for tests that need distances to
    actually differ by content -- e.g. retrieval ranking tests. Patches
    both call sites (ingestion AND retrieval each import embed_texts
    separately), so a chunk embedded at upload time and a query embedded
    at retrieval time land on comparable vectors.
    """
    from app.core.config import settings

    async def fake_embed_texts(texts):
        return [_topic_vector(t, settings.embedding_dimensions) for t in texts]

    monkeypatch.setattr("app.services.ingestion_service.embed_texts", fake_embed_texts)
    monkeypatch.setattr("app.services.retrieval_service.embed_texts", fake_embed_texts)
