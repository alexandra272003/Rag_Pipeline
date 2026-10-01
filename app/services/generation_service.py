import re

from openai import APIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ProviderError
from app.core.llm_client import get_chat_completion
from app.services import retrieval_service

SYSTEM_PROMPT = (
    "Answer ONLY using the numbered context below. If the context does not "
    "contain the answer, say exactly: \"I don't know based on the provided "
    "documents.\" Cite every claim with the matching source number in square "
    "brackets, like [1]. Never cite a number that is not in the context."
)

# Matches [1], [12], etc. in the model's answer.
CITATION_RE = re.compile(r"\[(\d+)\]")

NO_DOCUMENTS_ANSWER = "I don't know based on the provided documents."


def _build_context_block(results: list[tuple]) -> str:
    """
    Numbers each retrieved chunk and labels it with its source, matching
    the citation format the system prompt asks the model to use -- e.g.
    "[1] (handbook.pdf, page 4)". Page number is omitted for txt/md files,
    which have none.
    """
    lines = []
    for i, (chunk, filename, _distance) in enumerate(results, start=1):
        page = f", page {chunk.page_number}" if chunk.page_number is not None else ""
        lines.append(f"[{i}] ({filename}{page})\n{chunk.content}")
    return "\n\n".join(lines)


async def ask(
    session: AsyncSession,
    query: str,
    top_k: int = 5,
    document_id: int | None = None,
    source_type: str | None = None,
) -> dict:
    """
    Day 31: retrieve, build a numbered+cited prompt, generate an answer,
    then verify every citation the model produced actually refers to a
    chunk that was really retrieved -- a model can still fabricate a
    citation number even when told not to, so this is checked rather than
    trusted. A cited number outside the retrieved set is dropped from the
    structured `citations` list and flags `all_citations_valid=False`,
    rather than silently passed through as if it were real.

    If retrieval finds nothing, the LLM is never called: there's no
    evidence to ground an answer in, so the refusal is deterministic
    instead of hoping the model says the right thing.
    """
    results = await retrieval_service.retrieve(
        session, query, top_k=top_k, document_id=document_id, source_type=source_type
    )

    if not results:
        return {
            "answer": NO_DOCUMENTS_ANSWER,
            "citations": [],
            "all_citations_valid": True,
            "retrieved_count": 0,
        }

    context_block = _build_context_block(results)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context_block}\n\nQUESTION: {query}"},
    ]

    try:
        answer = await get_chat_completion(messages)
    except APIError as exc:
        raise ProviderError(
            "The language model provider failed to respond", details={"reason": str(exc)}
        ) from exc

    cited_numbers = {int(n) for n in CITATION_RE.findall(answer)}
    valid_numbers = set(range(1, len(results) + 1))
    all_valid = cited_numbers.issubset(valid_numbers)

    citations = [
        {
            "number": i,
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "filename": filename,
            "page_number": chunk.page_number,
            "distance": distance,
        }
        for i, (chunk, filename, distance) in enumerate(results, start=1)
        if i in cited_numbers
    ]

    return {
        "answer": answer,
        "citations": citations,
        "all_citations_valid": all_valid,
        "retrieved_count": len(results),
    }
