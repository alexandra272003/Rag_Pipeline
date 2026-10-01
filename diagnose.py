import asyncio
from sqlalchemy import select, func

from app.core.db import SessionLocal
from app.core.embeddings import embed_texts
from app.models import Chunk, Document
from app.services import retrieval_service


async def main():
    async with SessionLocal() as session:
        results = await retrieval_service.retrieve(session, "what does this project do", top_k=3)
        print("retrieve() returned", len(results), "results")
        for chunk, filename, dist in results:
            print(f"  chunk {chunk.id} from {filename}: distance={dist}")


asyncio.run(main())
