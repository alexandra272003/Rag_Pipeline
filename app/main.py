from fastapi import FastAPI

from app.core.errors import AppError, app_error_handler
from app.routers import ask, documents, lab, retrieval

app = FastAPI(title="RAG Pipeline")

app.add_exception_handler(AppError, app_error_handler)

app.include_router(lab.router)
app.include_router(documents.router)
app.include_router(retrieval.router)
app.include_router(ask.router)


@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "pong"}
