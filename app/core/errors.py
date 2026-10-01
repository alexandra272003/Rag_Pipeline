from fastapi import Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    status_code = 400
    code = "app_error"

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class DuplicateDocumentError(AppError):
    status_code = 409
    code = "duplicate_document"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"


class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "unsupported_file_type"


class InvalidDocumentError(AppError):
    """The file was readable as a request, but yielded no usable text."""
    status_code = 422
    code = "invalid_document"


class InvalidChunkParamsError(AppError):
    status_code = 422
    code = "invalid_chunk_params"


class EmbeddingError(AppError):
    """The embedding provider failed, timed out, or returned something unusable."""
    status_code = 502
    code = "embedding_error"


class ProviderError(AppError):
    """The LLM (generation) provider failed, timed out, or returned something unusable."""
    status_code = 502
    code = "provider_error"


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
    )
