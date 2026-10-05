from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger

logger = get_logger("http")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs method, path, status and latency per request. Bodies and headers are never logged."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Request crashed", extra={"request_id": request_id, "method": request.method,
                                                       "path": request.url.path})
            raise
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info("Request", extra={"request_id": request_id, "method": request.method, "path": request.url.path,
                                      "status": response.status_code, "duration_ms": duration_ms})
        response.headers["X-Request-ID"] = request_id
        return response
