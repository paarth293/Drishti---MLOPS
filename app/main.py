"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.logging_conf import configure_logging
from app.settings import load_settings


def create_app() -> FastAPI:
    """Create a configured FastAPI application."""

    configure_logging()
    load_settings()
    application = FastAPI(
        title="Drishti — AIOps & MLOps Copilot",
        version="0.1.0",
        description="Real-time anomaly detection and root cause analysis for microservices.",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        logger = logging.getLogger("aiops.request")
        response = await call_next(request)
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response

    application.include_router(router)
    return application


app = create_app()
