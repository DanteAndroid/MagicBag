"""FastAPI application entrypoint.

This file wires routers together and exposes the ASGI app object used by
uvicorn and Railway.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.routes import health, rag, stats
from app.core.config import settings
from app.core.logging import configure_logging


INDEX_FILE = Path(__file__).resolve().parents[1] / "index.html"


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize shared resources on startup."""
    configure_logging()
    # TODO: Initialize expensive shared clients here if you want eager startup.
    yield
    # TODO: Close shared clients here when you add long-lived connections.


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Magic knowledge RAG service skeleton.",
    lifespan=lifespan,
)

# Allow the standalone HTML client to call the API from file:// or any web host.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(stats.router, prefix="/stats", tags=["stats"])
app.include_router(rag.router, prefix="/api/v1/rag", tags=["rag"])


@app.get("/", summary="Root")
async def root() -> FileResponse:
    """Serve the single-page web client."""
    return FileResponse(INDEX_FILE)
