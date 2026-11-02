"""
FastAPI application entrypoint.
Provides REST API for the Quantum RAG Assistant.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os

import app.state as state
from app.routes.chat import router as chat_router
from app.routes.health import router as health_router
from app.routes.documents import router as documents_router
from app.routes.graph import router as graph_router
from app.routes.history import router as history_router
from app.routes.v3 import router as v3_router
from src.llm.qa_chain import QAChain
from src.ingestion.load_documents import DocumentLoader
from src.ingestion.chunker import TextChunker
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources at startup, clean up at shutdown."""
    log.info("🚀 Starting Quantum RAG Assistant API...")
    try:
        state.qa_chain = QAChain()
        graph_store = state.qa_chain.retriever.graph_store
        graph_store.load()
        if graph_store.graph.number_of_nodes() == 0:
            log.info("No knowledge graph found; building it from data/raw ...")
            documents = DocumentLoader().load_directory(settings.data_raw_dir)
            chunks = TextChunker().chunk_documents(documents)
            if chunks:
                graph_store.rebuild_from_chunks(chunks)
        log.info("✓ Knowledge graph ready: %s", graph_store.stats())
    except Exception as exc:
        log.exception("Knowledge graph initialization failed: %s", exc)
    yield
    log.info("🛑 Shutting down Quantum RAG Assistant API")


app = FastAPI(
    title="Quantum RAG Assistant",
    description=(
        "AI-powered Quantum Computing tutor using Retrieval-Augmented Generation. "
        "Answers are grounded in trusted quantum computing documents."
    ),
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)

# CORS — allow all for development, restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Diagram-Type", "X-Diagram-Title", "X-Diagram-Alt",
                    "X-Diagram-Confidence", "X-Diagram-Description"],
)

# Include routers
app.include_router(health_router, prefix="/api/v1", tags=["Health"])
app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
app.include_router(documents_router, prefix="/api/v1", tags=["Documents"])
app.include_router(graph_router, prefix="/api/v1", tags=["Graph"])
app.include_router(history_router, prefix="/api/v1", tags=["History"])
app.include_router(v3_router, prefix="/api/v1", tags=["V3 Intelligence"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the Quantum RAG web interface."""
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})
