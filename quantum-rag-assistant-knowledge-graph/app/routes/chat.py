"""
Chat endpoint — the core RAG question-answering API.
"""

import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

import app.state as state
from src.ingestion.load_documents import DocumentLoader
from src.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The quantum computing question to answer",
        examples=["What is quantum superposition?"],
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve",
    )
    threshold: Optional[float] = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for retrieval",
    )
    graph_enhanced: bool = Field(
        default=True,
        description="Expand retrieval through knowledge-graph relationships",
    )


class RetrievalInfo(BaseModel):
    source: str
    score: float
    document_id: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[int] = None
    metadata: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    retrieval_info: List[RetrievalInfo]
    processing_time_ms: float
    model_used: str
    has_context: bool
    graph_enhanced: bool = True
    attachments_used: List[str] = []
    citations: List[dict] = Field(default_factory=list)
    conversation_id: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question about Quantum Computing.

    The system will:
    1. Retrieve relevant passages from the knowledge base
    2. Use an LLM to generate a grounded, accurate answer
    3. Return the answer with source citations
    """
    if state.qa_chain is None:
        raise HTTPException(
            status_code=503,
            detail="QA chain not initialized. Check server logs.",
        )

    try:
        response = state.qa_chain.answer(
            question=request.question,
            top_k=request.top_k,
            threshold=request.threshold,
            use_graph=request.graph_enhanced,
        )

        return ChatResponse(
            question=response.question,
            answer=response.answer,
            sources=response.sources,
            retrieval_info=[
                RetrievalInfo(source=r.source, score=round(r.score, 4), document_id=r.document_id,
                              page=r.page, chunk_id=r.chunk_id, metadata=r.metadata or {})
                for r in response.retrieval_results
            ],
            processing_time_ms=round(response.processing_time_ms, 2),
            model_used=response.model_used,
            has_context=response.has_context,
            graph_enhanced=request.graph_enhanced,
            citations=__import__("app.routes.v3", fromlist=["_citations"])._citations(response.retrieval_results),
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Knowledge base not ready: {e}. Run: python scripts/build_index.py",
        )
    except Exception as e:
        log.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Multipart endpoint used by the web UI (supports temporary chat attachments)
# ---------------------------------------------------------------------------
MAX_ATTACHMENTS = 3
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
MAX_ATTACHMENT_CHARS = 8000
EXTRA_TEXT_SUFFIXES = {
    ".py", ".js", ".ts", ".json", ".csv", ".yaml", ".yml", ".html", ".css",
    ".java", ".c", ".cpp", ".h", ".go", ".rs", ".sh", ".ipynb", ".log", ".qasm",
}


def _extract_attachment_text(name: str, content: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in DocumentLoader.SUPPORTED_EXTENSIONS:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"attachment{suffix}"
            path.write_bytes(content)
            docs = DocumentLoader().load_file(path)
        return " ".join(d.content for d in docs)
    if suffix in EXTRA_TEXT_SUFFIXES:
        return content.decode("utf-8", errors="replace")
    raise HTTPException(status_code=400, detail=f"Unsupported attachment type: {name}")


@router.post("/chat/query", response_model=ChatResponse)
async def chat_query(
    question: str = Form("", max_length=1000),
    top_k: int = Form(8, ge=1, le=20),
    threshold: float = Form(0.3, ge=0.0, le=1.0),
    graph_enhanced: bool = Form(True),
    conversation_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[]),
):
    """Ask a question, optionally with temporary attachments used as extra context.

    Attachments are read for this request only; they are not added to the knowledge graph.
    """
    if state.qa_chain is None:
        raise HTTPException(status_code=503, detail="QA chain not initialized. Check server logs.")
    files = [f for f in files if f and f.filename]
    if len(files) > MAX_ATTACHMENTS:
        raise HTTPException(status_code=400, detail=f"Attach at most {MAX_ATTACHMENTS} files per message.")

    attachments = []
    for upload in files:
        name = Path(upload.filename).name
        content = await upload.read()
        await upload.close()
        if len(content) > MAX_ATTACHMENT_BYTES:
            raise HTTPException(status_code=413, detail=f"{name} exceeds the 10 MB attachment limit.")
        try:
            text = _extract_attachment_text(name, content).strip()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not read {name}: {exc}") from exc
        if text:
            attachments.append((name, text[:MAX_ATTACHMENT_CHARS]))

    question = question.strip() or "Summarize the attached file(s) and explain their key quantum computing concepts."
    try:
        from app.routes.v3 import _history, _save, _citations
        history = _history(conversation_id)
        response = state.qa_chain.answer(
            question=question, top_k=top_k, threshold=threshold,
            use_graph=graph_enhanced, attachments=attachments,
            conversation_history=history,
        )
    except ValueError as e:  # e.g. missing GROQ_API_KEY
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        question=response.question,
        answer=response.answer,
        sources=response.sources,
        retrieval_info=[RetrievalInfo(source=r.source, score=round(r.score, 4), document_id=r.document_id,
                              page=r.page, chunk_id=r.chunk_id, metadata=r.metadata or {}) for r in response.retrieval_results],
        processing_time_ms=round(response.processing_time_ms, 2),
        model_used=response.model_used,
        has_context=response.has_context,
        graph_enhanced=graph_enhanced,
        attachments_used=[name for name, _ in attachments],
        citations=_citations(response.retrieval_results),
        conversation_id=_save(conversation_id, question, response.answer, response.sources),
    )
