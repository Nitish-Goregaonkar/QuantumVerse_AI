"""Document upload and knowledge-graph ingestion endpoints."""
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

import app.state as state
from src.ingestion.chunker import TextChunker
from src.ingestion.load_documents import DocumentLoader
from src.utils.config import settings
from src.utils.logger import get_logger

router = APIRouter()
log = get_logger(__name__)
ALLOWED_EXTENSIONS = DocumentLoader.SUPPORTED_EXTENSIONS
MAX_FILE_BYTES = 25 * 1024 * 1024


@router.get("/documents/status")
async def documents_status():
    if state.qa_chain is None:
        return {"ready": False, "graph": {}}
    store = state.qa_chain.retriever.graph_store
    if not store.loaded:
        store.load()
    return {"ready": store.graph.number_of_nodes() > 0, "graph": store.stats()}


@router.get("/documents")
async def list_documents():
    """Documents indexed in the knowledge graph (for the Document Vault view)."""
    if state.qa_chain is None:
        return {"documents": [], "graph": {}}
    store = state.qa_chain.retriever.graph_store
    if not store.loaded:
        store.load()
    return {"documents": store.list_documents(), "graph": store.stats()}


@router.delete("/documents/{source:path}")
async def delete_document(source: str):
    if state.qa_chain is None:
        raise HTTPException(status_code=503, detail="Backend is still initializing.")
    store = state.qa_chain.retriever.graph_store
    try:
        result = store.delete_document(source)
    except KeyError:
        raise HTTPException(status_code=404, detail="Document not found in the knowledge graph.")
    return {"message": "Document removed.", **result, "graph": store.stats()}


@router.post("/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """Save files, extract text, chunk and embed them, then rebuild the local graph."""
    if state.qa_chain is None:
        raise HTTPException(status_code=503, detail="Backend is still initializing. Try again shortly.")
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one file.")

    upload_dir = settings.data_raw_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    try:
        for upload in files:
            original_name = Path(upload.filename or "document").name
            suffix = Path(original_name).suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
                raise HTTPException(status_code=400, detail=f"Unsupported file type for {original_name}. Supported: {supported}")
            content = await upload.read()
            if not content:
                continue
            if len(content) > MAX_FILE_BYTES:
                raise HTTPException(status_code=413, detail=f"{original_name} exceeds the 25 MB per-file limit.")
            safe_name = f"{uuid4().hex[:8]}_{original_name}"
            target = upload_dir / safe_name
            target.write_bytes(content)
            saved.append(target)

        if not saved:
            raise HTTPException(status_code=400, detail="No non-empty files were uploaded.")

        # Load the persistent upload corpus so re-indexing retains prior documents.
        # OCR failures are warnings; text-readable documents still proceed.
        documents = DocumentLoader().load_directory(settings.data_raw_dir)
        chunks = TextChunker().chunk_documents(documents)
        if not chunks:
            raise HTTPException(status_code=422, detail="No readable text found. Scanned PDFs may require OCR before upload.")

        graph_store = state.qa_chain.retriever.graph_store
        stats = graph_store.rebuild_from_chunks(chunks, replace=True)
        uploaded_names = {p.name for p in saved}
        loaded_sources = {doc.source for doc in documents}
        unreadable_uploads = sorted(uploaded_names - loaded_sources)
        message = "Documents indexed; vault and knowledge graph updated."
        if unreadable_uploads:
            message += " Some uploads produced no extractable text (OCR may be unavailable)."
        return {
            "message": message,
            "warnings": [f"No text extracted from: {name}" for name in unreadable_uploads],
            "uploaded_files": [p.name for p in saved],
            "documents_loaded": len(documents),
            "graph": stats,
            "embedding_model": settings.embedding_model,
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Document upload/ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc
    finally:
        for upload in files:
            await upload.close()
