"""V3 intelligence endpoints: SSE, scoped chat, comparison, study tools and previews."""
from __future__ import annotations
import asyncio, json, time, re, shutil, subprocess, tempfile, os
from pathlib import Path
from uuid import uuid4
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
import app.state as state
from app.routes.history import connect
from src.utils.config import settings

router = APIRouter()

class V3ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    document_ids: Optional[list[str]] = None
    top_k: int = Field(5, ge=1, le=20)
    threshold: float = Field(0.3, ge=0, le=1)
    graph_enhanced: bool = True
    debug: bool = False

class CompareRequest(BaseModel):
    document_ids: list[str] = Field(min_length=2, max_length=10)
    question: str = Field(min_length=1, max_length=1500)

class StudyRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1, max_length=10)
    kind: str = Field(pattern="^(summary|notes|flashcards|mcq|short_answer)$")
    difficulty: str = Field(default="intermediate", pattern="^(beginner|intermediate|advanced)$")
    count: int = Field(default=8, ge=1, le=30)

def _store():
    if state.qa_chain is None:
        raise HTTPException(503, "RAG engine is initializing.")
    store=state.qa_chain.retriever.graph_store
    if not store.loaded: store.load()
    return store

def _validate_docs(ids):
    store=_store()
    available={f"doc::{d.get('source')}" for _,d in store.graph.nodes(data=True) if d.get("node_type")=="document"}
    invalid=set(ids)-available
    if invalid: raise HTTPException(404, detail={"message":"Unknown document ID(s)","invalid":sorted(invalid)})
    return store

def _history(cid):
    if not cid: return []
    with connect() as c:
        exists=c.execute("SELECT id FROM conversations WHERE id=?",(cid,)).fetchone()
        if not exists: raise HTTPException(404,"Conversation not found.")
        rows=c.execute("SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT " + str(settings.conversation_memory_messages) + "",(cid,)).fetchall()
        return [dict(r) for r in reversed(rows)]

def _save(cid, question, answer, sources):
    now=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
    cid=cid or uuid4().hex
    with connect() as c:
        exists=c.execute("SELECT id FROM conversations WHERE id=?",(cid,)).fetchone()
        if not exists: c.execute("INSERT INTO conversations VALUES(?,?,?,?)",(cid,question[:64],now,now))
        c.executemany("INSERT INTO messages(conversation_id,role,content,sources,created_at) VALUES(?,?,?,?,?)",[
            (cid,"user",question,"[]",now),(cid,"assistant",answer,json.dumps(sources),now)])
        c.execute("UPDATE conversations SET updated_at=? WHERE id=?",(now,cid))
    return cid

def _citations(results):
    output=[]; seen=set()
    for r in results:
        key=(r.document_id or f"doc::{r.source}",r.page,r.chunk_id)
        if key in seen: continue
        seen.add(key)
        output.append({"document_id":key[0],"source":r.source,"page":r.page,
                       "chunk_id":r.chunk_id,"metadata":r.metadata or {},
                       "label":f"{r.source}"+(f" · page {r.page}" if r.page else "")})
    return output

def _suggestions(results, question):
    if not results: return []
    concepts=[]
    for r in results:
        for term in re.findall(r"\b[A-Z][A-Za-z-]{3,}(?:\s+[A-Z][A-Za-z-]{3,})?\b",r.text):
            if term.casefold() not in {x.casefold() for x in concepts}: concepts.append(term)
    candidates=[f"Explain {x} with a simple example." for x in concepts[:3]]
    if len(results)>1: candidates.append("How do these retrieved sections relate to each other?")
    return [x for x in candidates if x.casefold()!=question.casefold()][:4]

@router.post("/chat/stream")
def stream_chat(req: V3ChatRequest, request: Request):
    store=_store()
    if req.document_ids is not None: _validate_docs(req.document_ids)
    history=_history(req.conversation_id)
    retrieval_query=req.question
    prior_questions=[m.get("content","") for m in history if m.get("role")=="user"][-3:]
    if prior_questions: retrieval_query += "\\nPrior user context: " + " | ".join(prior_questions)
    from src.llm.qa_chain import is_basic_conversation
    conversational = is_basic_conversation(req.question)
    started=time.perf_counter()
    results=[] if conversational else state.qa_chain.retriever.retrieve(retrieval_query,req.top_k,req.threshold,
        req.graph_enhanced,document_ids=req.document_ids)
    retrieval_latency_ms=round((time.perf_counter()-started)*1000,2)
    def events():
        answer=[]
        try:
            yield "event: meta\ndata: "+json.dumps({"conversation_id":req.conversation_id})+"\n\n"
            for token in state.qa_chain.stream_answer(req.question,req.top_k,req.threshold,
                    req.graph_enhanced,history,req.document_ids):
                answer.append(token)
                yield "event: chunk\ndata: "+json.dumps({"text":token})+"\n\n"
            text="".join(answer)
            cites=_citations(results)
            cid=_save(req.conversation_id,req.question,text,[c["source"] for c in cites])
            yield "event: citations\ndata: "+json.dumps(cites)+"\n\n"
            suggestions=[] if conversational else _suggestions(results,req.question)
            yield "event: suggestions\ndata: "+json.dumps(suggestions)+"\n\n"
            if req.debug:
                yield "event: debug\ndata: "+json.dumps({"chunk_count":len(results),
                    "retrieval_latency_ms":retrieval_latency_ms,
                    "retrieval_method":"embedding similarity + lexical boost + optional graph expansion",
                    "scores":[{"source":r.source,"document_id":r.document_id,"page":r.page,
                               "chunk_id":r.chunk_id,"score":round(r.score,4),"text":r.text[:1200],
                               "metadata":r.metadata or {}} for r in results]})+"\n\n"
            yield "event: done\ndata: "+json.dumps({"conversation_id":cid,"has_context":bool(results),"conversational":conversational})+"\n\n"
        except GeneratorExit: raise
        except Exception as exc:
            yield "event: error\ndata: "+json.dumps({"message":str(exc)})+"\n\n"
    return StreamingResponse(events(),media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@router.get("/documents/{document_id:path}/preview")
def preview_document(document_id: str):
    store=_store()
    node=store.graph.nodes[document_id] if document_id in store.graph else None
    if not node or node.get("node_type")!="document": raise HTTPException(404,"Document not found.")
    source=Path(node.get("source","")).name
    candidates=[p for p in settings.data_raw_dir.rglob(source) if p.is_file()]
    if not candidates: raise HTTPException(404,"Original source file is unavailable.")
    path=candidates[0].resolve()
    base=settings.data_raw_dir.resolve()
    if base not in path.parents: raise HTTPException(403,"Invalid source path.")
    return FileResponse(path,filename=source,content_disposition_type="inline")

@router.post("/compare")
def compare_documents(req: CompareRequest):
    _validate_docs(req.document_ids)
    results=[]
    for did in req.document_ids:
        hits=state.qa_chain.retriever.retrieve(req.question,top_k=5,threshold=0.05,
            use_graph=False,document_ids=[did])
        results.extend(hits)
    if not results: raise HTTPException(422,"No relevant passages found in selected documents.")
    context=state.qa_chain.retriever.format_context(results)
    prompt=("Compare only the evidence below for this question: "+req.question+
        "\nReturn sections: Similarities, Differences, Key claims, Dates/figures/terms, "
        "Contradictions, Missing evidence. Cite each point as [Source: filename, page N] "
        "when page exists. State 'Not found in this document' rather than infer. Evidence:\n"+context)
    answer=state.qa_chain._generate("You are a careful document comparison analyst. Never invent facts.",prompt)
    return {"answer":answer,"citations":_citations(results),
            "documents":sorted({r.source for r in results})}

@router.post("/study")
def study_tools(req: StudyRequest):
    _validate_docs(req.document_ids)
    results=[]
    for did in req.document_ids:
        results.extend(state.qa_chain.retriever.retrieve(
            f"key concepts definitions facts examples {req.kind}",top_k=8,threshold=0.05,
            use_graph=False,document_ids=[did]))
    if not results: raise HTTPException(422,"No useful source passages were found.")
    context=state.qa_chain.retriever.format_context(results)
    instructions={
      "summary":"Create a concise structured summary with key takeaways.",
      "notes":"Create organized study notes with headings, definitions and examples.",
      "flashcards":f"Create exactly {req.count} concise front/back flashcards.",
      "mcq":f"Create {req.count} multiple-choice questions with four options. Put ALL questions first, then a clearly separated ANSWER KEY with correct options and explanations.",
      "short_answer":f"Create {req.count} short-answer questions first, then a clearly separated ANSWER KEY with model answers and explanations."
    }[req.kind]
    prompt=f"Difficulty: {req.difficulty}. {instructions}\nUse only the evidence. Include source/page references for each item; if evidence is absent, omit it.\n\nEvidence:\n{context}"
    answer=state.qa_chain._generate("You create accurate, source-grounded learning materials.",prompt)
    return {"kind":req.kind,"difficulty":req.difficulty,"output":answer,
            "citations":_citations(results),"source_count":len({r.source for r in results})}


@router.post("/video/transcribe")
async def transcribe_video(file: UploadFile = File(...)):
    """Extract speech from an uploaded video using FFmpeg + Groq Whisper.

    The video is processed in a temporary directory and is not persisted.
    The returned transcript can be attached to a normal RAG chat request.
    """
    if state.qa_chain is None:
        raise HTTPException(503, "RAG engine is initializing.")
    allowed = {".mp4", ".webm", ".mov", ".mkv", ".mpeg", ".mpg"}
    name = Path(file.filename or "video").name
    suffix = Path(name).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, "Unsupported video format. Use MP4, WebM, MOV, or MKV.")
    ffmpeg = shutil.which(os.getenv("FFMPEG_BINARY", "ffmpeg"))
    if not ffmpeg:
        raise HTTPException(503, "FFmpeg is required for video input. Install FFmpeg and add it to PATH.")
    content = await file.read()
    await file.close()
    if not content:
        raise HTTPException(400, "The uploaded video is empty.")
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(413, "Video exceeds the 100 MB upload limit.")
    try:
        with tempfile.TemporaryDirectory(prefix="qrag_video_") as tmp:
            source = Path(tmp) / ("input" + suffix)
            audio = Path(tmp) / "audio.mp3"
            source.write_bytes(content)
            proc = subprocess.run([ffmpeg, "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(audio)],
                capture_output=True, text=True, timeout=300)
            if proc.returncode != 0 or not audio.exists():
                raise HTTPException(422, "Could not extract audio from this video. Check that the file contains a supported audio track.")
            if audio.stat().st_size > 24 * 1024 * 1024:
                raise HTTPException(413, "Extracted audio exceeds Groq's 24 MB upload limit. Use a shorter or lower-resolution video.")
            client = state.qa_chain._get_llm_client()
            with audio.open("rb") as stream:
                result = client.audio.transcriptions.create(
                    file=(f"{Path(name).stem}.mp3", stream.read()),
                    model=os.getenv("VIDEO_TRANSCRIPTION_MODEL", "whisper-large-v3-turbo"),
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
            transcript = getattr(result, "text", "") or ""
            segments = getattr(result, "segments", None) or []
            if segments:
                lines=[]
                for seg in segments:
                    get = (lambda key: seg.get(key)) if isinstance(seg, dict) else (lambda key: getattr(seg, key, None))
                    start=get("start"); end=get("end"); text=get("text") or ""
                    stamp=f"[{float(start):.1f}s–{float(end):.1f}s] " if start is not None and end is not None else ""
                    if text: lines.append(stamp+text.strip())
                if lines: transcript="\n".join(lines)
            if not transcript.strip():
                raise HTTPException(422, "No speech was detected in this video.")
            return {"filename": name, "transcript": transcript, "model": os.getenv("VIDEO_TRANSCRIPTION_MODEL", "whisper-large-v3-turbo"), "timestamped": bool(segments)}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(408, "Video audio extraction timed out.") from exc
    except Exception as exc:
        log_message = f"Video transcription failed: {exc}"
        import logging
        logging.getLogger(__name__).exception(log_message)
        raise HTTPException(502, "Video transcription failed. Check Groq credentials, model availability, and video audio format.") from exc

# ---------------------------------------------------------------------------
# Automatic inline scientific diagrams for educational answers
# ---------------------------------------------------------------------------
class DiagramRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(default="", max_length=12000)

@router.post("/visualizations/auto")
def auto_visualization(req: DiagramRequest):
    """Return a locally rendered diagram when the question/answer names a supported concept."""
    from fastapi.responses import Response
    from app.quantum_visualizer import select_visualizations, render_diagram
    selection = select_visualizations(req.question, req.answer)
    kind = selection["visualization_type"]
    if not kind:
        return {"generated": False, "reason": "No supported visual concept detected."}
    try:
        png = render_diagram(kind)
        title = selection.get("title", kind.replace("_", " ").title())
        return Response(content=png, media_type="image/png", headers={
            "X-Diagram-Type": kind,
            "X-Diagram-Title": title,
            "X-Diagram-Alt": title,
            "X-Diagram-Confidence": str(selection["confidence"]),
            "X-Diagram-Description": selection["reason"],
            "Content-Disposition": f'inline; filename="{kind}.png"',
            "Cache-Control": "no-store",
        })
    except Exception as exc:
        import logging
        logging.getLogger(__name__).exception("Diagram generation failed: %s", exc)
        raise HTTPException(500, "Could not generate the requested scientific diagram.") from exc
