import json
from types import SimpleNamespace
import pytest
from fastapi import HTTPException
from app.routes import v3

class FakeResult:
    def __init__(self, source="notes.pdf", page=3, chunk_id=7, score=.81):
        self.source=source; self.page=page; self.chunk_id=chunk_id; self.score=score
        self.document_id="doc::"+source; self.metadata={"format":"pdf"}
        self.text="Quantum circuits use Hadamard gates and entanglement."

def test_citations_are_stable_and_deduplicated():
    items=v3._citations([FakeResult(),FakeResult()])
    assert len(items)==1
    assert items[0]["document_id"]=="doc::notes.pdf"
    assert items[0]["page"]==3
    assert items[0]["chunk_id"]==7

def test_suggestions_are_grounded_and_not_empty():
    suggestions=v3._suggestions([FakeResult()], "Explain quantum circuits")
    assert suggestions
    assert all(isinstance(x,str) and x for x in suggestions)

def test_suggestions_empty_without_retrieval():
    assert v3._suggestions([], "Anything?")==[]

def test_document_id_validation_rejects_unknown(monkeypatch):
    class Graph:
        nodes=staticmethod(lambda data=False: [])
    store=SimpleNamespace(graph=Graph())
    monkeypatch.setattr(v3,"_store",lambda:store)
    with pytest.raises(HTTPException) as err:
        v3._validate_docs(["doc::not-here.pdf"])
    assert err.value.status_code==404

def test_stream_request_schema_rejects_bad_limits():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        v3.V3ChatRequest(question="x",top_k=100)
    with pytest.raises(ValidationError):
        v3.StudyRequest(document_ids=["doc::a"],kind="not-a-kind")


def test_sse_event_contract_with_mocked_groq(monkeypatch):
    import asyncio
    class Nodes:
        def __call__(self, data=False):
            return [("doc::notes.pdf", {"node_type":"document","source":"notes.pdf"})]
    store=SimpleNamespace(graph=SimpleNamespace(nodes=Nodes()))
    result=FakeResult()
    fake_chain=SimpleNamespace(
        retriever=SimpleNamespace(retrieve=lambda *a,**k:[result]),
        stream_answer=lambda *a,**k:iter(["Hello"," world"]))
    monkeypatch.setattr(v3,"_store",lambda:store)
    monkeypatch.setattr(v3,"_history",lambda cid:[])
    monkeypatch.setattr(v3,"_save",lambda *a:"conversation-test")
    monkeypatch.setattr(v3.state,"qa_chain",fake_chain)
    response=v3.stream_chat(v3.V3ChatRequest(question="Explain it",debug=True),object())
    async def collect():
        chunks=[]
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk,bytes) else chunk)
        return "".join(chunks)
    payload=asyncio.run(collect())
    for event in ["event: meta","event: chunk","event: citations","event: suggestions","event: debug","event: done"]:
        assert event in payload
    assert '"text": "Hello"' in payload
    assert '"conversation_id": "conversation-test"' in payload
    assert "GROQ_API_KEY" not in payload and "your_groq_api_key" not in payload

def test_compare_and_study_use_selected_source_evidence(monkeypatch):
    result=FakeResult()
    class Retriever:
        def retrieve(self,*args,**kwargs): return [result]
        def format_context(self,items): return "Evidence from notes.pdf page 3"
    class Chain:
        retriever=Retriever()
        def _generate(self,system,prompt): return "Grounded output"
    class Nodes:
        def __call__(self,data=False):
            return [("doc::notes.pdf",{"node_type":"document","source":"notes.pdf"}),
                    ("doc::second.pdf",{"node_type":"document","source":"second.pdf"})]
    monkeypatch.setattr(v3,"_store",lambda:SimpleNamespace(graph=SimpleNamespace(nodes=Nodes())))
    monkeypatch.setattr(v3.state,"qa_chain",Chain())
    compared=v3.compare_documents(v3.CompareRequest(document_ids=["doc::notes.pdf","doc::second.pdf"],question="Compare"))
    studied=v3.study_tools(v3.StudyRequest(document_ids=["doc::notes.pdf"],kind="summary"))
    assert compared["answer"]=="Grounded output" and compared["citations"]
    assert studied["output"]=="Grounded output" and studied["kind"]=="summary"


def test_retrieval_document_filter_blocks_other_sources():
    import networkx as nx
    import numpy as np
    from src.retrieval.retriever import Retriever
    class Embed:
        def embed_query(self,text): return np.array([1.0,0.0])
    graph=nx.Graph()
    for source,vec in [("a.pdf",[1.0,0.0]),("b.pdf",[1.0,0.0])]:
        graph.add_node("chunk::"+source,node_type="chunk",source=source,text="quantum retrieval passage",
                       embedding=vec,page=2,chunk_id=1,metadata={"format":"pdf"})
    store=SimpleNamespace(graph=graph,loaded=True,embedding_engine=Embed())
    hits=Retriever(store).retrieve("quantum retrieval",top_k=5,threshold=0.1,use_graph=False,
                                   document_ids=["doc::a.pdf"])
    assert hits and {x.source for x in hits}=={"a.pdf"}


def test_conversation_memory_is_scoped_to_conversation(monkeypatch,tmp_path):
    import app.routes.history as history
    monkeypatch.setattr(history,"DB_PATH",tmp_path/"history.sqlite3")
    monkeypatch.setattr(v3,"connect",history.connect)
    cid=v3._save(None,"Question A","Answer A",[])
    assert v3._history(cid)[0]["content"]=="Question A"
    with pytest.raises(HTTPException) as err:
        v3._history("not-this-conversation")
    assert err.value.status_code==404

@pytest.mark.asyncio
async def test_video_transcription_rejects_unsupported_extension(monkeypatch):
    from io import BytesIO
    from starlette.datastructures import UploadFile as StarletteUploadFile
    monkeypatch.setattr(v3.state, "qa_chain", object())
    upload = StarletteUploadFile(filename="not-a-video.pdf", file=BytesIO(b"content"))
    with pytest.raises(HTTPException) as err:
        await v3.transcribe_video(upload)
    assert err.value.status_code == 400


def test_chat_ui_has_sidebar_memory_and_video_action():
    from pathlib import Path
    html = (Path(__file__).parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="conversation-list"' in html
    assert 'id="chat-video-input"' in html
    assert "handleVideoSelected" in html
    assert 'id="v3-toolbar"' not in html
