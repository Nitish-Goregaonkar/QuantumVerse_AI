"""Knowledge-graph endpoints used by the interactive graph canvas."""
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

import app.state as state

router = APIRouter()


class NewNode(BaseModel):
    label: str = Field(..., min_length=1, max_length=80)
    type: Literal["Concept", "Quantum Entity", "Document"] = "Concept"
    description: str = Field(..., min_length=1, max_length=1000)


def _store():
    if state.qa_chain is None:
        raise HTTPException(status_code=503, detail="Backend is still initializing.")
    store = state.qa_chain.retriever.graph_store
    if not store.loaded:
        store.load()
    return store


@router.get("/graph/nodes")
async def graph_nodes(limit: int = Query(60, ge=5, le=300)):
    """Documents plus the most-connected concepts, with the links between them."""
    return _store().overview(limit=limit)


@router.get("/graph/node/{node_id:path}")
async def graph_node(node_id: str):
    detail = _store().node_detail(node_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Node not found.")
    return detail


@router.post("/graph/node")
async def create_graph_node(node: NewNode):
    try:
        return _store().add_custom_node(node.label, node.type, node.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/graph/node/{node_id:path}")
async def delete_graph_node(node_id: str):
    """Delete document nodes with their chunks, or user-created custom nodes."""
    store = _store()
    data = store.graph.nodes.get(node_id)
    if not data:
        raise HTTPException(status_code=404, detail="Graph node not found.")
    if data.get("node_type") == "document":
        try:
            result = store.delete_document(data.get("source", node_id))
        except KeyError:
            raise HTTPException(status_code=404, detail="Document not found.")
        return {"message": "Document and its chunks removed.", **result, "graph": store.stats()}
    if data.get("custom"):
        store.graph.remove_node(node_id)
        store.save()
        return {"message": "Custom graph node removed.", "graph": store.stats()}
    raise HTTPException(status_code=400, detail="Only document nodes and user-created nodes can be deleted directly.")
