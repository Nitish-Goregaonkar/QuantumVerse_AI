"""Persistent conversation history API backed by SQLite."""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.config import settings

router = APIRouter(prefix="/history", tags=["History"])
DB_PATH = Path(settings.data_raw_dir).parent / "chat_history.sqlite3"

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
        role TEXT NOT NULL, content TEXT NOT NULL, sources TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL, FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE)""")
    conn.commit()
    return conn

class Exchange(BaseModel):
    conversation_id: Optional[str] = None
    question: str
    answer: str
    sources: list = Field(default_factory=list)

@router.get("")
def list_history():
    with connect() as c:
        rows=c.execute("""SELECT c.id,c.title,c.created_at,c.updated_at,
        (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) AS message_count
        FROM conversations c ORDER BY c.updated_at DESC""").fetchall()
        return [dict(r) for r in rows]

@router.get("/{conversation_id}")
def get_history(conversation_id: str):
    with connect() as c:
        conv=c.execute("SELECT * FROM conversations WHERE id=?",(conversation_id,)).fetchone()
        if not conv: raise HTTPException(404,"Conversation not found")
        msgs=c.execute("SELECT role,content,sources,created_at FROM messages WHERE conversation_id=? ORDER BY id",(conversation_id,)).fetchall()
        return {"conversation":dict(conv),"messages":[{**dict(m),"sources":__import__('json').loads(m["sources"])} for m in msgs]}

@router.post("")
def save_exchange(data: Exchange):
    now=datetime.now(timezone.utc).isoformat()
    cid=data.conversation_id or __import__('uuid').uuid4().hex
    title=data.question.strip()[:64] or "New conversation"
    with connect() as c:
        exists=c.execute("SELECT id FROM conversations WHERE id=?",(cid,)).fetchone()
        if not exists: c.execute("INSERT INTO conversations VALUES(?,?,?,?)",(cid,title,now,now))
        c.executemany("INSERT INTO messages(conversation_id,role,content,sources,created_at) VALUES(?,?,?,?,?)",[
            (cid,"user",data.question,"[]",now),(cid,"assistant",data.answer,__import__('json').dumps(data.sources),now)])
        c.execute("UPDATE conversations SET updated_at=? WHERE id=?",(now,cid))
        return {"id":cid,"title":title}

@router.patch("/{conversation_id}")
def rename_history(conversation_id: str, title: str):
    title=title.strip()
    if not title: raise HTTPException(400,"Title cannot be empty")
    with connect() as c:
        cur=c.execute("UPDATE conversations SET title=?,updated_at=? WHERE id=?",(title,datetime.now(timezone.utc).isoformat(),conversation_id))
        if not cur.rowcount: raise HTTPException(404,"Conversation not found")
        return {"ok":True,"title":title}

@router.delete("/{conversation_id}")
def delete_history(conversation_id: str):
    with connect() as c:
        c.execute("DELETE FROM messages WHERE conversation_id=?",(conversation_id,))
        cur=c.execute("DELETE FROM conversations WHERE id=?",(conversation_id,))
        if not cur.rowcount: raise HTTPException(404,"Conversation not found")
        return {"ok":True}
