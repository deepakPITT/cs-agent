import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import config
from draft import draft, retrieve

app = FastAPI()

class Ticket(BaseModel):
    text: str
    category: str | None = None

@app.post("/draft")
def make_draft(t: Ticket, x_api_key: str = Header(None)):
    if x_api_key != config.SERVICE_KEY:
        raise HTTPException(status_code=401, detail="bad key")
    hits = retrieve(t.text, t.category)
    return {
        "draft": draft(t.text, t.category),
        "sources": [{"id": i, "question": q, "distance": round(d, 3)} for i, q, m, d in hits],
    }