import sys

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

import config
from redact import redact

model = SentenceTransformer(config.EMBED_MODEL)
col = chromadb.PersistentClient(path=config.STORE_PATH).get_collection(config.COLLECTION)


def retrieve(ticket, category=None, exclude_id=None):
    """Retrieval only (no LLM, no API key). Returns [(id, question, meta, distance)]."""
    res = col.query(
        query_embeddings=model.encode([redact(ticket)]).tolist(),
        n_results=config.TOP_K + 1,
        where={"category": category} if category else None,
        include=["documents", "metadatas", "distances"],
    )
    hits = list(zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]))
    return [h for h in hits if h[0] != str(exclude_id)][: config.TOP_K]


def draft(ticket, category=None, exclude_id=None):
    hits = retrieve(ticket, category, exclude_id)
    if config.MAX_DIST is not None and hits[0][3] > config.MAX_DIST:
        return "NEEDS AGENT: no similar resolved ticket found."
    if config.MOCK_LLM:
        return f"[MOCK DRAFT: reply of closest ticket #{hits[0][0]}]\n{hits[0][2]['reply']}"
    examples = "\n\n".join(f"Ticket: {d}\nReply: {m['reply']}" for _, d, m, _ in hits)
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": (
                f"Past resolved tickets:\n\n{examples}\n\n"
                "Draft a reply to the new ticket below, matching the tone and "
                "resolutions above. Use ONLY facts in the examples. Never invent "
                "amounts, dates, timeframes, or links. If the examples do not cover "
                "the question, reply exactly 'NEEDS AGENT:' followed by what "
                "information is missing.\n\n"
                f"New ticket: {redact(ticket)}"
            ),
        }],
    )
    return msg.content[0].text


if __name__ == "__main__":
    print(draft(" ".join(sys.argv[1:]) or "I was charged twice this month"))
