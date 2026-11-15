import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

import config
from redact import redact

model = SentenceTransformer(config.EMBED_MODEL)
db = chromadb.PersistentClient(path=config.STORE_PATH)
col = db.get_or_create_collection(config.COLLECTION)

df = pd.read_csv(config.TRAIN_CSV)
df["question"] = df.question.map(redact)
df["reply"] = df.reply.map(redact)

col.upsert(
    ids=df.id.astype(str).tolist(),
    embeddings=model.encode(df.question.tolist()).tolist(),
    documents=df.question.tolist(),
    metadatas=[
        {"category": c, "intent": i, "reply": r}
        for c, i, r in zip(df.category, df.intent, df.reply)
    ],
)
print(f"Indexed {len(df)} tickets. Total in store: {col.count()}")
