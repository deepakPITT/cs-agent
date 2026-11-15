import chromadb
from sentence_transformers import SentenceTransformer
import config

col = chromadb.PersistentClient(path=config.STORE_PATH).get_collection(config.COLLECTION)
model = SentenceTransformer(config.EMBED_MODEL)

print("count:", col.count())

r = col.get(limit=2, include=["documents", "metadatas", "embeddings"])
print(r["documents"])
print(r["metadatas"])
print(len(r["embeddings"][0]))

res = col.query(
    query_embeddings=model.encode(["I was charged twice"]).tolist(),
    n_results=3,
)
for d, dist, m in zip(res["documents"][0], res["distances"][0], res["metadatas"][0]):
    print(round(dist, 3), "|", d, "->", m["reply"][:80])

print(col.get(where={"category": "refund"}, limit=3)["documents"])
print(col.get(ids=["5"]))