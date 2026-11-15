"""Leave-one-out retrieval test on train.csv. No LLM, no API key."""
import pandas as pd

import config
from draft import col, model
from redact import redact

df = pd.read_csv(config.TRAIN_CSV).pipe(lambda d: d.sample(min(200, len(d)), random_state=0))
K = config.TOP_K

res = col.query(
    query_embeddings=model.encode(df.question.map(redact).tolist()).tolist(),
    n_results=K + 1,
    include=["metadatas", "distances"],
)

top1_hit, share, cat_hit, d_ok, d_bad = [], [], [], [], []
for i, row in enumerate(df.itertuples()):
    hits = [
        (m, d)
        for t, m, d in zip(res["ids"][i], res["metadatas"][i], res["distances"][i])
        if t != str(row.id)
    ][:K]
    same = [m["intent"] == row.intent for m, _ in hits]
    top1_hit.append(same[0])
    share.append(sum(same) / K)
    cat_hit.append(hits[0][0]["category"] == row.category)
    (d_ok if same[0] else d_bad).append(hits[0][1])

n = len(df)
print(f"top-1 intent match:   {sum(top1_hit) / n:.0%}")
print(f"top-{K} intent share:    {sum(share) / n:.0%}")
print(f"top-1 category match: {sum(cat_hit) / n:.0%}")
print(
    f"avg distance | right: {sum(d_ok) / max(len(d_ok), 1):.3f}  "
    f"wrong: {sum(d_bad) / max(len(d_bad), 1):.3f}"
)
