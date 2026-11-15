import pandas as pd

import config
from draft import col, draft, model

test = pd.read_csv(config.TEST_CSV)
rows = []
for _, r in test.iterrows():
    d = draft(r.question)
    a, b = model.encode([d, r.reply], normalize_embeddings=True)
    top = col.query(query_embeddings=model.encode([r.question]).tolist(), n_results=1)
    rows.append({
        "question": r.question,
        "real_reply": r.reply,
        "draft": d,
        "similarity": round(float(a @ b), 2),
        "top_distance": round(top["distances"][0][0], 3),
        "rating_1to5": "",
    })

out = pd.DataFrame(rows)
out.to_csv(config.RESULTS_CSV, index=False)
print("avg similarity:", out.similarity.mean().round(3))
print("Saved", config.RESULTS_CSV, "- rate each draft 1-5 in rating_1to5")
