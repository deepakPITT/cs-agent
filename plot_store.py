import chromadb, numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import config

col = chromadb.PersistentClient(path=config.STORE_PATH).get_collection(config.COLLECTION)
r = col.get(include=["embeddings", "metadatas"])

xy = PCA(2).fit_transform(np.array(r["embeddings"]))
cats = [m["category"] for m in r["metadatas"]]
for c in sorted(set(cats)):
    i = [k for k, x in enumerate(cats) if x == c]
    plt.scatter(xy[i, 0], xy[i, 1], s=10, label=c)
plt.legend(fontsize=7)
plt.savefig("store_map.png", dpi=150)
print("saved store_map.png")