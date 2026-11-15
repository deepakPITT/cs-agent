"""Download Bitext, save raw copy, build processed train/test CSVs.

Run from project root:  python scripts/get_bitext.py
"""
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


def clean(t):
    t = re.sub(r"\{\{(.*?)\}\}", lambda m: f"[{m.group(1).lower()}]", str(t))
    return re.sub(r"\s+", " ", t).strip()


raw = load_dataset(NAME)["train"].to_pandas()
config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
raw.to_csv(config.RAW_CSV, index=False)
print("raw rows:", len(raw), "| columns:", list(raw.columns))

df = pd.DataFrame({
    "category": raw.category.str.lower(),
    "intent": raw.intent.str.lower(),
    "question": raw.instruction.map(clean),
    "reply": raw.response.map(clean),
})
df = df.drop_duplicates("question")
df = (
    df.sample(frac=1, random_state=1)
    .groupby("category")
    .head(config.SAMPLE_PER_CATEGORY)
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)
df.insert(0, "id", df.index + 1)

test, train = df.head(config.TEST_SIZE), df.iloc[config.TEST_SIZE:]
config.TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
train.to_csv(config.TRAIN_CSV, index=False)
test.to_csv(config.TEST_CSV, index=False)

print(f"train: {len(train)} | test: {len(test)}")
print(train.category.value_counts())
