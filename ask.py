"""Interactive tester.
  python ask.py                  retrieval + Claude draft
  python ask.py --mock           retrieval + mock draft (no API key)
  python ask.py --retrieval      retrieval only
Type ticket text, or #id to test a train.csv ticket (its own entry is excluded).
"""
import sys

import pandas as pd

import config

if "--mock" in sys.argv:
    config.MOCK_LLM = True
from draft import draft, retrieve

NO_DRAFT = "--retrieval" in sys.argv
train = pd.read_csv(config.TRAIN_CSV).set_index("id")
test = pd.read_csv(config.TEST_CSV).set_index("id")

while True:
    t = input("\nTicket text, or #id from train.csv (blank to quit): ").strip()
    if not t:
        break
    ex, row = None, None
    if t == "list":
        for i, r in test.iterrows():
            print(f"@{i}: {r.question[:70]}")
        continue
    if t[:1] in ("#", "@"):
        src = train if t[0] == "#" else test
        try:
            rid = int(t[1:])
            row = src.loc[rid]
        except (ValueError, KeyError):
            print("id not found")
            continue
        ex = rid if t[0] == "#" else None
        t = row.question
        print(f"\n[{row.category}/{row.intent}]\nQ: {t}\nReal reply: {row.reply}")
        try:
            ex = int(t[1:])
            row = train.loc[ex]
        except (ValueError, KeyError):
            print("id not found in train.csv")
            continue
        t = row.question
        print(f"\n[train #{ex}] {row.category}/{row.intent}\nQ: {t}\nReal reply: {row.reply}")

    print("\n--- Retrieved (own entry excluded) ---")
    for i, d, m, dist in retrieve(t, exclude_id=ex):
        ok = "" if row is None else (" OK" if m["intent"] == row.intent else " WRONG")
        print(f"[{dist:.3f}] #{i} {m['category']}/{m['intent']}{ok}\n   {d[:70]}\n   -> {m['reply'][:90]}")

    if not NO_DRAFT:
        print("\n--- Draft ---")
        print(draft(t, exclude_id=ex))