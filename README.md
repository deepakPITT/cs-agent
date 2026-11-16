Drafts replies to support tickets by retrieving similar past resolved tickets
(Chroma + sentence-transformers) and asking Claude to write a reply grounded in them.
Includes a CLI and a `/draft` HTTP API. No UI yet.

## How it works
```
ticket -> redact PII -> embed -> Chroma search (top-k similar tickets)
       -> prompt with examples -> Claude draft  (or NEEDS AGENT if not covered)
```
## Quick setup (Python 3.12)
```bash
git clone https://github.com/YOUR_USERNAME/cs-agent.git
cd cs-agent
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit .env
python scripts/get_bitext.py  # download data, build train/test CSVs
python index.py               # build the Chroma store
```

`.env`:
```
ANTHROPIC_API_KEY=sk-ant-...   # not needed in mock mode
SERVICE_KEY=any-long-random-string   # protects the /draft API
MOCK_LLM=0                     # 1 = no LLM, returns closest past reply
```
Generate a service key: `python -c "import secrets; print(secrets.token_urlsafe(32))"`

## Packages
chromadb, sentence-transformers, pandas, anthropic, datasets, python-dotenv,
matplotlib, fastapi, uvicorn (all in `requirements.txt`).

## Usage

**CLI (interactive tester)**
```bash
python ask.py --mock "I was charged twice"   # free, no API key
python ask.py "where is my package"          # real Claude draft
python ask.py --retrieval                    # retrieval only
```
Interactive prompt: type ticket text, `#id` for a train ticket (own entry excluded),
`@id` for a test ticket, `list` to show test questions.

**API**
```bash
uvicorn api:app --port 8000        # add MOCK_LLM=1 for no LLM
curl -X POST localhost:8000/draft -H "x-api-key: YOUR_SERVICE_KEY" \
  -H "Content-Type: application/json" -d '{"text": "I was charged twice"}'
```
Returns `{"draft": "...", "sources": [...]}`. Docs at `http://localhost:8000/docs`.

**Evaluate**
```bash
python tune.py              # leave-one-out retrieval test on train data (free)
python test_retrieval.py    # retrieval accuracy on held-out test tickets (free)
python eval.py              # drafts for test tickets -> data/processed/results.csv
python report.py data/processed/results_v1.csv   # summary after rating 1-5
```

## Project layout
```
config.py         settings and paths
redact.py         PII redaction
index.py          train.csv -> Chroma
draft.py          retrieve() and draft()
ask.py            CLI tester
api.py            FastAPI backend (/draft)
tune.py, eval.py, test_retrieval.py, report.py   evaluation
inspect_store.py, plot_store.py                  inspect the store
scripts/get_bitext.py   data download + train/test split
kb/               policy docs (planned second collection)
```

## Rebuild from scratch
Do this after changing data, `EMBED_MODEL`, or index metadata:
```bash
rm -rf store data/raw/*.csv data/processed/*.csv
python scripts/get_bitext.py && python index.py
```

## Contributing
Branch from `main`, open a pull request. Never commit `.env`, `store/`, or real ticket data.
