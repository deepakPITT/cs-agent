import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

RAW_CSV = ROOT / "data/raw/bitext.csv"
TRAIN_CSV = ROOT / "data/processed/train.csv"
TEST_CSV = ROOT / "data/processed/test.csv"
RESULTS_CSV = ROOT / "data/processed/results.csv"

STORE_PATH = str(ROOT / "store")
COLLECTION = "tickets"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "claude-sonnet-5-5"
TOP_K = 4
MAX_DIST = None  # set after checking distances; None = no cutoff
# Mock mode: draft() returns the best-matching past reply, no LLM call, no API key.
# Turn on with MOCK_LLM=1 in .env or the shell, or set True here.
MOCK_LLM = os.getenv("MOCK_LLM", "0") == "1"

SAMPLE_PER_CATEGORY = 100
TEST_SIZE = 20
SERVICE_KEY="t10iqIRReG0bRgWy72hOf9SNrVmajKGN2ARKBo5UQdE"
