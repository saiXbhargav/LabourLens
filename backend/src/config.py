"""Paths and settings shared by the pipeline. Values can be overridden in backend/.env."""

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND / ".env")

DATA = BACKEND / "data"
PROCESSED = DATA / "processed"
QDRANT_PATH = DATA / "qdrant"  # Qdrant local mode: an on-disk folder, no server needed

QDRANT_URL = os.getenv("QDRANT_URL") or None  # set to use a Qdrant server / Qdrant Cloud instead
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# "*-latest" aliases track Google's current models. Flash-Lite answers in ~1 s and has higher
# free-tier limits (evals make hundreds of calls); Flash is the fallback. Override with GEMINI_MODEL.
GEMINI_MODELS = [os.getenv("GEMINI_MODEL")] if os.getenv("GEMINI_MODEL") else [
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
]

DEFAULT_COLLECTION = "baseline"
DEFAULT_TOP_K = 5
