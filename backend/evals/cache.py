"""Disk cache and rate limiting for LLM calls during evaluation.

Re-running an experiment (or the judge) with identical prompts costs nothing and isn't
rate-limited; cached entries keep the latency and token counts of the original call.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time

from src.config import DATA

CACHE_PATH = DATA / "cache" / "llm_cache.sqlite"


class CachedLLM:
    def __init__(self, llm, min_interval_s: float = 0.0, path=CACHE_PATH):
        self.llm = llm
        self.min_interval_s = min_interval_s
        self._last_call = 0.0
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, value TEXT)")
        self.last_meta: dict = {}  # latency/usage/cached flag of the latest generate()

    def _key(self, prompt: str, system: str | None, json_mode: bool) -> str:
        spec = {"models": self.llm.models, "temperature": self.llm.temperature,
                "system": system, "prompt": prompt, "json": json_mode}
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False,
                 refresh: bool = False) -> tuple[str, str]:
        """refresh=True ignores (and replaces) a cached response, e.g. a truncated judge verdict."""
        key = self._key(prompt, system, json_mode)
        row = None if refresh else self.db.execute("SELECT value FROM calls WHERE key = ?", (key,)).fetchone()
        if row:
            v = json.loads(row[0])
            self.last_meta = {**v["meta"], "cached": True}
            return v["text"], v["model"]

        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        t0 = time.perf_counter()
        text, model = self.llm.generate(prompt, system=system, json_mode=json_mode)
        self._last_call = time.monotonic()
        meta = {"latency_ms": (time.perf_counter() - t0) * 1000, **getattr(self.llm, "last_usage", {})}
        self.db.execute("INSERT OR REPLACE INTO calls VALUES (?, ?)",
                        (key, json.dumps({"text": text, "model": model, "meta": meta})))
        self.db.commit()
        self.last_meta = {**meta, "cached": False}
        return text, model
