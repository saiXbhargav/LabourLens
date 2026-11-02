"""Gemini client with retries and model fallback."""

from __future__ import annotations

import time
from typing import Protocol

from src.config import GEMINI_API_KEY, GEMINI_MODELS


class LLM(Protocol):
    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> tuple[str, str]:
        """Return (text, model_used). With json_mode the model must reply with JSON."""
        ...


class GeminiLLM:
    def __init__(
        self, models: list[str] | None = None, temperature: float = 0.0, retries: int = 2, timeout_s: float = 120
    ):
        from google import genai
        from google.genai import types

        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set (add it to backend/.env)")
        # A per-request timeout so one stuck call on a slow network can't hang the pipeline.
        self.client = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        self.models = models or GEMINI_MODELS
        self.temperature = temperature
        self.retries = retries
        self.last_usage: dict = {}  # token counts of the most recent successful call

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> tuple[str, str]:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        errors = []
        for model in self.models:
            for attempt in range(self.retries + 1):
                try:
                    resp = self.client.models.generate_content(model=model, contents=prompt, config=config)
                    u = resp.usage_metadata
                    self.last_usage = {
                        "prompt_tokens": getattr(u, "prompt_token_count", None),
                        "output_tokens": getattr(u, "candidates_token_count", None),
                        "thinking_tokens": getattr(u, "thoughts_token_count", None),
                    }
                    return (resp.text or "").strip(), resp.model_version or model
                except Exception as e:
                    code = getattr(e, "code", None)
                    errors.append(f"{model} (attempt {attempt + 1}): {code or type(e).__name__}: {str(e)[:200]}")
                    if code == 429 and attempt < self.retries:  # rate limit: wait, then retry same model
                        time.sleep(5 * (attempt + 1))
                        continue
                    break  # overloaded (503/504), timeout, bad request: try the next model now
        raise RuntimeError("All Gemini models failed:\n" + "\n".join(errors))
