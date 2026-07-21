"""
gemini_safe.py
================
Shared, cost-safe Gemini model wrapper.

Policy (added 2026-06-26, after gemini-2.5-pro usage across this project's
many scripts ran up a ~550 EUR bill): every Gemini call in this codebase
must try gemini-2.5-flash-lite first; only on failure fall back to
gemini-2.5-flash; if that also fails, give up and return an empty string.
gemini-2.5-pro (or any other "pro" tier model) must never be called,
anywhere, regardless of how the failure looks.

Usage
-----
    from gemini_safe import SafeGeminiModel

    model = SafeGeminiModel(api_key=GEMINI_API_KEY,
                             generation_config={"temperature": 0, "max_output_tokens": 4096},
                             safety_settings=SAFETY_SETTINGS)
    text = model.generate_content_safe(prompt, timeout=180)  # "" on total failure
"""
from __future__ import annotations

import time
from typing import Optional

import google.generativeai as genai

FLASH_LITE = "gemini-2.5-flash-lite"
FLASH = "gemini-2.5-flash"


class SafeGeminiModel:
    def __init__(self, api_key: str, generation_config: Optional[dict] = None,
                 safety_settings: Optional[list] = None):
        genai.configure(api_key=api_key)
        self.generation_config = generation_config or {}
        self.safety_settings = safety_settings
        self._model_lite = genai.GenerativeModel(FLASH_LITE)
        self._model_flash = genai.GenerativeModel(FLASH)

    def _try_tier(self, model, prompt: str, retries: int, timeout: int) -> Optional[str]:
        """Returns the response text on success (possibly "" if content-filtered
        -- not retried further, since the same content will hit the same filter
        on any model), or None if every retry raised (signals: escalate to the
        next tier)."""
        for attempt in range(retries):
            try:
                resp = model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                    request_options={"timeout": timeout},
                )
                if not resp.candidates or resp.candidates[0].finish_reason not in (1, "STOP"):
                    return ""
                return resp.text or ""
            except Exception:
                if attempt == retries - 1:
                    return None
                time.sleep(5 * (attempt + 1))
        return None

    def generate_content_safe(self, prompt: str, retries_per_tier: int = 2,
                               timeout: int = 180) -> str:
        """flash-lite first; on total failure, flash; on total failure, give up
        ("" ). Never escalates to gemini-2.5-pro."""
        result = self._try_tier(self._model_lite, prompt, retries_per_tier, timeout)
        if result is not None:
            return result
        result = self._try_tier(self._model_flash, prompt, retries_per_tier, timeout)
        if result is not None:
            return result
        return ""
