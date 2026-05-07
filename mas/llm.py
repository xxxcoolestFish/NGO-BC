"""
mas/llm.py — Minimal LLM client. OpenAI SDK compatible.
"""

from __future__ import annotations

import os
from typing import Optional

import tiktoken
from openai import OpenAI


class LLMClient:
    """Minimal LLM wrapper with token counting and max_tokens control."""

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "",
        model_name: str = "",
        p_in: float = 0.0,
        p_out: float = 0.0,
        default_max_tokens: int = 4096,
    ):
        api_key = api_key or os.environ.get("LLM_API_KEY", "")
        base_url = base_url or os.environ.get("LLM_BASE_URL", "")
        model_name = model_name or os.environ.get("LLM_MODEL", "")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.p_in = p_in or float(os.environ.get("LLM_P_IN", "0"))
        self.p_out = p_out or float(os.environ.get("LLM_P_OUT", "0"))
        self.max_tokens = default_max_tokens

        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    def chat(self, system: str, user: str, max_tokens: int,
             temperature: float = 0.7) -> tuple[str, int]:
        """Single-turn chat. Returns (content, output_token_count)."""
        max_tokens = max(10, min(max_tokens, self.max_tokens))
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content or ""
        c_out = response.usage.completion_tokens if response.usage else len(content) // 4
        return content, c_out

    def count_tokens(self, *texts: str) -> int:
        """Token count for one or more strings."""
        if self._enc:
            return sum(len(self._enc.encode(t)) for t in texts)
        return sum(len(t) // 4 for t in texts)

    def count_messages(self, system: str, user: str) -> int:
        """Token count for system + user messages."""
        return self.count_tokens(system, user)
