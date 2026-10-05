"""Anthropic provider: Messages API."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class AnthropicProvider:
    def __init__(self, model: str = "claude-sonnet-5", api_base: str = "https://api.anthropic.com/v1", api_key: str | None = None, max_tokens: int = 32000):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

    def generate(self, prompt: str, *, reasoning_effort: str | None = None) -> str:
        payload: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if reasoning_effort is not None:
            # Claude 5 uses adaptive thinking plus an effort setting, not a token budget.
            payload["thinking"] = {"type": "adaptive"}
            payload["output_config"] = {"effort": reasoning_effort}

        req = urllib.request.Request(
            url=f"{self.api_base}/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                response = json.load(resp)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic API request failed with status {e.code}: {body}") from e

        texts = [block["text"] for block in response.get("content", []) if block.get("type") == "text"]
        text = "".join(texts).strip()
        if not text:
            raise ValueError("Model response did not contain text content")
        return text
