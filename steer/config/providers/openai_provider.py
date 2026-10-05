"""OpenAI provider: Responses API for standard models, Chat Completions for
reasoning models (o1/o3/o4-*, which need reasoning_effort and don't support
the Responses API's plain `input` field the same way)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class OpenAIProvider:
    def __init__(self, model: str = "o3", api_base: str = "https://api.openai.com/v1", api_key: str | None = None):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            url=f"{self.api_base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API request failed with status {e.code}: {body}") from e

    def generate(self, prompt: str, *, reasoning_effort: str | None = None) -> str:
        if reasoning_effort is not None:
            response = self._post(
                "/chat/completions",
                {"model": self.model, "messages": [{"role": "user", "content": prompt}], "reasoning_effort": reasoning_effort},
            )
            content = response["choices"][0]["message"].get("content") or ""
            text = content.strip()
            if not text:
                raise ValueError("Model response did not contain content")
            return text

        response = self._post("/responses", {"model": self.model, "input": prompt})
        texts: list[str] = []
        for item in response.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    texts.append(content.get("text", ""))
        text = "".join(texts).strip()
        if not text:
            raise ValueError("Model response did not contain output_text content")
        return text
