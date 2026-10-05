"""Provider protocol: one LLM call, given a prompt, returns raw text output."""

from __future__ import annotations

from typing import Protocol


class Provider(Protocol):
    def generate(self, prompt: str, *, reasoning_effort: str | None = None) -> str: ...
