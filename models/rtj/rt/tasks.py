"""The task description the sampler and dataloader consume (built by STEER's task registry)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    db_name: str
    table_name: str
    target_column: str
    task_type: str  # "clf" | "reg"
    split: str = ""  # "train" | "val" | "test"
