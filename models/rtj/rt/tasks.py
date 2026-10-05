"""The Task representation rt-j's sampler and dataloader consume. STEER builds
these from its own task registry (steer/tasks.py)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    db_name: str
    table_name: str
    target_column: str
    task_type: str  # "clf" | "reg"
    split: str = ""  # "train" | "val" | "test"
    leakage_columns: tuple[str, ...] = ()
