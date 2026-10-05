"""Sample size in tokens, for every model.

RT and RT-J build their context out of tokens, so their samplers report the
token count directly. Griffin feeds whole nodes to a graph neural network and
has no token count, so we estimate one: sampled nodes times the average
number of columns per table in the database
(`configs/dataset_avg_columns.json`). This is an approximation, used only to
put Griffin's sample size in the same unit as the other models.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from . import tasks as task_registry

AVG_COLUMNS_PATH = Path(__file__).resolve().parent.parent / "configs" / "dataset_avg_columns.json"


@lru_cache(maxsize=1)
def _avg_columns_per_table() -> dict[str, float]:
    return json.loads(AVG_COLUMNS_PATH.read_text())


def estimate_tokens(model: str, dataset: str, sample_size: dict) -> float:
    if model == task_registry.MODEL_GRIFFIN:
        return sample_size["avg_num_nodes"] * _avg_columns_per_table()[dataset]
    return sample_size["avg_filled_tokens"]
