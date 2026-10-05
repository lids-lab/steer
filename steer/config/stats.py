"""Per-column null-rate stats, used by correct.py's demotion rule.

Only null_rate is needed for the current approach (mode_concentration,
cardinality_ratio, and FK fanout/parent-coverage were only ever consumed by
corrections that aren't part of the approach anymore).
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

RELBENCH_CACHE = Path(os.environ.get("RELBENCH_CACHE", Path.home() / ".cache" / "relbench"))


def compute_null_rate(series: pl.Series) -> float:
    if series.len() == 0:
        return 1.0
    return round(series.null_count() / series.len(), 4)


def compute_column_null_rates(dataset: str) -> dict[str, float]:
    """null_rate for every "table.column" in a dataset's cached parquet tables."""
    db_dir = RELBENCH_CACHE / dataset / "db"
    stats: dict[str, float] = {}
    for path in sorted(db_dir.glob("*.parquet")):
        table_name = path.stem
        df = pl.read_parquet(path)
        for col_name in df.columns:
            stats[f"{table_name}.{col_name}"] = compute_null_rate(df[col_name])
    return stats
