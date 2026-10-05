"""The one correction step: demote high-null-rate columns in column_filter.

Any column currently in very_strong or strong with null_rate > max_null_rate
moves to moderate. Columns already at moderate/weak are left alone. A
mostly-null column may still carry signal in the rows where it has a value,
so it is demoted rather than dropped.
"""

from __future__ import annotations

from .schema import TieredConfig

_DEMOTE_FROM = ("very_strong", "strong")
_DEMOTE_TARGET = "moderate"


def demote_high_null_columns(
    config: TieredConfig, column_null_rates: dict[str, float], max_null_rate: float = 0.8
) -> TieredConfig:
    column_filter = {tier: list(cols) for tier, cols in config.column_filter.items()}

    for tier in _DEMOTE_FROM:
        keep: list[str] = []
        for col in column_filter[tier]:
            null_rate = column_null_rates.get(col)
            if null_rate is not None and null_rate > max_null_rate:
                column_filter[_DEMOTE_TARGET].append(col)
            else:
                keep.append(col)
        column_filter[tier] = keep

    return TieredConfig(
        column_filter=column_filter,
        p2f_edge={tier: list(cols) for tier, cols in config.p2f_edge.items()},
        f2p_edge={tier: list(cols) for tier, cols in config.f2p_edge.items()},
    )
