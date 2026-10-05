"""Names -> ids conversion for RT and rt-j's sampler config.

Both models' Rust samplers are keyed by integer table/column ids. RT ships
this mapping as text_map.json (table/column name -> id) directly; rt-j ships
the interned string list itself (text.json) -- a name's position in that
list IS its id.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import Profile, TieredConfig, ToggleMode

_TIER_ORDER = ("very_strong", "strong", "moderate", "weak")


def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def load_text_map(dataset_dir: Path) -> dict[str, int]:
    text_map_path = dataset_dir / "text_map.json"
    if text_map_path.exists():
        return _load_json(text_map_path)
    text_list_path = dataset_dir / "text.json"
    if text_list_path.exists():
        return {name: idx for idx, name in enumerate(_load_json(text_list_path))}
    raise FileNotFoundError(f"Neither text_map.json nor text.json found in {dataset_dir}")


def _resolve_table_id(table_name: str, text_map: dict[str, int], aliases: dict[str, str]) -> int:
    table_name = aliases.get(table_name, table_name)
    if table_name not in text_map:
        raise ValueError(f"Unknown table '{table_name}': not found in text_map")
    return text_map[table_name]


def _resolve_column_id(table_name: str, column_name: str, column_index: dict[str, int], aliases: dict[str, str]) -> int:
    table_name = aliases.get(table_name, table_name)
    key = f"{column_name} of {table_name}"
    if key not in column_index:
        raise ValueError(f"Unknown column '{column_name}' for table '{table_name}': key '{key}' not in column_index")
    return column_index[key]


def _convert_column_section(section: dict[str, list[str]], column_index: dict[str, int], aliases: dict[str, str]) -> dict[str, list[int]]:
    seen: dict[int, str] = {}
    out: dict[str, list[int]] = {t: [] for t in _TIER_ORDER}
    for tier in _TIER_ORDER:
        for entry in section.get(tier, []):
            if "." not in entry:
                raise ValueError(f"Column entry '{entry}' must be 'table_name.column_name'")
            table_name, column_name = entry.split(".", 1)
            col_id = _resolve_column_id(table_name, column_name, column_index, aliases)
            if col_id in seen:
                continue  # keep the higher tier already recorded
            seen[col_id] = tier
            out[tier].append(col_id)
    return out


def _convert_edge_section(section: dict[str, list[str]], text_map: dict[str, int], aliases: dict[str, str]) -> dict[str, list[str]]:
    # Edges are "src->dst" id strings; fk_col is dropped (only needed for
    # LLM disambiguation/validation, not at sampling time). Same table pair
    # in multiple tiers -> keep the highest tier.
    seen: dict[str, str] = {}
    out: dict[str, list[str]] = {t: [] for t in _TIER_ORDER}
    for tier in _TIER_ORDER:
        for edge in section.get(tier, []):
            src, rest = edge.split("->", 1)
            dst = rest.split(":", 1)[0]
            src_id = _resolve_table_id(src, text_map, aliases)
            dst_id = _resolve_table_id(dst, text_map, aliases)
            key = f"{src_id}->{dst_id}"
            if key in seen:
                continue
            seen[key] = tier
            out[tier].append(key)
    return out


def convert(
    config: TieredConfig,
    profile: Profile,
    toggle_mode: ToggleMode,
    dataset_dir: Path,
    task_table_aliases: dict[str, str] | None = None,
) -> dict:
    """`task_table_aliases` maps other spellings of the task table (a config
    may use another model's name for it -- rel-ratebeer's differ) to this
    model's own spelling, so such a config converts instead of failing."""
    aliases = task_table_aliases or {}
    text_map = load_text_map(dataset_dir)
    column_index = _load_json(dataset_dir / "column_index.json")

    return {
        "toggles": toggle_mode.toggles_dict(),
        "column_filter": _convert_column_section(config.column_filter, column_index, aliases),
        "p2f_edge_probabilistic_at_push": _convert_edge_section(config.p2f_edge, text_map, aliases),
        "f2p_edge_probabilistic_at_push": _convert_edge_section(config.f2p_edge, text_map, aliases),
        "keep_probabilities": {
            "column_filter": profile.column_filter,
            "p2f_edge": profile.p2f_edge,
            "f2p_edge": profile.f2p_edge,
        },
    }
