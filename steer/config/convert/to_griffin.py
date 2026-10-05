"""Names -> Griffin-native conversion.

Griffin's own graph metadata (metanode.yaml/metaadj.yaml) is already keyed
by human-readable strings, so this targets that format directly instead of
doing an RT-style names->ids pass -- see rt_griffin/relfm/griffin/hdataset_prob.py
for the consumer side.

Output (consumed by hdataset_prob.py's load_tier_config()):
    {
      "toggles": {"column_filter": bool, "p2f_edge_probabilistic_at_push": bool, "f2p_edge_probabilistic_at_push": bool},
      "edge_tier": {"<griffin edgetype string>": "very_strong"|"strong"|"moderate"|"weak", ...},
      "column_tier": {"<nodetype>\t<colname>": "very_strong"|..., ...},
      "keep_probabilities": {"column_filter": {...}, "p2f_edge": {...}, "f2p_edge": {...}},
    }
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import Profile, TieredConfig, ToggleMode

_TIER_ORDER = ("very_strong", "strong", "moderate", "weak")


def _load_yaml(path: Path):
    import yaml

    with path.open() as f:
        return yaml.safe_load(f)


class GriffinSchema:
    """Wraps one dataset's metanode.yaml/metaadj.yaml for name resolution."""

    def __init__(self, dataset: str, task_table: str, griffin_data_root: Path, task_table_aliases: frozenset[str] = frozenset()):
        ds_dir = griffin_data_root / dataset
        metanode = _load_yaml(ds_dir / "metanode.yaml")
        metaadj = _load_yaml(ds_dir / "metaadj.yaml")
        self.dataset = dataset
        # Griffin's own preprocessing namespaces the current task's task
        # table as "<task_table>__TASK__" in metanode.yaml/metaadj.yaml (a
        # dataset's metanode.yaml lists every task table it has, each
        # suffixed this way -- there's no plain, unsuffixed node for it).
        # The tiered config is written in RT-side names, which use the bare
        # task_table, so references to it need this remap; every other
        # table keeps its plain name. For rel-ratebeer, RT/rt-j's own
        # task_table spelling differs from Griffin's (e.g. "beer-rating-churn"
        # vs. "beer-churn") even though they name the same task -- the shared
        # tiered config is written in whichever spelling it was authored
        # against, so `task_table_aliases` lets other models' spellings also
        # resolve to this same task node.
        self._task_names = frozenset({task_table}) | task_table_aliases
        self._task_node = f"{task_table}__TASK__"

        # nodetype -> raw_colname -> [griffin internal colnames]
        self._col_lookup: dict[str, dict[str, list[str]]] = {}
        for nodetype, meta in metanode.items():
            lookup: dict[str, list[str]] = {}
            prefix = f"{nodetype}___"
            for griffin_col in meta.get("feat", []):
                # griffin_col is "<nodetype>___<col>", "<nodetype>___Griffin_text_<col>",
                # or "<nodetype>___TIMESTAMP(<col>)". Strip the known nodetype prefix
                # itself (not just the first "___") -- a task-table nodetype already
                # ends in "__TASK__", so naively splitting on the first "___" lands
                # inside that suffix instead of after the full nodetype.
                rest = griffin_col[len(prefix):] if griffin_col.startswith(prefix) else griffin_col.split("___", 1)[1]
                if rest.startswith("Griffin_text_"):
                    raw = rest[len("Griffin_text_"):]
                elif rest.startswith("TIMESTAMP(") and rest.endswith(")"):
                    raw = rest[len("TIMESTAMP("):-1]
                else:
                    raw = rest
                lookup.setdefault(raw, []).append(griffin_col)
            self._col_lookup[nodetype] = lookup
        self._edge_set: set[str] = set()
        for nodetype, adj in metaadj.items():
            self._edge_set.update(adj.get("out", []))
            self._edge_set.update(adj.get("in", []))

    def _node(self, table: str) -> str:
        return self._task_node if table in self._task_names else table

    def resolve_columns(self, table: str, raw_col: str) -> list[str]:
        return self._col_lookup.get(self._node(table), {}).get(raw_col, [])

    def resolve_f2p_edge(self, child: str, parent: str, fk_col: str) -> str | None:
        child, parent = self._node(child), self._node(parent)
        key = f"head of {child}:{child}-{fk_col}:{parent}"
        return key if key in self._edge_set else None

    def resolve_p2f_edge(self, parent: str, child: str, fk_col: str) -> str | None:
        parent, child = self._node(parent), self._node(child)
        key = f"tail of {child}:{child}-{fk_col}:{parent}"
        return key if key in self._edge_set else None


def _convert_column_filter(section: dict[str, list[str]], schema: GriffinSchema, warnings: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    seen: dict[str, str] = {}
    for tier in _TIER_ORDER:
        for entry in section.get(tier, []):
            table, _, raw_col = entry.partition(".")
            griffin_cols = schema.resolve_columns(table, raw_col)
            if not griffin_cols:
                warnings.append(f"column_filter: no Griffin column found for '{entry}' (dataset={schema.dataset})")
                continue
            for gc in griffin_cols:
                key = f"{schema._node(table)}\t{gc}"
                if key in seen:
                    continue
                seen[key] = tier
                out[key] = tier
    return out


def _convert_edge_section(section: dict[str, list[str]], schema: GriffinSchema, warnings: list[str], resolve, mech_name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    seen: dict[str, str] = {}
    for tier in _TIER_ORDER:
        for entry in section.get(tier, []):
            src_dst, _, fk_col = entry.partition(":")
            src, _, dst = src_dst.partition("->")
            key = resolve(src, dst, fk_col)
            if key is None:
                warnings.append(f"{mech_name}: no Griffin edge found for '{entry}' (dataset={schema.dataset})")
                continue
            if key in seen:
                continue
            seen[key] = tier
            out[key] = tier
    return out


def convert(
    config: TieredConfig, profile: Profile, toggle_mode: ToggleMode, dataset: str, task_table: str, griffin_data_root: Path,
    task_table_aliases: frozenset[str] = frozenset(),
) -> tuple[dict, list[str]]:
    """Returns (converted_config, warnings)."""
    schema = GriffinSchema(dataset, task_table, griffin_data_root, task_table_aliases)
    warnings: list[str] = []

    column_tier = _convert_column_filter(config.column_filter, schema, warnings)
    f2p_edge_tier = _convert_edge_section(config.f2p_edge, schema, warnings, schema.resolve_f2p_edge, "f2p_edge")
    p2f_edge_tier = _convert_edge_section(config.p2f_edge, schema, warnings, schema.resolve_p2f_edge, "p2f_edge")
    edge_tier = {**f2p_edge_tier, **p2f_edge_tier}

    converted = {
        "toggles": toggle_mode.toggles_dict(),
        "edge_tier": edge_tier,
        "column_tier": column_tier,
        "keep_probabilities": {
            "column_filter": profile.column_filter,
            "p2f_edge": profile.p2f_edge,
            "f2p_edge": profile.f2p_edge,
        },
    }
    return converted, warnings
