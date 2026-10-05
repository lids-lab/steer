"""Names -> Griffin-native conversion.

Griffin's graph metadata (metaadj.yaml) is keyed by readable strings,
so names are resolved against it directly. The output is read by hdataset_prob.py:
    {
      "toggles": {...},
      "edge_tier": {"<griffin edgetype string>": "very_strong"|"strong"|"moderate"|"weak", ...},
      "keep_probabilities": {"p2f_edge": {...}, "f2p_edge": {...}},
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
    """Wraps one dataset's metaadj.yaml for edge name resolution."""

    def __init__(self, dataset: str, task_table: str, griffin_data_root: Path, task_table_aliases: frozenset[str] = frozenset()):
        ds_dir = griffin_data_root / dataset
        metaadj = _load_yaml(ds_dir / "metaadj.yaml")
        self.dataset = dataset
        # Griffin names the task table "<task_table>__TASK__"; configs use the bare name.
        # task_table_aliases covers other models' spellings of it (rel-ratebeer).
        self._task_names = frozenset({task_table}) | task_table_aliases
        self._task_node = f"{task_table}__TASK__"

        self._edge_set: set[str] = set()
        for nodetype, adj in metaadj.items():
            self._edge_set.update(adj.get("out", []))
            self._edge_set.update(adj.get("in", []))

    def _node(self, table: str) -> str:
        return self._task_node if table in self._task_names else table

    def resolve_f2p_edge(self, child: str, parent: str, fk_col: str) -> str | None:
        child, parent = self._node(child), self._node(parent)
        key = f"head of {child}:{child}-{fk_col}:{parent}"
        return key if key in self._edge_set else None

    def resolve_p2f_edge(self, parent: str, child: str, fk_col: str) -> str | None:
        parent, child = self._node(parent), self._node(child)
        key = f"tail of {child}:{child}-{fk_col}:{parent}"
        return key if key in self._edge_set else None


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

    f2p_edge_tier = _convert_edge_section(config.f2p_edge, schema, warnings, schema.resolve_f2p_edge, "f2p_edge")
    p2f_edge_tier = _convert_edge_section(config.p2f_edge, schema, warnings, schema.resolve_p2f_edge, "p2f_edge")
    edge_tier = {**f2p_edge_tier, **p2f_edge_tier}

    converted = {
        "toggles": toggle_mode.toggles_dict(),
        "edge_tier": edge_tier,
        "keep_probabilities": {
            "p2f_edge": profile.p2f_edge,
            "f2p_edge": profile.f2p_edge,
        },
    }
    return converted, warnings
