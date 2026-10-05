"""Loads the per-task sampler config that hdataset.py uses for STEER's edge filter.
Config JSON: "edge_tier" (edge type -> tier), "toggles" (p2f/f2p_edge_probabilistic_at_push),
and optional "keep_probabilities" for "p2f_edge" / "f2p_edge" (tier -> keep probability).
With no config path (or a missing file), load_tier_config returns None and sampling is unchanged.
"""
import json
import os

DEFAULT_P2F_EDGE_KEEP = {"very_strong": 1.0, "strong": 0.6, "moderate": 0.0, "weak": 0.0}
DEFAULT_F2P_EDGE_KEEP = {"very_strong": 1.0, "strong": 1.0, "moderate": 0.0, "weak": 0.0}

_cache: dict = {}


def load_tier_config(taskname, config_path):
    """Return the parsed config for `taskname`, or None if there is no config to apply."""
    if taskname is None or config_path is None:
        return None
    cache_key = (taskname, config_path)
    if cache_key in _cache:
        return _cache[cache_key]

    if not os.path.exists(config_path):
        _cache[cache_key] = None
        return None

    with open(config_path) as f:
        raw = json.load(f)
    raw_toggles = raw.get("toggles", {})
    raw_keep_probs = raw.get("keep_probabilities", {})
    tier_config = {
        "edge_tier": raw.get("edge_tier", {}),
        "toggles": {
            "p2f_edge_probabilistic_at_push": raw_toggles.get("p2f_edge_probabilistic_at_push", True),
            "f2p_edge_probabilistic_at_push": raw_toggles.get("f2p_edge_probabilistic_at_push", True),
        },
        "p2f_edge_keep": {**DEFAULT_P2F_EDGE_KEEP, **raw_keep_probs.get("p2f_edge", {})},
        "f2p_edge_keep": {**DEFAULT_F2P_EDGE_KEEP, **raw_keep_probs.get("f2p_edge", {})},
    }
    _cache[cache_key] = tier_config
    return tier_config
