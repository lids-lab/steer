"""
Support module for hdataset.py's opt-in tiered probabilistic sampling
(the Griffin counterpart of RT's sampler mechanisms:
column_filter, p2f_edge_probabilistic_at_push,
f2p_edge_probabilistic_at_push -- see rustler/src/fly.rs).

hdataset.py calls load_tier_config(taskname, config_path) below with an
explicit config file path (STEER always knows exactly which config a given
run needs -- no env var, no directory lookup). When config_path is None, or
the file doesn't exist, load_tier_config() returns None and Graph/Node
behave exactly as before -- this module changes nothing by default.

Config file format (written by steer/config/convert/to_griffin.py from a
tiered config, a probability profile and the toggle mode):
    {
      "edge_tier": {"<griffin edgetype string>": "very_strong"|"strong"|"moderate"|"weak", ...},
      "column_tier": {"<nodetype>\t<colname>": "very_strong"|..., ...},
      "toggles": {"column_filter": bool, "p2f_edge_probabilistic_at_push": bool, "f2p_edge_probabilistic_at_push": bool},
      "keep_probabilities": {
        "p2f_edge": {"very_strong": 1.0, "strong": 0.6, "moderate": 0.0, "weak": 0.0},
        "f2p_edge": {"very_strong": 1.0, "strong": 1.0, "moderate": 0.0, "weak": 0.0}
      }
    }
("\t"-joined keys because JSON object keys must be strings; hdataset.py
splits them back into (nodetype, colname) tuples on load.)

"keep_probabilities" is optional and defaults to RT's own tuned constants
(DEFAULT_P2F_EDGE_KEEP/DEFAULT_F2P_EDGE_KEEP/DEFAULT_COLUMN_FILTER_KEEP
below) when absent or partial -- this is for manually sweeping the
tier->probability mapping itself, independent of which edges/columns the
LLM tiered into which bucket.

Matching fly.rs's actual design: each mechanism's toggle is an independent
gate checked at the call site, separate from whether its tier dict is
populated -- an empty tier dict with the toggle ON still means "everything
unassigned defaults to weak" (i.e. drop it), NOT "mechanism off". Toggles
default to True if the "toggles" key is absent, so a config file that only
has edge_tier/column_tier (no toggles block) behaves as fully enabled,
matching hand-written test configs used before toggles existed here.
"""
import json
import os

# Same constants as RT's rustler/src/fly.rs P2F_EDGE_PROBABILISTIC_KEEP_*,
# F2P_EDGE_PROBABILISTIC_KEEP_*, COLUMN_FILTER_KEEP_*. Used as defaults,
# overridable per-task via the config file's "keep_probabilities" block.
DEFAULT_P2F_EDGE_KEEP = {"very_strong": 1.0, "strong": 0.6, "moderate": 0.0, "weak": 0.0}
DEFAULT_F2P_EDGE_KEEP = {"very_strong": 1.0, "strong": 1.0, "moderate": 0.0, "weak": 0.0}
DEFAULT_COLUMN_FILTER_KEEP = {"very_strong": 1.0, "strong": 1.0, "moderate": 0.2, "weak": 0.0}

_cache: dict = {}


def load_tier_config(taskname, config_path):
    """Return {"edge_tier": {...}, "column_tier": {(nodetype, colname): tier}}
    for `taskname`, or None if config_path is None or doesn't exist -- in
    both cases callers must fall back to untiered behavior."""
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
        "column_tier": {
            tuple(k.split("\t")): v for k, v in raw.get("column_tier", {}).items()
        },
        "toggles": {
            "column_filter": raw_toggles.get("column_filter", True),
            "p2f_edge_probabilistic_at_push": raw_toggles.get("p2f_edge_probabilistic_at_push", True),
            "f2p_edge_probabilistic_at_push": raw_toggles.get("f2p_edge_probabilistic_at_push", True),
        },
        "p2f_edge_keep": {**DEFAULT_P2F_EDGE_KEEP, **raw_keep_probs.get("p2f_edge", {})},
        "f2p_edge_keep": {**DEFAULT_F2P_EDGE_KEEP, **raw_keep_probs.get("f2p_edge", {})},
        "column_filter_keep": {**DEFAULT_COLUMN_FILTER_KEEP, **raw_keep_probs.get("column_filter", {})},
    }
    _cache[cache_key] = tier_config
    return tier_config
