"""Load a released RT-J checkpoint: a directory with `model.safetensors` and `config.json` (or a weights file)."""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_FILE = "config.json"
MODEL_FILE = "model.safetensors"
MODEL_DIM_KEYS = ("num_blocks", "d_model", "d_text", "num_heads", "d_ff")


def load_model(path):
    """Load a flat tensor `state_dict` from a `.safetensors` file."""
    from safetensors.torch import load_file

    return load_file(str(path))


def resolve_checkpoint(spec) -> tuple[dict, Path]:
    """Return `(config, weights_path)` for a local weights file or checkpoint directory."""
    p = Path(spec).expanduser()
    if p.is_file():
        cfg_path = p.with_name(CONFIG_FILE)
        config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        return config, p
    if not p.is_dir():
        raise FileNotFoundError(f"checkpoint {spec!r} is not a local file or directory")
    config = json.loads((p / CONFIG_FILE).read_text())
    return config, p / config.get("checkpoint_file", MODEL_FILE)


def load_rt_model(spec, *, device: str = "cpu"):
    """Build the RelationalTransformer from a checkpoint and return `(model, config)`."""
    from rt.model import RelationalTransformer

    model = RelationalTransformer.from_pretrained(spec, device=device)
    return model, model.config
