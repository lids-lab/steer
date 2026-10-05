"""Adapter for the RT model: builds the subprocess command for
models/rt/steer_eval.py (RT's own pixi env) and reads back its JSON result.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "rt"
PYTHON = MODEL_DIR / ".pixi" / "envs" / "default" / "bin" / "python"


def run(task, spec, sampler_config_path: Path) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        json_out = Path(f.name)

    command = [
        str(PYTHON), str(MODEL_DIR / "steer_eval.py"),
        "--dataset", task.dataset,
        "--task-table", task.task_table,
        "--target-column", task.target_column,
        "--task-type", task.task_type,
        "--checkpoint", str(task.checkpoint_path("rt")),
        "--seq-len", str(spec.seq_len),
        "--max-eval-steps", str(spec.max_eval_steps),
        "--sampler-config-path", str(sampler_config_path),
        "--device", spec.device,
        "--split", spec.split,
        "--reg-metric", spec.reg_metric,
        "--json-out", str(json_out),
    ]
    subprocess.run(command, cwd=str(MODEL_DIR), check=True)
    result = json.loads(json_out.read_text())
    json_out.unlink(missing_ok=True)
    return result
