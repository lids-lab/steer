"""The single "run one task on one model" primitive used by the CLI."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import tasks as task_registry
from .config.schema import Profile, TieredConfig, ToggleMode
from .sample_size import estimate_tokens

STEER_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = STEER_ROOT / "configs"

# Defaults: the paper's universal probability profile and the Claude-sourced
# configs. Any other config/profile is just another path.
DEFAULT_PROFILE_PATH = CONFIGS_DIR / "profiles" / "universal.json"
DEFAULT_CONFIG_DIR = CONFIGS_DIR / "tasks" / "claude"

# The paper's evaluation budget is 1280 items per task. RT counts it in
# batches (40 steps); RT-J and Griffin count items.
EVAL_STEPS = {
    task_registry.MODEL_RT: 40,
    task_registry.MODEL_RTJ: 1280,
    task_registry.MODEL_GRIFFIN: 1280,
}

COLUMN_FILTER = "column_filter"


@dataclass
class RunSpec:
    model: str
    task_key: str
    toggles: str  # "baseline" or a '+'-joined combination of toggle keys (see config/schema.py)
    config_path: str | None = None  # default: configs/tasks/claude/<task_key>.json
    profile_path: str | None = None  # default: configs/profiles/universal.json
    device: str = "cuda"
    # Fixed paper settings -- not exposed on the command line.
    seq_len: int = 1024  # RT / RT-J context length (Griffin uses its own fanout 20, 2 hops)
    split: str = "test"
    reg_metric: str = "r2"
    max_eval_steps: int | None = None  # None -> EVAL_STEPS[model]


@dataclass
class RunResult:
    task_key: str
    model: str
    toggle_mode: str
    metric_name: str
    metric_value: float
    tokens: float  # RT / RT-J: measured; Griffin: estimated from sampled nodes (see sample_size.py)
    sample_size: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def load_config(spec: RunSpec, task: task_registry.TaskSpec) -> TieredConfig:
    path = Path(spec.config_path) if spec.config_path else DEFAULT_CONFIG_DIR / f"{task.task_key}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No config for task '{task.task_key}' at {path}. "
            f"Generate one with: steer-generate-config --task {task.task_key} --provider <anthropic|openai>"
        )
    return TieredConfig.from_dict(json.loads(path.read_text()))


def load_profile(profile_path: str | None) -> Profile:
    path = Path(profile_path) if profile_path else DEFAULT_PROFILE_PATH
    if not path.exists():
        raise FileNotFoundError(f"No profile at {path}")
    return Profile.from_dict(json.loads(path.read_text()))


def _check_ready(spec: RunSpec, task: task_registry.TaskSpec) -> None:
    """Fail early, with the fix, if the model environment or its downloads are missing."""
    model_dir = STEER_ROOT / "models" / spec.model
    if spec.model != task_registry.MODEL_GRIFFIN:
        sampler_hint = " and `pixi run build-sampler`"
    else:
        sampler_hint = ""
    if not (model_dir / ".pixi" / "envs" / "default" / "bin" / "python").exists():
        raise FileNotFoundError(
            f"The {spec.model} environment is not built. Run `pixi install`{sampler_hint} in models/{spec.model} "
            "(see models/" + spec.model + "/README.md)."
        )
    missing = [p for p in (task.checkpoint_path(spec.model), task.data_path(spec.model)) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(str(p.relative_to(STEER_ROOT)) for p in missing)}. "
            f"Download them with: steer setup {spec.model} --task {spec.task_key}"
        )


def _check_toggles(spec: RunSpec, toggle_mode: ToggleMode, config: TieredConfig) -> None:
    if not toggle_mode.is_active(COLUMN_FILTER):
        return
    if spec.model == task_registry.MODEL_GRIFFIN:
        raise ValueError("The column filter applies to RT and RT-J only (Griffin represents a row as a single node).")
    if not any(config.column_filter.values()):
        raise ValueError(
            "This config has no column-filter section. Regenerate it with: "
            f"steer-generate-config --task {spec.task_key} --provider <anthropic|openai> --column-filter"
        )


def run_task(spec: RunSpec) -> RunResult:
    task = task_registry.get_task_spec(spec.task_key, spec.model)
    config = load_config(spec, task)
    profile = load_profile(spec.profile_path)
    toggle_mode = ToggleMode.parse(spec.toggles)
    _check_toggles(spec, toggle_mode, config)
    _check_ready(spec, task)

    if spec.max_eval_steps is None:
        spec = replace(spec, max_eval_steps=EVAL_STEPS[spec.model])

    with tempfile.TemporaryDirectory(prefix="steer_sampler_config_") as tmp_dir:
        sampler_config_path = _write_runtime_config(spec, task, config, profile, toggle_mode, Path(tmp_dir))
        adapter = _get_adapter(spec.model)
        raw_result = adapter.run(task, spec, sampler_config_path)

    sample_size = raw_result.get("sample_size", {})
    return RunResult(
        task_key=task.task_key,
        model=spec.model,
        toggle_mode=spec.toggles,
        metric_name=raw_result["metric_name"],
        metric_value=raw_result["metric_value"],
        tokens=estimate_tokens(spec.model, task.dataset, sample_size),
        sample_size=sample_size,
        extra={k: v for k, v in raw_result.items() if k not in {"metric_name", "metric_value", "sample_size"}},
    )


def _task_table_aliases(task_key: str) -> frozenset[str]:
    """Every model's spelling of this task's table (they differ for 3 ratebeer tasks)."""
    return frozenset(
        task_registry.get_task_spec(task_key, m).task_table
        for m in task_registry.ALL_MODELS
        if task_key in {t.task_key for t in task_registry.get_task_specs_for_model(m)}
    )


def _write_runtime_config(spec: RunSpec, task, config: TieredConfig, profile: Profile, toggle_mode: ToggleMode, tmp_dir: Path) -> Path:
    aliases = _task_table_aliases(spec.task_key)
    if spec.model == task_registry.MODEL_GRIFFIN:
        from .config.convert.to_griffin import convert
        griffin_data_root = STEER_ROOT / "models" / "griffin" / "data" / "step3"
        runtime_config, warnings = convert(
            config, profile, toggle_mode, task.dataset, task.task_table, griffin_data_root, aliases
        )
        for w in warnings:
            print(f"Warning: {w}")
    else:
        from .config.convert.to_rt_ids import convert
        model_dir = "rt" if spec.model == task_registry.MODEL_RT else "rtj"
        dataset_dir = STEER_ROOT / "models" / model_dir / "data" / "pre" / task.dataset
        own_alias_map = {alias: task.task_table for alias in aliases if alias != task.task_table}
        runtime_config = convert(config, profile, toggle_mode, dataset_dir, own_alias_map)

    out_path = tmp_dir / f"{task.task_key}.json"
    out_path.write_text(json.dumps(runtime_config, indent=2, sort_keys=True))
    return out_path


def _get_adapter(model: str):
    if model == task_registry.MODEL_RT:
        from .adapters import rt
        return rt
    elif model == task_registry.MODEL_RTJ:
        from .adapters import rtj
        return rtj
    elif model == task_registry.MODEL_GRIFFIN:
        from .adapters import griffin
        return griffin
    raise ValueError(f"Unknown model '{model}'. Expected one of: {task_registry.ALL_MODELS}")
