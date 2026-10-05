"""`steer setup`: download a model's checkpoints and preprocessed data.

    steer setup rt --task rel_f1_driver_dnf     # just what that task needs
    steer setup rt --all                        # everything for the model (large)

Everything is hosted on Hugging Face (see HF_NAMESPACE), one checkpoints repo and
one data repo per model, laid out exactly like the local `models/<model>/checkpoints` and
`models/<model>/data` directories, so a task's files are selected by path.
"""

from __future__ import annotations

from pathlib import Path

from . import tasks as task_registry

# The Hugging Face account that hosts the checkpoints and data.
HF_NAMESPACE = "abdallam1007"

# Approximate download size of everything for a model, in GB (checkpoints + data).
APPROX_ALL_GB = {task_registry.MODEL_RT: 216, task_registry.MODEL_RTJ: 129, task_registry.MODEL_GRIFFIN: 158}

_CKPT_BASES = {
    task_registry.MODEL_RT: task_registry.RT_CKPT_BASE,
    task_registry.MODEL_RTJ: task_registry.RTJ_CKPT_BASE,
    task_registry.MODEL_GRIFFIN: task_registry.GRIFFIN_CKPT_BASE,
}
_DATA_BASES = {
    task_registry.MODEL_RT: task_registry.RT_DATA_BASE,
    task_registry.MODEL_RTJ: task_registry.RTJ_DATA_BASE,
    task_registry.MODEL_GRIFFIN: task_registry.GRIFFIN_DATA_BASE,
}


def hf_repo(model: str, kind: str) -> str:
    """kind: "checkpoints" or "data"."""
    return f"{HF_NAMESPACE}/steer-{model}-{kind}"


def _task_patterns(model: str, task: task_registry.TaskSpec) -> tuple[list[str], list[str]]:
    """File patterns, relative to each repo's root, covering one task."""
    ckpt = task.checkpoint_path(model).relative_to(_CKPT_BASES[model])
    # RT stores one checkpoint file per task; RT-J and Griffin store a directory.
    ckpt_pattern = str(ckpt) if model == task_registry.MODEL_RT else f"{ckpt}/*"
    data_pattern = f"{task.data_path(model).relative_to(_DATA_BASES[model])}/*"
    return [ckpt_pattern], [data_pattern]


def setup(model: str, task_key: str | None) -> None:
    from huggingface_hub import snapshot_download

    if task_key is None:
        ckpt_patterns = data_patterns = None
        print(f"Downloading everything for {model} (about {APPROX_ALL_GB[model]} GB).")
    else:
        task = task_registry.get_task_spec(task_key, model)
        ckpt_patterns, data_patterns = _task_patterns(model, task)

    for kind, repo_type, patterns, base in (
        ("checkpoints", "model", ckpt_patterns, _CKPT_BASES[model]),
        ("data", "dataset", data_patterns, _DATA_BASES[model]),
    ):
        repo = hf_repo(model, kind)
        print(f"{repo} -> {base.relative_to(task_registry.STEER_ROOT)}")
        snapshot_download(repo_id=repo, repo_type=repo_type, allow_patterns=patterns, local_dir=Path(base))
