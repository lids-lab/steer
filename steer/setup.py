"""`steer setup`: download a model's checkpoints and preprocessed data.

    steer setup rt --task rel_f1_driver_dnf     # just what that task needs
    steer setup rt --all                        # everything for the model (large)

Files are laid out on Hugging Face exactly like the local
`models/<model>/checkpoints` and `models/<model>/data` directories, so a task's
files are selected by path. Most come from this project's own repositories
(see HF_NAMESPACE). RT-J's checkpoints and the preprocessed data for six of the
RelBench databases are hosted by the RT-J authors, so they are downloaded from
there, pinned to the exact revisions our results were produced with.
"""

from __future__ import annotations

from pathlib import Path

from . import tasks as task_registry

# The Hugging Face account that hosts the checkpoints and data.
HF_NAMESPACE = "abdallam1007"

# Hosted by the RT-J authors (stanford-star), pinned to the revisions used for the paper.
# The RT-J checkpoints at that revision are CC BY-NC-SA 4.0; their LICENSE file is downloaded with them.
UPSTREAM_RTJ_CHECKPOINTS = ("stanford-star/rt-j", "1819386c64c7d1371cd4f82d308989c24384f685")
UPSTREAM_RTJ_DATA = ("stanford-star/relbench-preprocessed", "1016626ddb30c027b92458bf866903850cc205e1")
UPSTREAM_RTJ_DATASETS = ("rel-amazon", "rel-avito", "rel-f1", "rel-hm", "rel-stack", "rel-trial")

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
    """This project's own repository. kind: "checkpoints" or "data"."""
    return f"{HF_NAMESPACE}/steer-{model}-{kind}"


# One download is (repo_id, repo_type, revision, allow_patterns); patterns None means everything.
Download = tuple[str, str, "str | None", "list[str] | None"]


def _downloads(model: str, kind: str, task: task_registry.TaskSpec | None) -> list[Download]:
    own_type = "model" if kind == "checkpoints" else "dataset"
    base = (_CKPT_BASES if kind == "checkpoints" else _DATA_BASES)[model]

    if task is None:  # everything for the model
        if model == task_registry.MODEL_RTJ and kind == "checkpoints":
            repo, rev = UPSTREAM_RTJ_CHECKPOINTS
            return [(repo, "model", rev, ["classification/*", "regression/*", "LICENSE"])]
        if model == task_registry.MODEL_RTJ and kind == "data":
            repo, rev = UPSTREAM_RTJ_DATA
            return [(repo, "dataset", rev, [f"{d}/*" for d in UPSTREAM_RTJ_DATASETS]), (hf_repo(model, kind), own_type, None, None)]
        return [(hf_repo(model, kind), own_type, None, None)]

    if kind == "checkpoints":
        ckpt = task.checkpoint_path(model).relative_to(base)
        # RT stores one checkpoint file per task; RT-J and Griffin store a directory.
        pattern = str(ckpt) if model == task_registry.MODEL_RT else f"{ckpt}/*"
        if model == task_registry.MODEL_RTJ:
            repo, rev = UPSTREAM_RTJ_CHECKPOINTS
            return [(repo, "model", rev, [pattern, "LICENSE"])]
        return [(hf_repo(model, kind), own_type, None, [pattern])]

    pattern = f"{task.data_path(model).relative_to(base)}/*"
    if model == task_registry.MODEL_RTJ and task.dataset in UPSTREAM_RTJ_DATASETS:
        repo, rev = UPSTREAM_RTJ_DATA
        return [(repo, "dataset", rev, [pattern])]
    return [(hf_repo(model, kind), own_type, None, [pattern])]


def setup(model: str, task_key: str | None) -> None:
    from huggingface_hub import snapshot_download

    task = None
    if task_key is None:
        print(f"Downloading everything for {model} (about {APPROX_ALL_GB[model]} GB).")
    else:
        task = task_registry.get_task_spec(task_key, model)

    for kind in ("checkpoints", "data"):
        base = (_CKPT_BASES if kind == "checkpoints" else _DATA_BASES)[model]
        for repo, repo_type, revision, patterns in _downloads(model, kind, task):
            pin = f"@{revision[:10]}" if revision else ""
            print(f"{repo}{pin} -> {base.relative_to(task_registry.STEER_ROOT)}")
            snapshot_download(repo_id=repo, repo_type=repo_type, revision=revision, allow_patterns=patterns, local_dir=Path(base))
