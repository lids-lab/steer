"""The 24 RelBench tasks STEER is evaluated on, and where each model's
checkpoint for a task lives.

The models themselves carry no task list: `steer run` builds everything a
model needs from the TaskSpec returned here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODEL_RT = "rt"
MODEL_RTJ = "rtj"
MODEL_GRIFFIN = "griffin"
ALL_MODELS = (MODEL_RT, MODEL_RTJ, MODEL_GRIFFIN)

STEER_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = STEER_ROOT / "models"

# Where `steer setup` puts each model's checkpoints (see README).
RT_CKPT_BASE = MODELS_DIR / "rt" / "checkpoints"
RTJ_CKPT_BASE = MODELS_DIR / "rtj" / "checkpoints"
GRIFFIN_CKPT_BASE = MODELS_DIR / "griffin" / "checkpoints"

# Where `steer setup` puts each model's preprocessed data (one directory per dataset).
RT_DATA_BASE = MODELS_DIR / "rt" / "data" / "pre"
RTJ_DATA_BASE = MODELS_DIR / "rtj" / "data" / "pre"
GRIFFIN_DATA_BASE = MODELS_DIR / "griffin" / "data" / "step3"

# RT-J is zero-shot: one classification checkpoint and one regression
# checkpoint cover every task (RT and Griffin use per-database / per-task
# checkpoints).
RTJ_CHECKPOINTS = {
    "clf": RTJ_CKPT_BASE / "classification",
    "reg": RTJ_CKPT_BASE / "regression",
}


@dataclass(frozen=True)
class TaskSpec:
    task_key: str
    dataset: str
    task_table: str
    target_column: str
    task_type: str  # "clf" | "reg"

    def checkpoint_path(self, model: str) -> Path:
        if model == MODEL_RTJ:
            return RTJ_CHECKPOINTS[self.task_type]
        elif model == MODEL_RT:
            return RT_CKPT_BASE / f"lodo_{self.dataset}" / f"{self.dataset}_{self.task_table}_best.pt"
        elif model == MODEL_GRIFFIN:
            return GRIFFIN_CKPT_BASE / f"checkpoint-{self.dataset}-{self.task_table}-best"
        else:
            raise ValueError(f"Unknown model '{model}'. Expected one of: {ALL_MODELS}")

    def data_path(self, model: str) -> Path:
        base = {MODEL_RT: RT_DATA_BASE, MODEL_RTJ: RTJ_DATA_BASE, MODEL_GRIFFIN: GRIFFIN_DATA_BASE}.get(model)
        if base is None:
            raise ValueError(f"Unknown model '{model}'. Expected one of: {ALL_MODELS}")
        return base / self.dataset


_CLASSIFICATION_TASK_KEYS = frozenset({
    "rel_amazon_user_churn", "rel_amazon_item_churn",
    "rel_avito_user_visits", "rel_avito_user_clicks",
    "rel_f1_driver_dnf", "rel_f1_driver_top3",
    "rel_hm_user_churn",
    "rel_stack_user_badge", "rel_stack_user_engagement",
    "rel_trial_study_outcome",
    "rel_ratebeer_beer_rating_churn", "rel_ratebeer_user_rating_churn",
    "rel_ratebeer_brewer_dormant", "rel_arxiv_paper_citation",
})


def _spec(task_key: str, dataset: str, task_table: str, target_column: str) -> TaskSpec:
    task_type = "clf" if task_key in _CLASSIFICATION_TASK_KEYS else "reg"
    return TaskSpec(task_key, dataset, task_table, target_column, task_type)


# Tasks whose names are identical for every model.
_COMMON_TASK_SPECS: tuple[TaskSpec, ...] = (
    _spec("rel_amazon_user_churn", "rel-amazon", "user-churn", "churn"),
    _spec("rel_amazon_item_churn", "rel-amazon", "item-churn", "churn"),
    _spec("rel_amazon_user_ltv", "rel-amazon", "user-ltv", "ltv"),
    _spec("rel_amazon_item_ltv", "rel-amazon", "item-ltv", "ltv"),
    _spec("rel_hm_user_churn", "rel-hm", "user-churn", "churn"),
    _spec("rel_hm_item_sales", "rel-hm", "item-sales", "sales"),
    _spec("rel_stack_user_badge", "rel-stack", "user-badge", "WillGetBadge"),
    _spec("rel_stack_user_engagement", "rel-stack", "user-engagement", "contribution"),
    _spec("rel_stack_post_votes", "rel-stack", "post-votes", "popularity"),
    _spec("rel_avito_user_visits", "rel-avito", "user-visits", "num_click"),
    _spec("rel_avito_user_clicks", "rel-avito", "user-clicks", "num_click"),
    _spec("rel_avito_ad_ctr", "rel-avito", "ad-ctr", "num_click"),
    _spec("rel_trial_study_outcome", "rel-trial", "study-outcome", "outcome"),
    _spec("rel_trial_site_success", "rel-trial", "site-success", "success_rate"),
    _spec("rel_trial_study_adverse", "rel-trial", "study-adverse", "num_of_adverse_events"),
    _spec("rel_f1_driver_dnf", "rel-f1", "driver-dnf", "did_not_finish"),
    _spec("rel_f1_driver_top3", "rel-f1", "driver-top3", "qualifying"),
    _spec("rel_f1_driver_position", "rel-f1", "driver-position", "position"),
    _spec("rel_arxiv_paper_citation", "rel-arxiv", "paper-citation", "cited"),
    _spec("rel_arxiv_author_publication", "rel-arxiv", "author-publication", "publication_count"),
    _spec("rel_ratebeer_brewer_dormant", "rel-ratebeer", "brewer-dormant", "dormant"),
)

# rel-ratebeer's task-table names differ by model: RT and RT-J were
# preprocessed with the "-rating-" spellings, while Griffin's data and the
# current RelBench use the short ones.
_RATEBEER_RT_SPECS: tuple[TaskSpec, ...] = (
    _spec("rel_ratebeer_beer_rating_churn", "rel-ratebeer", "beer-rating-churn", "rating_churn"),
    _spec("rel_ratebeer_user_rating_churn", "rel-ratebeer", "user-rating-churn", "user_churn"),
    _spec("rel_ratebeer_user_rating_count", "rel-ratebeer", "user-rating-count", "num_ratings"),
)
_RATEBEER_GRIFFIN_SPECS: tuple[TaskSpec, ...] = (
    _spec("rel_ratebeer_beer_rating_churn", "rel-ratebeer", "beer-churn", "rating_churn"),
    _spec("rel_ratebeer_user_rating_churn", "rel-ratebeer", "user-churn", "user_churn"),
    _spec("rel_ratebeer_user_rating_count", "rel-ratebeer", "user-count", "num_ratings"),
)

# Per-model task lists (24 tasks each).
RT_TASK_SPECS: tuple[TaskSpec, ...] = _COMMON_TASK_SPECS + _RATEBEER_RT_SPECS
RTJ_TASK_SPECS: tuple[TaskSpec, ...] = _COMMON_TASK_SPECS + _RATEBEER_RT_SPECS
GRIFFIN_TASK_SPECS: tuple[TaskSpec, ...] = _COMMON_TASK_SPECS + _RATEBEER_GRIFFIN_SPECS

# Model-agnostic view, for fields that don't vary by model (dataset,
# target_column, task_type). Anything that needs a model-correct task_table
# must call get_task_spec(task_key, model).
ALL_TASK_SPECS: tuple[TaskSpec, ...] = GRIFFIN_TASK_SPECS
TASKS_BY_KEY: dict[str, TaskSpec] = {spec.task_key: spec for spec in ALL_TASK_SPECS}


def get_task_specs_for_model(model: str) -> tuple[TaskSpec, ...]:
    if model == MODEL_RT:
        return RT_TASK_SPECS
    elif model == MODEL_RTJ:
        return RTJ_TASK_SPECS
    elif model == MODEL_GRIFFIN:
        return GRIFFIN_TASK_SPECS
    else:
        raise ValueError(f"Unknown model '{model}'. Expected one of: {ALL_MODELS}")


def task_keys() -> list[str]:
    return sorted(TASKS_BY_KEY)


def get_task_spec(task_key: str, model: str | None = None) -> TaskSpec:
    """Look up a TaskSpec. Pass `model` whenever the result feeds a checkpoint
    path or data lookup -- task_table differs by model for the 3 ratebeer
    tasks above even though task_key is shared."""
    if model is not None:
        by_key = {spec.task_key: spec for spec in get_task_specs_for_model(model)}
        try:
            return by_key[task_key]
        except KeyError as exc:
            known = ", ".join(sorted(by_key))
            raise KeyError(f"Unknown task key '{task_key}' for model '{model}'. Known task keys: {known}") from exc
    try:
        return TASKS_BY_KEY[task_key]
    except KeyError as exc:
        known = ", ".join(task_keys())
        raise KeyError(f"Unknown task key '{task_key}'. Known task keys: {known}") from exc
