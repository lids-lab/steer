"""Metric and evaluator construction for steer_eval.py."""

from __future__ import annotations

CTX_SIZE = 8192  # outer context buffer (cells)
BFS_WIDTH = 32
TOKENS_PER_BATCH = 2**18


def metric_for(task_type: str, labels, preds, reg_metric: str = "r2") -> tuple[str, float]:
    """Metric on the normalized target scale: AUROC for clf, R2 or MAE for reg."""
    import sklearn.metrics as M

    if task_type == "reg":
        if reg_metric == "r2":
            return "r2", float(M.r2_score(labels, preds))
        return "mae", float(M.mean_absolute_error(labels, preds))
    return "roc_auc", float(M.roc_auc_score((labels > 0).astype(int), preds))


def build_evaluator(tasks, pre_dir, *, embedding_model, d_text, device, local_ctx_size,
                    items_per_task, sampler_config_path=None):
    from rt.evaluator import Evaluator

    return Evaluator(
        tasks=tasks, pre_dir=pre_dir, eval_bs=max(1, TOKENS_PER_BATCH // CTX_SIZE),
        ctx_size=CTX_SIZE, items_per_task=items_per_task, local_ctx_size=local_ctx_size,
        bfs_width=BFS_WIDTH, embedding_model=embedding_model, d_text=d_text, device=device,
        sampler_config_path=sampler_config_path,
    )
