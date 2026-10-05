"""Standalone inference for RT: load a checkpoint and evaluate one task's
split. No training loop, logging or distributed code."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from torch.utils.data import DataLoader

from data import RelationalDataset
from model import RelationalTransformer


def load_checkpoint(checkpoint_path: str, num_blocks: int, d_model: int, d_text: int, num_heads: int, d_ff: int, device: str) -> RelationalTransformer:
    net = RelationalTransformer(num_blocks=num_blocks, d_model=d_model, d_text=d_text, num_heads=num_heads, d_ff=d_ff)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    net.load_state_dict(state_dict)
    net = net.to(device).to(torch.bfloat16)
    net.eval()
    return net


def evaluate_checkpoint(
    net: RelationalTransformer,
    eval_tasks: list[tuple[str, str, str, str, list[str], str]],
    batch_size: int,
    seq_len: int,
    max_bfs_width: int,
    embedding_model: str,
    d_text: int,
    max_eval_steps: int,
    sampler_config_path: str | None = None,
    num_workers: int = 2,
    device: str = "cuda",
    reg_metric: str = "r2",
) -> dict[tuple[str, str, str], dict]:
    """eval_tasks: list of (db_name, table_name, target_column, split, columns_to_drop, task_type).

    Returns {(db_name, table_name, split): {metric_name, metric_value, sample_size: {avg_filled_tokens}, ...}}.
    """
    results: dict[tuple[str, str, str], dict] = {}

    for db_name, table_name, target_column, split, columns_to_drop, task_type in eval_tasks:
        eval_dataset = RelationalDataset(
            tasks=[(db_name, table_name, target_column, split, columns_to_drop)],
            batch_size=batch_size,
            seq_len=seq_len,
            rank=0,
            world_size=1,
            max_bfs_width=max_bfs_width,
            embedding_model=embedding_model,
            d_text=d_text,
            seed=0,
            sampler_config_path=sampler_config_path,
        )
        eval_dataset.sampler.shuffle_py(0)
        eval_loader = DataLoader(eval_dataset, batch_size=None, num_workers=num_workers, persistent_workers=False, pin_memory=True, in_order=True)

        preds, labels, losses = [], [], []
        filled_tokens, seq_utilizations = [], []

        with torch.inference_mode():
            batch_idx = 0
            for batch in eval_loader:
                true_batch_size = batch.pop("true_batch_size")
                for k in batch:
                    batch[k] = batch[k].to(device, non_blocking=True)

                batch["masks"][true_batch_size:, :] = False
                batch["is_targets"][true_batch_size:, :] = False
                batch["is_padding"][true_batch_size:, :] = True

                non_padding = (~batch["is_padding"][:true_batch_size]).sum(dim=1)
                utilization = non_padding.float() / batch["is_padding"].shape[1]
                filled_tokens.extend(non_padding.detach().cpu().tolist())
                seq_utilizations.extend(utilization.detach().cpu().tolist())

                loss, yhat_dict = net(batch)

                if task_type == "clf":
                    yhat = yhat_dict["boolean"][batch["is_targets"]]
                    y = batch["boolean_values"][batch["is_targets"]].flatten()
                else:
                    yhat = yhat_dict["number"][batch["is_targets"]]
                    y = batch["number_values"][batch["is_targets"]].flatten()

                losses.append(loss.item())
                preds.append(yhat.flatten())
                labels.append(y)

                batch_idx += 1
                if max_eval_steps > -1 and batch_idx >= max_eval_steps:
                    break

        preds = torch.cat(preds, dim=0).float().cpu().numpy()
        labels = torch.cat(labels, dim=0).float().cpu().numpy()

        if task_type == "reg":
            if reg_metric == "mae":
                metric_name, metric_value = "mae", float(mean_absolute_error(labels, preds))
            else:
                metric_name, metric_value = "r2", float(r2_score(labels, preds))
        else:
            metric_name = "auc"
            metric_value = float(roc_auc_score([int(x > 0) for x in labels], preds))

        results[(db_name, table_name, split)] = {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "sample_size": {"avg_filled_tokens": float(np.mean(filled_tokens))},
            "avg_seq_utilization": float(np.mean(seq_utilizations)),
        }

    return results
