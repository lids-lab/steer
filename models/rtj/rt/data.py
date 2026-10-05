import json
import math
from functools import cache
from pathlib import Path

import ml_dtypes  # noqa: F401  # registers the bfloat16 numpy dtype for rustler
import numpy as np
import torch
from torch.utils.data import Dataset

from rt._rustler import Sampler

MAX_F2P_NBRS = 5  # keep in sync with fly.rs

SEM_TYPE_NUMBER = 0
SEM_TYPE_BOOLEAN = 3

SPLIT_NAMES = {"train": "Train", "val": "Val", "test": "Test"}


@cache
def _load_column_index(db_name: str, pre_dir: str) -> dict:
    with open(f"{pre_dir}/{db_name}/column_index.json") as f:
        return json.load(f)


def get_column_index(
    column_name: str, table_name: str, db_name: str, pre_dir: str
) -> int:
    column_index = _load_column_index(db_name, pre_dir)
    target = f"{column_name} of {table_name}"

    if target not in column_index:
        raise ValueError(
            f'Column "{target}" not found in {pre_dir}/{db_name}/column_index.json.'
        )

    return column_index[target]


def process_batch(tup, d_text):
    """Turn the sampler's flat arrays into batched tensors; booleans are folded into numbers."""
    out = dict(tup)
    seq_len = out.pop("seq_len")

    for k, v in out.items():
        if k in [
            "number_values",
            "datetime_values",
            "text_values",
            "col_name_values",
            "boolean_values",
        ]:
            out[k] = torch.from_numpy(v.view(np.float16)).view(torch.bfloat16)
        else:
            out[k] = torch.from_numpy(v)

    out["node_idxs"] = out["node_idxs"].view(-1, seq_len)
    out["sem_types"] = out["sem_types"].view(-1, seq_len)
    out["is_targets"] = out["is_targets"].view(-1, seq_len)
    out["is_task_nodes"] = out["is_task_nodes"].view(-1, seq_len)
    out["is_padding"] = out["is_padding"].view(-1, seq_len)
    out["table_name_idxs"] = out["table_name_idxs"].view(-1, seq_len)
    out["col_name_idxs"] = out["col_name_idxs"].view(-1, seq_len)
    out["class_value_idxs"] = out["class_value_idxs"].view(-1, seq_len)
    out["timestamps"] = out["timestamps"].view(-1, seq_len)

    out["f2p_nbr_idxs"] = out["f2p_nbr_idxs"].view(-1, seq_len, MAX_F2P_NBRS)
    out["number_values"] = out["number_values"].view(-1, seq_len, 1)
    out["datetime_values"] = out["datetime_values"].view(-1, seq_len, 1)
    out["boolean_values"] = out["boolean_values"].view(-1, seq_len, 1).bfloat16()
    out["text_values"] = out["text_values"].view(-1, seq_len, d_text)
    out["col_name_values"] = out["col_name_values"].view(-1, seq_len, d_text)

    bool_mask = out["sem_types"] == SEM_TYPE_BOOLEAN
    out["number_values"][bool_mask] = out["boolean_values"][bool_mask]
    out["boolean_values"][bool_mask] = 0
    out["sem_types"][bool_mask] = SEM_TYPE_NUMBER

    return out


class RustlerDataset:
    """One task's items, sampled by the Rust sampler (with STEER's filters when configured)."""

    def __init__(
        self,
        task,
        pre_dir: str,
        local_ctx_size,
        bfs_width,
        embedding_model,
        d_text,
        items_per_task,
        sampler_config_path: str | None = None,
    ):
        pre_dir = str(Path(pre_dir).expanduser())
        if not Path(pre_dir).exists():
            raise FileNotFoundError(f"preprocessed data directory not found: {pre_dir}")

        with open(f"{pre_dir}/{task.db_name}/table_info.json") as f:
            table_info = json.load(f)
        key = f"{task.table_name}:Db"
        if key not in table_info:
            key = f"{task.table_name}:{SPLIT_NAMES[task.split]}"
        info = table_info[key]

        target_idx = get_column_index(
            task.target_column, task.table_name, task.db_name, pre_dir
        )

        self.sampler = Sampler(
            dataset_tuples=[
                (task.db_name, task.table_name, info["node_idx_offset"], info["num_nodes"])
            ],
            local_ctx_size=local_ctx_size,
            bfs_width=bfs_width,
            embedding_model=embedding_model,
            pre_dir=pre_dir,
            d_text=d_text,
            shuffle_seed=0,
            context_seed=0,
            target_columns=[target_idx],
            items_per_task=items_per_task,
            sampler_config_path=sampler_config_path,
        )
        self.num_items = self.sampler.num_items
        self.d_text = d_text


class EvalDataset(Dataset):
    """Fixed-size batches of one task's items, in sampler order."""

    def __init__(self, rustler_dataset: RustlerDataset, eval_bs, eval_ctx_size):
        self.rustler_dataset = rustler_dataset
        self.eval_bs = eval_bs
        self.eval_ctx_size = eval_ctx_size

    def __len__(self):
        return math.ceil(self.rustler_dataset.num_items / self.eval_bs)

    def __getitem__(self, i):
        ds = self.rustler_dataset
        return process_batch(
            ds.sampler.batch_py(i, self.eval_bs, self.eval_ctx_size), ds.d_text
        )
