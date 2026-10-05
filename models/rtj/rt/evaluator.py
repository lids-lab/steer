"""Runs a model over one or more tasks' items and collects labels and predictions."""

from itertools import islice

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rt.data import EvalDataset, RustlerDataset


class Evaluator:
    def __init__(
        self,
        *,
        tasks,
        pre_dir,
        eval_bs,
        ctx_size,
        items_per_task,
        local_ctx_size,
        bfs_width,
        embedding_model,
        d_text,
        device,
        sampler_config_path=None,
    ):
        self.eval_bs = eval_bs
        self.items_per_task = items_per_task
        self.device = device

        self.loaders = {}
        for task in tasks:
            rustler_dataset = RustlerDataset(
                task=task,
                pre_dir=pre_dir,
                local_ctx_size=local_ctx_size,
                bfs_width=bfs_width,
                embedding_model=embedding_model,
                d_text=d_text,
                items_per_task=items_per_task,
                sampler_config_path=sampler_config_path,
            )
            # no worker processes: they race on shared memory when evaluations share a node
            self.loaders[task] = DataLoader(
                EvalDataset(rustler_dataset, eval_bs, ctx_size),
                batch_size=None,
                num_workers=0,
                pin_memory=True,
            )

    def evaluate_raw(self, net):
        """Yield `(task, labels, preds, filled_tokens)` per task, one entry per real item.

        `filled_tokens` is the number of non-padding cells in each item's context.
        """
        net.eval()

        with torch.inference_mode():
            for task, loader in self.loaders.items():
                n_batches = min(
                    len(loader.dataset), max(1, self.items_per_task // self.eval_bs)
                )

                labels, preds, filled_tokens, batch_masks = [], [], [], []
                for batch in tqdm(
                    islice(loader, n_batches),
                    total=n_batches,
                    desc=f"{task.db_name}/{task.table_name}/{task.split}",
                    leave=False,
                ):
                    batch_masks.append(batch.pop("batch_mask"))
                    filled_tokens.append((~batch["is_padding"]).sum(dim=1).to(torch.int64))
                    preds.append(net.predict(batch, self.device))
                    # exactly one target cell per real row, so the masked sum picks out its label
                    y = batch["number_values"].squeeze(-1)
                    labels.append((y * batch["is_targets"].to(y.dtype)).sum(dim=1))

                # drop phantom rows (last-batch overshoot or dropped items)
                mask = torch.cat(batch_masks)
                yield (
                    task,
                    torch.cat(labels)[mask].float().numpy(),
                    torch.cat(preds)[mask].float().numpy(),
                    torch.cat(filled_tokens)[mask].numpy(),
                )
