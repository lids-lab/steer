#!/usr/bin/env python3
"""rt-j's clean inference entry point. Takes primitive args only (no STEER
registry knowledge, no recipe/get_tasks discovery), runs one task's
evaluation, writes one structured JSON result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from rt.checkpoints import load_rt_model
from rt.eval_utils import build_evaluator, metric_for
from rt.tasks import Task

CTX_SIZE = 8192
LOCAL_CTX_SIZE = 1024  # matches RT's context length
BFS_WIDTH = 32
NUM_WALKS = 10_000
WALK_LENGTH = 20
TOKENS_PER_GPU = 2**18


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="e.g. rel-f1")
    parser.add_argument("--task-table", required=True, help="e.g. driver-dnf")
    parser.add_argument("--target-column", required=True, help="e.g. did_not_finish")
    parser.add_argument("--task-type", required=True, choices=["clf", "reg"])
    parser.add_argument("--checkpoint", required=True, help="local path or HF Hub repo spec")
    parser.add_argument("--pre-dir", required=True, help="preprocessed RelBench data: local path or HF Hub repo spec")
    parser.add_argument("--max-eval-steps", type=int, default=-1, help="cap on items evaluated (-1 = full split)")
    parser.add_argument("--sampler-config-path", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--reg-metric", default="r2", choices=["mae", "r2"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-ctx-size", type=int, default=LOCAL_CTX_SIZE, help="override the fixed local context size (default: 1024)")
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    net, config = load_rt_model(args.checkpoint, device=device, compile=False)
    net = net.to(torch.bfloat16)

    task = Task(
        db_name=args.dataset,
        table_name=args.task_table,
        target_column=args.target_column,
        task_type=args.task_type,
        split=args.split,
    )

    items_per_task = 10_000_000 if args.max_eval_steps < 0 else args.max_eval_steps
    evaluator = build_evaluator(
        [task], args.pre_dir,
        embedding_model=config["embedding_model"], d_text=config["d_text"], device=device,
        ctx_size=CTX_SIZE, local_ctx_size=args.local_ctx_size, bfs_width=BFS_WIDTH,
        num_walks=NUM_WALKS, walk_length=WALK_LENGTH, tokens_per_gpu=TOKENS_PER_GPU,
        # num_workers=0: DataLoader worker processes hit a shared-memory
        # race ("could not unlink the shared memory file") when several
        # evaluations run on one node.
        items_per_task=items_per_task, num_workers=0,
        sampler_config_path=args.sampler_config_path,
    )

    result = None
    for eval_task, _ctx, labels, preds_by_prefix, _nl, filled_tokens in evaluator.evaluate_raw(
        [(net, "")], [CTX_SIZE], with_context_stats=True
    ):
        preds = preds_by_prefix[""]
        metric_name, metric_value = metric_for(eval_task.task_type, labels, preds, args.reg_metric)
        result = {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "sample_size": {"avg_filled_tokens": float(filled_tokens.mean()), "n": int(len(labels))},
        }

    if result is None:
        raise RuntimeError("evaluate_raw produced no results for this task")

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{result['metric_name']}={result['metric_value']:.4f}  sample_size={result['sample_size']}")


if __name__ == "__main__":
    main()
