#!/usr/bin/env python3
"""Evaluate one task with RT-J and write the metric and average token count as JSON."""

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

LOCAL_CTX_SIZE = 1024  # default local (BFS) context size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="e.g. rel-f1")
    parser.add_argument("--task-table", required=True, help="e.g. driver-dnf")
    parser.add_argument("--target-column", required=True, help="e.g. did_not_finish")
    parser.add_argument("--task-type", required=True, choices=["clf", "reg"])
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory")
    parser.add_argument("--pre-dir", required=True, help="preprocessed RelBench data directory")
    parser.add_argument("--max-eval-steps", type=int, default=-1, help="cap on items evaluated (-1 = full split)")
    parser.add_argument("--sampler-config-path", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--reg-metric", default="r2", choices=["mae", "r2"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-ctx-size", type=int, default=LOCAL_CTX_SIZE, help="local (BFS) context size")
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    net, config = load_rt_model(args.checkpoint, device=device)
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
        local_ctx_size=args.local_ctx_size, items_per_task=items_per_task,
        sampler_config_path=args.sampler_config_path,
    )

    result = None
    for eval_task, labels, preds, filled_tokens in evaluator.evaluate_raw(net):
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
