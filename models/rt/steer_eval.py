#!/usr/bin/env python3
"""RT's clean inference entry point. Takes primitive args only (no STEER
registry knowledge), runs one task's test-split evaluation, writes one
structured JSON result. No env vars, no stdout scraping needed by the
caller -- this script owns its own inputs and output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import evaluate_checkpoint, load_checkpoint

NUM_BLOCKS = 12  # RT is always run as RT-12
D_MODEL = 256
NUM_HEADS = 8
D_FF = 1024
EMBEDDING_MODEL = "all-MiniLM-L12-v2"
D_TEXT = 384
BATCH_SIZE = 32
MAX_BFS_WIDTH = 256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="e.g. rel-f1")
    parser.add_argument("--task-table", required=True, help="e.g. driver-dnf")
    parser.add_argument("--target-column", required=True, help="e.g. did_not_finish")
    parser.add_argument("--task-type", required=True, choices=["clf", "reg"])
    parser.add_argument("--checkpoint", required=True, help="local .pt path")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-eval-steps", type=int, default=40)
    parser.add_argument("--sampler-config-path", default=None, help="runtime sampler config JSON path (toggles + ids + keep_probabilities)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--reg-metric", default="r2", choices=["mae", "r2"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    net = load_checkpoint(args.checkpoint, NUM_BLOCKS, D_MODEL, D_TEXT, NUM_HEADS, D_FF, args.device)

    eval_tasks = [
        (args.dataset, args.task_table, args.target_column, args.split, [], args.task_type),
    ]
    results = evaluate_checkpoint(
        net,
        eval_tasks,
        batch_size=BATCH_SIZE,
        seq_len=args.seq_len,
        max_bfs_width=MAX_BFS_WIDTH,
        embedding_model=EMBEDDING_MODEL,
        d_text=D_TEXT,
        max_eval_steps=args.max_eval_steps,
        sampler_config_path=args.sampler_config_path,
        device=args.device,
        reg_metric=args.reg_metric,
    )
    result = next(iter(results.values()))

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{result['metric_name']}={result['metric_value']:.4f}  sample_size={result['sample_size']}")


if __name__ == "__main__":
    main()
