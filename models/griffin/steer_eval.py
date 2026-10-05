#!/usr/bin/env python3
"""Griffin's clean inference entry point. Takes primitive args only, runs
one task's evaluation on the given split, writes one structured JSON result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="e.g. rel-f1")
    parser.add_argument("--task-table", required=True, help="e.g. driver-dnf")
    parser.add_argument("--checkpoint", required=True, help="local checkpoint directory")
    parser.add_argument("--data-path", required=True, help="Griffin-preprocessed data dir for this dataset")
    parser.add_argument("--max-eval-steps", type=int, default=-1)
    parser.add_argument("--sampler-config-path", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--reg-metric", default=None, choices=["mae", "r2"], help="override the regression metric baked into this dataset's metatask.yaml; ignored for classification tasks")
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    result = evaluate_checkpoint(
        args.data_path, args.checkpoint, args.dataset, args.task_table, args.max_eval_steps,
        sampler_config_path=args.sampler_config_path, split=args.split, reg_metric=args.reg_metric,
    )

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{result['metric_name']}={result['metric_value']:.4f}  sample_size={result['sample_size']}")


if __name__ == "__main__":
    main()
