"""`steer-generate-config`: ask an LLM to rank a task's schema into STEER tiers.

This is the only part of STEER that calls an LLM API. It writes a tiered
config that `steer run --config PATH` can use.

    steer-generate-config --task rel_f1_driver_dnf --provider anthropic
    steer-generate-config --task rel_f1_driver_dnf --provider openai --column-filter

Needs ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment, and RelBench
access for the task's database. Two LLM calls are made (one per traversal
direction), plus a third for the column filter when --column-filter is given.
LLM output is not bit-for-bit reproducible, which is why the configs used in
the paper are shipped under configs/tasks/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import tasks as task_registry
from .run import STEER_ROOT

DEFAULT_OUT_DIR = STEER_ROOT / "generated"
EDGE_APPROACHES = ("p2f_edge", "f2p_edge")
ALL_APPROACHES = ("column_filter",) + EDGE_APPROACHES


def _make_provider(name: str, model: str | None):
    kwargs = {"model": model} if model else {}
    if name == "openai":
        from .config.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(**kwargs)
    from .config.providers.anthropic_provider import AnthropicProvider
    return AnthropicProvider(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(prog="steer-generate-config", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=task_registry.task_keys(), metavar="TASK")
    parser.add_argument("--provider", required=True, choices=["anthropic", "openai"])
    parser.add_argument("--model", default=None, help="Model name within the provider (default: the provider's own default)")
    parser.add_argument("--column-filter", action="store_true", help="Also rank columns (a third LLM call), for use with `steer run --column-filter`")
    parser.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high", "none"], help="'none' omits the setting, for models without reasoning (default: high, as in the paper)")
    parser.add_argument("--out", default=None, help="Output path (default: generated/<provider>/<task>.json)")
    parser.add_argument("--force", action="store_true", help="Overwrite the output file if it exists")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else DEFAULT_OUT_DIR / args.provider / f"{args.task}.json"
    if out_path.exists() and not args.force:
        sys.exit(f"error: {out_path} already exists (use --force to overwrite)")

    from .config.correct import demote_high_null_columns
    from .config.generate import build_task_input, generate_config
    from .config.schema import TieredConfig
    from .config.stats import compute_column_null_rates

    # Task tables are named as the current RelBench names them (the same
    # spelling Griffin uses); the run step maps other models' spellings.
    task = task_registry.get_task_spec(args.task, task_registry.MODEL_GRIFFIN)
    provider = _make_provider(args.provider, args.model)
    task_input = build_task_input(task.dataset, task.task_table)

    approaches = ALL_APPROACHES if args.column_filter else EDGE_APPROACHES
    config = generate_config(
        task_input,
        provider,
        reasoning_effort=None if args.reasoning_effort == "none" else args.reasoning_effort,
        approaches=approaches,
        existing_config=None if args.column_filter else TieredConfig(),
    )
    if args.column_filter:
        config = demote_high_null_columns(config, compute_column_null_rates(task.dataset))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out_path}")
    flag = " --column-filter" if args.column_filter else ""
    print(f"Use it with: steer run --model rt --task {args.task} --config {out_path}{flag}")


if __name__ == "__main__":
    main()
