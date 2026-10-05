"""The `steer` command line.

    steer setup rt --task rel_f1_driver_dnf
    steer run --model rt --task rel_f1_driver_dnf [--column-filter] [--compare]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import tasks as task_registry
from .run import DEFAULT_CONFIG_DIR, DEFAULT_PROFILE_PATH, STEER_ROOT, RunResult, RunSpec, run_task

STEER_TOGGLES = "p2f_edge_probabilistic_at_push+f2p_edge_probabilistic_at_push"
STEER_COLUMN_FILTER_TOGGLES = "column_filter+" + STEER_TOGGLES

_METRIC_LABELS = {"auc": "AUROC (%)", "roc_auc": "AUROC (%)", "r2": "R² (%)"}


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(STEER_ROOT))
    except ValueError:
        return str(path)


def _print_results(spec: RunSpec, config_path: Path, profile_path: Path, rows: list[tuple[str, RunResult]]) -> None:
    metric_label = _METRIC_LABELS.get(rows[0][1].metric_name, rows[0][1].metric_name)
    label_width = max(len(label) for label, _ in rows)
    print(f"{spec.task_key} | {spec.model} | config: {_display_path(config_path)} | profile: {_display_path(profile_path)}")
    print(f"{'':<{label_width}}  {metric_label:>10}  {'Tokens':>8}")
    for label, result in rows:
        print(f"{label:<{label_width}}  {result.metric_value * 100:>10.2f}  {result.tokens:>8.1f}")
    if spec.model == task_registry.MODEL_GRIFFIN:
        print("(Griffin tokens are estimated from the number of sampled nodes.)")


def _run(args: argparse.Namespace) -> None:
    toggles = STEER_COLUMN_FILTER_TOGGLES if args.column_filter else STEER_TOGGLES
    label = "STEER + column filter" if args.column_filter else "STEER"
    spec = RunSpec(
        model=args.model,
        task_key=args.task,
        toggles=toggles,
        config_path=args.config,
        profile_path=args.profile,
        device=args.device,
    )
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_DIR / f"{args.task}.json"
    profile_path = Path(args.profile) if args.profile else DEFAULT_PROFILE_PATH

    rows = [(label, run_task(spec))]
    if args.compare:
        rows.append(("baseline", run_task(RunSpec(**{**spec.__dict__, "toggles": "baseline"}))))
    _print_results(spec, config_path, profile_path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(prog="steer", description="Run STEER on a relational foundation model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one task on one model with STEER")
    run_parser.add_argument("--model", required=True, choices=list(task_registry.ALL_MODELS))
    run_parser.add_argument("--task", required=True, choices=task_registry.task_keys(), metavar="TASK")
    run_parser.add_argument("--column-filter", action="store_true", help="Also apply the column filter (RT and RT-J only)")
    run_parser.add_argument("--config", default=None, help="Path to a tiered config (default: the shipped Claude config for the task)")
    run_parser.add_argument("--profile", default=None, help="Path to a probability profile (default: configs/profiles/universal.json)")
    run_parser.add_argument("--compare", action="store_true", help="Also run the baseline sampler and print both results")
    run_parser.add_argument("--device", default="cuda")

    setup_parser = subparsers.add_parser("setup", help="Download a model's checkpoints and preprocessed data")
    setup_parser.add_argument("model", choices=list(task_registry.ALL_MODELS))
    scope = setup_parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--task", choices=task_registry.task_keys(), metavar="TASK", help="Download only what this task needs")
    scope.add_argument("--all", action="store_true", help="Download everything for the model (large)")

    args = parser.parse_args()
    try:
        if args.command == "run":
            _run(args)
        elif args.command == "setup":
            from .setup import setup
            setup(args.model, None if args.all else args.task)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
