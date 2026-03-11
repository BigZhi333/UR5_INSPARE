from __future__ import annotations

import argparse
from pathlib import Path

from fr5_rh56e2_dgrasp_rl.label_conversion import prepare_converted_labels
from fr5_rh56e2_dgrasp_rl.paths import configure_local_runtime_env
from fr5_rh56e2_dgrasp_rl.task_config import TaskConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare converted D-Grasp labels for FR5+RH56E2.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config" / "default_task.json")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    configure_local_runtime_env()
    args = build_arg_parser().parse_args()
    config = TaskConfig.from_json(args.config)
    goals = prepare_converted_labels(config, force_rebuild=args.force)
    print(f"Prepared {len(goals)} converted goals: {config.converted_goals_path}")


if __name__ == "__main__":
    main()
