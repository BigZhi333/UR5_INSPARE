from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fr5_rh56e2_dgrasp_rl.paths import configure_local_runtime_env
from fr5_rh56e2_dgrasp_rl.pose_driven_data import (
    pose_driven_report_path,
    pose_driven_samples_path,
    prepare_pose_driven_samples,
)
from fr5_rh56e2_dgrasp_rl.task_config import TaskConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare pose-driven D-Grasp samples for FR5+RH56E2.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config" / "default_task.json")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    configure_local_runtime_env()
    args = build_arg_parser().parse_args()
    config = TaskConfig.from_json(args.config)
    samples = prepare_pose_driven_samples(config, force_rebuild=args.force)
    source_tip_errors = np.asarray(
        [sample.fit_error.get("source_tip_rmse_m", sample.fit_error.get("tip_rmse_m", 0.0)) for sample in samples],
        dtype=np.float64,
    )
    penetration_errors = np.asarray(
        [sample.fit_error.get("projected_max_penetration_m", 0.0) for sample in samples],
        dtype=np.float64,
    )
    contact_hamming = np.asarray(
        [sample.fit_error.get("source_contact_hamming", 0.0) for sample in samples],
        dtype=np.float64,
    )
    matched_contacts = np.asarray(
        [sample.fit_error.get("projected_matched_target_contacts", 0.0) for sample in samples],
        dtype=np.float64,
    )
    required_contacts = np.asarray(
        [sample.fit_error.get("projected_required_target_contacts", 0.0) for sample in samples],
        dtype=np.float64,
    )
    valid_count = int(sum(sample.valid_execution for sample in samples))
    best_index = int(np.argmin(source_tip_errors))
    print(f"Prepared {len(samples)} pose-driven samples: {pose_driven_samples_path(config)}")
    print(f"Valid execution samples: {valid_count}/{len(samples)}")
    print(
        "Source tip RMSE after projection (m): "
        f"mean={source_tip_errors.mean():.4f}, std={source_tip_errors.std():.4f}, "
        f"min={source_tip_errors.min():.4f}, max={source_tip_errors.max():.4f}"
    )
    print(
        "Projected max penetration (m): "
        f"mean={penetration_errors.mean():.4f}, std={penetration_errors.std():.4f}, "
        f"min={penetration_errors.min():.4f}, max={penetration_errors.max():.4f}"
    )
    print(
        "Source contact hamming after projection: "
        f"mean={contact_hamming.mean():.2f}, std={contact_hamming.std():.2f}, "
        f"min={contact_hamming.min():.0f}, max={contact_hamming.max():.0f}"
    )
    print(
        "Matched target contacts after projection: "
        f"mean={matched_contacts.mean():.2f}, std={matched_contacts.std():.2f}, "
        f"min={matched_contacts.min():.0f}, max={matched_contacts.max():.0f}"
    )
    print(
        "Contact buckets: "
        f"zero={(matched_contacts < 0.5).sum()}, "
        f"one={((matched_contacts >= 0.5) & (matched_contacts < 1.5)).sum()}, "
        f"ge2={(matched_contacts >= 1.5).sum()}, "
        f"required_met={(matched_contacts + 1e-9 >= required_contacts).sum()}/{len(samples)}"
    )
    print(f"Best sample index: {best_index}, label_idx={samples[best_index].label_idx}")
    print(f"Report: {pose_driven_report_path(config)}")


if __name__ == "__main__":
    main()
