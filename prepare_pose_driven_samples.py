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
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N labels for a quick test run.")
    parser.add_argument("--no-save", action="store_true", help="Do not overwrite the default samples/report files.")
    return parser


def main() -> None:
    configure_local_runtime_env()
    args = build_arg_parser().parse_args()
    config = TaskConfig.from_json(args.config)
    samples = prepare_pose_driven_samples(
        config,
        force_rebuild=args.force,
        max_labels=args.limit,
        save_outputs=not args.no_save,
    )
    print(
        "Conversion policy: "
        f"screen={config.conversion.candidate_screening_mode}, "
        f"target={config.conversion.saved_target_mode}"
    )
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
    hold_drop = np.asarray(
        [sample.fit_error.get("projected_hold_object_drop_m", 0.0) for sample in samples],
        dtype=np.float64,
    )
    hold_hybrid_groups = np.asarray(
        [sample.fit_error.get("projected_hold_hybrid_contact_group_count", 0.0) for sample in samples],
        dtype=np.float64,
    )
    hold_hard_groups = np.asarray(
        [sample.fit_error.get("projected_hold_hard_contact_group_count", 0.0) for sample in samples],
        dtype=np.float64,
    )
    hold_opposition = np.asarray(
        [sample.fit_error.get("projected_hold_has_thumb_opposition", 0.0) for sample in samples],
        dtype=np.float64,
    )
    required_contacts = np.asarray(
        [sample.fit_error.get("projected_required_target_contacts", 0.0) for sample in samples],
        dtype=np.float64,
    )
    hold_tested = np.asarray(
        [sample.fit_error.get("hold_tested", 0.0) for sample in samples],
        dtype=np.float64,
    )
    valid_count = int(sum(sample.valid_execution for sample in samples))
    best_index = int(np.argmin(source_tip_errors))
    if args.no_save:
        print(f"Prepared {len(samples)} pose-driven samples (not saved).")
    else:
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
    if np.any(hold_tested > 0.5):
        print(
            "Hold-test object drop (m): "
            f"mean={hold_drop.mean():.4f}, std={hold_drop.std():.4f}, "
            f"min={hold_drop.min():.4f}, max={hold_drop.max():.4f}"
        )
        print(
            "Hold-test contact groups: "
            f"hybrid_mean={hold_hybrid_groups.mean():.2f}, "
            f"hard_mean={hold_hard_groups.mean():.2f}, "
            f"thumb_opposition={(hold_opposition > 0.5).sum()}/{len(samples)}"
        )
    else:
        print("Hold-test: disabled for this run (settle-only screening).")
    print(
        "Contact buckets: "
        f"zero={(matched_contacts < 0.5).sum()}, "
        f"one={((matched_contacts >= 0.5) & (matched_contacts < 1.5)).sum()}, "
        f"ge2={(matched_contacts >= 1.5).sum()}, "
        f"required_met={(matched_contacts + 1e-9 >= required_contacts).sum()}/{len(samples)}"
    )
    print(f"Best sample index: {best_index}, label_idx={samples[best_index].label_idx}")
    if samples[best_index].fit_error.get("genhand_debug_dir"):
        print(f"Best sample debug dir: {samples[best_index].fit_error.get('genhand_debug_dir')}")
        print(f"Best sample cluster plot: {samples[best_index].fit_error.get('genhand_cluster_plot')}")
        print(f"Best sample optimized plot: {samples[best_index].fit_error.get('genhand_optimized_plot')}")
    if args.no_save:
        print(f"Report not written (use without --no-save to save to {pose_driven_report_path(config)})")
    else:
        print(f"Report: {pose_driven_report_path(config)}")


if __name__ == "__main__":
    main()
