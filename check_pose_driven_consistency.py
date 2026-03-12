from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fr5_rh56e2_dgrasp_rl.kinematics import solve_arm_wrist_palm_ik
from fr5_rh56e2_dgrasp_rl.paths import configure_local_runtime_env
from fr5_rh56e2_dgrasp_rl.pose_driven_data import (
    PROJECTION_SETTLE_STEPS,
    PoseDrivenSample,
    load_pose_driven_samples,
    pose_driven_samples_path,
    prepare_pose_driven_samples,
)
from fr5_rh56e2_dgrasp_rl.robot_model import RobotSceneModel
from fr5_rh56e2_dgrasp_rl.scene_builder import build_training_scene
from fr5_rh56e2_dgrasp_rl.semantics import SEMANTIC_CONTACT_NAMES
from fr5_rh56e2_dgrasp_rl.task_config import TaskConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check pose-driven sample consistency between target labels and replayed MuJoCo state."
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config" / "default_task.json")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "build" / "pose_driven_consistency_report.json",
    )
    return parser


def _solve_final_actuated_qpos(runtime: RobotSceneModel, config: TaskConfig, sample: PoseDrivenSample) -> np.ndarray:
    runtime.reset()
    runtime.set_object_pose(np.asarray(sample.object_pose_goal, dtype=np.float64))
    arm_qpos = solve_arm_wrist_palm_ik(
        runtime=runtime,
        target_wrist_pose_world=np.asarray(sample.wrist_pose_goal_world, dtype=np.float64),
        initial_arm_qpos=runtime.get_actuated_qpos()[:6],
        hand_qpos=np.asarray(sample.hand_qpos_6, dtype=np.float64),
        iterations=config.conversion.arm_ik_iterations,
        damping=config.conversion.arm_ik_damping,
    )
    return np.concatenate([arm_qpos, np.asarray(sample.hand_qpos_6, dtype=np.float64)])


def analyze_pose_driven_consistency(
    config: TaskConfig,
    force_prepare: bool = False,
    valid_only: bool = False,
) -> dict[str, object]:
    if force_prepare:
        samples = prepare_pose_driven_samples(config, force_rebuild=True)
    else:
        samples_path = pose_driven_samples_path(config)
        samples = load_pose_driven_samples(samples_path) if samples_path.exists() else prepare_pose_driven_samples(config)

    if valid_only:
        samples = [sample for sample in samples if sample.valid_execution]
    if not samples:
        raise RuntimeError("No pose-driven samples were selected for analysis.")

    scene_xml, metadata_path = build_training_scene(config)
    runtime = RobotSceneModel(config, scene_xml=scene_xml, metadata_path=metadata_path)
    site_rmse_values: list[float] = []
    site_max_values: list[float] = []
    wrist_fit_diff: list[float] = []
    palm_fit_diff: list[float] = []
    tip_fit_diff: list[float] = []
    contact_hamming_values: list[int] = []
    per_label_matches = np.zeros(len(SEMANTIC_CONTACT_NAMES), dtype=np.float64)
    contact_exact_match_count = 0
    contact_tp = 0
    contact_fp = 0
    contact_fn = 0
    contact_tn = 0
    sample_rows: list[dict[str, object]] = []

    for sample_index, sample in enumerate(samples):
        final_actuated_qpos = _solve_final_actuated_qpos(runtime, config, sample)

        runtime.reset()
        runtime.set_object_pose(np.asarray(sample.object_pose_goal, dtype=np.float64))
        runtime.set_robot_actuated_qpos(final_actuated_qpos)
        current_sites = runtime.get_semantic_sites_world()
        target_sites = sample.semantic_sites_world()
        site_distances = np.linalg.norm(current_sites - target_sites, axis=1)

        wrist_error = float(site_distances[0])
        palm_error = float(site_distances[1])
        tip_rmse = float(np.sqrt(np.mean(np.sum((current_sites[2:] - target_sites[2:]) ** 2, axis=1))))
        site_rmse = float(np.sqrt(np.mean(np.sum((current_sites - target_sites) ** 2, axis=1))))
        site_max = float(site_distances.max())

        runtime.reset()
        runtime.set_object_pose(np.asarray(sample.object_pose_goal, dtype=np.float64))
        runtime.settle_actuated_pose(final_actuated_qpos, PROJECTION_SETTLE_STEPS)
        contact_diag = runtime.get_contact_diagnostics_12()
        actual_contact_mask = np.asarray(contact_diag["hybrid_mask"], dtype=np.float64)
        actual_contact_forces = np.asarray(contact_diag["forces"], dtype=np.float64)
        target_contact_mask = np.asarray(sample.contact_mask_12, dtype=np.float64)
        contact_hamming = int(np.abs(actual_contact_mask - target_contact_mask).sum())

        site_rmse_values.append(site_rmse)
        site_max_values.append(site_max)
        wrist_fit_diff.append(abs(wrist_error - sample.fit_error.get("wrist_error_m", 0.0)))
        palm_fit_diff.append(abs(palm_error - sample.fit_error.get("palm_error_m", 0.0)))
        tip_fit_diff.append(abs(tip_rmse - sample.fit_error.get("tip_rmse_m", 0.0)))
        contact_hamming_values.append(contact_hamming)
        per_label_matches += (actual_contact_mask == target_contact_mask).astype(np.float64)
        contact_exact_match_count += int(contact_hamming == 0)
        contact_tp += int(((actual_contact_mask > 0.5) & (target_contact_mask > 0.5)).sum())
        contact_fp += int(((actual_contact_mask > 0.5) & (target_contact_mask < 0.5)).sum())
        contact_fn += int(((actual_contact_mask < 0.5) & (target_contact_mask > 0.5)).sum())
        contact_tn += int(((actual_contact_mask < 0.5) & (target_contact_mask < 0.5)).sum())

        sample_rows.append(
            {
                "sample_index": sample_index,
                "label_idx": sample.label_idx,
                "valid_execution": sample.valid_execution,
                "site_rmse_m": site_rmse,
                "site_max_error_m": site_max,
                "wrist_error_m": wrist_error,
                "palm_error_m": palm_error,
                "tip_rmse_m": tip_rmse,
                "stored_fit_error": sample.fit_error,
                "contact_hamming": contact_hamming,
                "target_contact_mask_12": [int(round(v)) for v in target_contact_mask],
                "actual_contact_mask_12": [int(round(v)) for v in actual_contact_mask],
                "actual_contact_forces": [float(v) for v in actual_contact_forces],
            }
        )

    per_label_accuracy = per_label_matches / len(samples)
    summary = {
        "num_samples": len(samples),
        "num_valid_execution": int(sum(sample.valid_execution for sample in samples)),
        "site_rmse_mean_m": float(np.mean(site_rmse_values)),
        "site_rmse_std_m": float(np.std(site_rmse_values)),
        "site_rmse_max_m": float(np.max(site_rmse_values)),
        "site_max_error_mean_m": float(np.mean(site_max_values)),
        "fit_error_consistency": {
            "wrist_abs_diff_mean_m": float(np.mean(wrist_fit_diff)),
            "wrist_abs_diff_max_m": float(np.max(wrist_fit_diff)),
            "palm_abs_diff_mean_m": float(np.mean(palm_fit_diff)),
            "palm_abs_diff_max_m": float(np.max(palm_fit_diff)),
            "tip_rmse_abs_diff_mean_m": float(np.mean(tip_fit_diff)),
            "tip_rmse_abs_diff_max_m": float(np.max(tip_fit_diff)),
        },
        "contact_exact_match_count": int(contact_exact_match_count),
        "contact_exact_match_rate": float(contact_exact_match_count / len(samples)),
        "contact_hamming_mean": float(np.mean(contact_hamming_values)),
        "contact_hamming_max": int(np.max(contact_hamming_values)),
        "contact_bit_totals": {
            "num_samples": len(samples),
            "contact_labels_per_sample": len(SEMANTIC_CONTACT_NAMES),
            "total_contact_bits": int(len(samples) * len(SEMANTIC_CONTACT_NAMES)),
            "matched_bits": int(contact_tp + contact_tn),
            "matched_bit_rate": float((contact_tp + contact_tn) / max(len(samples) * len(SEMANTIC_CONTACT_NAMES), 1)),
            "mismatched_bits": int(contact_fp + contact_fn),
            "target_positive_contacts": int(contact_tp + contact_fn),
            "actual_positive_contacts": int(contact_tp + contact_fp),
            "true_positive_matches": int(contact_tp),
            "true_negative_matches": int(contact_tn),
            "false_positive": int(contact_fp),
            "false_negative": int(contact_fn),
        },
        "contact_precision": float(contact_tp / max(contact_tp + contact_fp, 1)),
        "contact_recall": float(contact_tp / max(contact_tp + contact_fn, 1)),
        "per_label_accuracy": {
            label_name: float(label_accuracy)
            for label_name, label_accuracy in zip(SEMANTIC_CONTACT_NAMES, per_label_accuracy)
        },
        "per_sample_contact_match_counts": [
            int(len(SEMANTIC_CONTACT_NAMES) - row["contact_hamming"])
            for row in sample_rows
        ],
        "best_site_sample": min(sample_rows, key=lambda row: row["site_rmse_m"]),
        "worst_contact_sample": max(sample_rows, key=lambda row: row["contact_hamming"]),
    }
    return {"summary": summary, "samples": sample_rows}


def main() -> None:
    configure_local_runtime_env()
    args = build_arg_parser().parse_args()
    config = TaskConfig.from_json(args.config)
    report = analyze_pose_driven_consistency(
        config=config,
        force_prepare=args.force_prepare,
        valid_only=args.valid_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(f"Report: {args.output}")
    print(f"Samples checked: {summary['num_samples']}")
    print(
        "Sites: "
        f"rmse_mean={summary['site_rmse_mean_m']:.4f} m, "
        f"rmse_max={summary['site_rmse_max_m']:.4f} m"
    )
    fit = summary["fit_error_consistency"]
    print(
        "Stored-vs-replayed fit: "
        f"wrist_mean_diff={fit['wrist_abs_diff_mean_m']:.4f} m, "
        f"palm_mean_diff={fit['palm_abs_diff_mean_m']:.4f} m, "
        f"tip_mean_diff={fit['tip_rmse_abs_diff_mean_m']:.4f} m"
    )
    print(
        "Contacts: "
        f"exact_match_rate={summary['contact_exact_match_rate']:.2%}, "
        f"hamming_mean={summary['contact_hamming_mean']:.2f}, "
        f"precision={summary['contact_precision']:.2%}, "
        f"recall={summary['contact_recall']:.2%}"
    )


if __name__ == "__main__":
    main()
