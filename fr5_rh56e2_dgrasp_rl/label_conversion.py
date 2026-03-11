from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .converted_goal import ConvertedGoal, load_converted_goals, save_converted_goals
from .kinematics import solve_arm_wrist_palm_ik
from .paths import ensure_runtime_dirs, locate_dgrasp_data_file
from .robot_model import RobotSceneModel
from .scene_builder import build_training_scene
from .semantics import (
    contact_mask_16_to_6,
    flatten_sites,
    mano_semantic_sites_from_keypoints,
    semantic_frame_from_sites,
    site_rmse,
)
from .task_config import TaskConfig
from .utils import rotation_angle_deg, transform_points


def _xyzw_pose_to_wxyz(pose: np.ndarray) -> np.ndarray:
    converted = pose.copy()
    converted[3:] = np.array([pose[6], pose[3], pose[4], pose[5]], dtype=np.float64)
    return converted


def _choose_pose_quaternion_order(pose: np.ndarray, ee_rel: np.ndarray, ee_world: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).copy()
    pose_wxyz = pose.copy()
    pose_xyzw = _xyzw_pose_to_wxyz(pose)
    err_wxyz = np.sqrt(np.mean(np.sum((transform_points(ee_rel, pose_wxyz) - ee_world) ** 2, axis=1)))
    err_xyzw = np.sqrt(np.mean(np.sum((transform_points(ee_rel, pose_xyzw) - ee_world) ** 2, axis=1)))
    return pose_wxyz if err_wxyz <= err_xyzw else pose_xyzw


def _load_raw_label_block(config: TaskConfig) -> dict:
    labels_path = locate_dgrasp_data_file("dexycb_train_labels.pkl")
    labels = joblib.load(labels_path)
    block = labels[config.object_id]
    if not isinstance(block, dict):
        raise TypeError(f"Unexpected label block type: {type(block)}")
    return block


def _optimize_goal_qpos(
    runtime: RobotSceneModel,
    config: TaskConfig,
    target_sites_world: np.ndarray,
    initial_guess: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("scipy is required for label conversion.") from exc

    arm_guess = runtime.clamp_arm(initial_guess[:6])
    hand_guess = runtime.clamp_hand(initial_guess[6:])

    arm_qpos = solve_arm_wrist_palm_ik(
        runtime=runtime,
        target_sites_world=target_sites_world,
        initial_arm_qpos=arm_guess,
        hand_qpos=np.zeros_like(hand_guess),
        iterations=config.conversion.arm_ik_iterations,
        damping=config.conversion.arm_ik_damping,
    )

    def hand_objective(hand_qpos: np.ndarray) -> np.ndarray:
        hand_qpos = runtime.clamp_hand(hand_qpos)
        runtime.set_robot_actuated_qpos(np.concatenate([arm_qpos, hand_qpos]))
        current_sites = runtime.get_semantic_sites_world()
        fingertip_error = (current_sites[2:] - target_sites_world[2:]).reshape(-1)
        regularizer = 0.01 * (hand_qpos - runtime.home_actuated[6:])
        return np.concatenate([fingertip_error, regularizer])

    hand_opt = least_squares(
        hand_objective,
        hand_guess,
        bounds=(runtime.hand_lower, runtime.hand_upper),
        max_nfev=config.conversion.finger_opt_max_nfev,
    )
    refined_hand = runtime.clamp_hand(hand_opt.x)

    def joint_objective(actuated_qpos: np.ndarray) -> np.ndarray:
        actuated_qpos = runtime.clamp_actuated(actuated_qpos)
        runtime.set_robot_actuated_qpos(actuated_qpos)
        current_sites = runtime.get_semantic_sites_world()
        current_frame = semantic_frame_from_sites(current_sites)
        target_frame = semantic_frame_from_sites(target_sites_world)
        site_error = (current_sites - target_sites_world).reshape(-1)
        frame_error = 0.25 * (current_frame - target_frame).reshape(-1)
        regularizer = 0.02 * (actuated_qpos - np.concatenate([arm_qpos, refined_hand]))
        return np.concatenate([site_error, frame_error, regularizer])

    refined_qpos = np.concatenate([arm_qpos, refined_hand])
    joint_opt = least_squares(
        joint_objective,
        refined_qpos,
        bounds=(runtime.actuated_lower, runtime.actuated_upper),
        max_nfev=config.conversion.joint_opt_max_nfev,
    )
    final_qpos = runtime.clamp_actuated(joint_opt.x)
    runtime.set_robot_actuated_qpos(final_qpos)

    current_sites = runtime.get_semantic_sites_world()
    current_frame = semantic_frame_from_sites(current_sites)
    target_frame = semantic_frame_from_sites(target_sites_world)
    errors = {
        "wrist_error_m": float(np.linalg.norm(current_sites[0] - target_sites_world[0])),
        "wrist_rot_error_deg": float(rotation_angle_deg(current_frame, target_frame)),
        "site_rmse_m": float(site_rmse(current_sites, target_sites_world, include_wrist=False)),
        "optimizer_cost": float(joint_opt.cost),
    }
    return final_qpos, errors


def prepare_converted_labels(config: TaskConfig, force_rebuild: bool = False) -> list[ConvertedGoal]:
    ensure_runtime_dirs()
    if config.converted_goals_path.exists() and not force_rebuild:
        return load_converted_goals(config.converted_goals_path)

    scene_xml, metadata_path = build_training_scene(config, force_rebuild=force_rebuild)
    runtime = RobotSceneModel(config, scene_xml=scene_xml, metadata_path=metadata_path)
    raw_block = _load_raw_label_block(config)

    final_obj_pos = np.asarray(raw_block["final_obj_pos"], dtype=np.float64)
    obj_pose_reset = np.asarray(raw_block["obj_pose_reset"], dtype=np.float64)
    final_ee = np.asarray(raw_block["final_ee"], dtype=np.float64).reshape(-1, 21, 3)
    final_ee_rel = np.asarray(raw_block["final_ee_rel"], dtype=np.float64).reshape(-1, 21, 3)
    final_contacts = np.asarray(raw_block["final_contacts"], dtype=np.float64)
    workspace_translation = np.asarray(config.workspace_translation, dtype=np.float64)

    goals: list[ConvertedGoal] = []
    candidate_goals: list[ConvertedGoal] = []
    report: list[dict[str, float | int]] = []
    previous_guess = runtime.home_actuated.copy()

    for label_idx in range(final_obj_pos.shape[0]):
        raw_goal_pose = final_obj_pos[label_idx]
        goal_pose = _choose_pose_quaternion_order(raw_goal_pose, final_ee_rel[label_idx], final_ee[label_idx])
        if np.allclose(goal_pose[3:], raw_goal_pose[3:]):
            init_pose = obj_pose_reset[label_idx].copy()
        else:
            init_pose = _xyzw_pose_to_wxyz(obj_pose_reset[label_idx])
        goal_pose[:3] += workspace_translation
        init_pose[:3] += workspace_translation
        semantic_sites_obj = mano_semantic_sites_from_keypoints(final_ee_rel[label_idx])
        semantic_sites_world = transform_points(semantic_sites_obj, goal_pose)
        target_contact_mask = contact_mask_16_to_6(final_contacts[label_idx])

        try:
            target_qpos, fit_error = _optimize_goal_qpos(runtime, config, semantic_sites_world, previous_guess)
        except Exception as exc:  # pragma: no cover - diagnostic path
            report.append({"label_idx": label_idx, "status": "failed", "message": str(exc)})
            continue

        valid = (
            fit_error["wrist_error_m"] <= config.conversion.wrist_error_threshold_m
            and fit_error["wrist_rot_error_deg"] <= config.conversion.wrist_rot_threshold_deg
            and fit_error["site_rmse_m"] <= config.conversion.site_rmse_threshold_m
            and np.all(np.isfinite(target_qpos))
        )

        report_item = {
            "label_idx": label_idx,
            "status": "accepted" if valid else "rejected",
            **fit_error,
        }
        report.append(report_item)

        goal = ConvertedGoal(
            object_id=config.object_id,
            label_idx=label_idx,
            object_pose_init=[float(v) for v in init_pose],
            object_pose_goal=[float(v) for v in goal_pose],
            target_qpos_12=[float(v) for v in target_qpos],
            target_sites_obj_21=flatten_sites(semantic_sites_obj),
            target_contact_mask_6=[float(v) for v in target_contact_mask],
            fit_error=fit_error,
        )
        candidate_goals.append(goal)
        if not valid:
            continue
        goals.append(goal)
        previous_guess = target_qpos

    report_path = config.project_dir / "data" / f"{config.object_name}_conversion_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if len(goals) < config.conversion.min_valid_goals:
        if config.manual_goals_path.exists() and not force_rebuild:
            manual_goals = load_converted_goals(config.manual_goals_path)
            save_converted_goals(config.converted_goals_path, manual_goals)
            return manual_goals

        if len(candidate_goals) < config.conversion.min_valid_goals:
            raise RuntimeError(
                f"Only {len(candidate_goals)} optimized goals were produced. "
                f"Expected at least {config.conversion.min_valid_goals}. "
                f"Add manual goals to {config.manual_goals_path} or relax the conversion thresholds."
            )

        fallback_goals = sorted(
            candidate_goals,
            key=lambda goal: (
                goal.fit_error["site_rmse_m"],
                goal.fit_error["wrist_error_m"],
                goal.fit_error["wrist_rot_error_deg"],
            ),
        )[: config.conversion.min_valid_goals]
        save_converted_goals(config.manual_goals_path, fallback_goals)
        save_converted_goals(config.converted_goals_path, fallback_goals)
        return fallback_goals

    save_converted_goals(config.converted_goals_path, goals)
    return goals
