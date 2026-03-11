from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .kinematics import solve_arm_wrist_palm_ik
from .paths import ensure_runtime_dirs, locate_dgrasp_data_file
from .robot_model import RobotSceneModel
from .scene_builder import build_training_scene
from .semantics import (
    ROBOT_WRIST_TO_PALM_OFFSET_M,
    contact_mask_16_to_12,
    flatten_sites,
    mano_semantic_sites_from_keypoints,
    unflatten_sites,
)
from .task_config import TaskConfig
from .utils import (
    compose_pose7,
    inverse_pose7,
    matrix_to_pose7,
    normalize,
    quat_wxyz_to_matrix,
    rotation_angle_deg,
    transform_points,
)

PROJECTION_SETTLE_STEPS = 32


@dataclass
class PoseDrivenSample:
    object_id: int
    label_idx: int
    object_pose_init: list[float]
    object_pose_goal: list[float]
    wrist_pose_goal_world: list[float]
    wrist_pose_goal_object: list[float]
    source_root_pose_world: list[float]
    hand_qpos_6: list[float]
    semantic_sites_goal_world_21: list[float]
    semantic_sites_goal_object_21: list[float]
    source_contact_mask_12: list[float]
    contact_mask_12: list[float]
    valid_execution: bool
    fit_error: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "label_idx": self.label_idx,
            "object_pose_init": self.object_pose_init,
            "object_pose_goal": self.object_pose_goal,
            "wrist_pose_goal_world": self.wrist_pose_goal_world,
            "wrist_pose_goal_object": self.wrist_pose_goal_object,
            "source_root_pose_world": self.source_root_pose_world,
            "hand_qpos_6": self.hand_qpos_6,
            "semantic_sites_goal_world_21": self.semantic_sites_goal_world_21,
            "semantic_sites_goal_object_21": self.semantic_sites_goal_object_21,
            "source_contact_mask_12": self.source_contact_mask_12,
            "contact_mask_12": self.contact_mask_12,
            "valid_execution": self.valid_execution,
            "fit_error": self.fit_error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PoseDrivenSample":
        source_contact_mask_12 = payload.get("source_contact_mask_12")
        contact_mask_12 = payload.get("contact_mask_12")
        if contact_mask_12 is None:
            legacy_mask_6 = np.asarray(payload["contact_mask_6"], dtype=np.float64).reshape(6)
            contact_mask_12 = [
                float(legacy_mask_6[0] > 0.5),
                float(legacy_mask_6[1] > 0.5),
                float(legacy_mask_6[1] > 0.5),
                float(legacy_mask_6[1] > 0.5),
                float(legacy_mask_6[2] > 0.5),
                float(legacy_mask_6[2] > 0.5),
                float(legacy_mask_6[3] > 0.5),
                float(legacy_mask_6[3] > 0.5),
                float(legacy_mask_6[4] > 0.5),
                float(legacy_mask_6[4] > 0.5),
                float(legacy_mask_6[5] > 0.5),
                float(legacy_mask_6[5] > 0.5),
            ]
        if source_contact_mask_12 is None:
            source_contact_mask_12 = contact_mask_12
        return cls(
            object_id=int(payload["object_id"]),
            label_idx=int(payload["label_idx"]),
            object_pose_init=[float(v) for v in payload["object_pose_init"]],
            object_pose_goal=[float(v) for v in payload["object_pose_goal"]],
            wrist_pose_goal_world=[float(v) for v in payload["wrist_pose_goal_world"]],
            wrist_pose_goal_object=[float(v) for v in payload["wrist_pose_goal_object"]],
            source_root_pose_world=[float(v) for v in payload["source_root_pose_world"]],
            hand_qpos_6=[float(v) for v in payload["hand_qpos_6"]],
            semantic_sites_goal_world_21=[float(v) for v in payload["semantic_sites_goal_world_21"]],
            semantic_sites_goal_object_21=[float(v) for v in payload["semantic_sites_goal_object_21"]],
            source_contact_mask_12=[float(v) for v in source_contact_mask_12],
            contact_mask_12=[float(v) for v in contact_mask_12],
            valid_execution=bool(payload["valid_execution"]),
            fit_error={str(k): float(v) for k, v in payload["fit_error"].items()},
        )

    def semantic_sites_world(self) -> np.ndarray:
        return unflatten_sites(self.semantic_sites_goal_world_21)

    def semantic_sites_object(self) -> np.ndarray:
        return unflatten_sites(self.semantic_sites_goal_object_21)


def pose_driven_samples_path(config: TaskConfig) -> Path:
    return config.project_dir / "data" / f"{config.object_name}_pose_driven_samples.json"


def pose_driven_report_path(config: TaskConfig) -> Path:
    return config.project_dir / "data" / f"{config.object_name}_pose_driven_report.json"


def save_pose_driven_samples(path: Path, samples: list[PoseDrivenSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [sample.to_dict() for sample in samples]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_pose_driven_samples(path: Path) -> list[PoseDrivenSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [PoseDrivenSample.from_dict(item) for item in payload]


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


def _source_root_pose_world(raw_qpos: np.ndarray, raw_pose: np.ndarray, translation: np.ndarray) -> np.ndarray:
    root_pose = np.zeros(7, dtype=np.float64)
    root_pose[:3] = np.asarray(raw_qpos[:3], dtype=np.float64) + translation
    quat_xyzw = Rotation.from_euler("xyz", np.asarray(raw_pose[:3], dtype=np.float64)).as_quat()
    root_pose[3:] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
    return root_pose


def _right_handed_wrist_frame(semantic_sites: np.ndarray) -> np.ndarray:
    wrist = semantic_sites[0]
    palm = semantic_sites[1]
    index_tip = semantic_sites[3]
    little_tip = semantic_sites[6]
    approach = normalize(palm - wrist)
    across = normalize(index_tip - little_tip)
    normal = normalize(np.cross(approach, across))
    across = normalize(np.cross(normal, approach))
    return np.column_stack((across, normal, approach))


def wrist_pose_from_semantic_sites(semantic_sites: np.ndarray) -> np.ndarray:
    pose = np.zeros(7, dtype=np.float64)
    pose[:3] = semantic_sites[0]
    pose[3:] = matrix_to_pose7(
        np.vstack(
            [
                np.hstack([_right_handed_wrist_frame(semantic_sites), semantic_sites[0:1].T]),
                np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
            ]
        )
    )[3:]
    return pose


def wrist_pose_to_target_sites(wrist_pose: np.ndarray) -> np.ndarray:
    wrist_pose = np.asarray(wrist_pose, dtype=np.float64)
    wrist = wrist_pose[:3]
    rotation = quat_wxyz_to_matrix(wrist_pose[3:])
    palm = wrist + rotation[:, 2] * ROBOT_WRIST_TO_PALM_OFFSET_M
    return np.vstack([wrist, palm])


def _load_raw_label_block(config: TaskConfig) -> dict[str, Any]:
    labels_path = locate_dgrasp_data_file("dexycb_train_labels.pkl")
    labels = joblib.load(labels_path)
    block = labels[config.object_id]
    if not isinstance(block, dict):
        raise TypeError(f"Unexpected label block type: {type(block)}")
    return block


def _optimize_hand_qpos(
    runtime: RobotSceneModel,
    arm_qpos: np.ndarray,
    target_sites_world: np.ndarray,
    initial_hand_qpos: np.ndarray,
    max_nfev: int,
) -> tuple[np.ndarray, float]:
    initial_hand_qpos = runtime.clamp_hand(np.asarray(initial_hand_qpos, dtype=np.float64))

    def objective(hand_qpos: np.ndarray) -> np.ndarray:
        hand_qpos = runtime.clamp_hand(hand_qpos)
        runtime.set_robot_actuated_qpos(np.concatenate([arm_qpos, hand_qpos]))
        current_sites = runtime.get_semantic_sites_world()
        fingertip_error = (current_sites[2:] - target_sites_world[2:]).reshape(-1)
        regularizer = 0.01 * (hand_qpos - runtime.home_actuated[6:])
        return np.concatenate([fingertip_error, regularizer])

    result = least_squares(
        objective,
        initial_hand_qpos,
        bounds=(runtime.hand_lower, runtime.hand_upper),
        max_nfev=max_nfev,
    )
    return runtime.clamp_hand(result.x), float(result.cost)


def _finger_contact_bias_from_mask(source_contact_mask_12: np.ndarray) -> np.ndarray:
    contact_mask = np.asarray(source_contact_mask_12, dtype=np.float64).reshape(12)
    active = np.array(
        [
            float(contact_mask[1:4].max() > 0.5),
            float(contact_mask[1:4].max() > 0.5),
            float(contact_mask[4:6].max() > 0.5),
            float(contact_mask[6:8].max() > 0.5),
            float(contact_mask[8:10].max() > 0.5),
            float(contact_mask[10:12].max() > 0.5),
        ],
        dtype=np.float64,
    )
    squeeze = np.array([0.04, 0.14, 0.12, 0.12, 0.12, 0.12], dtype=np.float64) * active
    open_non_target = np.array([0.0, 0.06, 0.08, 0.08, 0.08, 0.08], dtype=np.float64) * (1.0 - active)
    return squeeze - open_non_target


def _evaluate_projected_candidate(
    runtime: RobotSceneModel,
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_wrist_pose_world: np.ndarray,
    source_semantic_sites_world: np.ndarray,
    source_contact_mask_12: np.ndarray,
    wrist_translation_local: np.ndarray,
    hand_qpos: np.ndarray,
) -> dict[str, Any]:
    source_wrist_pose_world = np.asarray(source_wrist_pose_world, dtype=np.float64)
    source_frame = quat_wxyz_to_matrix(source_wrist_pose_world[3:])
    wrist_translation_local = np.asarray(wrist_translation_local, dtype=np.float64)
    candidate_wrist_pose = source_wrist_pose_world.copy()
    candidate_wrist_pose[:3] = candidate_wrist_pose[:3] + source_frame @ wrist_translation_local
    candidate_hand_qpos = runtime.clamp_hand(np.asarray(hand_qpos, dtype=np.float64))

    candidate_arm_qpos = solve_arm_wrist_palm_ik(
        runtime=runtime,
        target_sites_world=wrist_pose_to_target_sites(candidate_wrist_pose),
        initial_arm_qpos=runtime.home_actuated[:6],
        hand_qpos=candidate_hand_qpos,
        iterations=max(48, int(config.conversion.arm_ik_iterations * 0.6)),
        damping=config.conversion.arm_ik_damping,
    )
    target_actuated_qpos = np.concatenate([candidate_arm_qpos, candidate_hand_qpos])

    runtime.reset()
    runtime.set_object_pose(object_pose_goal)
    runtime.set_robot_actuated_qpos(target_actuated_qpos)
    kinematic_sites = runtime.get_semantic_sites_world()
    kinematic_wrist_pose = wrist_pose_from_semantic_sites(kinematic_sites)
    kinematic_frame = quat_wxyz_to_matrix(kinematic_wrist_pose[3:])

    runtime.reset()
    runtime.set_object_pose(object_pose_goal)
    runtime.settle_actuated_pose(target_actuated_qpos, PROJECTION_SETTLE_STEPS)
    settled_contact_diag = runtime.get_contact_diagnostics_12()
    settled_contact_mask = np.asarray(settled_contact_diag["mask"], dtype=np.float64)
    settled_object_pose = runtime.get_object_pose()
    goal_object_frame = quat_wxyz_to_matrix(np.asarray(object_pose_goal[3:], dtype=np.float64))
    settled_object_frame = quat_wxyz_to_matrix(settled_object_pose[3:])

    wrist_error = float(np.linalg.norm(kinematic_sites[0] - source_semantic_sites_world[0]))
    palm_error = float(np.linalg.norm(kinematic_sites[1] - source_semantic_sites_world[1]))
    tip_rmse = float(
        np.sqrt(np.mean(np.sum((kinematic_sites[2:] - source_semantic_sites_world[2:]) ** 2, axis=1)))
    )
    site_rmse = float(
        np.sqrt(np.mean(np.sum((kinematic_sites - source_semantic_sites_world) ** 2, axis=1)))
    )
    frame_error_deg = float(rotation_angle_deg(kinematic_frame, source_frame))
    target_contact_misses = float(np.sum((source_contact_mask_12 > 0.5) & (settled_contact_mask < 0.5)))
    extra_contacts = float(np.sum((source_contact_mask_12 < 0.5) & (settled_contact_mask > 0.5)))
    matched_target_contacts = float(np.sum((source_contact_mask_12 > 0.5) & (settled_contact_mask > 0.5)))
    translation_norm = float(np.linalg.norm(wrist_translation_local))
    object_translation_drift_m = float(np.linalg.norm(settled_object_pose[:3] - np.asarray(object_pose_goal[:3], dtype=np.float64)))
    object_rotation_drift_deg = float(rotation_angle_deg(settled_object_frame, goal_object_frame))

    score = (
        site_rmse
        + 28.0 * float(settled_contact_diag["total_penetration_m"])
        + 48.0 * float(settled_contact_diag["max_penetration_m"])
        + 0.018 * target_contact_misses
        + 0.010 * extra_contacts
        - 0.004 * matched_target_contacts
        + 0.02 * float(bool(settled_contact_diag["table_contact"]))
        + 0.20 * translation_norm
        + 1.25 * object_translation_drift_m
        + 0.0008 * object_rotation_drift_deg
        + 0.01 * float(np.linalg.norm(candidate_hand_qpos - runtime.home_actuated[6:]))
        + 0.0002 * frame_error_deg
    )

    return {
        "score": float(score),
        "wrist_translation_local": wrist_translation_local.copy(),
        "arm_qpos": candidate_arm_qpos.copy(),
        "hand_qpos": candidate_hand_qpos.copy(),
        "semantic_sites_world": kinematic_sites.copy(),
        "wrist_pose_world": kinematic_wrist_pose.copy(),
        "object_pose_goal": np.asarray(object_pose_goal, dtype=np.float64).copy(),
        "contact_mask_12": settled_contact_mask.copy(),
        "contact_forces_12": np.asarray(settled_contact_diag["forces"], dtype=np.float64).copy(),
        "table_contact": bool(settled_contact_diag["table_contact"]),
        "total_penetration_m": float(settled_contact_diag["total_penetration_m"]),
        "max_penetration_m": float(settled_contact_diag["max_penetration_m"]),
        "source_wrist_error_m": wrist_error,
        "source_palm_error_m": palm_error,
        "source_tip_rmse_m": tip_rmse,
        "source_site_rmse_m": site_rmse,
        "source_semantic_frame_error_deg": frame_error_deg,
        "source_contact_hamming": float(np.abs(settled_contact_mask - source_contact_mask_12).sum()),
        "source_contact_misses": target_contact_misses,
        "source_extra_contacts": extra_contacts,
        "source_matched_target_contacts": matched_target_contacts,
        "object_translation_drift_m": object_translation_drift_m,
        "object_rotation_drift_deg": object_rotation_drift_deg,
    }


def _project_feasible_target(
    runtime: RobotSceneModel,
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_wrist_pose_world: np.ndarray,
    source_semantic_sites_world: np.ndarray,
    source_contact_mask_12: np.ndarray,
    initial_arm_qpos: np.ndarray,
    initial_hand_qpos: np.ndarray,
) -> dict[str, Any]:
    del initial_arm_qpos

    base_hand_qpos = runtime.clamp_hand(np.asarray(initial_hand_qpos, dtype=np.float64))
    contact_biased_hand = runtime.clamp_hand(base_hand_qpos + _finger_contact_bias_from_mask(source_contact_mask_12))
    seed_states = [
        (np.array([0.0, 0.0, 0.0], dtype=np.float64), base_hand_qpos),
        (np.array([0.0, 0.0, -0.006], dtype=np.float64), contact_biased_hand),
        (np.array([0.0, 0.0, -0.003], dtype=np.float64), contact_biased_hand),
    ]

    translation_step_schedule = [
        np.array([0.004, 0.004, 0.006], dtype=np.float64),
        np.array([0.0015, 0.0015, 0.0025], dtype=np.float64),
    ]
    hand_step_schedule = [
        np.array([0.14, 0.10, 0.14, 0.14, 0.14, 0.14], dtype=np.float64),
        np.array([0.05, 0.035, 0.05, 0.05, 0.05, 0.05], dtype=np.float64),
    ]

    global_best: dict[str, Any] | None = None

    for seed_translation, seed_hand_qpos in seed_states:
        best = _evaluate_projected_candidate(
            runtime=runtime,
            config=config,
            object_pose_goal=object_pose_goal,
            source_wrist_pose_world=source_wrist_pose_world,
            source_semantic_sites_world=source_semantic_sites_world,
            source_contact_mask_12=source_contact_mask_12,
            wrist_translation_local=seed_translation,
            hand_qpos=seed_hand_qpos,
        )

        for translation_steps, hand_steps in zip(translation_step_schedule, hand_step_schedule):
            for _ in range(2):
                improved = False

                for axis in range(3):
                    for direction in (-1.0, 1.0):
                        candidate_translation = np.asarray(best["wrist_translation_local"], dtype=np.float64).copy()
                        candidate_translation[axis] += direction * translation_steps[axis]
                        candidate = _evaluate_projected_candidate(
                            runtime=runtime,
                            config=config,
                            object_pose_goal=object_pose_goal,
                            source_wrist_pose_world=source_wrist_pose_world,
                            source_semantic_sites_world=source_semantic_sites_world,
                            source_contact_mask_12=source_contact_mask_12,
                            wrist_translation_local=candidate_translation,
                            hand_qpos=np.asarray(best["hand_qpos"], dtype=np.float64),
                        )
                        if candidate["score"] < best["score"] - 1e-9:
                            best = candidate
                            improved = True

                if not improved:
                    break

                for joint_idx in range(len(base_hand_qpos)):
                    for direction in (-1.0, 1.0):
                        candidate_hand_qpos = np.asarray(best["hand_qpos"], dtype=np.float64).copy()
                        candidate_hand_qpos[joint_idx] += direction * hand_steps[joint_idx]
                        candidate = _evaluate_projected_candidate(
                            runtime=runtime,
                            config=config,
                            object_pose_goal=object_pose_goal,
                            source_wrist_pose_world=source_wrist_pose_world,
                            source_semantic_sites_world=source_semantic_sites_world,
                            source_contact_mask_12=source_contact_mask_12,
                            wrist_translation_local=np.asarray(best["wrist_translation_local"], dtype=np.float64),
                            hand_qpos=candidate_hand_qpos,
                        )
                        if candidate["score"] < best["score"] - 1e-9:
                            best = candidate
                            improved = True

        if global_best is None or best["score"] < global_best["score"]:
            global_best = best

    if global_best is None:
        raise RuntimeError("Failed to project a feasible pose-driven target.")

    global_best["retreat_m"] = float(max(0.0, -float(np.asarray(global_best["wrist_translation_local"])[2])))
    global_best["hand_open_blend"] = float(
        np.linalg.norm(np.asarray(global_best["hand_qpos"]) - base_hand_qpos) / max(np.linalg.norm(runtime.hand_upper - runtime.hand_lower), 1e-8)
    )
    return global_best


def prepare_pose_driven_samples(config: TaskConfig, force_rebuild: bool = False) -> list[PoseDrivenSample]:
    ensure_runtime_dirs()
    samples_path = pose_driven_samples_path(config)
    if samples_path.exists() and not force_rebuild:
        return load_pose_driven_samples(samples_path)

    scene_xml, metadata_path = build_training_scene(config, force_rebuild=force_rebuild)
    runtime = RobotSceneModel(config, scene_xml=scene_xml, metadata_path=metadata_path)
    raw_block = _load_raw_label_block(config)

    final_qpos = np.asarray(raw_block["final_qpos"], dtype=np.float64)
    final_pose = np.asarray(raw_block["final_pose"], dtype=np.float64)
    final_obj_pos = np.asarray(raw_block["final_obj_pos"], dtype=np.float64)
    obj_pose_reset = np.asarray(raw_block["obj_pose_reset"], dtype=np.float64)
    final_ee = np.asarray(raw_block["final_ee"], dtype=np.float64).reshape(-1, 21, 3)
    final_ee_rel = np.asarray(raw_block["final_ee_rel"], dtype=np.float64).reshape(-1, 21, 3)
    final_contacts = np.asarray(raw_block["final_contacts"], dtype=np.float64)
    workspace_translation = np.asarray(config.workspace_translation, dtype=np.float64)

    samples: list[PoseDrivenSample] = []
    report: list[dict[str, Any]] = []
    previous_arm_qpos = runtime.home_actuated[:6].copy()
    previous_hand_qpos = runtime.home_actuated[6:].copy()

    for label_idx in range(final_obj_pos.shape[0]):
        raw_goal_pose = final_obj_pos[label_idx]
        goal_pose = _choose_pose_quaternion_order(raw_goal_pose, final_ee_rel[label_idx], final_ee[label_idx])
        if np.allclose(goal_pose[3:], raw_goal_pose[3:]):
            init_pose = obj_pose_reset[label_idx].copy()
        else:
            init_pose = _xyzw_pose_to_wxyz(obj_pose_reset[label_idx])
        goal_pose[:3] += workspace_translation
        init_pose[:3] += workspace_translation

        source_semantic_sites_obj = mano_semantic_sites_from_keypoints(final_ee_rel[label_idx])
        source_semantic_sites_world = transform_points(source_semantic_sites_obj, goal_pose)
        source_wrist_pose_goal_world = wrist_pose_from_semantic_sites(source_semantic_sites_world)
        source_root_pose_world = _source_root_pose_world(final_qpos[label_idx], final_pose[label_idx], workspace_translation)
        source_contact_mask = contact_mask_16_to_12(final_contacts[label_idx])

        arm_qpos = solve_arm_wrist_palm_ik(
            runtime=runtime,
            target_sites_world=wrist_pose_to_target_sites(source_wrist_pose_goal_world),
            initial_arm_qpos=previous_arm_qpos,
            hand_qpos=previous_hand_qpos,
            iterations=config.conversion.arm_ik_iterations,
            damping=config.conversion.arm_ik_damping,
        )
        hand_qpos, optimizer_cost = _optimize_hand_qpos(
            runtime=runtime,
            arm_qpos=arm_qpos,
            target_sites_world=source_semantic_sites_world,
            initial_hand_qpos=previous_hand_qpos,
            max_nfev=config.conversion.finger_opt_max_nfev,
        )

        projected = _project_feasible_target(
            runtime=runtime,
            config=config,
            object_pose_goal=goal_pose,
            source_wrist_pose_world=source_wrist_pose_goal_world,
            source_semantic_sites_world=source_semantic_sites_world,
            source_contact_mask_12=source_contact_mask,
            initial_arm_qpos=arm_qpos,
            initial_hand_qpos=hand_qpos,
        )
        projected_semantic_sites_world = np.asarray(projected["semantic_sites_world"], dtype=np.float64)
        projected_wrist_pose_world = np.asarray(projected["wrist_pose_world"], dtype=np.float64)
        projected_object_pose_goal = np.asarray(projected["object_pose_goal"], dtype=np.float64)
        projected_semantic_sites_object = transform_points(projected_semantic_sites_world, inverse_pose7(projected_object_pose_goal))
        projected_wrist_pose_object = compose_pose7(inverse_pose7(projected_object_pose_goal), projected_wrist_pose_world)
        fit_error = {
            "wrist_error_m": 0.0,
            "palm_error_m": 0.0,
            "tip_rmse_m": 0.0,
            "semantic_frame_error_deg": 0.0,
            "source_root_translation_gap_m": float(
                np.linalg.norm(source_root_pose_world[:3] - source_wrist_pose_goal_world[:3])
            ),
            "source_wrist_error_m": float(projected["source_wrist_error_m"]),
            "source_palm_error_m": float(projected["source_palm_error_m"]),
            "source_tip_rmse_m": float(projected["source_tip_rmse_m"]),
            "source_site_rmse_m": float(projected["source_site_rmse_m"]),
            "source_semantic_frame_error_deg": float(projected["source_semantic_frame_error_deg"]),
            "source_contact_hamming": float(projected["source_contact_hamming"]),
            "source_contact_misses": float(projected["source_contact_misses"]),
            "source_extra_contacts": float(projected["source_extra_contacts"]),
            "projected_total_penetration_m": float(projected["total_penetration_m"]),
            "projected_max_penetration_m": float(projected["max_penetration_m"]),
            "projected_object_translation_drift_m": float(projected["object_translation_drift_m"]),
            "projected_object_rotation_drift_deg": float(projected["object_rotation_drift_deg"]),
            "projected_retreat_m": float(projected["retreat_m"]),
            "projected_hand_open_blend": float(projected["hand_open_blend"]),
            "projection_score": float(projected["score"]),
            "optimizer_cost": optimizer_cost,
        }
        valid_execution = bool(
            fit_error["projected_max_penetration_m"] <= 0.0025
            and fit_error["source_tip_rmse_m"] <= 0.08
            and fit_error["projected_object_translation_drift_m"] <= 0.04
            and not bool(projected["table_contact"])
        )

        samples.append(
            PoseDrivenSample(
                object_id=config.object_id,
                label_idx=label_idx,
                object_pose_init=[float(v) for v in init_pose],
                object_pose_goal=[float(v) for v in projected_object_pose_goal],
                wrist_pose_goal_world=[float(v) for v in projected_wrist_pose_world],
                wrist_pose_goal_object=[float(v) for v in projected_wrist_pose_object],
                source_root_pose_world=[float(v) for v in source_root_pose_world],
                hand_qpos_6=[float(v) for v in projected["hand_qpos"]],
                semantic_sites_goal_world_21=flatten_sites(projected_semantic_sites_world),
                semantic_sites_goal_object_21=flatten_sites(projected_semantic_sites_object),
                source_contact_mask_12=[float(v) for v in source_contact_mask],
                contact_mask_12=[float(v) for v in projected["contact_mask_12"]],
                valid_execution=valid_execution,
                fit_error=fit_error,
            )
        )
        report.append(
            {
                "label_idx": label_idx,
                "valid_execution": valid_execution,
                "source_contact_mask_12": [int(round(v)) for v in source_contact_mask],
                "projected_contact_mask_12": [int(round(v)) for v in projected["contact_mask_12"]],
                **fit_error,
            }
        )

        previous_arm_qpos = np.asarray(projected["arm_qpos"], dtype=np.float64)
        previous_hand_qpos = np.asarray(projected["hand_qpos"], dtype=np.float64)

    save_pose_driven_samples(samples_path, samples)
    pose_driven_report_path(config).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return samples
