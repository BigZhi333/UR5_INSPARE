from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import hdbscan
import numpy as np
import torch
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial.transform import Rotation

from .kinematics import solve_arm_wrist_palm_ik
from .paths import BUNDLED_DGRASP_DIR, PROJECT_DIR, WORKSPACE_DIR, ensure_runtime_dirs
from .robot_model import RobotSceneModel
from .scene_builder import build_training_scene
from .semantics import (
    GENHAND_CONTACT_FAMILY_NAMES,
    GENHAND_ROBOT_CONTACT_CANDIDATES,
    ROBOT_WRIST_TO_PALM_OFFSET_M,
    SEMANTIC_CONTACT_DISTANCE_THRESHOLDS_M,
    contact_mask_16_to_12,
    flatten_sites,
    mano_semantic_sites_from_keypoints,
    semantic_frame_from_sites,
    unflatten_sites,
)
from .task_config import TaskConfig
from .utils import (
    apply_local_pose_delta,
    compose_pose7,
    inverse_pose7,
    matrix_to_pose7,
    normalize,
    quat_wxyz_to_matrix,
    rotation_angle_deg,
    transform_points,
)

PROJECTION_SETTLE_STEPS = 48
PROJECTION_SOFT_MAX_PENETRATION_M = 0.0035
PROJECTION_SOFT_TOTAL_PENETRATION_M = 0.0080
PROJECTION_SOFT_OBJECT_TRANSLATION_DRIFT_M = 0.040
PROJECTION_SOFT_OBJECT_ROTATION_DRIFT_DEG = 12.0
PROJECTION_HARD_MAX_PENETRATION_M = 0.0045
PROJECTION_HARD_TOTAL_PENETRATION_M = 0.0100
PROJECTION_HARD_OBJECT_TRANSLATION_DRIFT_M = 0.050
PROJECTION_HARD_OBJECT_ROTATION_DRIFT_DEG = 18.0
PROJECTION_HOLD_TEST_STEPS = 36
PROJECTION_HOLD_PRELOAD_STEPS = 10
PROJECTION_HOLD_ARM_KP_SCALE = 2.5
PROJECTION_HOLD_SUPPORT_FORCE_SCALE = 10.0
PROJECTION_HOLD_MAX_DROP_M = 0.028
PROJECTION_HOLD_MAX_TRANSLATION_M = 0.040
PROJECTION_HOLD_MAX_ROTATION_DEG = 16.0
PROJECTION_CYLINDER_PALM_CLEARANCE_TARGET_M = 0.014
PROJECTION_CYLINDER_PALM_CLEARANCE_TOL_M = 0.010
PROJECTION_CYLINDER_HOLD_PALM_CLEARANCE_TOL_M = 0.015
PROJECTION_CYLINDER_OPPOSITION_TARGET_COS = -0.20
PROJECTION_CYLINDER_HOLD_OPPOSITION_TARGET_COS = -0.10
GENHAND_ANCHOR_POSITION_WEIGHT = 6.5
GENHAND_ACTIVE_DISTANCE_WEIGHT = 9.0
GENHAND_DIRECTION_WEIGHT = 2.2
GENHAND_PALM_DIRECTION_WEIGHT = 1.6
GENHAND_PALM_POSITION_WEIGHT = 1.8
GENHAND_WRIST_POSITION_WEIGHT = 0.6
GENHAND_PENETRATION_WEIGHT = 18.0
GENHAND_TABLE_WEIGHT = 10.0
GENHAND_REACH_OBJECT_WEIGHT = 2.5
GENHAND_REACH_BASE_WEIGHT = 2.5
GENHAND_REACH_DOWNWARD_WEIGHT = 2.0
GENHAND_ARM_TABLE_WEIGHT = 8.0
GENHAND_WRIST_REG_WEIGHT = 0.30
GENHAND_HAND_REG_WEIGHT = 0.10
GENHAND_MAX_NFEV = 56
GENHAND_CONTACT_SURFACE_MAX_DIST_M = 0.030
GENHAND_ANTIPODAL_MIN_COS = 0.18
GENHAND_PACKED_MAX_PER_FINGER = 16
GENHAND_PACKED_FALLBACK_PER_FINGER = 8
GENHAND_FRICTION_COEFF = 0.7
GENHAND_FORCE_CLOSURE_SIGMA_WEIGHT = 0.10
GENHAND_FORCE_CLOSURE_SPREAD_WEIGHT = 0.05
GENHAND_HDBSCAN_NORMAL_MIN_CLUSTER = 10
GENHAND_HDBSCAN_POSITION_MIN_CLUSTER = 4
GENHAND_FC_MAX_ITERS = 120
GENHAND_GF_MAX_ITERS = 60
GENHAND_FC_LR = 1.0e-3
GENHAND_GF_LR = 5.0e-3
GENHAND_CONTACT_TARGET_TOL_M = 0.010
GENHAND_HAND_SEED_RELAXED_COEFF = np.array([0.18, 0.22, 0.18, 0.18, 0.18, 0.16], dtype=np.float64)
GENHAND_HAND_SEED_BALANCED_COEFF = np.array([0.34, 0.42, 0.30, 0.30, 0.30, 0.26], dtype=np.float64)
GENHAND_HAND_SEED_AGGRESSIVE_COEFF = np.array([0.48, 0.60, 0.42, 0.42, 0.42, 0.36], dtype=np.float64)
GENHAND_HAND_SEED_THUMB_LEAD_COEFF = np.array([0.42, 0.56, 0.24, 0.24, 0.24, 0.20], dtype=np.float64)
GENHAND_ROOT_DIR = WORKSPACE_DIR / "Generalised_Human_Grasp_Kinematic_Retargeting"
GENHAND_MANOPTH_DIR = GENHAND_ROOT_DIR / "network" / "manopth"
REACHABILITY_TARGET_WRIST_XY_M = np.array([0.66, -0.52], dtype=np.float64)
REACHABILITY_MAX_SHIFT_XY_M = np.array([0.10, 0.14], dtype=np.float64)
REACHABILITY_TABLE_MARGIN_XY_M = np.array([0.06, 0.08], dtype=np.float64)
REACHABILITY_MIN_OBJECT_FACING_COS = 0.45
REACHABILITY_MAX_BASE_FACING_COS = 0.20
REACHABILITY_MIN_DOWNWARD_APPROACH = 0.15
REACHABILITY_MIN_ARM_TABLE_CLEARANCE_M = 0.045
DGRASP_MANO_TRANSLATION_OFFSET = np.array([0.09566993, 0.00638343, 0.00618631], dtype=np.float64)
GENHAND_MANO_CONTACT_ZONES = {
    "index": [350, 355, 329, 332, 349, 354, 343, 327, 351, 353, 348, 328, 347, 326, 337, 346],
    "middle": [462, 439, 467, 442, 461, 466, 455, 437, 459, 438, 465, 460, 436, 463, 449, 434],
    "ring": [573, 550, 578, 577, 553, 572, 566, 570, 576, 549, 548, 547, 574, 571, 546, 569],
    "little": [690, 689, 667, 695, 670, 694, 664, 683, 688, 665, 693, 691, 666, 687, 661, 663],
    "thumb": [743, 738, 768, 740, 763, 737, 766, 767, 764, 734, 735, 762, 745, 761, 717, 765],
}
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
    physical_contact_mask_12: list[float]
    proximity_contact_mask_12: list[float]
    contact_proximity_score_12: list[float]
    contact_distance_12_m: list[float]
    contact_mask_12: list[float]
    valid_execution: bool
    fit_error: dict[str, Any]

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
            "physical_contact_mask_12": self.physical_contact_mask_12,
            "proximity_contact_mask_12": self.proximity_contact_mask_12,
            "contact_proximity_score_12": self.contact_proximity_score_12,
            "contact_distance_12_m": self.contact_distance_12_m,
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
        physical_contact_mask_12 = payload.get("physical_contact_mask_12")
        if physical_contact_mask_12 is None:
            physical_contact_mask_12 = contact_mask_12
        proximity_contact_mask_12 = payload.get("proximity_contact_mask_12")
        if proximity_contact_mask_12 is None:
            proximity_contact_mask_12 = contact_mask_12
        contact_proximity_score_12 = payload.get("contact_proximity_score_12")
        if contact_proximity_score_12 is None:
            contact_proximity_score_12 = contact_mask_12
        contact_distance_12_m = payload.get("contact_distance_12_m")
        if contact_distance_12_m is None:
            contact_distance_12_m = [0.0 if float(v) > 0.5 else 1.0 for v in contact_mask_12]
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
            physical_contact_mask_12=[float(v) for v in physical_contact_mask_12],
            proximity_contact_mask_12=[float(v) for v in proximity_contact_mask_12],
            contact_proximity_score_12=[float(v) for v in contact_proximity_score_12],
            contact_distance_12_m=[float(v) for v in contact_distance_12_m],
            contact_mask_12=[float(v) for v in contact_mask_12],
            valid_execution=bool(payload["valid_execution"]),
            fit_error={str(k): _coerce_fit_error_value(v) for k, v in payload["fit_error"].items()},
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


def _coerce_fit_error_value(value: Any) -> Any:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


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


def _compute_reachability_alignment_delta(
    config: TaskConfig,
    goal_pose: np.ndarray,
    init_pose: np.ndarray,
    source_wrist_pose_world: np.ndarray,
) -> np.ndarray:
    goal_pose = np.asarray(goal_pose, dtype=np.float64)
    init_pose = np.asarray(init_pose, dtype=np.float64)
    source_wrist_pose_world = np.asarray(source_wrist_pose_world, dtype=np.float64)

    desired_delta_xy = REACHABILITY_TARGET_WRIST_XY_M - source_wrist_pose_world[:2]
    desired_delta_xy = np.clip(desired_delta_xy, -REACHABILITY_MAX_SHIFT_XY_M, REACHABILITY_MAX_SHIFT_XY_M)

    table_center_xy = np.asarray(config.table_center[:2], dtype=np.float64)
    table_half_xy = np.asarray(config.table_size[:2], dtype=np.float64)
    table_min_xy = table_center_xy - table_half_xy + REACHABILITY_TABLE_MARGIN_XY_M
    table_max_xy = table_center_xy + table_half_xy - REACHABILITY_TABLE_MARGIN_XY_M

    lower_xy = np.maximum(table_min_xy - goal_pose[:2], table_min_xy - init_pose[:2])
    upper_xy = np.minimum(table_max_xy - goal_pose[:2], table_max_xy - init_pose[:2])
    delta_xy = np.minimum(np.maximum(desired_delta_xy, lower_xy), upper_xy)
    return np.array([delta_xy[0], delta_xy[1], 0.0], dtype=np.float64)


def _reachability_metrics_world(
    object_pose_goal: np.ndarray,
    semantic_sites_world: np.ndarray,
    arm_table_clearance_m: np.ndarray,
) -> dict[str, float]:
    object_pose_goal = np.asarray(object_pose_goal, dtype=np.float64)
    semantic_sites_world = np.asarray(semantic_sites_world, dtype=np.float64).reshape(7, 3)
    arm_table_clearance_m = np.asarray(arm_table_clearance_m, dtype=np.float64).reshape(-1)

    wrist = semantic_sites_world[0]
    palm = semantic_sites_world[1]
    approach = normalize(palm - wrist)
    wrist_to_object = normalize(object_pose_goal[:3] - wrist)
    object_to_base = normalize(-object_pose_goal[:3])

    object_facing_cos = float(np.dot(approach, wrist_to_object))
    base_facing_cos = float(np.dot(approach, object_to_base))
    downward_component = float(-approach[2])
    min_arm_clearance = float(np.min(arm_table_clearance_m)) if arm_table_clearance_m.size > 0 else 1.0

    return {
        "object_facing_cos": object_facing_cos,
        "base_facing_cos": base_facing_cos,
        "downward_component": downward_component,
        "min_arm_table_clearance_m": min_arm_clearance,
    }


def wrist_pose_to_target_sites(wrist_pose: np.ndarray) -> np.ndarray:
    wrist_pose = np.asarray(wrist_pose, dtype=np.float64)
    wrist = wrist_pose[:3]
    rotation = quat_wxyz_to_matrix(wrist_pose[3:])
    palm = wrist + rotation[:, 2] * ROBOT_WRIST_TO_PALM_OFFSET_M
    return np.vstack([wrist, palm])


def _load_raw_label_block(config: TaskConfig) -> dict[str, Any]:
    candidate_paths: list[Path] = []
    if BUNDLED_DGRASP_DIR.exists():
        candidate_paths.extend(BUNDLED_DGRASP_DIR.rglob("dexycb_train_labels.pkl"))
    candidate_paths.extend(WORKSPACE_DIR.rglob("dexycb_train_labels.pkl"))

    seen: set[Path] = set()
    for labels_path in candidate_paths:
        labels_path = Path(labels_path).resolve()
        if labels_path in seen:
            continue
        seen.add(labels_path)
        labels = joblib.load(labels_path)
        if config.object_id not in labels:
            continue
        block = labels[config.object_id]
        if not isinstance(block, dict):
            raise TypeError(f"Unexpected label block type: {type(block)}")
        return block
    raise KeyError(f"Object id {config.object_id} was not found in any dexycb_train_labels.pkl")


def _contact_group_presence(contact_mask_12: np.ndarray) -> np.ndarray:
    contact_mask = np.asarray(contact_mask_12, dtype=np.float64).reshape(12)
    return np.array(
        [
            float(contact_mask[0] > 0.5),
            float(contact_mask[1:4].max() > 0.5),
            float(contact_mask[4:6].max() > 0.5),
            float(contact_mask[6:8].max() > 0.5),
            float(contact_mask[8:10].max() > 0.5),
            float(contact_mask[10:12].max() > 0.5),
        ],
        dtype=np.float64,
    )


def _contact_group_count(contact_mask_12: np.ndarray) -> int:
    return int(np.sum(_contact_group_presence(contact_mask_12) > 0.5))


def _has_thumb_opposition(contact_mask_12: np.ndarray) -> bool:
    group_presence = _contact_group_presence(contact_mask_12)
    thumb_active = bool(group_presence[1] > 0.5)
    opposing_finger_active = bool(np.any(group_presence[2:] > 0.5))
    return thumb_active and opposing_finger_active


def _run_projection_hold_test(
    runtime: RobotSceneModel,
    config: TaskConfig,
    ctrl_target: np.ndarray,
) -> dict[str, Any]:
    start_object_pose = runtime.get_object_pose()
    start_frame = quat_wxyz_to_matrix(start_object_pose[3:])

    runtime.set_arm_hold_mode(True, arm_kp_scale=PROJECTION_HOLD_ARM_KP_SCALE)
    support_force_world = np.array(
        [0.0, 0.0, float(config.object_mass_kg) * PROJECTION_HOLD_SUPPORT_FORCE_SCALE],
        dtype=np.float64,
    )
    for _ in range(PROJECTION_HOLD_PRELOAD_STEPS):
        runtime.step(ctrl_target, config.frame_skip, arm_support_force_world=support_force_world)
    runtime.set_table_height(0.0)
    for _ in range(PROJECTION_HOLD_TEST_STEPS):
        runtime.step(ctrl_target, config.frame_skip, arm_support_force_world=support_force_world)

    final_object_pose = runtime.get_object_pose()
    final_contact_diag = runtime.get_contact_diagnostics_12()
    object_drop_m = float(max(0.0, start_object_pose[2] - final_object_pose[2]))
    object_translation_m = float(np.linalg.norm(final_object_pose[:3] - start_object_pose[:3]))
    object_rotation_deg = float(rotation_angle_deg(quat_wxyz_to_matrix(final_object_pose[3:]), start_frame))
    return {
        "object_pose": final_object_pose.copy(),
        "semantic_sites_world": runtime.get_semantic_sites_world(),
        "wrist_pose_world": wrist_pose_from_semantic_sites(runtime.get_semantic_sites_world()),
        "contact_diag": final_contact_diag,
        "object_drop_m": object_drop_m,
        "object_translation_m": object_translation_m,
        "object_rotation_deg": object_rotation_deg,
    }


def _pose_delta_local(source_pose: np.ndarray, target_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_pose = np.asarray(source_pose, dtype=np.float64)
    target_pose = np.asarray(target_pose, dtype=np.float64)
    source_rotation = quat_wxyz_to_matrix(source_pose[3:])
    target_rotation = quat_wxyz_to_matrix(target_pose[3:])
    translation_local = source_rotation.T @ (target_pose[:3] - source_pose[:3])
    delta_rotation_local = source_rotation.T @ target_rotation
    rotvec_local = Rotation.from_matrix(delta_rotation_local).as_rotvec()
    return translation_local, rotvec_local


def _genhand_hand_seed_from_coeff(runtime: RobotSceneModel, coeff: np.ndarray) -> np.ndarray:
    coeff = np.asarray(coeff, dtype=np.float64).reshape(runtime.hand_upper.shape)
    coeff = np.clip(coeff, 0.0, 1.0)
    seed = runtime.hand_lower + coeff * (runtime.hand_upper - runtime.hand_lower)
    return runtime.clamp_hand(seed)


def _genhand_hand_seed_variants(runtime: RobotSceneModel, base_hand_qpos: np.ndarray) -> dict[str, np.ndarray]:
    base = runtime.clamp_hand(np.asarray(base_hand_qpos, dtype=np.float64))
    relaxed = np.maximum(base, _genhand_hand_seed_from_coeff(runtime, GENHAND_HAND_SEED_RELAXED_COEFF))
    balanced = np.maximum(base, _genhand_hand_seed_from_coeff(runtime, GENHAND_HAND_SEED_BALANCED_COEFF))
    aggressive = np.maximum(base, _genhand_hand_seed_from_coeff(runtime, GENHAND_HAND_SEED_AGGRESSIVE_COEFF))
    thumb_lead = np.maximum(base, _genhand_hand_seed_from_coeff(runtime, GENHAND_HAND_SEED_THUMB_LEAD_COEFF))
    return {
        "relaxed": runtime.clamp_hand(relaxed),
        "balanced": runtime.clamp_hand(balanced),
        "aggressive": runtime.clamp_hand(aggressive),
        "thumb_lead": runtime.clamp_hand(thumb_lead),
    }


def _cylinder_hand_wrap_seed(runtime: RobotSceneModel, strength: float = 1.0) -> np.ndarray:
    relaxed = _genhand_hand_seed_from_coeff(runtime, GENHAND_HAND_SEED_RELAXED_COEFF)
    balanced = _genhand_hand_seed_from_coeff(runtime, GENHAND_HAND_SEED_BALANCED_COEFF)
    aggressive = _genhand_hand_seed_from_coeff(runtime, GENHAND_HAND_SEED_AGGRESSIVE_COEFF)
    blend = float(np.clip(strength - 1.0, 0.0, 0.5) / 0.5)
    base = (1.0 - blend) * balanced + blend * aggressive
    base = 0.15 * relaxed + 0.85 * base
    return runtime.clamp_hand(base)


def _cylinder_grasp_seed_states(
    runtime: RobotSceneModel,
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_wrist_pose_world: np.ndarray,
    source_semantic_sites_world: np.ndarray,
    base_hand_qpos: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if config.object_geom_type != "cylinder":
        return []

    object_pose_goal = np.asarray(object_pose_goal, dtype=np.float64)
    object_center = object_pose_goal[:3]
    object_rotation = quat_wxyz_to_matrix(object_pose_goal[3:])
    cylinder_axis = normalize(object_rotation[:, 2])
    if np.linalg.norm(cylinder_axis) < 1e-8:
        return []

    radius = float(config.object_dims_m[0])
    height = float(config.object_dims_m[1])
    palm_source = np.asarray(source_semantic_sites_world[1], dtype=np.float64)
    radial = palm_source - object_center - cylinder_axis * float(np.dot(palm_source - object_center, cylinder_axis))
    if np.linalg.norm(radial) < 1e-8:
        radial = object_rotation[:, 0]
    radial = normalize(radial)
    tangent = normalize(np.cross(cylinder_axis, radial))
    if np.linalg.norm(tangent) < 1e-8:
        tangent = object_rotation[:, 1]
    tangent = normalize(tangent)

    palm_height = float(np.clip(np.dot(palm_source - object_center, cylinder_axis), -0.28 * height, 0.28 * height))
    seed_specs = [
        (0.012, 0.0, 0.0, 1.05),
        (0.016, 0.0, 0.0, 1.20),
        (0.018, 0.010, 0.0, 1.15),
        (0.018, -0.010, 0.0, 1.15),
        (0.016, 0.0, 0.014, 1.10),
        (0.016, 0.0, -0.014, 1.10),
    ]
    across_candidates = [
        tangent,
        -tangent,
        cylinder_axis,
        -cylinder_axis,
    ]

    seeds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for radial_clearance, tangent_offset, height_offset, squeeze_strength in seed_specs:
        palm_target = (
            object_center
            + cylinder_axis * (palm_height + height_offset)
            + radial * (radius + radial_clearance)
            + tangent * tangent_offset
        )
        approach = -radial
        wrap_seed = _cylinder_hand_wrap_seed(runtime, strength=squeeze_strength)
        blended_hand = runtime.clamp_hand(0.20 * np.asarray(base_hand_qpos, dtype=np.float64) + 0.80 * wrap_seed)
        for across_hint in across_candidates:
            across = across_hint - approach * float(np.dot(across_hint, approach))
            if np.linalg.norm(across) < 1e-8:
                continue
            across = normalize(across)
            normal = normalize(np.cross(across, approach))
            if np.linalg.norm(normal) < 1e-8:
                continue
            across = normalize(np.cross(approach, normal))
            target_wrist_matrix = np.eye(4, dtype=np.float64)
            target_wrist_matrix[:3, :3] = np.column_stack((across, normal, approach))
            target_wrist_matrix[:3, 3] = palm_target - approach * ROBOT_WRIST_TO_PALM_OFFSET_M
            target_wrist_pose = matrix_to_pose7(target_wrist_matrix)
            translation_local, rotvec_local = _pose_delta_local(source_wrist_pose_world, target_wrist_pose)
            seeds.append((translation_local, rotvec_local, blended_hand))
    return seeds


def _cylinder_object_frame_seed_poses(
    runtime: RobotSceneModel,
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_keypoints_obj: np.ndarray,
    base_hand_qpos: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if config.object_geom_type != "cylinder":
        return []

    object_pose_goal = np.asarray(object_pose_goal, dtype=np.float64)
    object_center = object_pose_goal[:3]
    object_rotation = quat_wxyz_to_matrix(object_pose_goal[3:])
    cylinder_axis_obj = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cylinder_axis_world = normalize(object_rotation[:, 2])
    radius = float(config.object_dims_m[0])
    height = float(config.object_dims_m[1])

    contact_targets_obj = _human_contact_target_points_obj(source_keypoints_obj)
    palm_obj = np.asarray(contact_targets_obj[0], dtype=np.float64)
    thumb_targets = np.asarray(contact_targets_obj[1:4], dtype=np.float64)
    finger_targets = np.asarray(contact_targets_obj[4:], dtype=np.float64)
    thumb_center_obj = np.mean(thumb_targets, axis=0)
    finger_center_obj = np.mean(finger_targets, axis=0)

    palm_radial_obj = palm_obj - cylinder_axis_obj * float(np.dot(palm_obj, cylinder_axis_obj))
    if np.linalg.norm(palm_radial_obj) < 1e-8:
        palm_radial_obj = 0.5 * (thumb_center_obj + finger_center_obj)
        palm_radial_obj = palm_radial_obj - cylinder_axis_obj * float(np.dot(palm_radial_obj, cylinder_axis_obj))
    if np.linalg.norm(palm_radial_obj) < 1e-8:
        palm_radial_obj = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    palm_radial_obj = normalize(palm_radial_obj)

    opposition_obj = finger_center_obj - thumb_center_obj
    opposition_obj = opposition_obj - cylinder_axis_obj * float(np.dot(opposition_obj, cylinder_axis_obj))
    if np.linalg.norm(opposition_obj) < 1e-8:
        opposition_obj = np.cross(cylinder_axis_obj, palm_radial_obj)
    if np.linalg.norm(opposition_obj) < 1e-8:
        opposition_obj = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    opposition_obj = normalize(opposition_obj)

    palm_height_obj = float(np.clip(palm_obj[2], -0.28 * height, 0.28 * height))
    clearance_values = [0.012, 0.018]
    height_offsets = [0.0]
    roll_offsets_deg = [-12.0, 0.0, 12.0]

    seed_variants = _genhand_hand_seed_variants(runtime, base_hand_qpos)
    base_wrap = 0.30 * seed_variants["relaxed"] + 0.70 * seed_variants["balanced"]
    stronger_wrap = 0.20 * seed_variants["balanced"] + 0.80 * seed_variants["aggressive"]

    seeds: list[tuple[np.ndarray, np.ndarray]] = []
    for clearance in clearance_values:
        for height_offset in height_offsets:
            palm_target_obj = np.array(
                [
                    palm_radial_obj[0] * (radius + clearance),
                    palm_radial_obj[1] * (radius + clearance),
                    palm_height_obj + height_offset,
                ],
                dtype=np.float64,
            )
            approach_obj = -palm_radial_obj
            across_candidates_obj = [
                opposition_obj,
                -opposition_obj,
            ]
            for across_hint_obj in across_candidates_obj:
                across_obj = across_hint_obj - approach_obj * float(np.dot(across_hint_obj, approach_obj))
                if np.linalg.norm(across_obj) < 1e-8:
                    continue
                across_obj = normalize(across_obj)
                normal_obj = normalize(np.cross(across_obj, approach_obj))
                if np.linalg.norm(normal_obj) < 1e-8:
                    continue
                across_obj = normalize(np.cross(approach_obj, normal_obj))
                base_rotation_obj = np.column_stack((across_obj, normal_obj, approach_obj))
                for roll_deg in roll_offsets_deg:
                    roll_rotation = Rotation.from_rotvec(np.deg2rad(roll_deg) * approach_obj).as_matrix()
                    wrist_rotation_obj = base_rotation_obj @ roll_rotation
                    wrist_rotation_world = object_rotation @ wrist_rotation_obj
                    palm_target_world = object_center + object_rotation @ palm_target_obj
                    wrist_target_world = palm_target_world - wrist_rotation_world[:, 2] * ROBOT_WRIST_TO_PALM_OFFSET_M
                    wrist_matrix = np.eye(4, dtype=np.float64)
                    wrist_matrix[:3, :3] = wrist_rotation_world
                    wrist_matrix[:3, 3] = wrist_target_world
                    hand_qpos = stronger_wrap if abs(roll_deg) <= 8.0 else base_wrap
                    seeds.append((matrix_to_pose7(wrist_matrix), hand_qpos.copy()))
    return seeds


def _cylinder_grasp_geometry_metrics(
    config: TaskConfig,
    object_pose: np.ndarray,
    semantic_sites_world: np.ndarray,
    source_contact_mask_12: np.ndarray,
) -> tuple[float, float]:
    if config.object_geom_type != "cylinder":
        return 0.0, -1.0

    object_pose = np.asarray(object_pose, dtype=np.float64)
    object_center = object_pose[:3]
    object_rotation = quat_wxyz_to_matrix(object_pose[3:])
    cylinder_axis = normalize(object_rotation[:, 2])
    radius = float(config.object_dims_m[0])

    palm = np.asarray(semantic_sites_world[1], dtype=np.float64)
    palm_radial = palm - object_center - cylinder_axis * float(np.dot(palm - object_center, cylinder_axis))
    palm_clearance_error_m = abs(float(np.linalg.norm(palm_radial)) - (radius + PROJECTION_CYLINDER_PALM_CLEARANCE_TARGET_M))

    group_presence = _contact_group_presence(source_contact_mask_12)
    active_tip_indices = []
    if group_presence[2] > 0.5:
        active_tip_indices.append(3)
    if group_presence[3] > 0.5:
        active_tip_indices.append(4)
    if group_presence[4] > 0.5:
        active_tip_indices.append(5)
    if group_presence[5] > 0.5:
        active_tip_indices.append(6)
    if not active_tip_indices:
        active_tip_indices = [3, 4, 5, 6]

    thumb_tip = np.asarray(semantic_sites_world[2], dtype=np.float64)
    finger_mean = np.mean(np.asarray(semantic_sites_world[active_tip_indices], dtype=np.float64), axis=0)
    thumb_radial = thumb_tip - object_center - cylinder_axis * float(np.dot(thumb_tip - object_center, cylinder_axis))
    finger_radial = finger_mean - object_center - cylinder_axis * float(np.dot(finger_mean - object_center, cylinder_axis))
    if np.linalg.norm(thumb_radial) < 1e-8 or np.linalg.norm(finger_radial) < 1e-8:
        opposition_cos = 1.0
    else:
        opposition_cos = float(np.dot(normalize(thumb_radial), normalize(finger_radial)))
    return palm_clearance_error_m, opposition_cos


def _human_contact_target_points_obj(source_keypoints_obj: np.ndarray) -> np.ndarray:
    keypoints = np.asarray(source_keypoints_obj, dtype=np.float64).reshape(21, 3)
    targets = np.zeros((12, 3), dtype=np.float64)
    semantic_sites = mano_semantic_sites_from_keypoints(keypoints)
    targets[0] = semantic_sites[1]

    def midpoint(a: int, b: int) -> np.ndarray:
        return 0.5 * (keypoints[a] + keypoints[b])

    def merged_distal(a: int, b: int, c: int) -> np.ndarray:
        mid_center = midpoint(a, b)
        distal_center = midpoint(b, c)
        return 0.5 * (mid_center + distal_center)

    targets[1] = midpoint(17, 18)
    targets[2] = midpoint(18, 19)
    targets[3] = midpoint(19, 20)
    targets[4] = midpoint(1, 2)
    targets[5] = merged_distal(2, 3, 4)
    targets[6] = midpoint(5, 6)
    targets[7] = merged_distal(6, 7, 8)
    targets[8] = midpoint(9, 10)
    targets[9] = merged_distal(10, 11, 12)
    targets[10] = midpoint(13, 14)
    targets[11] = merged_distal(14, 15, 16)
    return targets


@lru_cache(maxsize=1)
def _locate_mano_models_dir() -> Path | None:
    candidates = [
        GENHAND_MANOPTH_DIR / "mano" / "models",
        WORKSPACE_DIR,
        PROJECT_DIR,
    ]
    for candidate in candidates:
        if (candidate / "MANO_RIGHT.pkl").exists() and (candidate / "MANO_LEFT.pkl").exists():
            return candidate
        if (candidate / "MANO_RIGHT.pkl").exists():
            return candidate
    matches = list(WORKSPACE_DIR.rglob("MANO_RIGHT.pkl"))
    if matches:
        return matches[0].parent
    return None


@lru_cache(maxsize=1)
def _load_genhand_mano_layer():
    if not GENHAND_MANOPTH_DIR.exists():
        return None
    mano_models_dir = _locate_mano_models_dir()
    if mano_models_dir is None:
        return None
    manopth_path = str(GENHAND_MANOPTH_DIR)
    if manopth_path not in sys.path:
        sys.path.insert(0, manopth_path)
    # Vendored manopth depends on chumpy, which still expects legacy NumPy aliases.
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]
    if not hasattr(np, "complex"):
        np.complex = complex  # type: ignore[attr-defined]
    if not hasattr(np, "object"):
        np.object = object  # type: ignore[attr-defined]
    if not hasattr(np, "str"):
        np.str = str  # type: ignore[attr-defined]
    if not hasattr(np, "unicode"):
        np.unicode = str  # type: ignore[attr-defined]
    try:
        from manopth.manolayer import ManoLayer  # type: ignore
    except Exception:
        return None
    layer = ManoLayer(
        mano_root=str(mano_models_dir),
        use_pca=False,
        ncomps=45,
        side="right",
        flat_hand_mean=True,
    )
    layer.eval()
    return layer


@lru_cache(maxsize=1)
def _load_genhand_fc_loss():
    root_path = str(GENHAND_ROOT_DIR)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    try:
        from optimisation.loss import FCLoss  # type: ignore
    except Exception:
        return None
    try:
        return FCLoss(device="cpu")
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_genhand_icp_module():
    root_path = str(GENHAND_ROOT_DIR)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    try:
        from optimisation import icp  # type: ignore
    except Exception:
        return None
    return icp


def _dgrasp_qpos_world_to_mano_pose_translation(dgrasp_qpos_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(dgrasp_qpos_world, dtype=np.float64).reshape(51)
    translation = qpos[:3] - DGRASP_MANO_TRANSLATION_OFFSET
    global_axisang = Rotation.from_euler("XYZ", qpos[3:6]).as_rotvec()
    joint_axisang = Rotation.from_euler("XYZ", qpos[6:].reshape(15, 3)).as_rotvec()
    # In D-Grasp the ring/little finger joint blocks are swapped relative to MANO.
    temp = joint_axisang[6:9].copy()
    joint_axisang[6:9] = joint_axisang[9:12]
    joint_axisang[9:12] = temp
    pose = np.concatenate([global_axisang, joint_axisang.reshape(-1)])
    return pose.astype(np.float64), translation.astype(np.float64)


def _mesh_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(vertices, dtype=np.float64)
    if faces.size == 0:
        return normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return normals / norms


def _mano_vertices_normals_from_dgrasp_qpos_world(
    dgrasp_qpos_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    mano_layer = _load_genhand_mano_layer()
    if mano_layer is None:
        return None
    pose, translation = _dgrasp_qpos_world_to_mano_pose_translation(dgrasp_qpos_world)
    pose_tensor = torch.tensor(pose, dtype=torch.float32).view(1, -1)
    trans_tensor = torch.tensor(translation, dtype=torch.float32).view(1, 3)
    with torch.no_grad():
        verts, joints, *_ = mano_layer(pose_tensor, th_trans=trans_tensor)
    verts_world = np.asarray(verts[0].detach().cpu().numpy(), dtype=np.float64)
    joints_world = np.asarray(joints[0].detach().cpu().numpy(), dtype=np.float64)
    faces = np.asarray(mano_layer.th_faces.detach().cpu().numpy(), dtype=np.int64)
    normals_world = _mesh_vertex_normals(verts_world, faces)
    return verts_world, normals_world, joints_world


def _contact_tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normalize(np.asarray(normal, dtype=np.float64))
    if abs(float(normal[2])) < 0.9:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    tangent1 = normalize(np.cross(normal, ref))
    tangent2 = normalize(np.cross(normal, tangent1))
    return tangent1, tangent2


def _force_closure_combo_metrics(points_obj: np.ndarray, normals_obj: np.ndarray) -> tuple[float, float]:
    points_obj = np.asarray(points_obj, dtype=np.float64).reshape(-1, 3)
    normals_obj = np.asarray(normals_obj, dtype=np.float64).reshape(-1, 3)
    if points_obj.shape[0] < 2:
        return 0.0, 0.0
    wrench_columns: list[np.ndarray] = []
    for point, normal in zip(points_obj, normals_obj, strict=True):
        tangent1, tangent2 = _contact_tangent_basis(normal)
        for sx, sy in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
            force_dir = normalize(normal + GENHAND_FRICTION_COEFF * (sx * tangent1 + sy * tangent2))
            torque = np.cross(point, force_dir)
            wrench_columns.append(np.concatenate([force_dir, torque]))
    if not wrench_columns:
        return 0.0, 0.0
    grasp_matrix = np.column_stack(wrench_columns)
    singular_values = np.linalg.svd(grasp_matrix, compute_uv=False)
    sigma_min = float(singular_values[-1]) if singular_values.size >= 6 else 0.0
    if points_obj.shape[0] == 1:
        spread = 0.0
    else:
        pairwise = np.linalg.norm(points_obj[:, None, :] - points_obj[None, :, :], axis=2)
        spread = float(np.min(pairwise[np.triu_indices(points_obj.shape[0], k=1)]))
    return sigma_min, spread


def _human_finger_directions_obj(source_keypoints_obj: np.ndarray) -> np.ndarray:
    keypoints = np.asarray(source_keypoints_obj, dtype=np.float64).reshape(21, 3)
    finger_pairs = (
        (17, 20),  # thumb
        (1, 4),    # index
        (5, 8),    # middle
        (9, 12),   # ring
        (13, 16),  # little
    )
    directions = []
    for start_idx, end_idx in finger_pairs:
        direction = normalize(keypoints[end_idx] - keypoints[start_idx])
        directions.append(direction)
    return np.asarray(directions, dtype=np.float64)


def _project_point_to_box_surface(point_obj: np.ndarray, dims_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half_extents = 0.5 * np.asarray(dims_m, dtype=np.float64)
    point_obj = np.asarray(point_obj, dtype=np.float64).reshape(3)
    best_distance = float("inf")
    best_point = np.zeros(3, dtype=np.float64)
    best_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for axis in range(3):
        for sign in (-1.0, 1.0):
            candidate = np.clip(point_obj, -half_extents, half_extents)
            candidate[axis] = sign * half_extents[axis]
            distance = float(np.linalg.norm(candidate - point_obj))
            if distance < best_distance:
                best_distance = distance
                best_point = candidate
                best_normal = np.zeros(3, dtype=np.float64)
                best_normal[axis] = sign
    return best_point, best_normal


def _project_point_to_cylinder_surface(point_obj: np.ndarray, dims_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = float(dims_m[0])
    half_height = 0.5 * float(dims_m[1])
    point_obj = np.asarray(point_obj, dtype=np.float64).reshape(3)
    xy = point_obj[:2]
    xy_norm = float(np.linalg.norm(xy))

    if xy_norm > 1e-8:
        radial_dir = xy / xy_norm
    else:
        radial_dir = np.array([1.0, 0.0], dtype=np.float64)

    side_point = np.array(
        [
            radial_dir[0] * radius,
            radial_dir[1] * radius,
            float(np.clip(point_obj[2], -half_height, half_height)),
        ],
        dtype=np.float64,
    )
    side_normal = np.array([radial_dir[0], radial_dir[1], 0.0], dtype=np.float64)

    if xy_norm > radius:
        cap_xy = radial_dir * radius
    else:
        cap_xy = xy.copy()

    top_point = np.array([cap_xy[0], cap_xy[1], half_height], dtype=np.float64)
    bottom_point = np.array([cap_xy[0], cap_xy[1], -half_height], dtype=np.float64)
    candidates = (
        (side_point, side_normal),
        (top_point, np.array([0.0, 0.0, 1.0], dtype=np.float64)),
        (bottom_point, np.array([0.0, 0.0, -1.0], dtype=np.float64)),
    )
    best_point = side_point
    best_normal = side_normal
    best_distance = float("inf")
    for candidate_point, candidate_normal in candidates:
        distance = float(np.linalg.norm(candidate_point - point_obj))
        if distance < best_distance:
            best_distance = distance
            best_point = candidate_point
            best_normal = candidate_normal
    return best_point, best_normal


def _project_targets_to_object_surface(
    config: TaskConfig,
    target_points_obj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_points_obj = np.asarray(target_points_obj, dtype=np.float64).reshape(-1, 3)
    projected_points = np.zeros_like(target_points_obj)
    projected_normals = np.zeros_like(target_points_obj)
    for idx, point in enumerate(target_points_obj):
        if config.object_geom_type == "box":
            projected_point, projected_normal = _project_point_to_box_surface(point, np.asarray(config.object_dims_m, dtype=np.float64))
        elif config.object_geom_type == "cylinder":
            projected_point, projected_normal = _project_point_to_cylinder_surface(point, np.asarray(config.object_dims_m, dtype=np.float64))
        else:
            projected_point = np.asarray(point, dtype=np.float64)
            projected_normal = normalize(projected_point)
            if np.linalg.norm(projected_normal) < 1e-8:
                projected_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        projected_points[idx] = projected_point
        projected_normals[idx] = normalize(projected_normal)
    return projected_points, projected_normals


def _box_contact_normal_obj(point_obj: np.ndarray, dims_m: np.ndarray) -> np.ndarray:
    half_extents = 0.5 * np.asarray(dims_m, dtype=np.float64)
    scaled = np.divide(
        np.asarray(point_obj, dtype=np.float64),
        np.maximum(half_extents, 1e-8),
        out=np.zeros(3, dtype=np.float64),
        where=np.maximum(half_extents, 1e-8) > 0.0,
    )
    axis = int(np.argmax(np.abs(scaled)))
    normal = np.zeros(3, dtype=np.float64)
    normal[axis] = 1.0 if scaled[axis] >= 0.0 else -1.0
    return normal


def _dense_mano_contact_candidates_obj(
    config: TaskConfig,
    source_dgrasp_qpos_world: np.ndarray,
    object_pose_goal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    mano_pack = _mano_vertices_normals_from_dgrasp_qpos_world(source_dgrasp_qpos_world)
    if mano_pack is None:
        return None
    verts_world, normals_world, _ = mano_pack
    inv_pose = inverse_pose7(object_pose_goal)
    inv_object_rotation = quat_wxyz_to_matrix(np.asarray(inv_pose[3:], dtype=np.float64))
    verts_obj = transform_points(verts_world, inv_pose)
    normals_obj = (inv_object_rotation @ normals_world.T).T

    packed_points: list[np.ndarray] = []
    packed_normals: list[np.ndarray] = []
    packed_families: list[int] = []
    for family_index, family_name in enumerate(GENHAND_CONTACT_FAMILY_NAMES):
        seed_indices = np.asarray(GENHAND_MANO_CONTACT_ZONES[family_name], dtype=np.int64)
        zone_points = verts_obj[seed_indices]
        zone_normals = np.asarray([normalize(n) for n in normals_obj[seed_indices]], dtype=np.float64)
        packed_points.extend(zone_points.tolist())
        packed_normals.extend(zone_normals.tolist())
        packed_families.extend([family_index] * len(seed_indices))

    if not packed_points:
        return None
    packed_points_arr = np.asarray(packed_points, dtype=np.float64).reshape(-1, 3)
    packed_normals_arr = np.asarray(packed_normals, dtype=np.float64).reshape(-1, 3)
    packed_family_arr = np.asarray(packed_families, dtype=np.int64).reshape(-1)

    projected_points, object_normals = _project_targets_to_object_surface(config, packed_points_arr)
    surface_distance = np.linalg.norm(packed_points_arr - projected_points, axis=1)
    antipodal_cos = np.sum((-packed_normals_arr) * object_normals, axis=1)
    candidate_scores = surface_distance + 0.020 * np.maximum(GENHAND_ANTIPODAL_MIN_COS - antipodal_cos, 0.0)
    valid_mask = (surface_distance <= GENHAND_CONTACT_SURFACE_MAX_DIST_M) & (antipodal_cos >= GENHAND_ANTIPODAL_MIN_COS)
    if np.any(valid_mask):
        base_keep_indices = np.flatnonzero(valid_mask)
        per_family_limit = GENHAND_PACKED_MAX_PER_FINGER
    else:
        fallback_count = max(20, min(40, packed_points_arr.shape[0] // 2))
        base_keep_indices = np.argsort(candidate_scores)[:fallback_count]
        per_family_limit = GENHAND_PACKED_FALLBACK_PER_FINGER

    balanced_keep: list[int] = []
    for family_index in range(len(GENHAND_CONTACT_FAMILY_NAMES)):
        family_indices = base_keep_indices[packed_family_arr[base_keep_indices] == family_index]
        if family_indices.size == 0:
            continue
        family_order = family_indices[np.argsort(candidate_scores[family_indices])]
        balanced_keep.extend(family_order[:per_family_limit].tolist())

    if balanced_keep:
        keep_indices = np.asarray(
            sorted(set(balanced_keep), key=lambda idx: float(candidate_scores[idx])),
            dtype=np.int64,
        )
    else:
        keep_indices = np.asarray(base_keep_indices, dtype=np.int64)

    return (
        np.asarray(projected_points[keep_indices], dtype=np.float64),
        np.asarray(object_normals[keep_indices], dtype=np.float64),
        np.asarray(packed_family_arr[keep_indices], dtype=np.int64),
        np.asarray(candidate_scores[keep_indices], dtype=np.float64),
    )


def _object_surface_query_torch(
    config: TaskConfig,
    points_obj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    points = points_obj
    if points.dim() == 2:
        points = points.unsqueeze(0)
    if config.object_geom_type == "box":
        half = torch.tensor(np.asarray(config.object_dims_m, dtype=np.float32) * 0.5, dtype=torch.float32, device=points.device)
        abs_points = torch.abs(points)
        q = abs_points - half.view(1, 1, 3)
        outside = torch.clamp(q, min=0.0)
        outside_dist = torch.linalg.norm(outside, dim=-1)
        inside_dist = torch.clamp(torch.max(q, dim=-1).values, max=0.0)
        sdf = outside_dist + inside_dist
        projected = torch.clamp(points, -half.view(1, 1, 3), half.view(1, 1, 3))
        scaled = projected / torch.clamp(half.view(1, 1, 3), min=1e-8)
        axis = torch.argmax(torch.abs(scaled), dim=-1)
        normals = torch.zeros_like(points)
        axis_sign = torch.where(
            torch.gather(projected, -1, axis.unsqueeze(-1)).squeeze(-1) >= 0.0,
            torch.ones_like(axis, dtype=torch.float32, device=points.device),
            -torch.ones_like(axis, dtype=torch.float32, device=points.device),
        )
        normals.scatter_(-1, axis.unsqueeze(-1), axis_sign.unsqueeze(-1))
        return sdf, normals
    if config.object_geom_type == "cylinder":
        radius = float(config.object_dims_m[0])
        half_height = 0.5 * float(config.object_dims_m[1])
        xy = points[..., :2]
        z = points[..., 2]
        xy_norm = torch.linalg.norm(xy, dim=-1)
        side_sdf = xy_norm - radius
        cap_sdf = torch.abs(z) - half_height
        outside = torch.stack([torch.clamp(side_sdf, min=0.0), torch.clamp(cap_sdf, min=0.0)], dim=-1)
        outside_dist = torch.linalg.norm(outside, dim=-1)
        inside_dist = torch.clamp(torch.maximum(side_sdf, cap_sdf), max=0.0)
        sdf = outside_dist + inside_dist
        radial_dir = xy / torch.clamp(xy_norm.unsqueeze(-1), min=1e-8)
        side_mask = side_sdf >= cap_sdf
        normals = torch.zeros_like(points)
        normals[..., :2] = radial_dir
        cap_sign = torch.where(z >= 0.0, 1.0, -1.0)
        cap_normals = torch.zeros_like(points)
        cap_normals[..., 2] = cap_sign
        normals = torch.where(side_mask.unsqueeze(-1), normals, cap_normals)
        return sdf, normals
    sdf = torch.zeros(points.shape[:-1], dtype=torch.float32, device=points.device)
    normals = torch.nn.functional.normalize(points, dim=-1, eps=1e-8)
    return sdf, normals


def _minimum_internal_distance_torch(points_obj: torch.Tensor) -> torch.Tensor:
    points = points_obj.view(-1, 3)
    if points.size(0) <= 1:
        return torch.zeros((), dtype=torch.float32, device=points.device)
    dist_matrix = torch.cdist(points, points, p=2)
    eye = torch.eye(points.size(0), device=points.device) * 1e6
    min_dists = torch.min(dist_matrix + eye, dim=1).values
    return torch.relu(torch.tensor(0.01, dtype=torch.float32, device=points.device) - min_dists).sum()


def _select_contact_targets_by_count_np(
    contact_points: np.ndarray,
    contact_normals: np.ndarray,
    target_count: int,
) -> np.ndarray:
    contact_points = np.asarray(contact_points, dtype=np.float64).reshape(-1, 3)
    contact_normals = np.asarray(contact_normals, dtype=np.float64).reshape(-1, 3)
    if contact_points.shape[0] == 0:
        return contact_points
    if target_count <= 1 or contact_points.shape[0] == 1:
        return contact_points[:1]
    if target_count >= contact_points.shape[0]:
        return contact_points.copy()
    if target_count == 2:
        normal_dot = contact_normals @ contact_normals.T
        normal_dot = normal_dot + np.eye(normal_dot.shape[0], dtype=np.float64) * 10.0
        i, j = np.unravel_index(np.argmin(normal_dot), normal_dot.shape)
        return np.vstack([contact_points[i], contact_points[j]])

    selected = [0]
    remaining = list(range(1, contact_points.shape[0]))
    while len(selected) < target_count and remaining:
        sel_pts = contact_points[selected]
        rem_pts = contact_points[remaining]
        dmat = np.linalg.norm(rem_pts[:, None, :] - sel_pts[None, :, :], axis=2)
        score = np.min(dmat, axis=1)
        best_idx = remaining[int(np.argmax(score))]
        selected.append(best_idx)
        remaining.remove(best_idx)
    return contact_points[selected]


def _cluster_genhand_contact_targets_obj(
    candidate_points_obj: np.ndarray,
    candidate_normals_obj: np.ndarray,
    robot_contact_count: int,
) -> np.ndarray:
    contact_points = np.asarray(candidate_points_obj, dtype=np.float64).reshape(-1, 3)
    contact_normals = np.asarray(candidate_normals_obj, dtype=np.float64).reshape(-1, 3)
    if contact_points.shape[0] == 0:
        return contact_points
    if contact_points.shape[0] < max(GENHAND_HDBSCAN_NORMAL_MIN_CLUSTER, GENHAND_HDBSCAN_POSITION_MIN_CLUSTER):
        return _select_contact_targets_by_count_np(contact_points, contact_normals, robot_contact_count)
    try:
        clusterer_n = hdbscan.HDBSCAN(min_cluster_size=GENHAND_HDBSCAN_NORMAL_MIN_CLUSTER)
        clusterer_p = hdbscan.HDBSCAN(min_cluster_size=GENHAND_HDBSCAN_POSITION_MIN_CLUSTER)
        labels_n = np.asarray(clusterer_n.fit_predict(contact_normals), dtype=np.int64)
        labels_p = np.asarray(clusterer_p.fit_predict(contact_points), dtype=np.int64)
    except Exception:
        return _select_contact_targets_by_count_np(contact_points, contact_normals, robot_contact_count)

    valid_mask = (labels_n != -1) & (labels_p != -1)
    if not np.any(valid_mask):
        return _select_contact_targets_by_count_np(contact_points, contact_normals, robot_contact_count)
    contact_points = contact_points[valid_mask]
    contact_normals = contact_normals[valid_mask]
    labels_n = labels_n[valid_mask]
    labels_p = labels_p[valid_mask]

    uniq_labels_n = np.unique(labels_n)
    uniq_labels_p = np.unique(labels_p)
    n_normal_components = int(uniq_labels_n.size)
    n_hand_contacts = int(uniq_labels_p.size)
    if n_normal_components == 0 or n_hand_contacts == 0:
        return _select_contact_targets_by_count_np(contact_points, contact_normals, robot_contact_count)

    centroid_n = {}
    centroid_p = {}
    p_collection = {}
    for label in uniq_labels_n:
        indices = np.where(labels_n == label)[0]
        p_in_n = labels_p[indices]
        valid_labels, counts = np.unique(p_in_n, return_counts=True)
        if valid_labels.size > 1:
            keep_labels = valid_labels[counts >= 2]
            if keep_labels.size > 0:
                keep_mask = np.isin(p_in_n, keep_labels)
                indices = indices[keep_mask]
                p_in_n = p_in_n[keep_mask]
        centroid_n[label] = np.mean(contact_normals[indices], axis=0)
        p_centroids = []
        p_collect: list[np.ndarray] = []
        for p_label in np.unique(p_in_n):
            indices_p = np.where(p_in_n == p_label)[0]
            p_centroids.append(np.mean(contact_points[indices[indices_p]], axis=0))
            p_collect.append(contact_points[indices[indices_p]])
        centroid_p[label] = np.asarray(p_centroids, dtype=np.float64)
        p_collection[label] = p_collect

    centroid_n_arr = np.asarray([centroid_n[label] for label in uniq_labels_n], dtype=np.float64)
    centroid_p_arr = np.vstack([centroid_p[label] for label in uniq_labels_n])
    centroid_p_n = np.vstack([centroid_p[label].mean(axis=0) for label in uniq_labels_n])

    if n_normal_components > 2:
        dist_matrix = np.linalg.norm(centroid_n_arr[:, None, :] - centroid_n_arr[None, :, :], axis=2)
        i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        smaller = i if centroid_p[uniq_labels_n[i]].shape[0] <= centroid_p[uniq_labels_n[j]].shape[0] else j
        larger = j if smaller == i else i
        contact_0 = centroid_p[uniq_labels_n[smaller]].mean(axis=0)
        contact_1 = centroid_p[uniq_labels_n[larger]].mean(axis=0)
        pca_base = np.vstack([cluster_points for label in uniq_labels_n for cluster_points in p_collection[label]])
        remaining_labels = [label for k, label in enumerate(uniq_labels_n) if k not in (i, j)]
    else:
        pair = [0, min(1, len(uniq_labels_n) - 1)]
        contact_0 = centroid_p[uniq_labels_n[pair[0]]].mean(axis=0)
        contact_1 = centroid_p[uniq_labels_n[pair[1]]].mean(axis=0)
        pca_base = np.vstack([cluster_points for label in uniq_labels_n for cluster_points in p_collection[label]])
        remaining_labels = [label for k, label in enumerate(uniq_labels_n) if k not in pair]

    if robot_contact_count == 2:
        return np.vstack([contact_0, contact_1])
    if robot_contact_count > 2 and robot_contact_count <= n_normal_components:
        targets = [contact_0, contact_1]
        for label in remaining_labels[: robot_contact_count - 2]:
            label_idx = int(np.where(uniq_labels_n == label)[0][0])
            targets.append(centroid_p_n[label_idx])
        return np.asarray(targets, dtype=np.float64)
    if robot_contact_count > 2 and robot_contact_count >= n_hand_contacts:
        return centroid_p_arr[:robot_contact_count].copy()

    return _select_contact_targets_by_count_np(contact_points, contact_normals, robot_contact_count)


def _optimize_force_closure_targets_obj(
    config: TaskConfig,
    initial_targets_obj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    initial_targets = np.asarray(initial_targets_obj, dtype=np.float64).reshape(-1, 3)
    if initial_targets.shape[0] <= 1:
        projected_points, projected_normals = _project_targets_to_object_surface(config, initial_targets)
        return projected_points, projected_normals, {
            "fc_loss": 0.0,
            "fc_net_wrench": 0.0,
            "fc_lin_ind": 0.0,
            "fc_intfc": 0.0,
            "fc_distance": 0.0,
            "fc_inter_dist": 0.0,
        }

    fc_loss = _load_genhand_fc_loss()
    if fc_loss is None:
        projected_points, projected_normals = _project_targets_to_object_surface(config, initial_targets)
        sigma_min, spread = _force_closure_combo_metrics(projected_points, projected_normals)
        return projected_points, projected_normals, {
            "fc_loss": 0.0,
            "fc_net_wrench": 0.0,
            "fc_lin_ind": float(max(0.0, 1.0 - sigma_min)),
            "fc_intfc": float(max(0.0, 0.01 - spread)),
            "fc_distance": 0.0,
            "fc_inter_dist": 0.0,
        }

    x = torch.tensor(initial_targets, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
    w = (torch.ones((1, initial_targets.shape[0], 4), dtype=torch.float32) * 0.25).requires_grad_(True)
    contact_target = torch.tensor(initial_targets, dtype=torch.float32)
    relu = torch.nn.ReLU()

    def fc_terms(x_tensor: torch.Tensor, w_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        sdf_val, surface_normals = _object_surface_query_torch(config, x_tensor)
        x_norm = x_tensor / torch.clamp(torch.linalg.norm(x_tensor, dim=-1).amax(), min=1e-8)
        G = fc_loss.x_to_G(x_norm)
        lin_ind = fc_loss.lin_ind(G)
        f, _edge_forces = fc_loss.linearized_cone(surface_normals, w_tensor)
        net_wrench = fc_loss.net_wrench(f, G)
        int_fc = fc_loss.inter_fc(w_tensor)
        sdf = (sdf_val**2).mean()
        e_dist = relu(torch.linalg.norm(x_tensor.squeeze(0) - contact_target, dim=-1) - GENHAND_CONTACT_TARGET_TOL_M).sum()
        inter_dist = _minimum_internal_distance_torch(x_tensor)
        total = sdf + net_wrench + lin_ind + int_fc + 10.0 * e_dist + inter_dist
        return {
            "loss": total,
            "sdf": sdf,
            "net_wrench": net_wrench,
            "lin_ind": lin_ind,
            "int_fc": int_fc,
            "distance": e_dist,
            "inter_dist": inter_dist,
        }

    opt_fc = torch.optim.Adam([x, w], lr=GENHAND_FC_LR)
    best = None
    best_loss = float("inf")
    for _ in range(GENHAND_FC_MAX_ITERS):
        opt_fc.zero_grad()
        data = fc_terms(x, w)
        data["loss"].backward()
        opt_fc.step()
        loss_value = float(data["loss"].detach().cpu().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best = (
                x.detach().clone(),
                w.detach().clone(),
                {k: float(v.detach().cpu().item()) for k, v in data.items()},
            )
        if (
            torch.allclose(data["sdf"], torch.tensor(0.0, dtype=torch.float32), atol=5e-3)
            and torch.allclose(data["net_wrench"], torch.tensor(0.0, dtype=torch.float32), atol=1e-3)
            and torch.allclose(data["distance"], torch.tensor(0.0, dtype=torch.float32), atol=2e-2)
            and torch.allclose(data["lin_ind"], torch.tensor(0.0, dtype=torch.float32), atol=1e-3)
        ):
            break

    if best is not None:
        x = best[0].requires_grad_(False)
        w = best[1].requires_grad_(True)
    opt_gf = torch.optim.Adam([w], lr=GENHAND_GF_LR)
    for _ in range(GENHAND_GF_MAX_ITERS):
        opt_gf.zero_grad()
        data = fc_terms(x, w)
        gf_loss = data["net_wrench"] + data["lin_ind"] + data["int_fc"]
        gf_loss.backward()
        opt_gf.step()
        if torch.allclose(data["net_wrench"], torch.tensor(0.0, dtype=torch.float32), atol=1e-3):
            break

    final_points = np.asarray(x.detach().cpu().numpy()[0], dtype=np.float64)
    projected_points, projected_normals = _project_targets_to_object_surface(config, final_points)
    final_terms = fc_terms(x, w)
    metrics = {
        "fc_loss": float(final_terms["loss"].detach().cpu().item()),
        "fc_net_wrench": float(final_terms["net_wrench"].detach().cpu().item()),
        "fc_lin_ind": float(final_terms["lin_ind"].detach().cpu().item()),
        "fc_intfc": float(final_terms["int_fc"].detach().cpu().item()),
        "fc_distance": float(final_terms["distance"].detach().cpu().item()),
        "fc_inter_dist": float(final_terms["inter_dist"].detach().cpu().item()),
    }
    return projected_points, projected_normals, metrics


def _build_genhand_contact_target_obj(
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_dgrasp_qpos_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]] | None:
    dense_pack = _dense_mano_contact_candidates_obj(
        config=config,
        source_dgrasp_qpos_world=source_dgrasp_qpos_world,
        object_pose_goal=object_pose_goal,
    )
    if dense_pack is None:
        return None
    candidate_points, candidate_normals, _candidate_families, candidate_scores = dense_pack
    target_count = len(GENHAND_ROBOT_CONTACT_CANDIDATES)
    clustered_targets = _cluster_genhand_contact_targets_obj(
        candidate_points_obj=candidate_points,
        candidate_normals_obj=candidate_normals,
        robot_contact_count=target_count,
    )
    if clustered_targets.size == 0:
        return None
    optimized_targets, optimized_normals, fc_metrics = _optimize_force_closure_targets_obj(
        config=config,
        initial_targets_obj=clustered_targets,
    )
    target_scores = np.linalg.norm(candidate_points[:, None, :] - optimized_targets[None, :, :], axis=2).min(axis=0)
    fc_metrics.update(
        {
            "cluster_candidate_count": float(candidate_points.shape[0]),
            "cluster_score_mean": float(np.mean(candidate_scores)) if candidate_scores.size > 0 else 0.0,
            "target_anchor_count": float(optimized_targets.shape[0]),
            "target_anchor_rmse_m": float(np.sqrt(np.mean(np.square(target_scores)))) if target_scores.size > 0 else 0.0,
        }
    )
    return optimized_targets, optimized_normals, fc_metrics


def _human_surface_contact_anchors_obj(
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_dgrasp_qpos_world: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if source_dgrasp_qpos_world is None:
        raise RuntimeError("GenHand retargeting requires source D-Grasp qpos to reconstruct MANO contacts.")
    built = _build_genhand_contact_target_obj(
        config=config,
        object_pose_goal=object_pose_goal,
        source_dgrasp_qpos_world=source_dgrasp_qpos_world,
    )
    if built is None:
        raise RuntimeError("Failed to build GenHand contact target from MANO candidates.")
    return built


def _kabsch_rigid_transform(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    if source.shape[0] == 0 or target.shape[0] == 0 or source.shape[0] != target.shape[0]:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    h = (source - source_centroid).T @ (target - target_centroid)
    u, _s, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    return rotation, translation


def _normalize_point_cloud(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    cloud = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if cloud.size == 0:
        return cloud.copy(), np.zeros(3, dtype=np.float64), 1.0
    centroid = np.mean(cloud, axis=0)
    centered = cloud - centroid
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale < 1e-8:
        scale = 1.0
    return centered / scale, centroid, scale


def _apply_homogeneous_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if cloud.size == 0:
        return cloud.copy()
    rotation = np.asarray(transform[:3, :3], dtype=np.float64)
    translation = np.asarray(transform[:3, 3], dtype=np.float64)
    return (rotation @ cloud.T).T + translation


def _kabsch_hungarian_assignment(
    channel_centers: np.ndarray,
    channel_normals: np.ndarray,
    human_targets: np.ndarray,
    human_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    aligned_centers = channel_centers.copy()
    assigned_targets = np.arange(channel_centers.shape[0], dtype=np.int64)
    for _ in range(2):
        position_cost = np.linalg.norm(aligned_centers[:, None, :] - human_targets[None, :, :], axis=2)
        normal_cost = 0.05 * np.maximum(0.0, 1.0 - channel_normals @ human_normals.T)
        row_ind, col_ind = linear_sum_assignment(position_cost + normal_cost)
        row_ind = np.asarray(row_ind, dtype=np.int64)
        col_ind = np.asarray(col_ind, dtype=np.int64)
        assigned_targets[row_ind] = col_ind
        rotation, translation = _kabsch_rigid_transform(channel_centers[row_ind], human_targets[col_ind])
        aligned_centers = (rotation @ channel_centers.T).T + translation
    return assigned_targets, aligned_centers


def _laicp_channel_assignment(
    channel_centers: np.ndarray,
    channel_normals: np.ndarray,
    human_targets: np.ndarray,
    human_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    icp_module = _load_genhand_icp_module()
    if icp_module is None:
        assigned_targets, aligned_centers = _kabsch_hungarian_assignment(
            channel_centers=channel_centers,
            channel_normals=channel_normals,
            human_targets=human_targets,
            human_normals=human_normals,
        )
        return assigned_targets, aligned_centers, {"anchor_assignment_norm_cost": 0.0, "anchor_assignment_method": 0.0}

    current_centers = np.asarray(channel_centers, dtype=np.float64).copy()
    assigned_targets = np.arange(channel_centers.shape[0], dtype=np.int64)
    aligned_centers = current_centers.copy()
    assignment_cost = 0.0

    for _ in range(3):
        robot_normed, _robot_centroid, _robot_scale = _normalize_point_cloud(current_centers)
        target_normed, _target_centroid, _target_scale = _normalize_point_cloud(human_targets)

        try:
            transform, _distances, _iters, _indices = icp_module.icp(  # type: ignore[attr-defined]
                robot_normed,
                target_normed,
                max_iterations=100,
                tolerance=1e-6,
            )
        except Exception:
            assigned_targets, aligned_centers = _kabsch_hungarian_assignment(
                channel_centers=channel_centers,
                channel_normals=channel_normals,
                human_targets=human_targets,
                human_normals=human_normals,
            )
            return assigned_targets, aligned_centers, {"anchor_assignment_norm_cost": 0.0, "anchor_assignment_method": 0.0}

        robot_normed_aligned = _apply_homogeneous_transform(robot_normed, np.asarray(transform, dtype=np.float64))
        position_cost = np.linalg.norm(robot_normed_aligned[:, None, :] - target_normed[None, :, :], axis=2)
        normal_cost = 0.05 * np.maximum(0.0, 1.0 - channel_normals @ human_normals.T)
        total_cost = position_cost + normal_cost
        row_ind, col_ind = linear_sum_assignment(total_cost)
        row_ind = np.asarray(row_ind, dtype=np.int64)
        col_ind = np.asarray(col_ind, dtype=np.int64)

        new_assigned_targets = np.arange(channel_centers.shape[0], dtype=np.int64)
        new_assigned_targets[row_ind] = col_ind
        rotation, translation = _kabsch_rigid_transform(channel_centers[row_ind], human_targets[col_ind])
        new_aligned_centers = (rotation @ channel_centers.T).T + translation
        new_assignment_cost = float(np.mean(total_cost[row_ind, col_ind])) if row_ind.size else 0.0

        assigned_targets = new_assigned_targets
        aligned_centers = new_aligned_centers
        assignment_cost = new_assignment_cost
        if np.array_equal(current_centers, aligned_centers):
            break
        current_centers = aligned_centers

    return assigned_targets, aligned_centers, {"anchor_assignment_norm_cost": assignment_cost, "anchor_assignment_method": 1.0}


def _assigned_contact_anchor_residuals(
    robot_contact_point_sets_obj: np.ndarray,
    robot_contact_normal_sets_obj: np.ndarray,
    human_contact_targets_obj: np.ndarray,
    human_contact_normals_obj: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, float]]:
    robot_point_sets = np.asarray(robot_contact_point_sets_obj, dtype=np.float64).reshape(-1, robot_contact_point_sets_obj.shape[1], 3)
    robot_normal_sets = np.asarray(robot_contact_normal_sets_obj, dtype=np.float64).reshape(-1, robot_contact_normal_sets_obj.shape[1], 3)
    human_targets = np.asarray(human_contact_targets_obj, dtype=np.float64).reshape(-1, 3)
    human_normals = np.asarray(human_contact_normals_obj, dtype=np.float64).reshape(-1, 3)

    channel_count = robot_point_sets.shape[0]
    target_count = human_targets.shape[0]
    if channel_count == 0 or target_count == 0:
        return [], np.zeros((0,), dtype=np.int64), {"anchor_rmse_m": 0.0, "anchor_max_error_m": 0.0, "anchor_pairs": 0.0}

    if target_count != channel_count:
        reduced_targets = _select_contact_targets_by_count_np(human_targets, human_normals, min(channel_count, target_count))
        reduction_cost = np.linalg.norm(human_targets[:, None, :] - reduced_targets[None, :, :], axis=2)
        keep_rows, keep_cols = linear_sum_assignment(reduction_cost)
        keep_indices = keep_rows[np.argsort(keep_cols)]
        human_targets = human_targets[keep_indices]
        human_normals = human_normals[keep_indices]
        target_count = human_targets.shape[0]
    if channel_count != target_count:
        channel_count = min(channel_count, target_count)
        robot_point_sets = robot_point_sets[:channel_count]
        robot_normal_sets = robot_normal_sets[:channel_count]
        human_targets = human_targets[:channel_count]
        human_normals = human_normals[:channel_count]

    channel_centers = np.mean(robot_point_sets, axis=1)
    channel_normals = np.asarray([normalize(np.mean(points, axis=0)) for points in robot_normal_sets], dtype=np.float64)

    assigned_targets, aligned_centers, assignment_metrics = _laicp_channel_assignment(
        channel_centers=channel_centers,
        channel_normals=channel_normals,
        human_targets=human_targets,
        human_normals=human_normals,
    )

    residuals: list[np.ndarray] = []
    assigned_error_norms: list[float] = []
    for channel_idx, target_idx in enumerate(assigned_targets.tolist()):
        target_point = human_targets[target_idx]
        target_normal = human_normals[target_idx]
        local_points = robot_point_sets[channel_idx]
        local_normals = robot_normal_sets[channel_idx]
        point_cost = np.linalg.norm(local_points - target_point[None, :], axis=1)
        normal_cost = 0.02 * np.maximum(0.0, 1.0 - np.sum(local_normals * target_normal[None, :], axis=1))
        best_local_idx = int(np.argmin(point_cost + normal_cost))
        diff = local_points[best_local_idx] - target_point
        residuals.append(diff)
        assigned_error_norms.append(float(np.linalg.norm(diff)))

    metrics = {
        "anchor_rmse_m": float(np.sqrt(np.mean(np.square(assigned_error_norms)))) if assigned_error_norms else 0.0,
        "anchor_max_error_m": float(np.max(assigned_error_norms)) if assigned_error_norms else 0.0,
        "anchor_pairs": float(len(residuals)),
        **assignment_metrics,
    }
    return residuals, assigned_targets, metrics


def _contact_anchor_metrics_obj(
    robot_contact_point_sets_obj: np.ndarray,
    robot_contact_normal_sets_obj: np.ndarray,
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_dgrasp_qpos_world: np.ndarray | None,
) -> dict[str, float]:
    human_targets, human_normals, fc_metrics = _human_surface_contact_anchors_obj(
        config=config,
        object_pose_goal=object_pose_goal,
        source_dgrasp_qpos_world=source_dgrasp_qpos_world,
    )
    _residuals, _assigned_targets, metrics = _assigned_contact_anchor_residuals(
        robot_contact_point_sets_obj=robot_contact_point_sets_obj,
        robot_contact_normal_sets_obj=robot_contact_normal_sets_obj,
        human_contact_targets_obj=human_targets,
        human_contact_normals_obj=human_normals,
    )
    metrics.update({f"genhand_{key}": float(value) for key, value in fc_metrics.items()})
    return metrics


def _candidate_contact_anchor_metrics(
    candidate: dict[str, Any],
    object_pose_goal: np.ndarray,
    human_contact_targets_obj: np.ndarray,
    human_contact_normals_obj: np.ndarray,
) -> dict[str, float]:
    point_sets_world = np.asarray(candidate["contact_candidate_point_sets_world"], dtype=np.float64)
    normal_sets_world = np.asarray(candidate["contact_candidate_normal_sets_world"], dtype=np.float64)
    point_sets_obj = transform_points(
        point_sets_world.reshape(-1, 3),
        inverse_pose7(object_pose_goal),
    ).reshape(point_sets_world.shape)
    inv_object_rotation = quat_wxyz_to_matrix(inverse_pose7(object_pose_goal)[3:])
    normal_sets_obj = np.einsum("ij,ksj->ksi", inv_object_rotation, normal_sets_world)
    _residuals, _assigned_targets, metrics = _assigned_contact_anchor_residuals(
        robot_contact_point_sets_obj=point_sets_obj,
        robot_contact_normal_sets_obj=normal_sets_obj,
        human_contact_targets_obj=human_contact_targets_obj,
        human_contact_normals_obj=human_contact_normals_obj,
    )
    return {str(k): float(v) for k, v in metrics.items()}


def _box_genhand_seed_states(
    runtime: RobotSceneModel,
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_wrist_pose_world: np.ndarray,
    source_semantic_sites_world: np.ndarray,
    source_keypoints_obj: np.ndarray,
    source_contact_mask_12: np.ndarray,
    source_dgrasp_qpos_world: np.ndarray | None,
    base_hand_qpos: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    del source_contact_mask_12
    del source_keypoints_obj
    source_wrist_pose_object = compose_pose7(inverse_pose7(object_pose_goal), source_wrist_pose_world)
    source_semantic_sites_obj = transform_points(source_semantic_sites_world, inverse_pose7(object_pose_goal))
    source_frame_obj = semantic_frame_from_sites(source_semantic_sites_obj)
    source_contact_targets_obj, source_contact_normals_obj, _fc_metrics = _human_surface_contact_anchors_obj(
        config=config,
        object_pose_goal=object_pose_goal,
        source_dgrasp_qpos_world=source_dgrasp_qpos_world,
    )
    contact_centroid = np.mean(source_contact_targets_obj, axis=0) if source_contact_targets_obj.size else source_semantic_sites_obj[1]
    averaged_normal = normalize(np.mean(source_contact_normals_obj, axis=0)) if source_contact_normals_obj.size else -source_frame_obj[:, 2]
    if np.linalg.norm(averaged_normal) < 1e-8:
        averaged_normal = -source_frame_obj[:, 2]
    desired_approach = -averaged_normal
    source_across = source_frame_obj[:, 0]
    across = source_across - desired_approach * float(np.dot(source_across, desired_approach))
    if np.linalg.norm(across) < 1e-8:
        across = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    across = normalize(across)
    normal = normalize(np.cross(across, desired_approach))
    across = normalize(np.cross(desired_approach, normal))
    seed_variants = _genhand_hand_seed_variants(runtime, base_hand_qpos)
    relaxed_hand = seed_variants["relaxed"]
    contact_biased_hand = seed_variants["balanced"]
    aggressive_hand = seed_variants["aggressive"]
    thumb_lead_hand = seed_variants["thumb_lead"]

    seeds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [
        (np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), relaxed_hand),
        (np.array([0.0, 0.0, -0.004], dtype=np.float64), np.zeros(3, dtype=np.float64), contact_biased_hand),
        (np.array([0.0, 0.0, 0.004], dtype=np.float64), np.zeros(3, dtype=np.float64), aggressive_hand),
        (np.array([0.006, 0.0, 0.0], dtype=np.float64), np.deg2rad(np.array([0.0, -6.0, 0.0], dtype=np.float64)), contact_biased_hand),
        (np.array([-0.006, 0.0, 0.0], dtype=np.float64), np.deg2rad(np.array([0.0, 6.0, 0.0], dtype=np.float64)), contact_biased_hand),
        (np.array([0.0, 0.008, 0.0], dtype=np.float64), np.deg2rad(np.array([4.0, 0.0, 0.0], dtype=np.float64)), contact_biased_hand),
        (np.array([0.0, -0.008, 0.0], dtype=np.float64), np.deg2rad(np.array([-4.0, 0.0, 0.0], dtype=np.float64)), contact_biased_hand),
        (np.array([0.0, 0.0, -0.002], dtype=np.float64), np.deg2rad(np.array([0.0, 0.0, 8.0], dtype=np.float64)), thumb_lead_hand),
    ]

    target_pose_object = np.eye(4, dtype=np.float64)
    target_pose_object[:3, :3] = np.column_stack((across, normal, desired_approach))
    palm_target = 0.65 * source_semantic_sites_obj[1] + 0.35 * (contact_centroid - desired_approach * 0.018)
    target_pose_object[:3, 3] = palm_target - desired_approach * ROBOT_WRIST_TO_PALM_OFFSET_M
    target_pose_object_7 = matrix_to_pose7(target_pose_object)
    translation_local, rotvec_local = _pose_delta_local(source_wrist_pose_object, target_pose_object_7)
    seeds.append((translation_local, rotvec_local, contact_biased_hand))
    seeds.append((translation_local, rotvec_local, aggressive_hand))
    return seeds


def _optimize_genhand_seed(
    runtime: RobotSceneModel,
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_wrist_pose_world: np.ndarray,
    source_semantic_sites_world: np.ndarray,
    source_keypoints_obj: np.ndarray,
    source_contact_mask_12: np.ndarray,
    source_dgrasp_qpos_world: np.ndarray | None,
    initial_translation_local: np.ndarray,
    initial_rotvec_local: np.ndarray,
    initial_hand_qpos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    source_wrist_pose_object = compose_pose7(inverse_pose7(object_pose_goal), source_wrist_pose_world)
    source_semantic_sites_obj = transform_points(source_semantic_sites_world, inverse_pose7(object_pose_goal))
    source_contact_targets_obj, source_contact_normals_obj, _fc_metrics = _human_surface_contact_anchors_obj(
        config=config,
        object_pose_goal=object_pose_goal,
        source_dgrasp_qpos_world=source_dgrasp_qpos_world,
    )
    source_finger_dirs_obj = _human_finger_directions_obj(source_keypoints_obj)
    object_rotation = quat_wxyz_to_matrix(np.asarray(object_pose_goal[3:], dtype=np.float64))
    inv_object_rotation = object_rotation.T
    table_top_z = runtime.get_table_top_z()

    initial_translation_local = np.asarray(initial_translation_local, dtype=np.float64)
    initial_rotvec_local = np.asarray(initial_rotvec_local, dtype=np.float64)
    initial_hand_qpos = runtime.clamp_hand(np.asarray(initial_hand_qpos, dtype=np.float64))
    x0 = np.concatenate([initial_translation_local, initial_rotvec_local, initial_hand_qpos])
    lower = np.concatenate(
        [
            initial_translation_local - np.array([0.05, 0.05, 0.06], dtype=np.float64),
            initial_rotvec_local - np.deg2rad(np.array([24.0, 24.0, 36.0], dtype=np.float64)),
            runtime.hand_lower,
        ]
    )
    upper = np.concatenate(
        [
            initial_translation_local + np.array([0.05, 0.05, 0.06], dtype=np.float64),
            initial_rotvec_local + np.deg2rad(np.array([24.0, 24.0, 36.0], dtype=np.float64)),
            runtime.hand_upper,
        ]
    )

    def objective(x: np.ndarray) -> np.ndarray:
        translation_local = np.asarray(x[:3], dtype=np.float64)
        rotvec_local = np.asarray(x[3:6], dtype=np.float64)
        hand_qpos = runtime.clamp_hand(np.asarray(x[6:], dtype=np.float64))

        target_wrist_pose_object = apply_local_pose_delta(
            source_wrist_pose_object,
            translation_local,
            rotvec_local,
        )
        target_wrist_pose_world = compose_pose7(object_pose_goal, target_wrist_pose_object)
        arm_qpos = solve_arm_wrist_palm_ik(
            runtime=runtime,
            target_wrist_pose_world=target_wrist_pose_world,
            initial_arm_qpos=runtime.home_actuated[:6],
            hand_qpos=hand_qpos,
            iterations=max(42, int(config.conversion.arm_ik_iterations * 0.45)),
            damping=config.conversion.arm_ik_damping,
        )

        runtime.reset()
        runtime.set_object_pose(object_pose_goal)
        runtime.set_robot_actuated_qpos(np.concatenate([arm_qpos, hand_qpos]))

        contact_candidate_point_sets_world, contact_candidate_normal_sets_world, _contact_candidate_families = runtime.get_contact_candidate_point_sets_world()
        contact_candidate_point_sets_obj = transform_points(
            contact_candidate_point_sets_world.reshape(-1, 3),
            inverse_pose7(object_pose_goal),
        ).reshape(contact_candidate_point_sets_world.shape)
        contact_candidate_normal_sets_obj = np.einsum(
            "ij,ksj->ksi",
            inv_object_rotation,
            np.asarray(contact_candidate_normal_sets_world, dtype=np.float64),
        )
        finger_dirs_world = runtime.get_finger_directions_world()
        finger_dirs_obj = (inv_object_rotation @ finger_dirs_world.T).T
        semantic_sites_world = runtime.get_semantic_sites_world()
        semantic_sites_obj = transform_points(semantic_sites_world, inverse_pose7(object_pose_goal))
        semantic_frame_obj = semantic_frame_from_sites(semantic_sites_obj)
        diag = runtime.get_contact_diagnostics_12()
        distances = np.asarray(diag["distances_m"], dtype=np.float64)
        table_clearance = runtime.get_proxy_table_clearance_12()
        arm_table_clearance = runtime.get_arm_sweep_table_clearance()
        reachability_metrics = _reachability_metrics_world(
            object_pose_goal=object_pose_goal,
            semantic_sites_world=semantic_sites_world,
            arm_table_clearance_m=arm_table_clearance,
        )

        anchor_residuals, assigned_targets, _ = _assigned_contact_anchor_residuals(
            robot_contact_point_sets_obj=contact_candidate_point_sets_obj,
            robot_contact_normal_sets_obj=contact_candidate_normal_sets_obj,
            human_contact_targets_obj=source_contact_targets_obj,
            human_contact_normals_obj=source_contact_normals_obj,
        )

        residuals: list[float] = []
        for diff in anchor_residuals:
            residuals.extend((GENHAND_ANCHOR_POSITION_WEIGHT * diff).tolist())

        for finger_idx in range(min(len(finger_dirs_obj), len(source_finger_dirs_obj))):
            residuals.extend((GENHAND_DIRECTION_WEIGHT * (finger_dirs_obj[finger_idx] - source_finger_dirs_obj[finger_idx])).tolist())

        desired_palm_approach = -normalize(np.mean(source_contact_normals_obj, axis=0)) if source_contact_normals_obj.size else semantic_frame_from_sites(source_semantic_sites_obj)[:, 2]
        current_palm_approach = semantic_frame_obj[:, 2]
        residuals.extend((GENHAND_PALM_DIRECTION_WEIGHT * (current_palm_approach - desired_palm_approach)).tolist())
        residuals.extend((GENHAND_PALM_POSITION_WEIGHT * (semantic_sites_obj[1] - source_semantic_sites_obj[1])).tolist())
        residuals.extend((GENHAND_WRIST_POSITION_WEIGHT * (semantic_sites_obj[0] - source_semantic_sites_obj[0])).tolist())

        if source_contact_targets_obj.size:
            aligned_targets = source_contact_targets_obj[np.asarray(assigned_targets, dtype=np.int64)]
            aligned_centers = np.mean(contact_candidate_point_sets_obj, axis=1)
            residuals.extend(
                (
                    GENHAND_ACTIVE_DISTANCE_WEIGHT
                    * np.linalg.norm(aligned_centers - aligned_targets, axis=1)
                ).tolist()
            )

        penetration = np.maximum(-distances, 0.0)
        residuals.extend((GENHAND_PENETRATION_WEIGHT * penetration).tolist())
        residuals.append(GENHAND_PENETRATION_WEIGHT * float(diag["max_penetration_m"]))

        semantic_clearance = np.array(
            [
                semantic_sites_world[0, 2] - table_top_z,
                semantic_sites_world[1, 2] - table_top_z,
            ],
            dtype=np.float64,
        )
        table_violation = np.maximum(0.003 - np.concatenate([table_clearance, semantic_clearance]), 0.0)
        residuals.extend((GENHAND_TABLE_WEIGHT * table_violation).tolist())
        residuals.append(
            GENHAND_REACH_OBJECT_WEIGHT
            * max(0.0, REACHABILITY_MIN_OBJECT_FACING_COS - reachability_metrics["object_facing_cos"])
        )
        residuals.append(
            GENHAND_REACH_BASE_WEIGHT
            * max(0.0, reachability_metrics["base_facing_cos"] - REACHABILITY_MAX_BASE_FACING_COS)
        )
        residuals.append(
            GENHAND_REACH_DOWNWARD_WEIGHT
            * max(0.0, REACHABILITY_MIN_DOWNWARD_APPROACH - reachability_metrics["downward_component"])
        )
        arm_clearance_violation = np.maximum(
            REACHABILITY_MIN_ARM_TABLE_CLEARANCE_M - arm_table_clearance,
            0.0,
        )
        residuals.extend((GENHAND_ARM_TABLE_WEIGHT * arm_clearance_violation).tolist())

        residuals.extend((GENHAND_WRIST_REG_WEIGHT * (translation_local - initial_translation_local)).tolist())
        residuals.extend((0.15 * (rotvec_local - initial_rotvec_local)).tolist())
        residuals.extend((GENHAND_HAND_REG_WEIGHT * (hand_qpos - initial_hand_qpos)).tolist())
        return np.asarray(residuals, dtype=np.float64)

    result = least_squares(
        objective,
        x0,
        bounds=(lower, upper),
        max_nfev=GENHAND_MAX_NFEV,
    )
    optimized = np.asarray(result.x, dtype=np.float64)
    return (
        optimized[:3].copy(),
        optimized[3:6].copy(),
        runtime.clamp_hand(optimized[6:]).copy(),
        float(result.cost),
    )


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


def _required_target_contact_count(source_contact_mask_12: np.ndarray | None = None) -> int:
    del source_contact_mask_12
    return min(2, len(GENHAND_CONTACT_FAMILY_NAMES))


def _projection_rank_key(candidate: dict[str, Any], required_contacts: int) -> tuple[float, ...]:
    is_cylinder = "cylinder_palm_clearance_error_m" in candidate
    matched_contacts = float(candidate["source_matched_target_contacts"])
    extra_contacts = float(candidate["source_extra_contacts"])
    table_contact = 1.0 if bool(candidate["table_contact"]) else 0.0
    max_penetration = float(candidate["max_penetration_m"])
    total_penetration = float(candidate["total_penetration_m"])
    object_translation_drift = float(candidate["object_translation_drift_m"])
    object_rotation_drift = float(candidate["object_rotation_drift_deg"])
    soft_translation_limit = 0.120 if is_cylinder else PROJECTION_SOFT_OBJECT_TRANSLATION_DRIFT_M
    hard_translation_limit = 0.180 if is_cylinder else PROJECTION_HARD_OBJECT_TRANSLATION_DRIFT_M
    soft_rotation_limit = 35.0 if is_cylinder else PROJECTION_SOFT_OBJECT_ROTATION_DRIFT_DEG
    hard_rotation_limit = 55.0 if is_cylinder else PROJECTION_HARD_OBJECT_ROTATION_DRIFT_DEG

    soft_penetration_over = max(0.0, max_penetration - PROJECTION_SOFT_MAX_PENETRATION_M)
    soft_total_penetration_over = max(0.0, total_penetration - PROJECTION_SOFT_TOTAL_PENETRATION_M)
    soft_translation_drift_over = max(0.0, object_translation_drift - soft_translation_limit)
    soft_rotation_drift_over = max(0.0, object_rotation_drift - soft_rotation_limit)
    soft_safety_violation = (
        10.0 * table_contact
        + 600.0 * soft_penetration_over
        + 120.0 * soft_total_penetration_over
        + 80.0 * soft_translation_drift_over
        + 2.0 * soft_rotation_drift_over
    )
    hard_penetration_over = max(0.0, max_penetration - PROJECTION_HARD_MAX_PENETRATION_M)
    hard_total_penetration_over = max(0.0, total_penetration - PROJECTION_HARD_TOTAL_PENETRATION_M)
    hard_translation_drift_over = max(0.0, object_translation_drift - hard_translation_limit)
    hard_rotation_drift_over = max(0.0, object_rotation_drift - hard_rotation_limit)
    hard_safety_violation = (
        10.0 * table_contact
        + 600.0 * hard_penetration_over
        + 120.0 * hard_total_penetration_over
        + 80.0 * hard_translation_drift_over
        + 2.0 * hard_rotation_drift_over
    )
    settled_contact_shortfall = max(0.0, 2.0 - float(candidate.get("settled_contact_group_count", 0.0)))
    settled_hard_shortfall = max(0.0, 1.0 - float(candidate.get("settled_hard_contact_group_count", 0.0)))
    settled_opposition_missing = 0.0 if bool(candidate.get("settled_has_thumb_opposition", False)) else 1.0
    hold_drop_over = max(0.0, float(candidate["hold_object_drop_m"]) - PROJECTION_HOLD_MAX_DROP_M)
    hold_translation_over = max(0.0, float(candidate["hold_object_translation_m"]) - PROJECTION_HOLD_MAX_TRANSLATION_M)
    hold_rotation_over = max(0.0, float(candidate["hold_object_rotation_deg"]) - PROJECTION_HOLD_MAX_ROTATION_DEG)
    hold_table_contact = 1.0 if bool(candidate["hold_table_contact"]) else 0.0
    hold_hybrid_shortfall = max(0.0, 2.0 - float(candidate["hold_hybrid_contact_group_count"]))
    hold_hard_shortfall = max(0.0, 1.0 - float(candidate["hold_hard_contact_group_count"]))
    hold_opposition_missing = 0.0 if bool(candidate["hold_has_thumb_opposition"]) else 1.0
    reach_object_shortfall = max(
        0.0,
        REACHABILITY_MIN_OBJECT_FACING_COS - float(candidate.get("reach_object_facing_cos", 1.0)),
    )
    reach_base_violation = max(
        0.0,
        float(candidate.get("reach_base_facing_cos", -1.0)) - REACHABILITY_MAX_BASE_FACING_COS,
    )
    reach_downward_violation = max(
        0.0,
        REACHABILITY_MIN_DOWNWARD_APPROACH - float(candidate.get("reach_downward_component", 1.0)),
    )
    reach_arm_table_violation = max(
        0.0,
        REACHABILITY_MIN_ARM_TABLE_CLEARANCE_M - float(candidate.get("reach_min_arm_table_clearance_m", 1.0)),
    )
    reachability_violation = (
        8.0 * reach_object_shortfall
        + 8.0 * reach_base_violation
        + 6.0 * reach_downward_violation
        + 40.0 * reach_arm_table_violation
    )
    hold_quality_violation = (
        100.0 * hold_table_contact
        + 50.0 * hold_drop_over
        + 10.0 * hold_translation_over
        + 0.5 * hold_rotation_over
        + 6.0 * hold_hybrid_shortfall
        + 4.0 * hold_hard_shortfall
        + 3.0 * hold_opposition_missing
    )
    fc_quality_violation = (
        16.0 * float(candidate.get("genhand_target_anchor_rmse_m", 0.0))
        + 1.5 * float(candidate.get("genhand_cluster_score_mean", 0.0))
        + 0.50 * float(candidate.get("genhand_fc_net_wrench", 0.0))
        + 0.50 * float(candidate.get("genhand_fc_lin_ind", 0.0))
        + 0.50 * float(candidate.get("genhand_fc_intfc", 0.0))
        + 4.0 * float(candidate.get("anchor_assignment_norm_cost", 0.0))
        + 2.0 * float(candidate.get("anchor_rmse_m", 0.0))
    )
    kinematic_quality_violation = (
        0.10 * float(candidate.get("teacher_cost", 0.0))
        + 2.0 * float(candidate["source_site_rmse_m"])
        + 0.010 * float(candidate["source_semantic_frame_error_deg"])
    )
    cylinder_wrap_violation = 0.0
    if is_cylinder:
        cylinder_wrap_violation += 60.0 * max(
            0.0,
            float(candidate["cylinder_palm_clearance_error_m"]) - PROJECTION_CYLINDER_PALM_CLEARANCE_TOL_M,
        )
        cylinder_wrap_violation += 8.0 * max(
            0.0,
            float(candidate["cylinder_opposition_cos"]) - PROJECTION_CYLINDER_OPPOSITION_TARGET_COS,
        )
        cylinder_wrap_violation += 80.0 * max(
            0.0,
            float(candidate["hold_cylinder_palm_clearance_error_m"]) - PROJECTION_CYLINDER_HOLD_PALM_CLEARANCE_TOL_M,
        )
        cylinder_wrap_violation += 10.0 * max(
            0.0,
            float(candidate["hold_cylinder_opposition_cos"]) - PROJECTION_CYLINDER_HOLD_OPPOSITION_TARGET_COS,
        )
        cylinder_wrap_violation += 2.0 * max(
            0.0,
            2.0 - float(candidate.get("settled_contact_group_count", 0.0)),
        )
        cylinder_wrap_violation += 2.0 * max(
            0.0,
            1.0 - float(candidate.get("settled_hard_contact_group_count", 0.0)),
        )
        cylinder_wrap_violation += 1.5 * float(not bool(candidate.get("settled_has_thumb_opposition", False)))

    if hard_safety_violation <= 1e-9:
        return (
            0.0,
            reachability_violation,
            fc_quality_violation,
            kinematic_quality_violation,
            hold_quality_violation,
            cylinder_wrap_violation,
            settled_contact_shortfall,
            settled_hard_shortfall,
            settled_opposition_missing,
            soft_safety_violation,
            object_translation_drift,
            max_penetration,
            float(np.linalg.norm(np.asarray(candidate["wrist_translation_local"], dtype=np.float64))),
            float(np.linalg.norm(np.asarray(candidate["wrist_rotvec_local"], dtype=np.float64))),
        )
    return (
        1.0,
        hard_safety_violation,
        reachability_violation,
        fc_quality_violation,
        kinematic_quality_violation,
        hold_quality_violation,
        cylinder_wrap_violation,
        settled_contact_shortfall,
        settled_hard_shortfall,
        settled_opposition_missing,
        soft_safety_violation,
        object_translation_drift,
        max_penetration,
        float(np.linalg.norm(np.asarray(candidate["wrist_translation_local"], dtype=np.float64))),
        float(np.linalg.norm(np.asarray(candidate["wrist_rotvec_local"], dtype=np.float64))),
    )


def _projection_preshortlist_rank_key(candidate: dict[str, Any], required_contacts: int) -> tuple[float, ...]:
    table_contact = 1.0 if bool(candidate["table_contact"]) else 0.0
    hard_safety_violation = (
        10.0 * table_contact
        + 600.0 * max(0.0, float(candidate["max_penetration_m"]) - PROJECTION_HARD_MAX_PENETRATION_M)
        + 120.0 * max(0.0, float(candidate["total_penetration_m"]) - PROJECTION_HARD_TOTAL_PENETRATION_M)
        + 80.0 * max(0.0, float(candidate["object_translation_drift_m"]) - PROJECTION_HARD_OBJECT_TRANSLATION_DRIFT_M)
        + 2.0 * max(0.0, float(candidate["object_rotation_drift_deg"]) - PROJECTION_HARD_OBJECT_ROTATION_DRIFT_DEG)
    )
    reachability_violation = (
        8.0 * max(0.0, REACHABILITY_MIN_OBJECT_FACING_COS - float(candidate.get("reach_object_facing_cos", 1.0)))
        + 8.0 * max(0.0, float(candidate.get("reach_base_facing_cos", -1.0)) - REACHABILITY_MAX_BASE_FACING_COS)
        + 6.0 * max(0.0, REACHABILITY_MIN_DOWNWARD_APPROACH - float(candidate.get("reach_downward_component", 1.0)))
        + 40.0 * max(0.0, REACHABILITY_MIN_ARM_TABLE_CLEARANCE_M - float(candidate.get("reach_min_arm_table_clearance_m", 1.0)))
    )
    fc_quality_violation = (
        16.0 * float(candidate.get("genhand_target_anchor_rmse_m", 0.0))
        + 1.5 * float(candidate.get("genhand_cluster_score_mean", 0.0))
        + 0.50 * float(candidate.get("genhand_fc_net_wrench", 0.0))
        + 0.50 * float(candidate.get("genhand_fc_lin_ind", 0.0))
        + 0.50 * float(candidate.get("genhand_fc_intfc", 0.0))
        + 4.0 * float(candidate.get("anchor_assignment_norm_cost", 0.0))
        + 2.0 * float(candidate.get("anchor_rmse_m", 0.0))
    )
    kinematic_quality_violation = (
        0.10 * float(candidate.get("teacher_cost", 0.0))
        + 2.0 * float(candidate["source_site_rmse_m"])
        + 0.010 * float(candidate["source_semantic_frame_error_deg"])
    )
    settled_contact_shortfall = max(0.0, 2.0 - float(candidate.get("settled_contact_group_count", 0.0)))
    settled_hard_shortfall = max(0.0, 1.0 - float(candidate.get("settled_hard_contact_group_count", 0.0)))
    settled_opposition_missing = 0.0 if bool(candidate.get("settled_has_thumb_opposition", False)) else 1.0
    return (
        hard_safety_violation,
        reachability_violation,
        fc_quality_violation,
        kinematic_quality_violation,
        settled_contact_shortfall,
        settled_hard_shortfall,
        settled_opposition_missing,
        float(candidate["object_translation_drift_m"]),
        float(candidate["max_penetration_m"]),
        float(np.linalg.norm(np.asarray(candidate["wrist_translation_local"], dtype=np.float64))),
        float(np.linalg.norm(np.asarray(candidate["wrist_rotvec_local"], dtype=np.float64))),
    )


def _is_better_projected_candidate(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    required_contacts: int,
) -> bool:
    if incumbent is None:
        return True
    return _projection_rank_key(candidate, required_contacts) < _projection_rank_key(incumbent, required_contacts)


def _evaluate_projected_candidate(
    runtime: RobotSceneModel,
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_wrist_pose_world: np.ndarray,
    source_semantic_sites_world: np.ndarray,
    source_contact_mask_12: np.ndarray,
    wrist_translation_local: np.ndarray,
    wrist_rotvec_local: np.ndarray,
    hand_qpos: np.ndarray,
    run_hold_test: bool = True,
) -> dict[str, Any]:
    wrist_translation_local = np.asarray(wrist_translation_local, dtype=np.float64)
    wrist_rotvec_local = np.asarray(wrist_rotvec_local, dtype=np.float64)
    source_wrist_pose_world = np.asarray(source_wrist_pose_world, dtype=np.float64)
    source_frame = quat_wxyz_to_matrix(source_wrist_pose_world[3:])
    candidate_wrist_pose = apply_local_pose_delta(
        source_wrist_pose_world,
        wrist_translation_local,
        wrist_rotvec_local,
    )
    candidate_hand_qpos = runtime.clamp_hand(np.asarray(hand_qpos, dtype=np.float64))

    candidate_arm_qpos = solve_arm_wrist_palm_ik(
        runtime=runtime,
        target_wrist_pose_world=candidate_wrist_pose,
        initial_arm_qpos=runtime.home_actuated[:6],
        hand_qpos=candidate_hand_qpos,
        iterations=max(48, int(config.conversion.arm_ik_iterations * 0.6)),
        damping=config.conversion.arm_ik_damping,
    )
    target_actuated_qpos = np.concatenate([candidate_arm_qpos, candidate_hand_qpos])

    runtime.reset()
    runtime.set_object_pose(object_pose_goal)
    runtime.settle_actuated_pose(target_actuated_qpos, PROJECTION_SETTLE_STEPS)
    settled_actuated_qpos = runtime.get_actuated_qpos()
    settled_sites = runtime.get_semantic_sites_world()
    settled_wrist_pose = wrist_pose_from_semantic_sites(settled_sites)
    settled_frame = quat_wxyz_to_matrix(settled_wrist_pose[3:])
    settled_contact_diag = runtime.get_contact_diagnostics_12()
    settled_hard_contact_mask = np.asarray(settled_contact_diag["hard_mask"], dtype=np.float64)
    settled_proximity_mask = np.asarray(settled_contact_diag["proximity_mask"], dtype=np.float64)
    settled_contact_mask = np.asarray(settled_contact_diag["hybrid_mask"], dtype=np.float64)
    settled_proximity_scores = np.asarray(settled_contact_diag["proximity_scores"], dtype=np.float64)
    settled_contact_distances = np.asarray(settled_contact_diag["distances_m"], dtype=np.float64)
    settled_object_pose = runtime.get_object_pose()
    goal_object_frame = quat_wxyz_to_matrix(np.asarray(object_pose_goal[3:], dtype=np.float64))
    if run_hold_test:
        hold_result = _run_projection_hold_test(
            runtime=runtime,
            config=config,
            ctrl_target=settled_actuated_qpos,
        )
        final_contact_diag = hold_result["contact_diag"]
        hold_object_drop_m = float(hold_result["object_drop_m"])
        hold_object_translation_m = float(hold_result["object_translation_m"])
        hold_object_rotation_deg = float(hold_result["object_rotation_deg"])
        hold_table_contact = bool(final_contact_diag["table_contact"])
    else:
        final_contact_diag = settled_contact_diag
        hold_object_drop_m = 0.0
        hold_object_translation_m = 0.0
        hold_object_rotation_deg = 0.0
        hold_table_contact = bool(final_contact_diag["table_contact"])
    final_actuated_qpos = settled_actuated_qpos.copy()
    final_sites = settled_sites.copy()
    final_wrist_pose = settled_wrist_pose.copy()
    final_frame = settled_frame.copy()
    final_object_pose = settled_object_pose.copy()
    final_hard_contact_mask = settled_hard_contact_mask.copy()
    final_proximity_mask = settled_proximity_mask.copy()
    final_contact_mask = settled_contact_mask.copy()
    final_proximity_scores = settled_proximity_scores.copy()
    final_contact_distances = settled_contact_distances.copy()
    final_contact_points_world = runtime.get_contact_proxy_points_world_12().copy()
    (
        final_contact_candidate_point_sets_world,
        final_contact_candidate_normal_sets_world,
        final_contact_candidate_families,
    ) = runtime.get_contact_candidate_point_sets_world()
    final_object_frame = quat_wxyz_to_matrix(final_object_pose[3:])

    wrist_error = float(np.linalg.norm(final_sites[0] - source_semantic_sites_world[0]))
    palm_error = float(np.linalg.norm(final_sites[1] - source_semantic_sites_world[1]))
    tip_rmse = float(
        np.sqrt(np.mean(np.sum((final_sites[2:] - source_semantic_sites_world[2:]) ** 2, axis=1)))
    )
    site_rmse = float(
        np.sqrt(np.mean(np.sum((final_sites - source_semantic_sites_world) ** 2, axis=1)))
    )
    frame_error_deg = float(rotation_angle_deg(final_frame, source_frame))
    target_contact_misses = float(np.sum((source_contact_mask_12 > 0.5) & (final_contact_mask < 0.5)))
    extra_contacts = float(np.sum((source_contact_mask_12 < 0.5) & (final_contact_mask > 0.5)))
    matched_target_contacts = float(np.sum((source_contact_mask_12 > 0.5) & (final_contact_mask > 0.5)))
    required_contacts = _required_target_contact_count(source_contact_mask_12)
    contact_shortfall = max(0.0, float(required_contacts) - matched_target_contacts)
    translation_norm = float(np.linalg.norm(wrist_translation_local))
    rotation_norm = float(np.linalg.norm(wrist_rotvec_local))
    object_translation_drift_m = float(np.linalg.norm(final_object_pose[:3] - np.asarray(object_pose_goal[:3], dtype=np.float64)))
    object_rotation_drift_deg = float(rotation_angle_deg(final_object_frame, goal_object_frame))
    settled_contact_group_count = float(_contact_group_count(final_contact_mask))
    settled_hard_contact_group_count = float(_contact_group_count(final_hard_contact_mask))
    settled_has_thumb_opposition = bool(_has_thumb_opposition(final_contact_mask))
    cylinder_palm_clearance_error_m, cylinder_opposition_cos = _cylinder_grasp_geometry_metrics(
        config,
        final_object_pose,
        final_sites,
        source_contact_mask_12,
    )
    hold_contact_mask = np.asarray(final_contact_diag["hybrid_mask"], dtype=np.float64)
    hold_hard_contact_mask = np.asarray(final_contact_diag["hard_mask"], dtype=np.float64)
    hold_matched_target_contacts = float(np.sum((source_contact_mask_12 > 0.5) & (hold_contact_mask > 0.5)))
    hold_hybrid_contact_group_count = float(_contact_group_count(hold_contact_mask))
    hold_hard_contact_group_count = float(_contact_group_count(hold_hard_contact_mask))
    hold_has_thumb_opposition = bool(_has_thumb_opposition(hold_contact_mask))
    hold_cylinder_palm_clearance_error_m, hold_cylinder_opposition_cos = _cylinder_grasp_geometry_metrics(
        config,
        np.asarray(hold_result["object_pose"], dtype=np.float64) if run_hold_test else final_object_pose,
        np.asarray(hold_result["semantic_sites_world"], dtype=np.float64) if run_hold_test else final_sites,
        source_contact_mask_12,
    )
    arm_table_clearance_m = runtime.get_arm_sweep_table_clearance()
    reachability_metrics = _reachability_metrics_world(
        object_pose_goal=object_pose_goal,
        semantic_sites_world=final_sites,
        arm_table_clearance_m=arm_table_clearance_m,
    )
    object_facing_shortfall = max(0.0, REACHABILITY_MIN_OBJECT_FACING_COS - reachability_metrics["object_facing_cos"])
    base_facing_violation = max(0.0, reachability_metrics["base_facing_cos"] - REACHABILITY_MAX_BASE_FACING_COS)
    downward_violation = max(0.0, REACHABILITY_MIN_DOWNWARD_APPROACH - reachability_metrics["downward_component"])
    arm_clearance_violation_sum = float(
        np.sum(np.maximum(REACHABILITY_MIN_ARM_TABLE_CLEARANCE_M - arm_table_clearance_m, 0.0))
    )

    object_translation_weight = 2.00 if config.object_geom_type == "cylinder" else 1.25
    object_rotation_weight = 0.0200 if config.object_geom_type == "cylinder" else 0.0008
    source_frame_weight = 0.0001 if config.object_geom_type == "cylinder" else 0.0002
    score = (
        site_rmse
        + 28.0 * float(final_contact_diag["total_penetration_m"])
        + 48.0 * float(final_contact_diag["max_penetration_m"])
        + 0.02 * float(bool(final_contact_diag["table_contact"]))
        + 0.20 * translation_norm
        + 0.05 * rotation_norm
        + object_translation_weight * object_translation_drift_m
        + object_rotation_weight * object_rotation_drift_deg
        + 0.01 * float(np.linalg.norm(candidate_hand_qpos - runtime.home_actuated[6:]))
        + source_frame_weight * frame_error_deg
        + 0.45 * float(max(0.0, 2.0 - settled_contact_group_count))
        + 0.25 * float(max(0.0, 1.0 - settled_hard_contact_group_count))
        + 0.45 * float(not settled_has_thumb_opposition)
        + 1.20 * object_facing_shortfall
        + 1.20 * base_facing_violation
        + 0.80 * downward_violation
        + 3.50 * arm_clearance_violation_sum
        + 4.0 * max(0.0, cylinder_palm_clearance_error_m - PROJECTION_CYLINDER_PALM_CLEARANCE_TOL_M)
        + 1.50 * max(0.0, cylinder_opposition_cos - PROJECTION_CYLINDER_OPPOSITION_TARGET_COS)
        + 14.0 * hold_object_drop_m
        + 2.5 * hold_object_translation_m
        + 0.050 * hold_object_rotation_deg
        + 1.50 * float(max(0.0, 2.0 - hold_hybrid_contact_group_count))
        + 1.00 * float(max(0.0, 1.0 - hold_hard_contact_group_count))
        + 1.50 * float(not hold_has_thumb_opposition)
        + 6.0 * max(0.0, hold_cylinder_palm_clearance_error_m - PROJECTION_CYLINDER_HOLD_PALM_CLEARANCE_TOL_M)
        + 2.00 * max(0.0, hold_cylinder_opposition_cos - PROJECTION_CYLINDER_HOLD_OPPOSITION_TARGET_COS)
        - 0.03 * hold_hard_contact_group_count
    )

    return {
        "score": float(score),
        "wrist_translation_local": wrist_translation_local.copy(),
        "wrist_rotvec_local": wrist_rotvec_local.copy(),
        "arm_qpos": final_actuated_qpos[:6].copy(),
        "hand_qpos": final_actuated_qpos[6:].copy(),
        "semantic_sites_world": final_sites.copy(),
        "wrist_pose_world": final_wrist_pose.copy(),
        # Keep the training target in the original object frame. Settled object drift is
        # measured separately and should penalize bad candidates rather than redefine the target.
        "object_pose_goal": np.asarray(object_pose_goal, dtype=np.float64).copy(),
        "settled_object_pose": final_object_pose.copy(),
        "contact_mask_12": final_contact_mask.copy(),
        "physical_contact_mask_12": final_hard_contact_mask.copy(),
        "proximity_contact_mask_12": final_proximity_mask.copy(),
        "contact_proximity_score_12": final_proximity_scores.copy(),
        "contact_distance_12_m": final_contact_distances.copy(),
        "contact_points_world_12": final_contact_points_world.copy(),
        "contact_candidate_point_sets_world": np.asarray(final_contact_candidate_point_sets_world, dtype=np.float64).copy(),
        "contact_candidate_normal_sets_world": np.asarray(final_contact_candidate_normal_sets_world, dtype=np.float64).copy(),
        "contact_candidate_points_world": np.asarray(final_contact_candidate_point_sets_world, dtype=np.float64).reshape(-1, 3).copy(),
        "contact_candidate_family_indices": np.asarray(final_contact_candidate_families, dtype=np.int64).copy(),
        "contact_forces_12": np.asarray(final_contact_diag["forces"], dtype=np.float64).copy(),
        "table_contact": bool(final_contact_diag["table_contact"]),
        "total_penetration_m": float(final_contact_diag["total_penetration_m"]),
        "max_penetration_m": float(final_contact_diag["max_penetration_m"]),
        "source_wrist_error_m": wrist_error,
        "source_palm_error_m": palm_error,
        "source_tip_rmse_m": tip_rmse,
        "source_site_rmse_m": site_rmse,
        "source_semantic_frame_error_deg": frame_error_deg,
        "source_contact_hamming": float(np.abs(final_contact_mask - source_contact_mask_12).sum()),
        "source_contact_misses": target_contact_misses,
        "source_extra_contacts": extra_contacts,
        "source_matched_target_contacts": matched_target_contacts,
        "required_target_contacts": float(required_contacts),
        "settled_contact_group_count": settled_contact_group_count,
        "settled_hard_contact_group_count": settled_hard_contact_group_count,
        "settled_has_thumb_opposition": settled_has_thumb_opposition,
        "reach_object_facing_cos": float(reachability_metrics["object_facing_cos"]),
        "reach_base_facing_cos": float(reachability_metrics["base_facing_cos"]),
        "reach_downward_component": float(reachability_metrics["downward_component"]),
        "reach_min_arm_table_clearance_m": float(reachability_metrics["min_arm_table_clearance_m"]),
        "cylinder_palm_clearance_error_m": float(cylinder_palm_clearance_error_m),
        "cylinder_opposition_cos": float(cylinder_opposition_cos),
        "object_translation_drift_m": object_translation_drift_m,
        "object_rotation_drift_deg": object_rotation_drift_deg,
        "settled_total_penetration_m": float(settled_contact_diag["total_penetration_m"]),
        "settled_max_penetration_m": float(settled_contact_diag["max_penetration_m"]),
        "hold_table_contact": bool(final_contact_diag["table_contact"]),
        "hold_object_drop_m": hold_object_drop_m,
        "hold_object_translation_m": hold_object_translation_m,
        "hold_object_rotation_deg": hold_object_rotation_deg,
        "hold_matched_target_contacts": hold_matched_target_contacts,
        "hold_hybrid_contact_group_count": hold_hybrid_contact_group_count,
        "hold_hard_contact_group_count": hold_hard_contact_group_count,
        "hold_has_thumb_opposition": hold_has_thumb_opposition,
        "hold_cylinder_palm_clearance_error_m": float(hold_cylinder_palm_clearance_error_m),
        "hold_cylinder_opposition_cos": float(hold_cylinder_opposition_cos),
        "hold_tested": bool(run_hold_test),
    }


def _project_genhand_target(
    runtime: RobotSceneModel,
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_wrist_pose_world: np.ndarray,
    source_semantic_sites_world: np.ndarray,
    source_keypoints_obj: np.ndarray,
    source_contact_mask_12: np.ndarray,
    source_dgrasp_qpos_world: np.ndarray | None,
    initial_hand_qpos: np.ndarray,
) -> dict[str, Any]:
    base_hand_qpos = runtime.clamp_hand(np.asarray(initial_hand_qpos, dtype=np.float64))
    required_contacts = _required_target_contact_count(source_contact_mask_12)
    source_contact_targets_obj, source_contact_normals_obj, source_fc_metrics = _human_surface_contact_anchors_obj(
        config=config,
        object_pose_goal=object_pose_goal,
        source_dgrasp_qpos_world=source_dgrasp_qpos_world,
    )
    seed_states = _box_genhand_seed_states(
        runtime=runtime,
        config=config,
        object_pose_goal=object_pose_goal,
        source_wrist_pose_world=source_wrist_pose_world,
        source_semantic_sites_world=source_semantic_sites_world,
        source_keypoints_obj=source_keypoints_obj,
        source_contact_mask_12=source_contact_mask_12,
        source_dgrasp_qpos_world=source_dgrasp_qpos_world,
        base_hand_qpos=base_hand_qpos,
    )

    candidate_pool: list[dict[str, Any]] = []
    global_best: dict[str, Any] | None = None
    for seed_translation, seed_rotvec, seed_hand_qpos in seed_states:
        candidate: dict[str, Any] | None = None
        teacher_cost = float("inf")
        try:
            optimized_translation, optimized_rotvec, optimized_hand_qpos, teacher_cost = _optimize_genhand_seed(
                runtime=runtime,
                config=config,
                object_pose_goal=object_pose_goal,
                source_wrist_pose_world=source_wrist_pose_world,
                source_semantic_sites_world=source_semantic_sites_world,
                source_keypoints_obj=source_keypoints_obj,
                source_contact_mask_12=source_contact_mask_12,
                source_dgrasp_qpos_world=source_dgrasp_qpos_world,
                initial_translation_local=seed_translation,
                initial_rotvec_local=seed_rotvec,
                initial_hand_qpos=seed_hand_qpos,
            )
            candidate = _evaluate_projected_candidate(
                runtime=runtime,
                config=config,
                object_pose_goal=object_pose_goal,
                source_wrist_pose_world=source_wrist_pose_world,
                source_semantic_sites_world=source_semantic_sites_world,
                source_contact_mask_12=source_contact_mask_12,
                wrist_translation_local=optimized_translation,
                wrist_rotvec_local=optimized_rotvec,
                hand_qpos=optimized_hand_qpos,
                run_hold_test=False,
            )
        except Exception:
            try:
                candidate = _evaluate_projected_candidate(
                    runtime=runtime,
                    config=config,
                    object_pose_goal=object_pose_goal,
                    source_wrist_pose_world=source_wrist_pose_world,
                    source_semantic_sites_world=source_semantic_sites_world,
                    source_contact_mask_12=source_contact_mask_12,
                    wrist_translation_local=np.asarray(seed_translation, dtype=np.float64),
                    wrist_rotvec_local=np.asarray(seed_rotvec, dtype=np.float64),
                    hand_qpos=np.asarray(seed_hand_qpos, dtype=np.float64),
                    run_hold_test=False,
                )
            except Exception:
                candidate = None
        if candidate is None:
            continue
        candidate["teacher_cost"] = float(teacher_cost)
        candidate.update({f"genhand_{k}": float(v) for k, v in source_fc_metrics.items()})
        candidate.update(
            _candidate_contact_anchor_metrics(
                candidate=candidate,
                object_pose_goal=object_pose_goal,
                human_contact_targets_obj=source_contact_targets_obj,
                human_contact_normals_obj=source_contact_normals_obj,
            )
        )
        candidate_pool.append(candidate)
        if _is_better_projected_candidate(candidate, global_best, required_contacts):
            global_best = candidate

    if global_best is None:
        raise RuntimeError("GenHand-style retargeting failed to produce a candidate.")

    unique_shortlist: list[dict[str, Any]] = []
    seen_keys: set[tuple[float, ...]] = set()
    for candidate in sorted(candidate_pool, key=lambda item: _projection_preshortlist_rank_key(item, required_contacts)):
        key = tuple(
            np.round(
                np.concatenate(
                    [
                        np.asarray(candidate["wrist_translation_local"], dtype=np.float64),
                        np.asarray(candidate["wrist_rotvec_local"], dtype=np.float64),
                        np.asarray(candidate["hand_qpos"], dtype=np.float64),
                    ]
                ),
                5,
            ).tolist()
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_shortlist.append(candidate)
        if len(unique_shortlist) >= 6:
            break

    hold_validated_best: dict[str, Any] | None = None
    for candidate in unique_shortlist:
        validated = _evaluate_projected_candidate(
            runtime=runtime,
            config=config,
            object_pose_goal=object_pose_goal,
            source_wrist_pose_world=source_wrist_pose_world,
            source_semantic_sites_world=source_semantic_sites_world,
            source_contact_mask_12=source_contact_mask_12,
            wrist_translation_local=np.asarray(candidate["wrist_translation_local"], dtype=np.float64),
            wrist_rotvec_local=np.asarray(candidate["wrist_rotvec_local"], dtype=np.float64),
            hand_qpos=np.asarray(candidate["hand_qpos"], dtype=np.float64),
            run_hold_test=True,
        )
        validated["teacher_cost"] = float(candidate.get("teacher_cost", 0.0))
        for key, value in candidate.items():
            if isinstance(key, str) and (key.startswith("genhand_") or key.startswith("anchor_")):
                validated[key] = value
        if _is_better_projected_candidate(validated, hold_validated_best, required_contacts):
            hold_validated_best = validated

    if hold_validated_best is None:
        raise RuntimeError("GenHand-style retargeting failed during hold validation.")

    hold_validated_best["retreat_m"] = float(
        max(0.0, -float(np.asarray(hold_validated_best["wrist_translation_local"], dtype=np.float64)[2]))
    )
    hold_validated_best["hand_open_blend"] = float(
        np.linalg.norm(np.asarray(hold_validated_best["hand_qpos"], dtype=np.float64) - base_hand_qpos)
        / max(np.linalg.norm(runtime.hand_upper - runtime.hand_lower), 1e-8)
    )
    return hold_validated_best


def _project_feasible_target(
    runtime: RobotSceneModel,
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_wrist_pose_world: np.ndarray,
    source_semantic_sites_world: np.ndarray,
    source_keypoints_obj: np.ndarray,
    source_contact_mask_12: np.ndarray,
    source_dgrasp_qpos_world: np.ndarray | None,
    initial_arm_qpos: np.ndarray,
    initial_hand_qpos: np.ndarray,
) -> dict[str, Any]:
    del initial_arm_qpos
    projected = _project_genhand_target(
        runtime=runtime,
        config=config,
        object_pose_goal=object_pose_goal,
        source_wrist_pose_world=source_wrist_pose_world,
        source_semantic_sites_world=source_semantic_sites_world,
        source_keypoints_obj=source_keypoints_obj,
        source_contact_mask_12=source_contact_mask_12,
        source_dgrasp_qpos_world=source_dgrasp_qpos_world,
        initial_hand_qpos=initial_hand_qpos,
    )
    projected["retarget_method"] = "genhand_direct"
    return projected


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
        source_dgrasp_qpos_world = final_qpos[label_idx].copy()
        source_dgrasp_qpos_world[:3] += workspace_translation
        reachability_delta = _compute_reachability_alignment_delta(
            config=config,
            goal_pose=goal_pose,
            init_pose=init_pose,
            source_wrist_pose_world=source_wrist_pose_goal_world,
        )
        goal_pose[:3] += reachability_delta
        init_pose[:3] += reachability_delta
        source_semantic_sites_world = source_semantic_sites_world + reachability_delta[None, :]
        source_wrist_pose_goal_world[:3] += reachability_delta
        source_root_pose_world[:3] += reachability_delta
        source_dgrasp_qpos_world[:3] += reachability_delta

        arm_qpos = solve_arm_wrist_palm_ik(
            runtime=runtime,
            target_wrist_pose_world=source_wrist_pose_goal_world,
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
            source_keypoints_obj=final_ee_rel[label_idx],
            source_contact_mask_12=source_contact_mask,
            source_dgrasp_qpos_world=source_dgrasp_qpos_world,
            initial_arm_qpos=arm_qpos,
            initial_hand_qpos=hand_qpos,
        )
        projected_semantic_sites_world = np.asarray(projected["semantic_sites_world"], dtype=np.float64)
        projected_wrist_pose_world = np.asarray(projected["wrist_pose_world"], dtype=np.float64)
        projected_object_pose_goal = np.asarray(projected["object_pose_goal"], dtype=np.float64)
        projected_semantic_sites_object = transform_points(projected_semantic_sites_world, inverse_pose7(projected_object_pose_goal))
        projected_wrist_pose_object = compose_pose7(inverse_pose7(projected_object_pose_goal), projected_wrist_pose_world)
        projected_contact_point_sets_object = transform_points(
            np.asarray(projected["contact_candidate_point_sets_world"], dtype=np.float64).reshape(-1, 3),
            inverse_pose7(projected_object_pose_goal),
        ).reshape(np.asarray(projected["contact_candidate_point_sets_world"], dtype=np.float64).shape)
        projected_contact_normal_sets_object = np.einsum(
            "ij,ksj->ksi",
            quat_wxyz_to_matrix(inverse_pose7(projected_object_pose_goal)[3:]),
            np.asarray(projected["contact_candidate_normal_sets_world"], dtype=np.float64),
        )
        anchor_metrics = _contact_anchor_metrics_obj(
            robot_contact_point_sets_obj=projected_contact_point_sets_object,
            robot_contact_normal_sets_obj=projected_contact_normal_sets_object,
            config=config,
            object_pose_goal=projected_object_pose_goal,
            source_dgrasp_qpos_world=source_dgrasp_qpos_world,
        )
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
            "projected_matched_target_contacts": float(projected["source_matched_target_contacts"]),
            "projected_required_target_contacts": float(projected["required_target_contacts"]),
            "projected_hard_contact_count": float(np.sum(np.asarray(projected["physical_contact_mask_12"], dtype=np.float64) > 0.5)),
            "projected_hybrid_contact_count": float(np.sum(np.asarray(projected["contact_mask_12"], dtype=np.float64) > 0.5)),
            "projected_contact_group_count": float(projected["settled_contact_group_count"]),
            "projected_hard_contact_group_count": float(projected["settled_hard_contact_group_count"]),
            "projected_has_thumb_opposition": float(bool(projected["settled_has_thumb_opposition"])),
            "projected_reach_object_facing_cos": float(projected["reach_object_facing_cos"]),
            "projected_reach_base_facing_cos": float(projected["reach_base_facing_cos"]),
            "projected_reach_downward_component": float(projected["reach_downward_component"]),
            "projected_reach_min_arm_table_clearance_m": float(projected["reach_min_arm_table_clearance_m"]),
            "projected_cylinder_palm_clearance_error_m": float(projected["cylinder_palm_clearance_error_m"]),
            "projected_cylinder_opposition_cos": float(projected["cylinder_opposition_cos"]),
            "projected_min_contact_distance_m": float(np.min(np.asarray(projected["contact_distance_12_m"], dtype=np.float64))),
            "projected_total_penetration_m": float(projected["total_penetration_m"]),
            "projected_max_penetration_m": float(projected["max_penetration_m"]),
            "projected_object_translation_drift_m": float(projected["object_translation_drift_m"]),
            "projected_object_rotation_drift_deg": float(projected["object_rotation_drift_deg"]),
            "projected_retreat_m": float(projected["retreat_m"]),
            "projected_hand_open_blend": float(projected["hand_open_blend"]),
            "projected_hold_object_drop_m": float(projected["hold_object_drop_m"]),
            "projected_hold_object_translation_m": float(projected["hold_object_translation_m"]),
            "projected_hold_object_rotation_deg": float(projected["hold_object_rotation_deg"]),
            "projected_hold_table_contact": float(bool(projected["hold_table_contact"])),
            "projected_hold_hybrid_contact_group_count": float(projected["hold_hybrid_contact_group_count"]),
            "projected_hold_hard_contact_group_count": float(projected["hold_hard_contact_group_count"]),
            "projected_hold_has_thumb_opposition": float(bool(projected["hold_has_thumb_opposition"])),
            "projected_hold_cylinder_palm_clearance_error_m": float(projected["hold_cylinder_palm_clearance_error_m"]),
            "projected_hold_cylinder_opposition_cos": float(projected["hold_cylinder_opposition_cos"]),
            "projected_hold_matched_target_contacts": float(projected["hold_matched_target_contacts"]),
            "projected_anchor_rmse_m": float(anchor_metrics["anchor_rmse_m"]),
            "projected_anchor_max_error_m": float(anchor_metrics["anchor_max_error_m"]),
            "projected_anchor_pairs": float(anchor_metrics["anchor_pairs"]),
            "projected_anchor_assignment_norm_cost": float(anchor_metrics.get("anchor_assignment_norm_cost", 0.0)),
            "projected_anchor_assignment_method": float(anchor_metrics.get("anchor_assignment_method", 0.0)),
            "genhand_fc_loss": float(anchor_metrics.get("genhand_fc_loss", projected.get("genhand_fc_loss", 0.0))),
            "genhand_fc_net_wrench": float(anchor_metrics.get("genhand_fc_net_wrench", projected.get("genhand_fc_net_wrench", 0.0))),
            "genhand_fc_lin_ind": float(anchor_metrics.get("genhand_fc_lin_ind", projected.get("genhand_fc_lin_ind", 0.0))),
            "genhand_fc_intfc": float(anchor_metrics.get("genhand_fc_intfc", projected.get("genhand_fc_intfc", 0.0))),
            "genhand_cluster_candidate_count": float(anchor_metrics.get("genhand_cluster_candidate_count", projected.get("genhand_cluster_candidate_count", 0.0))),
            "genhand_cluster_score_mean": float(anchor_metrics.get("genhand_cluster_score_mean", projected.get("genhand_cluster_score_mean", 0.0))),
            "genhand_target_anchor_rmse_m": float(anchor_metrics.get("genhand_target_anchor_rmse_m", projected.get("genhand_target_anchor_rmse_m", 0.0))),
            "projection_score": float(projected["score"]),
            "retarget_method": str(projected.get("retarget_method", "genhand_direct")),
            "teacher_cost": float(projected.get("teacher_cost", 0.0)),
            "optimizer_cost": optimizer_cost,
        }
        source_tip_threshold = 0.12 if config.object_geom_type == "cylinder" else 0.085
        object_translation_threshold = 0.12 if config.object_geom_type == "cylinder" else PROJECTION_HARD_OBJECT_TRANSLATION_DRIFT_M
        object_rotation_threshold = 45.0 if config.object_geom_type == "cylinder" else PROJECTION_HARD_OBJECT_ROTATION_DRIFT_DEG
        hold_drop_threshold = 0.08 if config.object_geom_type == "cylinder" else PROJECTION_HOLD_MAX_DROP_M
        hold_translation_threshold = 0.10 if config.object_geom_type == "cylinder" else PROJECTION_HOLD_MAX_TRANSLATION_M
        hold_rotation_threshold = 35.0 if config.object_geom_type == "cylinder" else PROJECTION_HOLD_MAX_ROTATION_DEG
        valid_execution = bool(
            fit_error["projected_max_penetration_m"] <= PROJECTION_HARD_MAX_PENETRATION_M
            and fit_error["source_tip_rmse_m"] <= source_tip_threshold
            and fit_error["projected_object_translation_drift_m"] <= object_translation_threshold
            and fit_error["projected_object_rotation_drift_deg"] <= object_rotation_threshold
            and not bool(projected["table_contact"])
            and fit_error["projected_hold_object_drop_m"] <= hold_drop_threshold
            and fit_error["projected_hold_object_translation_m"] <= hold_translation_threshold
            and fit_error["projected_hold_object_rotation_deg"] <= hold_rotation_threshold
            and not bool(projected["hold_table_contact"])
            and fit_error["projected_hold_hybrid_contact_group_count"] >= 2.0
            and fit_error["projected_hold_hard_contact_group_count"] >= 1.0
            and bool(projected["hold_has_thumb_opposition"])
            and fit_error["projected_contact_group_count"] >= 2.0
            and fit_error["projected_hard_contact_group_count"] >= 1.0
            and bool(projected["settled_has_thumb_opposition"])
            and fit_error["projected_reach_object_facing_cos"] >= REACHABILITY_MIN_OBJECT_FACING_COS
            and fit_error["projected_reach_base_facing_cos"] <= REACHABILITY_MAX_BASE_FACING_COS
            and fit_error["projected_reach_downward_component"] >= REACHABILITY_MIN_DOWNWARD_APPROACH
            and fit_error["projected_reach_min_arm_table_clearance_m"] >= REACHABILITY_MIN_ARM_TABLE_CLEARANCE_M
            and (
                config.object_geom_type != "cylinder"
                or (
                    fit_error["projected_cylinder_palm_clearance_error_m"] <= 0.018
                    and fit_error["projected_cylinder_opposition_cos"] <= -0.05
                    and fit_error["projected_hold_cylinder_palm_clearance_error_m"] <= 0.022
                    and fit_error["projected_hold_cylinder_opposition_cos"] <= 0.05
                )
            )
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
                physical_contact_mask_12=[float(v) for v in projected["physical_contact_mask_12"]],
                proximity_contact_mask_12=[float(v) for v in projected["proximity_contact_mask_12"]],
                contact_proximity_score_12=[float(v) for v in projected["contact_proximity_score_12"]],
                contact_distance_12_m=[float(v) for v in projected["contact_distance_12_m"]],
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
                "projected_physical_contact_mask_12": [int(round(v)) for v in projected["physical_contact_mask_12"]],
                "projected_proximity_contact_mask_12": [int(round(v)) for v in projected["proximity_contact_mask_12"]],
                "projected_contact_mask_12": [int(round(v)) for v in projected["contact_mask_12"]],
                "projected_contact_distance_12_m": [float(v) for v in projected["contact_distance_12_m"]],
                "projected_hold_object_drop_m": float(projected["hold_object_drop_m"]),
                "projected_hold_hybrid_contact_group_count": float(projected["hold_hybrid_contact_group_count"]),
                "projected_hold_hard_contact_group_count": float(projected["hold_hard_contact_group_count"]),
                "projected_hold_has_thumb_opposition": bool(projected["hold_has_thumb_opposition"]),
                "projected_contact_group_count": float(projected["settled_contact_group_count"]),
                "projected_hard_contact_group_count": float(projected["settled_hard_contact_group_count"]),
                "projected_has_thumb_opposition": bool(projected["settled_has_thumb_opposition"]),
                "projected_reach_object_facing_cos": float(projected["reach_object_facing_cos"]),
                "projected_reach_base_facing_cos": float(projected["reach_base_facing_cos"]),
                "projected_reach_downward_component": float(projected["reach_downward_component"]),
                "projected_reach_min_arm_table_clearance_m": float(projected["reach_min_arm_table_clearance_m"]),
                "projected_cylinder_palm_clearance_error_m": float(projected["cylinder_palm_clearance_error_m"]),
                "projected_cylinder_opposition_cos": float(projected["cylinder_opposition_cos"]),
                "projected_hold_cylinder_palm_clearance_error_m": float(projected["hold_cylinder_palm_clearance_error_m"]),
                "projected_hold_cylinder_opposition_cos": float(projected["hold_cylinder_opposition_cos"]),
                "projected_anchor_rmse_m": float(anchor_metrics["anchor_rmse_m"]),
                "projected_anchor_max_error_m": float(anchor_metrics["anchor_max_error_m"]),
                "projected_anchor_pairs": float(anchor_metrics["anchor_pairs"]),
                "projected_anchor_assignment_norm_cost": float(anchor_metrics.get("anchor_assignment_norm_cost", 0.0)),
                "projected_anchor_assignment_method": float(anchor_metrics.get("anchor_assignment_method", 0.0)),
                "retarget_method": str(projected.get("retarget_method", "genhand_direct")),
                **fit_error,
            }
        )

        previous_arm_qpos = np.asarray(projected["arm_qpos"], dtype=np.float64)
        previous_hand_qpos = np.asarray(projected["hand_qpos"], dtype=np.float64)

    save_pose_driven_samples(samples_path, samples)
    pose_driven_report_path(config).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return samples
