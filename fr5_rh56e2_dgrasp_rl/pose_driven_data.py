from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial.transform import Rotation

from .kinematics import solve_arm_wrist_palm_ik
from .paths import BUNDLED_DGRASP_DIR, WORKSPACE_DIR, ensure_runtime_dirs
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
CONTACT_DRIVEN_MAX_NFEV = 42
CONTACT_DRIVEN_POSITION_WEIGHT = 4.0
CONTACT_DRIVEN_DIRECTION_WEIGHT = 1.5
CONTACT_DRIVEN_PALM_WEIGHT = 2.0
CONTACT_DRIVEN_PENETRATION_WEIGHT = 18.0
CONTACT_DRIVEN_TABLE_WEIGHT = 12.0
CONTACT_DRIVEN_WRIST_REG_WEIGHT = 0.35
CONTACT_DRIVEN_HAND_REG_WEIGHT = 0.10
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
REACHABILITY_TARGET_WRIST_XY_M = np.array([0.66, -0.52], dtype=np.float64)
REACHABILITY_MAX_SHIFT_XY_M = np.array([0.10, 0.14], dtype=np.float64)
REACHABILITY_TABLE_MARGIN_XY_M = np.array([0.06, 0.08], dtype=np.float64)
REACHABILITY_MIN_OBJECT_FACING_COS = 0.45
REACHABILITY_MAX_BASE_FACING_COS = 0.20
REACHABILITY_MIN_DOWNWARD_APPROACH = 0.15
REACHABILITY_MIN_ARM_TABLE_CLEARANCE_M = 0.045
CONTACT_FAMILY_GROUPS_12: tuple[tuple[int, ...], ...] = (
    (0,),
    (1, 2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
)
GENHAND_ROBOT_FAMILY_CAPACITY = {
    family_name: sum(1 for _, family in GENHAND_ROBOT_CONTACT_CANDIDATES if family == family_name)
    for family_name in GENHAND_CONTACT_FAMILY_NAMES
}
CONTACT_FAMILY_TO_FINGER_INDEX = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
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

    runtime.set_table_height(0.0)
    runtime.set_arm_hold_mode(True, arm_kp_scale=PROJECTION_HOLD_ARM_KP_SCALE)
    support_force_world = np.array(
        [0.0, 0.0, float(config.object_mass_kg) * PROJECTION_HOLD_SUPPORT_FORCE_SCALE],
        dtype=np.float64,
    )
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


def _cylinder_hand_wrap_seed(runtime: RobotSceneModel, source_contact_mask_12: np.ndarray, strength: float = 1.0) -> np.ndarray:
    group_presence = _contact_group_presence(source_contact_mask_12)
    base_coeff = np.array([0.30, 0.40, 0.46, 0.46, 0.44, 0.40], dtype=np.float64)
    contact_bonus = np.array(
        [
            0.12 if group_presence[1] > 0.5 else 0.04,
            0.18 if group_presence[1] > 0.5 else 0.08,
            0.16 if group_presence[2] > 0.5 else 0.06,
            0.16 if group_presence[3] > 0.5 else 0.06,
            0.14 if group_presence[4] > 0.5 else 0.05,
            0.14 if group_presence[5] > 0.5 else 0.05,
        ],
        dtype=np.float64,
    )
    coeff = np.clip(base_coeff + strength * contact_bonus, 0.0, 0.92)
    wrap_seed = runtime.home_actuated[6:] + coeff * (runtime.hand_upper - runtime.home_actuated[6:])
    return runtime.clamp_hand(wrap_seed)


def _cylinder_grasp_seed_states(
    runtime: RobotSceneModel,
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_wrist_pose_world: np.ndarray,
    source_semantic_sites_world: np.ndarray,
    source_contact_mask_12: np.ndarray,
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
        wrap_seed = _cylinder_hand_wrap_seed(runtime, source_contact_mask_12, strength=squeeze_strength)
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
    source_contact_mask_12: np.ndarray,
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
    source_contact_mask_12 = np.asarray(source_contact_mask_12, dtype=np.float64)

    active_thumb = thumb_targets[source_contact_mask_12[1:4] > 0.5]
    active_fingers = finger_targets[source_contact_mask_12[4:] > 0.5]
    thumb_center_obj = np.mean(active_thumb, axis=0) if active_thumb.size else np.mean(thumb_targets, axis=0)
    finger_center_obj = np.mean(active_fingers, axis=0) if active_fingers.size else np.mean(finger_targets, axis=0)

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

    base_wrap = _cylinder_hand_wrap_seed(runtime, source_contact_mask_12, strength=1.15)
    stronger_wrap = _cylinder_hand_wrap_seed(runtime, source_contact_mask_12, strength=1.35)

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


def _source_contact_family_mask(source_contact_mask_12: np.ndarray) -> np.ndarray:
    mask_12 = np.asarray(source_contact_mask_12, dtype=np.float64).reshape(12)
    family_mask = np.zeros(len(GENHAND_CONTACT_FAMILY_NAMES), dtype=np.float64)
    family_mask[0] = float(mask_12[0] > 0.5)
    family_mask[1] = float(np.max(mask_12[1:4]) > 0.5)
    family_mask[2] = float(np.max(mask_12[4:6]) > 0.5)
    family_mask[3] = float(np.max(mask_12[6:8]) > 0.5)
    family_mask[4] = float(np.max(mask_12[8:10]) > 0.5)
    family_mask[5] = float(np.max(mask_12[10:12]) > 0.5)
    return family_mask


def _human_contact_candidate_points_obj(source_keypoints_obj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.asarray(source_keypoints_obj, dtype=np.float64).reshape(21, 3)
    semantic_sites = mano_semantic_sites_from_keypoints(keypoints)
    points: list[np.ndarray] = []
    family_indices: list[int] = []

    def add(point: np.ndarray, family_index: int) -> None:
        points.append(np.asarray(point, dtype=np.float64))
        family_indices.append(int(family_index))

    def midpoint(a: int, b: int) -> np.ndarray:
        return 0.5 * (keypoints[a] + keypoints[b])

    # Palm candidates: keep a small set of broad palm anchors rather than a single fixed palm slot.
    add(semantic_sites[1], 0)
    add(np.mean(keypoints[[1, 5]], axis=0), 0)
    add(np.mean(keypoints[[9, 13]], axis=0), 0)
    add(np.mean(keypoints[[1, 5, 9, 13]], axis=0), 0)

    finger_chains = (
        (1, (17, 18, 19, 20)),  # thumb
        (2, (1, 2, 3, 4)),      # index
        (3, (5, 6, 7, 8)),      # middle
        (4, (9, 10, 11, 12)),   # ring
        (5, (13, 14, 15, 16)),  # little
    )
    for family_index, chain in finger_chains:
        add(midpoint(chain[0], chain[1]), family_index)
        add(midpoint(chain[1], chain[2]), family_index)
        add(midpoint(chain[2], chain[3]), family_index)
        add(keypoints[chain[3]], family_index)

    return np.asarray(points, dtype=np.float64), np.asarray(family_indices, dtype=np.int64)


def _cluster_points_to_surface_anchors(
    points_obj: np.ndarray,
    normals_obj: np.ndarray,
    family_indices: np.ndarray,
    family_mask: np.ndarray,
    family_capacities: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clustered_points: list[np.ndarray] = []
    clustered_normals: list[np.ndarray] = []
    clustered_families: list[int] = []

    points_obj = np.asarray(points_obj, dtype=np.float64)
    normals_obj = np.asarray(normals_obj, dtype=np.float64)
    family_indices = np.asarray(family_indices, dtype=np.int64)
    family_mask = np.asarray(family_mask, dtype=np.float64)

    for family_index, family_name in enumerate(GENHAND_CONTACT_FAMILY_NAMES):
        if family_mask[family_index] <= 0.5:
            continue
        member_indices = np.flatnonzero(family_indices == family_index)
        if member_indices.size == 0:
            continue
        family_points = points_obj[member_indices]
        family_normals = normals_obj[member_indices]
        unique_points = np.unique(np.round(family_points, decimals=6), axis=0)
        target_clusters = max(
            1,
            min(
                int(family_capacities[family_name]),
                int(member_indices.size),
                int(unique_points.shape[0]),
            ),
        )

        if target_clusters == 1:
            centroid = np.mean(family_points, axis=0)
            nearest_index = int(np.argmin(np.linalg.norm(family_points - centroid, axis=1)))
            clustered_points.append(family_points[nearest_index])
            clustered_normals.append(normalize(np.mean(family_normals, axis=0)))
            clustered_families.append(family_index)
            continue

        try:
            centroids, labels = kmeans2(family_points, target_clusters, minit="points")
            labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        except Exception:
            labels = np.arange(family_points.shape[0], dtype=np.int64) % target_clusters
            centroids = np.vstack(
                [
                    np.mean(family_points[labels == cluster_index], axis=0)
                    for cluster_index in range(target_clusters)
                ]
            )
        for cluster_index in range(target_clusters):
            cluster_member_indices = np.flatnonzero(labels == cluster_index)
            if cluster_member_indices.size == 0:
                continue
            cluster_points = family_points[cluster_member_indices]
            cluster_normals = family_normals[cluster_member_indices]
            centroid = np.asarray(centroids[cluster_index], dtype=np.float64)
            nearest_local_index = int(np.argmin(np.linalg.norm(cluster_points - centroid, axis=1)))
            clustered_points.append(cluster_points[nearest_local_index])
            clustered_normals.append(normalize(np.mean(cluster_normals, axis=0)))
            clustered_families.append(family_index)

    if not clustered_points:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.asarray(clustered_points, dtype=np.float64),
        np.asarray(clustered_normals, dtype=np.float64),
        np.asarray(clustered_families, dtype=np.int64),
    )


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


def _human_contact_target_normals_obj(config: TaskConfig, source_contact_targets_obj: np.ndarray) -> np.ndarray:
    targets = np.asarray(source_contact_targets_obj, dtype=np.float64).reshape(12, 3)
    normals = np.zeros_like(targets)
    if config.object_geom_type == "box":
        dims = np.asarray(config.object_dims_m, dtype=np.float64)
        for idx in range(targets.shape[0]):
            normals[idx] = _box_contact_normal_obj(targets[idx], dims)
        return normals
    if config.object_geom_type == "cylinder":
        for idx, point in enumerate(targets):
            radial = np.asarray([point[0], point[1], 0.0], dtype=np.float64)
            normals[idx] = normalize(radial)
        return normals
    return normals


def _human_surface_contact_anchors_obj(
    config: TaskConfig,
    source_keypoints_obj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_points, candidate_families = _human_contact_candidate_points_obj(source_keypoints_obj)
    projected_points, projected_normals = _project_targets_to_object_surface(config, candidate_points)
    family_mask = np.ones(len(GENHAND_CONTACT_FAMILY_NAMES), dtype=np.float64)
    return _cluster_points_to_surface_anchors(
        projected_points,
        projected_normals,
        candidate_families,
        family_mask,
        GENHAND_ROBOT_FAMILY_CAPACITY,
    )


def _assigned_contact_anchor_residuals(
    robot_contact_points_obj: np.ndarray,
    robot_contact_family_indices: np.ndarray,
    human_contact_targets_obj: np.ndarray,
    human_contact_normals_obj: np.ndarray,
    human_contact_family_indices: np.ndarray,
) -> tuple[list[np.ndarray], dict[int, np.ndarray], dict[str, float]]:
    robot_contact_points_obj = np.asarray(robot_contact_points_obj, dtype=np.float64).reshape(-1, 3)
    robot_contact_family_indices = np.asarray(robot_contact_family_indices, dtype=np.int64).reshape(-1)
    human_contact_targets_obj = np.asarray(human_contact_targets_obj, dtype=np.float64).reshape(-1, 3)
    human_contact_normals_obj = np.asarray(human_contact_normals_obj, dtype=np.float64).reshape(-1, 3)
    human_contact_family_indices = np.asarray(human_contact_family_indices, dtype=np.int64).reshape(-1)

    residuals: list[np.ndarray] = []
    family_normal_targets: dict[int, np.ndarray] = {}
    assigned_error_norms: list[float] = []
    assigned_pairs = 0

    for family_idx, _family_name in enumerate(GENHAND_CONTACT_FAMILY_NAMES):
        active_human_indices = np.flatnonzero(human_contact_family_indices == family_idx)
        if active_human_indices.size == 0:
            continue
        family_robot_indices = np.flatnonzero(robot_contact_family_indices == family_idx)
        if family_robot_indices.size == 0:
            continue
        family_robot_points = robot_contact_points_obj[family_robot_indices]
        family_human_targets = human_contact_targets_obj[active_human_indices]
        cost = np.linalg.norm(
            family_human_targets[:, None, :] - family_robot_points[None, :, :],
            axis=2,
        )
        row_ind, col_ind = linear_sum_assignment(cost)
        row_ind = np.asarray(row_ind, dtype=np.int64)
        col_ind = np.asarray(col_ind, dtype=np.int64)
        kept_pairs = min(len(row_ind), len(col_ind))
        if kept_pairs <= 0:
            continue
        robot_indices = family_robot_indices[col_ind[:kept_pairs]]
        human_indices = active_human_indices[row_ind[:kept_pairs]]
        diffs = robot_contact_points_obj[robot_indices] - human_contact_targets_obj[human_indices]
        residuals.extend(list(diffs))
        assigned_error_norms.extend(np.linalg.norm(diffs, axis=1).tolist())
        assigned_pairs += int(len(robot_indices))
        mean_normal = normalize(np.mean(human_contact_normals_obj[human_indices], axis=0))
        if np.linalg.norm(mean_normal) > 1e-8:
            family_normal_targets[family_idx] = mean_normal

    metrics = {
        "anchor_rmse_m": float(np.sqrt(np.mean(np.square(assigned_error_norms)))) if assigned_error_norms else 0.0,
        "anchor_max_error_m": float(np.max(assigned_error_norms)) if assigned_error_norms else 0.0,
        "anchor_pairs": float(assigned_pairs),
    }
    return residuals, family_normal_targets, metrics


def _contact_anchor_metrics_obj(
    robot_contact_points_obj: np.ndarray,
    robot_contact_family_indices: np.ndarray,
    source_keypoints_obj: np.ndarray,
    source_contact_mask_12: np.ndarray,
    config: TaskConfig,
) -> dict[str, float]:
    human_targets, human_normals, human_families = _human_surface_contact_anchors_obj(
        config,
        source_keypoints_obj,
    )
    family_mask = _source_contact_family_mask(source_contact_mask_12)
    active_human_indices = np.flatnonzero(family_mask[human_families] > 0.5)
    if active_human_indices.size > 0:
        human_targets = human_targets[active_human_indices]
        human_normals = human_normals[active_human_indices]
        human_families = human_families[active_human_indices]
    _, _, metrics = _assigned_contact_anchor_residuals(
        robot_contact_points_obj=robot_contact_points_obj,
        robot_contact_family_indices=robot_contact_family_indices,
        human_contact_targets_obj=human_targets,
        human_contact_normals_obj=human_normals,
        human_contact_family_indices=human_families,
    )
    return metrics


def _box_genhand_seed_states(
    runtime: RobotSceneModel,
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_wrist_pose_world: np.ndarray,
    source_semantic_sites_world: np.ndarray,
    source_keypoints_obj: np.ndarray,
    source_contact_mask_12: np.ndarray,
    base_hand_qpos: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    source_wrist_pose_object = compose_pose7(inverse_pose7(object_pose_goal), source_wrist_pose_world)
    source_semantic_sites_obj = transform_points(source_semantic_sites_world, inverse_pose7(object_pose_goal))
    source_frame_obj = semantic_frame_from_sites(source_semantic_sites_obj)
    source_contact_targets_obj, source_contact_normals_obj, source_contact_families = _human_surface_contact_anchors_obj(
        config,
        source_keypoints_obj,
    )
    family_mask = _source_contact_family_mask(source_contact_mask_12)
    active_anchor_indices = np.flatnonzero(family_mask[source_contact_families] > 0.5)
    contact_centroid = (
        np.mean(source_contact_targets_obj[active_anchor_indices], axis=0)
        if active_anchor_indices.size > 0
        else source_semantic_sites_obj[1]
    )
    if active_anchor_indices.size > 0:
        averaged_normal = normalize(np.mean(source_contact_normals_obj[active_anchor_indices], axis=0))
    else:
        averaged_normal = -source_frame_obj[:, 2]
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
    contact_biased_hand = runtime.clamp_hand(np.asarray(base_hand_qpos, dtype=np.float64) + _finger_contact_bias_from_mask(source_contact_mask_12))
    aggressive_hand = runtime.clamp_hand(np.asarray(base_hand_qpos, dtype=np.float64) + 1.25 * _finger_contact_bias_from_mask(source_contact_mask_12))

    seeds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [
        (np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), runtime.clamp_hand(base_hand_qpos)),
        (np.array([0.0, 0.0, -0.004], dtype=np.float64), np.zeros(3, dtype=np.float64), contact_biased_hand),
        (np.array([0.0, 0.0, 0.004], dtype=np.float64), np.zeros(3, dtype=np.float64), aggressive_hand),
        (np.array([0.006, 0.0, 0.0], dtype=np.float64), np.deg2rad(np.array([0.0, -6.0, 0.0], dtype=np.float64)), contact_biased_hand),
        (np.array([-0.006, 0.0, 0.0], dtype=np.float64), np.deg2rad(np.array([0.0, 6.0, 0.0], dtype=np.float64)), contact_biased_hand),
        (np.array([0.0, 0.008, 0.0], dtype=np.float64), np.deg2rad(np.array([4.0, 0.0, 0.0], dtype=np.float64)), contact_biased_hand),
        (np.array([0.0, -0.008, 0.0], dtype=np.float64), np.deg2rad(np.array([-4.0, 0.0, 0.0], dtype=np.float64)), contact_biased_hand),
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


def _optimize_contact_direction_seed(
    runtime: RobotSceneModel,
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_wrist_pose_world: np.ndarray,
    source_semantic_sites_world: np.ndarray,
    source_keypoints_obj: np.ndarray,
    source_contact_mask_12: np.ndarray,
    initial_translation_local: np.ndarray,
    initial_rotvec_local: np.ndarray,
    initial_hand_qpos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    source_contact_targets_obj = _human_contact_target_points_obj(source_keypoints_obj)
    source_finger_dirs_obj = _human_finger_directions_obj(source_keypoints_obj)
    source_semantic_sites_obj = transform_points(source_semantic_sites_world, inverse_pose7(object_pose_goal))
    object_rotation = quat_wxyz_to_matrix(np.asarray(object_pose_goal[3:], dtype=np.float64))
    inv_object_rotation = object_rotation.T
    table_top_z = runtime.get_table_top_z()
    group_presence = _contact_group_presence(source_contact_mask_12)
    active_contact_indices = np.flatnonzero(np.asarray(source_contact_mask_12, dtype=np.float64) > 0.5)
    active_fingers = [
        bool(group_presence[1] > 0.5),
        bool(group_presence[2] > 0.5),
        bool(group_presence[3] > 0.5),
        bool(group_presence[4] > 0.5),
        bool(group_presence[5] > 0.5),
    ]

    initial_translation_local = np.asarray(initial_translation_local, dtype=np.float64)
    initial_rotvec_local = np.asarray(initial_rotvec_local, dtype=np.float64)
    initial_hand_qpos = runtime.clamp_hand(np.asarray(initial_hand_qpos, dtype=np.float64))
    x0 = np.concatenate([initial_translation_local, initial_rotvec_local, initial_hand_qpos])

    lower = np.concatenate(
        [
            initial_translation_local - np.array([0.06, 0.06, 0.08], dtype=np.float64),
            initial_rotvec_local - np.deg2rad(np.array([35.0, 35.0, 55.0], dtype=np.float64)),
            runtime.hand_lower,
        ]
    )
    upper = np.concatenate(
        [
            initial_translation_local + np.array([0.06, 0.06, 0.08], dtype=np.float64),
            initial_rotvec_local + np.deg2rad(np.array([35.0, 35.0, 55.0], dtype=np.float64)),
            runtime.hand_upper,
        ]
    )

    def objective(x: np.ndarray) -> np.ndarray:
        translation_local = np.asarray(x[:3], dtype=np.float64)
        rotvec_local = np.asarray(x[3:6], dtype=np.float64)
        hand_qpos = runtime.clamp_hand(np.asarray(x[6:], dtype=np.float64))
        target_wrist_pose = apply_local_pose_delta(source_wrist_pose_world, translation_local, rotvec_local)
        arm_qpos = solve_arm_wrist_palm_ik(
            runtime=runtime,
            target_wrist_pose_world=target_wrist_pose,
            initial_arm_qpos=runtime.home_actuated[:6],
            hand_qpos=hand_qpos,
            iterations=max(36, int(config.conversion.arm_ik_iterations * 0.4)),
            damping=config.conversion.arm_ik_damping,
        )
        runtime.reset()
        runtime.set_object_pose(object_pose_goal)
        runtime.set_robot_actuated_qpos(np.concatenate([arm_qpos, hand_qpos]))

        contact_points_world = runtime.get_contact_proxy_points_world_12()
        contact_points_obj = transform_points(contact_points_world, inverse_pose7(object_pose_goal))
        finger_dirs_world = runtime.get_finger_directions_world()
        finger_dirs_obj = (inv_object_rotation @ finger_dirs_world.T).T
        semantic_sites_world = runtime.get_semantic_sites_world()
        semantic_sites_obj = transform_points(semantic_sites_world, inverse_pose7(object_pose_goal))
        diag = runtime.get_contact_diagnostics_12()
        distances = np.asarray(diag["distances_m"], dtype=np.float64)
        table_clearance = runtime.get_proxy_table_clearance_12()

        residuals: list[float] = []
        for group_idx in active_contact_indices.tolist():
            residuals.extend(
                (
                    CONTACT_DRIVEN_POSITION_WEIGHT
                    * (contact_points_obj[group_idx] - source_contact_targets_obj[group_idx])
                ).tolist()
            )

        residuals.extend(
            (CONTACT_DRIVEN_PALM_WEIGHT * (semantic_sites_obj[1] - source_semantic_sites_obj[1])).tolist()
        )

        for finger_idx, is_active in enumerate(active_fingers):
            if not is_active:
                continue
            residuals.extend(
                (
                    CONTACT_DRIVEN_DIRECTION_WEIGHT
                    * (finger_dirs_obj[finger_idx] - source_finger_dirs_obj[finger_idx])
                ).tolist()
            )

        penetration = np.maximum(-distances, 0.0)
        residuals.extend((CONTACT_DRIVEN_PENETRATION_WEIGHT * penetration).tolist())

        semantic_clearance = np.array(
            [
                semantic_sites_world[0, 2] - table_top_z,
                semantic_sites_world[1, 2] - table_top_z,
            ],
            dtype=np.float64,
        )
        table_violation = np.maximum(0.002 - np.concatenate([table_clearance, semantic_clearance]), 0.0)
        residuals.extend((CONTACT_DRIVEN_TABLE_WEIGHT * table_violation).tolist())

        residuals.extend((CONTACT_DRIVEN_WRIST_REG_WEIGHT * (translation_local - initial_translation_local)).tolist())
        residuals.extend((0.20 * (rotvec_local - initial_rotvec_local)).tolist())
        residuals.extend((CONTACT_DRIVEN_HAND_REG_WEIGHT * (hand_qpos - initial_hand_qpos)).tolist())

        if config.object_geom_type == "cylinder":
            palm_clearance_error_m, opposition_cos = _cylinder_grasp_geometry_metrics(
                config,
                object_pose_goal,
                semantic_sites_world,
                source_contact_mask_12,
            )
            residuals.append(6.0 * max(0.0, palm_clearance_error_m - PROJECTION_CYLINDER_PALM_CLEARANCE_TOL_M))
            residuals.append(2.0 * max(0.0, opposition_cos - PROJECTION_CYLINDER_OPPOSITION_TARGET_COS))

        return np.asarray(residuals, dtype=np.float64)

    result = least_squares(
        objective,
        x0,
        bounds=(lower, upper),
        max_nfev=CONTACT_DRIVEN_MAX_NFEV,
    )
    optimized = np.asarray(result.x, dtype=np.float64)
    return (
        optimized[:3].copy(),
        optimized[3:6].copy(),
        runtime.clamp_hand(optimized[6:]).copy(),
        float(result.cost),
    )


def _optimize_genhand_seed(
    runtime: RobotSceneModel,
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_wrist_pose_world: np.ndarray,
    source_semantic_sites_world: np.ndarray,
    source_keypoints_obj: np.ndarray,
    source_contact_mask_12: np.ndarray,
    initial_translation_local: np.ndarray,
    initial_rotvec_local: np.ndarray,
    initial_hand_qpos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    source_wrist_pose_object = compose_pose7(inverse_pose7(object_pose_goal), source_wrist_pose_world)
    source_semantic_sites_obj = transform_points(source_semantic_sites_world, inverse_pose7(object_pose_goal))
    source_contact_targets_obj, source_contact_normals_obj, source_contact_families = _human_surface_contact_anchors_obj(
        config,
        source_keypoints_obj,
    )
    family_mask = _source_contact_family_mask(source_contact_mask_12)
    active_anchor_indices = np.flatnonzero(family_mask[source_contact_families] > 0.5)
    if active_anchor_indices.size > 0:
        source_contact_targets_obj = source_contact_targets_obj[active_anchor_indices]
        source_contact_normals_obj = source_contact_normals_obj[active_anchor_indices]
        source_contact_families = source_contact_families[active_anchor_indices]
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

        contact_candidate_points_world, contact_candidate_families = runtime.get_contact_candidate_points_world()
        contact_candidate_points_obj = transform_points(contact_candidate_points_world, inverse_pose7(object_pose_goal))
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

        anchor_residuals, family_normal_targets, _ = _assigned_contact_anchor_residuals(
            robot_contact_points_obj=contact_candidate_points_obj,
            robot_contact_family_indices=contact_candidate_families,
            human_contact_targets_obj=source_contact_targets_obj,
            human_contact_normals_obj=source_contact_normals_obj,
            human_contact_family_indices=source_contact_families,
        )

        residuals: list[float] = []
        for diff in anchor_residuals:
            residuals.extend((GENHAND_ANCHOR_POSITION_WEIGHT * diff).tolist())

        for family_idx, mean_normal in family_normal_targets.items():
            if family_idx == 0:
                continue
            finger_idx = CONTACT_FAMILY_TO_FINGER_INDEX[family_idx]
            desired_dir = normalize(0.60 * source_finger_dirs_obj[finger_idx] + 0.40 * (-mean_normal))
            residuals.extend((GENHAND_DIRECTION_WEIGHT * (finger_dirs_obj[finger_idx] - desired_dir)).tolist())

        if family_normal_targets:
            active_normals = np.asarray(list(family_normal_targets.values()), dtype=np.float64)
            desired_palm_approach = -normalize(np.mean(active_normals, axis=0))
        else:
            desired_palm_approach = semantic_frame_from_sites(source_semantic_sites_obj)[:, 2]
        current_palm_approach = semantic_frame_obj[:, 2]
        residuals.extend((GENHAND_PALM_DIRECTION_WEIGHT * (current_palm_approach - desired_palm_approach)).tolist())
        residuals.extend((GENHAND_PALM_POSITION_WEIGHT * (semantic_sites_obj[1] - source_semantic_sites_obj[1])).tolist())
        residuals.extend((GENHAND_WRIST_POSITION_WEIGHT * (semantic_sites_obj[0] - source_semantic_sites_obj[0])).tolist())

        active_distance_penalty = []
        for family_idx, group_indices in enumerate(CONTACT_FAMILY_GROUPS_12):
            active_group_indices = [idx for idx in group_indices if np.asarray(source_contact_mask_12, dtype=np.float64)[idx] > 0.5]
            if not active_group_indices:
                continue
            active_distance_penalty.extend(
                np.maximum(
                    distances[active_group_indices]
                    - np.asarray([SEMANTIC_CONTACT_DISTANCE_THRESHOLDS_M[idx] for idx in active_group_indices], dtype=np.float64),
                    0.0,
                ).tolist()
            )
        residuals.extend((GENHAND_ACTIVE_DISTANCE_WEIGHT * np.asarray(active_distance_penalty, dtype=np.float64)).tolist())

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


def _required_target_contact_count(source_contact_mask_12: np.ndarray) -> int:
    positive_contacts = int(np.sum(np.asarray(source_contact_mask_12, dtype=np.float64) > 0.5))
    return min(2, positive_contacts)


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
    required_contact_shortfall = max(0.0, float(required_contacts) - matched_contacts)
    hold_drop_over = max(0.0, float(candidate["hold_object_drop_m"]) - PROJECTION_HOLD_MAX_DROP_M)
    hold_translation_over = max(0.0, float(candidate["hold_object_translation_m"]) - PROJECTION_HOLD_MAX_TRANSLATION_M)
    hold_rotation_over = max(0.0, float(candidate["hold_object_rotation_deg"]) - PROJECTION_HOLD_MAX_ROTATION_DEG)
    hold_table_contact = 1.0 if bool(candidate["hold_table_contact"]) else 0.0
    hold_hybrid_shortfall = max(0.0, 2.0 - float(candidate["hold_hybrid_contact_group_count"]))
    hold_hard_shortfall = max(0.0, 1.0 - float(candidate["hold_hard_contact_group_count"]))
    hold_opposition_missing = 0.0 if bool(candidate["hold_has_thumb_opposition"]) else 1.0
    hold_match_shortfall = max(0.0, float(required_contacts) - float(candidate["hold_matched_target_contacts"]))
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
        + 2.0 * hold_match_shortfall
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
            hold_quality_violation,
            cylinder_wrap_violation,
            required_contact_shortfall,
            -matched_contacts,
            soft_safety_violation,
            extra_contacts,
            float(candidate["source_site_rmse_m"]),
            float(candidate["source_semantic_frame_error_deg"]),
            object_translation_drift,
            max_penetration,
            float(np.linalg.norm(np.asarray(candidate["wrist_translation_local"], dtype=np.float64))),
            float(np.linalg.norm(np.asarray(candidate["wrist_rotvec_local"], dtype=np.float64))),
        )
    return (
        1.0,
        hard_safety_violation,
        reachability_violation,
        hold_quality_violation,
        cylinder_wrap_violation,
        required_contact_shortfall,
        -matched_contacts,
        soft_safety_violation,
        extra_contacts,
        float(candidate["source_site_rmse_m"]),
        float(candidate["source_semantic_frame_error_deg"]),
        object_translation_drift,
        max_penetration,
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
    final_contact_candidate_points_world, final_contact_candidate_families = runtime.get_contact_candidate_points_world()
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
        + 0.200 * contact_shortfall
        + 0.030 * target_contact_misses
        + 0.010 * extra_contacts
        - 0.020 * matched_target_contacts
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
        + 0.40 * float(max(0.0, float(required_contacts) - hold_matched_target_contacts))
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
        "contact_candidate_points_world": np.asarray(final_contact_candidate_points_world, dtype=np.float64).copy(),
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


def _coordinate_descent_projected_candidate(
    runtime: RobotSceneModel,
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_wrist_pose_world: np.ndarray,
    source_semantic_sites_world: np.ndarray,
    source_contact_mask_12: np.ndarray,
    best: dict[str, Any],
    required_contacts: int,
    translation_steps: np.ndarray,
    rotation_steps: np.ndarray,
    hand_steps: np.ndarray,
    passes: int,
) -> dict[str, Any]:
    for _ in range(max(1, int(passes))):
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
                    wrist_rotvec_local=np.asarray(best["wrist_rotvec_local"], dtype=np.float64),
                    hand_qpos=np.asarray(best["hand_qpos"], dtype=np.float64),
                    run_hold_test=False,
                )
                if _is_better_projected_candidate(candidate, best, required_contacts):
                    best = candidate
                    improved = True

        for axis in range(3):
            for direction in (-1.0, 1.0):
                candidate_rotvec = np.asarray(best["wrist_rotvec_local"], dtype=np.float64).copy()
                candidate_rotvec[axis] += direction * rotation_steps[axis]
                candidate = _evaluate_projected_candidate(
                    runtime=runtime,
                    config=config,
                    object_pose_goal=object_pose_goal,
                    source_wrist_pose_world=source_wrist_pose_world,
                    source_semantic_sites_world=source_semantic_sites_world,
                    source_contact_mask_12=source_contact_mask_12,
                    wrist_translation_local=np.asarray(best["wrist_translation_local"], dtype=np.float64),
                    wrist_rotvec_local=candidate_rotvec,
                    hand_qpos=np.asarray(best["hand_qpos"], dtype=np.float64),
                    run_hold_test=False,
                )
                if _is_better_projected_candidate(candidate, best, required_contacts):
                    best = candidate
                    improved = True

        for joint_idx in range(len(hand_steps)):
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
                    wrist_rotvec_local=np.asarray(best["wrist_rotvec_local"], dtype=np.float64),
                    hand_qpos=candidate_hand_qpos,
                    run_hold_test=False,
                )
                if _is_better_projected_candidate(candidate, best, required_contacts):
                    best = candidate
                    improved = True

        if not improved:
            break

    return best


def _project_genhand_target(
    runtime: RobotSceneModel,
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_wrist_pose_world: np.ndarray,
    source_semantic_sites_world: np.ndarray,
    source_keypoints_obj: np.ndarray,
    source_contact_mask_12: np.ndarray,
    initial_hand_qpos: np.ndarray,
) -> dict[str, Any]:
    base_hand_qpos = runtime.clamp_hand(np.asarray(initial_hand_qpos, dtype=np.float64))
    required_contacts = _required_target_contact_count(source_contact_mask_12)
    seed_states = _box_genhand_seed_states(
        runtime=runtime,
        config=config,
        object_pose_goal=object_pose_goal,
        source_wrist_pose_world=source_wrist_pose_world,
        source_semantic_sites_world=source_semantic_sites_world,
        source_keypoints_obj=source_keypoints_obj,
        source_contact_mask_12=source_contact_mask_12,
        base_hand_qpos=base_hand_qpos,
    )

    candidate_pool: list[dict[str, Any]] = []
    global_best: dict[str, Any] | None = None
    for seed_translation, seed_rotvec, seed_hand_qpos in seed_states:
        try:
            optimized_translation, optimized_rotvec, optimized_hand_qpos, teacher_cost = _optimize_genhand_seed(
                runtime=runtime,
                config=config,
                object_pose_goal=object_pose_goal,
                source_wrist_pose_world=source_wrist_pose_world,
                source_semantic_sites_world=source_semantic_sites_world,
                source_keypoints_obj=source_keypoints_obj,
                source_contact_mask_12=source_contact_mask_12,
                initial_translation_local=seed_translation,
                initial_rotvec_local=seed_rotvec,
                initial_hand_qpos=seed_hand_qpos,
            )
        except Exception:
            continue

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
        candidate["teacher_cost"] = float(teacher_cost)
        candidate = _coordinate_descent_projected_candidate(
            runtime=runtime,
            config=config,
            object_pose_goal=object_pose_goal,
            source_wrist_pose_world=source_wrist_pose_world,
            source_semantic_sites_world=source_semantic_sites_world,
            source_contact_mask_12=source_contact_mask_12,
            best=candidate,
            required_contacts=required_contacts,
            translation_steps=np.array([0.0015, 0.0015, 0.0020], dtype=np.float64),
            rotation_steps=np.deg2rad(np.array([1.0, 1.0, 1.5], dtype=np.float64)),
            hand_steps=np.array([0.04, 0.03, 0.04, 0.04, 0.04, 0.04], dtype=np.float64),
            passes=2,
        )
        candidate["teacher_cost"] = float(teacher_cost)
        candidate_pool.append(candidate)
        if _is_better_projected_candidate(candidate, global_best, required_contacts):
            global_best = candidate

    if global_best is None:
        raise RuntimeError("GenHand-style retargeting failed to produce a candidate.")

    unique_shortlist: list[dict[str, Any]] = []
    seen_keys: set[tuple[float, ...]] = set()
    for candidate in sorted(candidate_pool, key=lambda item: _projection_rank_key(item, required_contacts)):
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


def _project_legacy_feasible_target(
    runtime: RobotSceneModel,
    config: TaskConfig,
    object_pose_goal: np.ndarray,
    source_wrist_pose_world: np.ndarray,
    source_semantic_sites_world: np.ndarray,
    source_keypoints_obj: np.ndarray,
    source_contact_mask_12: np.ndarray,
    initial_arm_qpos: np.ndarray,
    initial_hand_qpos: np.ndarray,
) -> dict[str, Any]:
    del initial_arm_qpos

    base_hand_qpos = runtime.clamp_hand(np.asarray(initial_hand_qpos, dtype=np.float64))
    contact_biased_hand = runtime.clamp_hand(base_hand_qpos + _finger_contact_bias_from_mask(source_contact_mask_12))
    aggressive_contact_hand = runtime.clamp_hand(base_hand_qpos + 1.45 * _finger_contact_bias_from_mask(source_contact_mask_12))
    required_contacts = _required_target_contact_count(source_contact_mask_12)
    generic_seed_states = [
        (np.array([0.0, 0.0, 0.0], dtype=np.float64), np.zeros(3, dtype=np.float64), base_hand_qpos),
        (np.array([0.0, 0.0, -0.006], dtype=np.float64), np.zeros(3, dtype=np.float64), contact_biased_hand),
        (np.array([0.0, 0.0, -0.003], dtype=np.float64), np.zeros(3, dtype=np.float64), contact_biased_hand),
        (np.array([0.0, 0.0, 0.002], dtype=np.float64), np.zeros(3, dtype=np.float64), contact_biased_hand),
        (np.array([0.0, 0.0, 0.001], dtype=np.float64), np.deg2rad(np.array([0.0, -4.0, 0.0], dtype=np.float64)), aggressive_contact_hand),
        (np.array([0.0, 0.0, 0.001], dtype=np.float64), np.deg2rad(np.array([0.0, 4.0, 0.0], dtype=np.float64)), aggressive_contact_hand),
        (np.array([0.0, 0.0, 0.0], dtype=np.float64), np.deg2rad(np.array([3.0, 0.0, 0.0], dtype=np.float64)), contact_biased_hand),
        (np.array([0.0, 0.0, 0.0], dtype=np.float64), np.deg2rad(np.array([-3.0, 0.0, 0.0], dtype=np.float64)), contact_biased_hand),
    ]
    cylinder_seed_states = _cylinder_grasp_seed_states(
        runtime=runtime,
        config=config,
        object_pose_goal=object_pose_goal,
        source_wrist_pose_world=source_wrist_pose_world,
        source_semantic_sites_world=source_semantic_sites_world,
        source_contact_mask_12=source_contact_mask_12,
        base_hand_qpos=base_hand_qpos,
    )
    cylinder_absolute_seed_poses = _cylinder_object_frame_seed_poses(
        runtime=runtime,
        config=config,
        object_pose_goal=object_pose_goal,
        source_keypoints_obj=source_keypoints_obj,
        source_contact_mask_12=source_contact_mask_12,
        base_hand_qpos=base_hand_qpos,
    )
    seed_states = cylinder_seed_states + generic_seed_states

    contact_direction_seed_specs = [
        (np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), contact_biased_hand),
        (np.array([0.0, 0.0, 0.004], dtype=np.float64), np.zeros(3, dtype=np.float64), aggressive_contact_hand),
    ]
    if config.object_geom_type == "cylinder":
        contact_direction_seed_specs.extend(
            [
                (np.array([0.0, 0.0, 0.006], dtype=np.float64), np.zeros(3, dtype=np.float64), _cylinder_hand_wrap_seed(runtime, source_contact_mask_12, strength=1.10)),
                (np.array([0.0, 0.0, 0.008], dtype=np.float64), np.deg2rad(np.array([0.0, 0.0, 10.0], dtype=np.float64)), _cylinder_hand_wrap_seed(runtime, source_contact_mask_12, strength=1.25)),
                (np.array([0.0, 0.0, 0.008], dtype=np.float64), np.deg2rad(np.array([0.0, 0.0, -10.0], dtype=np.float64)), _cylinder_hand_wrap_seed(runtime, source_contact_mask_12, strength=1.25)),
            ]
        )
    optimized_seed_states: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for seed_translation, seed_rotvec, seed_hand_qpos in contact_direction_seed_specs:
        try:
            optimized_translation, optimized_rotvec, optimized_hand_qpos, _ = _optimize_contact_direction_seed(
                runtime=runtime,
                config=config,
                object_pose_goal=object_pose_goal,
                source_wrist_pose_world=source_wrist_pose_world,
                source_semantic_sites_world=source_semantic_sites_world,
                source_keypoints_obj=source_keypoints_obj,
                source_contact_mask_12=source_contact_mask_12,
                initial_translation_local=seed_translation,
                initial_rotvec_local=seed_rotvec,
                initial_hand_qpos=seed_hand_qpos,
            )
            optimized_seed_states.append((optimized_translation, optimized_rotvec, optimized_hand_qpos))
        except Exception:
            continue

    for seed_wrist_pose_world, seed_hand_qpos in cylinder_absolute_seed_poses:
        try:
            optimized_translation_seed, optimized_rotvec_seed, optimized_hand_qpos, _ = _optimize_contact_direction_seed(
                runtime=runtime,
                config=config,
                object_pose_goal=object_pose_goal,
                source_wrist_pose_world=seed_wrist_pose_world,
                source_semantic_sites_world=source_semantic_sites_world,
                source_keypoints_obj=source_keypoints_obj,
                source_contact_mask_12=source_contact_mask_12,
                initial_translation_local=np.zeros(3, dtype=np.float64),
                initial_rotvec_local=np.zeros(3, dtype=np.float64),
                initial_hand_qpos=seed_hand_qpos,
            )
            optimized_wrist_pose_world = apply_local_pose_delta(
                seed_wrist_pose_world,
                optimized_translation_seed,
                optimized_rotvec_seed,
            )
            optimized_translation, optimized_rotvec = _pose_delta_local(
                source_wrist_pose_world,
                optimized_wrist_pose_world,
            )
            optimized_seed_states.append((optimized_translation, optimized_rotvec, optimized_hand_qpos))
        except Exception:
            continue

    if config.object_geom_type == "cylinder":
        # For cylindrical grasps, prioritize object-frame contact/direction solutions and
        # only keep a very small amount of generic fallback exploration.
        seed_states = optimized_seed_states + cylinder_seed_states + generic_seed_states[:2]
    else:
        seed_states = optimized_seed_states + seed_states

    translation_step_schedule = [
        np.array([0.004, 0.004, 0.006], dtype=np.float64),
        np.array([0.0015, 0.0015, 0.0025], dtype=np.float64),
    ]
    rotation_step_schedule = [
        np.deg2rad(np.array([4.0, 4.0, 6.0], dtype=np.float64)),
        np.deg2rad(np.array([1.5, 1.5, 2.5], dtype=np.float64)),
    ]
    hand_step_schedule = [
        np.array([0.14, 0.10, 0.14, 0.14, 0.14, 0.14], dtype=np.float64),
        np.array([0.05, 0.035, 0.05, 0.05, 0.05, 0.05], dtype=np.float64),
    ] 

    global_best: dict[str, Any] | None = None
    shortlist: list[dict[str, Any]] = []

    for seed_translation, seed_rotvec, seed_hand_qpos in seed_states:
        best = _evaluate_projected_candidate(
            runtime=runtime,
            config=config,
            object_pose_goal=object_pose_goal,
            source_wrist_pose_world=source_wrist_pose_world,
            source_semantic_sites_world=source_semantic_sites_world,
            source_contact_mask_12=source_contact_mask_12,
            wrist_translation_local=seed_translation,
            wrist_rotvec_local=seed_rotvec,
            hand_qpos=seed_hand_qpos,
            run_hold_test=False,
        )

        for translation_steps, rotation_steps, hand_steps in zip(
            translation_step_schedule,
            rotation_step_schedule,
            hand_step_schedule,
        ):
            best = _coordinate_descent_projected_candidate(
                runtime=runtime,
                config=config,
                object_pose_goal=object_pose_goal,
                source_wrist_pose_world=source_wrist_pose_world,
                source_semantic_sites_world=source_semantic_sites_world,
                source_contact_mask_12=source_contact_mask_12,
                best=best,
                required_contacts=required_contacts,
                translation_steps=translation_steps,
                rotation_steps=rotation_steps,
                hand_steps=hand_steps,
                passes=3,
            )

        if _is_better_projected_candidate(best, global_best, required_contacts):
            global_best = best
        shortlist.append(best)

    if global_best is None:
        raise RuntimeError("Failed to project a feasible pose-driven target.")

    shortlist.append(global_best)
    unique_shortlist: list[dict[str, Any]] = []
    seen_keys: set[tuple[float, ...]] = set()
    for candidate in sorted(shortlist, key=lambda item: _projection_rank_key(item, required_contacts)):
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
        shortlist_limit = 10 if config.object_geom_type == "cylinder" else 6
        if len(unique_shortlist) >= shortlist_limit:
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
        if _is_better_projected_candidate(validated, hold_validated_best, required_contacts):
            hold_validated_best = validated

    if hold_validated_best is None:
        raise RuntimeError("Failed to validate any projected candidate with hold-test.")

    hold_validated_best["retreat_m"] = float(max(0.0, -float(np.asarray(hold_validated_best["wrist_translation_local"])[2])))
    hold_validated_best["hand_open_blend"] = float(
        np.linalg.norm(np.asarray(hold_validated_best["hand_qpos"]) - base_hand_qpos) / max(np.linalg.norm(runtime.hand_upper - runtime.hand_lower), 1e-8)
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
            initial_arm_qpos=arm_qpos,
            initial_hand_qpos=hand_qpos,
        )
        projected_semantic_sites_world = np.asarray(projected["semantic_sites_world"], dtype=np.float64)
        projected_wrist_pose_world = np.asarray(projected["wrist_pose_world"], dtype=np.float64)
        projected_object_pose_goal = np.asarray(projected["object_pose_goal"], dtype=np.float64)
        projected_semantic_sites_object = transform_points(projected_semantic_sites_world, inverse_pose7(projected_object_pose_goal))
        projected_wrist_pose_object = compose_pose7(inverse_pose7(projected_object_pose_goal), projected_wrist_pose_world)
        projected_contact_points_object = transform_points(
            np.asarray(projected["contact_candidate_points_world"], dtype=np.float64),
            inverse_pose7(projected_object_pose_goal),
        )
        anchor_metrics = _contact_anchor_metrics_obj(
            robot_contact_points_obj=projected_contact_points_object,
            robot_contact_family_indices=np.asarray(projected["contact_candidate_family_indices"], dtype=np.int64),
            source_keypoints_obj=final_ee_rel[label_idx],
            source_contact_mask_12=source_contact_mask,
            config=config,
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
            "projection_score": float(projected["score"]),
            "retarget_method": 1.0 if str(projected.get("retarget_method", "genhand_direct")).startswith("genhand") else 0.0,
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
            and fit_error["projected_hold_matched_target_contacts"] >= fit_error["projected_required_target_contacts"]
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
                "retarget_method": str(projected.get("retarget_method", "genhand_direct")),
                **fit_error,
            }
        )

        previous_arm_qpos = np.asarray(projected["arm_qpos"], dtype=np.float64)
        previous_hand_qpos = np.asarray(projected["hand_qpos"], dtype=np.float64)

    save_pose_driven_samples(samples_path, samples)
    pose_driven_report_path(config).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return samples
