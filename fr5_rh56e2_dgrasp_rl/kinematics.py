from __future__ import annotations

import numpy as np
import mujoco

from .robot_model import RobotSceneModel
from .utils import damped_least_squares, quat_wxyz_to_matrix, normalize


def solve_arm_wrist_palm_ik(
    runtime: RobotSceneModel,
    target_wrist_pose_world: np.ndarray,
    initial_arm_qpos: np.ndarray,
    hand_qpos: np.ndarray | None = None,
    iterations: int = 120,
    damping: float = 5e-4,
) -> np.ndarray:
    arm_qpos = runtime.clamp_arm(np.asarray(initial_arm_qpos, dtype=np.float64).copy())
    if hand_qpos is None:
        hand_qpos = np.zeros(len(runtime.hand_joints), dtype=np.float64)
    hand_qpos = runtime.clamp_hand(np.asarray(hand_qpos, dtype=np.float64))

    target_wrist_pose_world = np.asarray(target_wrist_pose_world, dtype=np.float64)
    target_wrist_rotation = quat_wxyz_to_matrix(target_wrist_pose_world[3:])
    arm_dof_indices = np.array([runtime.qvel_index_by_joint[name] for name in runtime.arm_joints], dtype=np.int32)
    anchor_site_names = ("wrist_mount", "palm_center", "index_tip", "little_tip")
    anchor_semantic_indices = (0, 1, 3, 6)
    anchor_site_ids = [runtime.site_id_by_name[name] for name in anchor_site_names]

    runtime.set_robot_actuated_qpos(np.concatenate([arm_qpos, hand_qpos]))
    reference_sites = runtime.get_semantic_sites_world()
    wrist = reference_sites[0]
    palm = reference_sites[1]
    index_tip = reference_sites[3]
    little_tip = reference_sites[6]
    approach = normalize(palm - wrist)
    across = normalize(index_tip - little_tip)
    normal = normalize(np.cross(approach, across))
    across = normalize(np.cross(normal, approach))
    semantic_wrist_rotation = np.column_stack((across, normal, approach))
    local_anchor_offsets = (semantic_wrist_rotation.T @ (reference_sites[list(anchor_semantic_indices)] - wrist).T).T

    for _ in range(iterations):
        runtime.set_robot_actuated_qpos(np.concatenate([arm_qpos, hand_qpos]))
        current_sites = runtime.get_semantic_sites_world()
        current_anchor_sites = current_sites[list(anchor_semantic_indices)]
        target_anchor_sites = (target_wrist_rotation @ local_anchor_offsets.T).T + target_wrist_pose_world[:3]
        error = (target_anchor_sites - current_anchor_sites).reshape(-1)
        if float(np.linalg.norm(error)) < 1e-4:
            break

        jacobian_rows = []
        for site_id in anchor_site_ids:
            jacp = np.zeros((3, runtime.model.nv), dtype=np.float64)
            jacr = np.zeros((3, runtime.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(runtime.model, runtime.data, jacp, jacr, site_id)
            jacobian_rows.append(jacp[:, arm_dof_indices])
        jacobian = np.vstack(jacobian_rows)
        delta = damped_least_squares(jacobian, error, damping)
        arm_qpos = runtime.clamp_arm(arm_qpos + delta)

    return arm_qpos
