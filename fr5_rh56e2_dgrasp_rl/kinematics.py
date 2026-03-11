from __future__ import annotations

import numpy as np
import mujoco

from .robot_model import RobotSceneModel
from .utils import damped_least_squares


def solve_arm_wrist_palm_ik(
    runtime: RobotSceneModel,
    target_sites_world: np.ndarray,
    initial_arm_qpos: np.ndarray,
    hand_qpos: np.ndarray | None = None,
    iterations: int = 120,
    damping: float = 5e-4,
) -> np.ndarray:
    arm_qpos = runtime.clamp_arm(np.asarray(initial_arm_qpos, dtype=np.float64).copy())
    if hand_qpos is None:
        hand_qpos = np.zeros(len(runtime.hand_joints), dtype=np.float64)
    hand_qpos = runtime.clamp_hand(np.asarray(hand_qpos, dtype=np.float64))

    wrist_id = runtime.site_id_by_name["wrist_mount"]
    palm_id = runtime.site_id_by_name["palm_center"]
    arm_dof_indices = np.array([runtime.qvel_index_by_joint[name] for name in runtime.arm_joints], dtype=np.int32)

    for _ in range(iterations):
        runtime.set_robot_actuated_qpos(np.concatenate([arm_qpos, hand_qpos]))
        current_sites = runtime.get_semantic_sites_world()
        error = np.concatenate(
            [target_sites_world[0] - current_sites[0], target_sites_world[1] - current_sites[1]]
        )
        if float(np.linalg.norm(error)) < 1e-4:
            break

        jacp_wrist = np.zeros((3, runtime.model.nv), dtype=np.float64)
        jacr_wrist = np.zeros((3, runtime.model.nv), dtype=np.float64)
        jacp_palm = np.zeros((3, runtime.model.nv), dtype=np.float64)
        jacr_palm = np.zeros((3, runtime.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(runtime.model, runtime.data, jacp_wrist, jacr_wrist, wrist_id)
        mujoco.mj_jacSite(runtime.model, runtime.data, jacp_palm, jacr_palm, palm_id)

        jacobian = np.vstack((jacp_wrist[:, arm_dof_indices], jacp_palm[:, arm_dof_indices]))
        delta = damped_least_squares(jacobian, error, damping)
        arm_qpos = runtime.clamp_arm(arm_qpos + delta)

    return arm_qpos
