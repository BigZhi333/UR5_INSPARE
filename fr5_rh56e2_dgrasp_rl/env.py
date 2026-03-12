from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import gymnasium as gym
import numpy as np

from .kinematics import solve_arm_wrist_palm_ik
from .pose_driven_data import (
    PoseDrivenSample,
    load_pose_driven_samples,
    pose_driven_samples_path,
    prepare_pose_driven_samples,
    wrist_pose_from_semantic_sites,
)
from .robot_model import RobotSceneModel
from .scene_builder import build_training_scene
from .task_config import TaskConfig
from .utils import (
    apply_local_pose_delta,
    compose_pose7,
    interpolate_pose7,
    inverse_pose7,
    quat_wxyz_to_matrix,
    rotation_6d_from_matrix,
    rotation_angle_deg,
    transform_points,
)


@dataclass
class EpisodeMetrics:
    reward_sum: float = 0.0
    displacement_trace: list[float] | None = None

    def __post_init__(self) -> None:
        if self.displacement_trace is None:
            self.displacement_trace = []


class FR5LowLevelGraspEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: TaskConfig | None = None,
        drop_table_after_pregrasp: bool = True,
        deterministic_goal_index: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or TaskConfig.default()
        scene_xml, metadata_path = build_training_scene(self.config)
        self.runtime = RobotSceneModel(self.config, scene_xml=scene_xml, metadata_path=metadata_path)
        self.goals = self._load_goals()
        self.drop_table_after_pregrasp = drop_table_after_pregrasp
        self.deterministic_goal_index = deterministic_goal_index
        self.current_goal: PoseDrivenSample | None = None
        self.current_goal_index = 0
        self.current_ctrl = self.runtime.home_actuated.copy()
        self.current_wrist_target_pose = wrist_pose_from_semantic_sites(self.runtime.get_semantic_sites_world())
        self.current_hand_target_qpos = self.runtime.get_actuated_qpos()[6:].copy()
        self.pregrasp_wrist_pose_world = self.current_wrist_target_pose.copy()
        self.pregrasp_hand_qpos = self.current_hand_target_qpos.copy()
        self.previous_action = np.zeros(12, dtype=np.float64)
        self.step_count = 0
        self.table_dropped = False
        self.slipped = False
        self.first_slip_step: int | None = None
        self.last_object_pos = np.zeros(3, dtype=np.float64)
        self.metrics = EpisodeMetrics()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(119,), dtype=np.float32)

    def _load_goals(self) -> list[PoseDrivenSample]:
        samples_path = pose_driven_samples_path(self.config)
        if samples_path.exists():
            return load_pose_driven_samples(samples_path)
        return prepare_pose_driven_samples(self.config)

    def _current_goal_sites_obj(self) -> np.ndarray:
        if self.current_goal is None:
            raise RuntimeError("Environment has no active goal.")
        return self.current_goal.semantic_sites_object()

    def _current_goal_wrist_pose_object(self) -> np.ndarray:
        if self.current_goal is None:
            raise RuntimeError("Environment has no active goal.")
        return np.asarray(self.current_goal.wrist_pose_goal_object, dtype=np.float64)

    def _target_wrist_world(self, object_pose: np.ndarray) -> np.ndarray:
        return compose_pose7(object_pose, self._current_goal_wrist_pose_object())

    def _solve_actuated_targets(
        self,
        wrist_pose_world: np.ndarray,
        hand_qpos: np.ndarray,
        initial_arm_qpos: np.ndarray,
    ) -> np.ndarray:
        hand_qpos = self.runtime.clamp_hand(np.asarray(hand_qpos, dtype=np.float64))
        arm_qpos = solve_arm_wrist_palm_ik(
            runtime=self.runtime,
            target_wrist_pose_world=wrist_pose_world,
            initial_arm_qpos=np.asarray(initial_arm_qpos, dtype=np.float64),
            hand_qpos=hand_qpos,
            iterations=self.config.conversion.arm_ik_iterations,
            damping=self.config.conversion.arm_ik_damping,
        )
        return self.runtime.clamp_actuated(np.concatenate([arm_qpos, hand_qpos]))

    def _compute_pregrasp_targets(self, object_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target_wrist_world = self._target_wrist_world(object_pose)
        target_rotation = quat_wxyz_to_matrix(target_wrist_world[3:])
        offset = -target_rotation[:, 2] * self.config.pregrasp_backoff_m + np.array(
            [0.0, 0.0, self.config.pregrasp_lift_m],
            dtype=np.float64,
        )
        pregrasp_wrist_world = np.asarray(target_wrist_world, dtype=np.float64).copy()
        pregrasp_wrist_world[:3] = pregrasp_wrist_world[:3] + offset
        target_hand_qpos = (
            np.zeros(6, dtype=np.float64)
            if self.current_goal is None
            else np.asarray(self.current_goal.hand_qpos_6, dtype=np.float64)
        )
        pregrasp_hand_qpos = self.runtime.clamp_hand(0.2 * target_hand_qpos)
        return pregrasp_wrist_world, pregrasp_hand_qpos

    def _choose_goal_index(self) -> int:
        if self.deterministic_goal_index is not None:
            return self.deterministic_goal_index % len(self.goals)
        return int(self.np_random.integers(0, len(self.goals)))

    def _goal_from_index(self, index: int) -> PoseDrivenSample:
        return self.goals[index % len(self.goals)]

    def _phase_one_hot(self) -> np.ndarray:
        if self.step_count < self.config.pre_grasp_steps:
            return np.array([1.0, 0.0], dtype=np.float64)
        return np.array([0.0, 1.0], dtype=np.float64)

    def _phase_progress(self) -> float:
        if self.config.pre_grasp_steps <= 0:
            return 1.0
        return float(np.clip((self.step_count + 1) / float(self.config.pre_grasp_steps), 0.0, 1.0))

    def _wrist_pose_world(self, semantic_sites_world: np.ndarray) -> np.ndarray:
        return wrist_pose_from_semantic_sites(semantic_sites_world)

    def _relative_body_pos_wrist(
        self,
        current_sites_world: np.ndarray,
        object_pose: np.ndarray,
        wrist_pose_world: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        wrist_pose_world = self._wrist_pose_world(current_sites_world) if wrist_pose_world is None else wrist_pose_world
        current_sites_obj = transform_points(current_sites_world, inverse_pose7(object_pose))
        target_sites_obj = self._current_goal_sites_obj()
        object_rotation = quat_wxyz_to_matrix(object_pose[3:])
        wrist_rotation = quat_wxyz_to_matrix(wrist_pose_world[3:])
        error_obj = target_sites_obj - current_sites_obj
        error_world = (object_rotation @ error_obj.T).T
        error_wrist = (wrist_rotation.T @ error_world.T).T
        return error_wrist, current_sites_obj

    def _object_relative_pose_features(self, wrist_pose_world: np.ndarray, object_pose: np.ndarray) -> np.ndarray:
        wrist_frame = quat_wxyz_to_matrix(wrist_pose_world[3:])
        object_rotation = quat_wxyz_to_matrix(object_pose[3:])
        # Match the original D-Grasp convention: encode the object pose in the wrist frame.
        rel_position = wrist_frame.T @ (wrist_pose_world[:3] - object_pose[:3])
        rel_rotation = wrist_frame.T @ object_rotation
        return np.concatenate([rel_position, rotation_6d_from_matrix(rel_rotation)])

    def _object_displacement_wrist(self, wrist_pose_world: np.ndarray, object_pose: np.ndarray) -> np.ndarray:
        if self.current_goal is None:
            raise RuntimeError("Environment has no active goal.")
        wrist_frame = quat_wxyz_to_matrix(wrist_pose_world[3:])
        init_object_pose = np.asarray(self.current_goal.object_pose_init, dtype=np.float64)
        return wrist_frame.T @ (init_object_pose[:3] - object_pose[:3])

    def _table_clearance_wrist(self, wrist_pose_world: np.ndarray) -> np.ndarray:
        wrist_frame = quat_wxyz_to_matrix(wrist_pose_world[3:])
        table_center_z = float(self.runtime.model.geom_pos[self.runtime.table_geom_id, 2])
        table_half_z = float(self.runtime.model.geom_size[self.runtime.table_geom_id, 2])
        table_top_z = table_center_z + table_half_z
        clearance_world = np.array(
            [0.0, 0.0, wrist_pose_world[2] - table_top_z],
            dtype=np.float64,
        )
        return wrist_frame.T @ clearance_world

    def _pose_goal_features(self, current_wrist_pose_object: np.ndarray) -> np.ndarray:
        target_wrist_pose_object = self._current_goal_wrist_pose_object()
        current_rotation = quat_wxyz_to_matrix(current_wrist_pose_object[3:])
        target_rotation = quat_wxyz_to_matrix(target_wrist_pose_object[3:])
        delta_translation = current_rotation.T @ (target_wrist_pose_object[:3] - current_wrist_pose_object[:3])
        delta_rotation = current_rotation.T @ target_rotation
        return np.concatenate([delta_translation, rotation_6d_from_matrix(delta_rotation)])

    def _guided_targets(self, object_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.current_goal is None:
            raise RuntimeError("Environment has no active goal.")
        target_wrist_world = self._target_wrist_world(object_pose)
        target_hand_qpos = np.asarray(self.current_goal.hand_qpos_6, dtype=np.float64)
        alpha = self._phase_progress()
        guided_wrist_pose = interpolate_pose7(self.pregrasp_wrist_pose_world, target_wrist_world, alpha)
        guided_hand_qpos = self.runtime.clamp_hand(
            (1.0 - alpha) * self.pregrasp_hand_qpos + alpha * target_hand_qpos
        )
        return guided_wrist_pose, guided_hand_qpos

    def _observation(self) -> np.ndarray:
        if self.current_goal is None:
            raise RuntimeError("Environment has no active goal.")
        semantic_sites_world = self.runtime.get_semantic_sites_world()
        object_pose = self.runtime.get_object_pose()
        wrist_pose_world = self._wrist_pose_world(semantic_sites_world)
        current_wrist_pose_object = compose_pose7(
            inverse_pose7(object_pose),
            wrist_pose_world,
        )
        rel_body_pos_wrist, _ = self._relative_body_pos_wrist(
            semantic_sites_world,
            object_pose,
            wrist_pose_world=wrist_pose_world,
        )
        object_linear_vel, object_angular_vel = self.runtime.get_object_velocity()
        contact_diag = self.runtime.get_contact_diagnostics_12()
        target_contact_mask = np.asarray(self.current_goal.contact_mask_12, dtype=np.float64)
        contact_forces = np.clip(np.asarray(contact_diag["forces"], dtype=np.float64) / 25.0, 0.0, 1.0)
        achieved_target_contacts = np.asarray(contact_diag["hybrid_mask"], dtype=np.float64) * target_contact_mask
        wrist_frame = quat_wxyz_to_matrix(wrist_pose_world[3:])
        object_velocity = np.concatenate(
            [wrist_frame.T @ object_linear_vel, wrist_frame.T @ object_angular_vel]
        )
        object_displacement_wrist = self._object_displacement_wrist(wrist_pose_world, object_pose)
        table_clearance_wrist = self._table_clearance_wrist(wrist_pose_world)
        actuated_qpos = self.runtime.get_actuated_qpos()
        target_hand_delta = np.asarray(self.current_goal.hand_qpos_6, dtype=np.float64) - actuated_qpos[6:]
        obs = np.concatenate(
            [
                actuated_qpos,
                self.runtime.get_actuated_qvel(),
                self._object_relative_pose_features(wrist_pose_world, object_pose),
                object_velocity,
                object_displacement_wrist,
                table_clearance_wrist,
                self._pose_goal_features(current_wrist_pose_object),
                target_hand_delta,
                rel_body_pos_wrist.reshape(-1),
                target_contact_mask,
                contact_forces,
                achieved_target_contacts,
                self._phase_one_hot(),
            ]
        )
        return obs.astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        if self.current_goal is None:
            raise RuntimeError("Environment has no active goal.")

        current_sites_world = self.runtime.get_semantic_sites_world()
        object_pose = self.runtime.get_object_pose()
        current_wrist_pose_world = self._wrist_pose_world(current_sites_world)
        rel_body_pos_wrist, _ = self._relative_body_pos_wrist(
            current_sites_world,
            object_pose,
            wrist_pose_world=current_wrist_pose_world,
        )

        current_wrist_pose_object = compose_pose7(
            inverse_pose7(object_pose),
            current_wrist_pose_world,
        )
        target_wrist_pose_object = self._current_goal_wrist_pose_object()

        current_wrist_frame = quat_wxyz_to_matrix(current_wrist_pose_object[3:])
        target_wrist_frame = quat_wxyz_to_matrix(target_wrist_pose_object[3:])

        contact_diag = self.runtime.get_contact_diagnostics_12()
        hard_contact_mask = np.asarray(contact_diag["hard_mask"], dtype=np.float64)
        proximity_contact_mask = np.asarray(contact_diag["proximity_mask"], dtype=np.float64)
        actual_contact_mask = np.asarray(contact_diag["hybrid_mask"], dtype=np.float64)
        contact_forces = np.asarray(contact_diag["forces"], dtype=np.float64)
        contact_distances = np.asarray(contact_diag["distances_m"], dtype=np.float64)
        proximity_scores = np.asarray(contact_diag["proximity_scores"], dtype=np.float64)
        table_contact = bool(contact_diag["table_contact"])
        target_contact_mask = np.asarray(self.current_goal.contact_mask_12, dtype=np.float64)

        current_qpos = self.runtime.get_actuated_qpos()
        current_hand_qpos = current_qpos[6:]
        target_hand_qpos = np.asarray(self.current_goal.hand_qpos_6, dtype=np.float64)
        commanded_hand_qpos = np.asarray(self.current_hand_target_qpos, dtype=np.float64)

        object_linear_vel, object_angular_vel = self.runtime.get_object_velocity()
        wrist_linear_vel, wrist_angular_vel = self.runtime.get_site_velocity("wrist_mount")
        object_displacement_wrist = self._object_displacement_wrist(current_wrist_pose_world, object_pose)
        table_clearance_wrist = self._table_clearance_wrist(current_wrist_pose_world)

        object_height = float(object_pose[2])
        if self.table_dropped and (table_contact or object_height < self.config.eval.success_height_threshold_m):
            if not self.slipped:
                self.first_slip_step = self.step_count
            self.slipped = True

        wrist_translation_error = float(np.linalg.norm(rel_body_pos_wrist[0]))
        wrist_rotation_error_deg = float(rotation_angle_deg(current_wrist_frame, target_wrist_frame))
        semantic_site_rmse = float(
            np.sqrt(np.mean(np.sum(rel_body_pos_wrist[1:] * rel_body_pos_wrist[1:], axis=1)))
        )
        hand_pose_error = float(np.mean(np.abs(current_hand_qpos - target_hand_qpos)))
        wrist_command_translation_error = float(
            np.linalg.norm(current_wrist_pose_world[:3] - self.current_wrist_target_pose[:3])
        )
        wrist_command_rotation_error_deg = float(
            rotation_angle_deg(
                quat_wxyz_to_matrix(current_wrist_pose_world[3:]),
                quat_wxyz_to_matrix(self.current_wrist_target_pose[3:]),
            )
        )
        hand_command_error = float(np.mean(np.abs(current_hand_qpos - commanded_hand_qpos)))

        target_contacts = max(float(target_contact_mask.sum()), 1.0)
        non_target_contacts = max(float(len(target_contact_mask) - target_contact_mask.sum()), 1.0)
        matched_positive = float((actual_contact_mask * target_contact_mask).sum() / target_contacts)
        false_positive = float((actual_contact_mask * (1.0 - target_contact_mask)).sum() / non_target_contacts)
        contact_bit_match = float(1.0 - np.mean(np.abs(actual_contact_mask - target_contact_mask)))
        contact_term = matched_positive - 0.25 * false_positive
        hard_matched_positive = float((hard_contact_mask * target_contact_mask).sum() / target_contacts)
        hard_false_positive = float((hard_contact_mask * (1.0 - target_contact_mask)).sum() / non_target_contacts)
        hard_contact_bit_match = float(1.0 - np.mean(np.abs(hard_contact_mask - target_contact_mask)))
        proximity_matched_positive = float((proximity_contact_mask * target_contact_mask).sum() / target_contacts)
        proximity_false_positive = float((proximity_contact_mask * (1.0 - target_contact_mask)).sum() / non_target_contacts)
        proximity_contact_bit_match = float(1.0 - np.mean(np.abs(proximity_contact_mask - target_contact_mask)))
        target_contact_selector = target_contact_mask > 0.5
        target_contact_distances = contact_distances[target_contact_selector]
        target_contact_scores = proximity_scores[target_contact_selector]

        active_forces = contact_forces[target_contact_mask > 0.5]
        impulse_term = float(np.clip(np.mean(active_forces) / 25.0, 0.0, 1.0)) if active_forces.size else 0.0
        mean_target_contact_force = float(np.mean(active_forces)) if active_forces.size else 0.0
        max_target_contact_force = float(np.max(active_forces)) if active_forces.size else 0.0

        site_term = float(np.exp(-40.0 * semantic_site_rmse))
        pose_term = float(np.exp(-4.0 * hand_pose_error))
        wrist_pose_term = float(np.exp(-30.0 * wrist_translation_error))
        wrist_align_term = float(np.exp(-wrist_rotation_error_deg / 20.0))
        falling_term = -1.0 if self.slipped else 0.0
        obj_displacement_term = float(np.linalg.norm(object_displacement_wrist))
        penetration_term = float(contact_diag["total_penetration_m"] + 2.0 * contact_diag["max_penetration_m"])
        rel_obj_term = float(
            np.dot(object_linear_vel, object_linear_vel) + 0.25 * np.dot(object_angular_vel, object_angular_vel)
        )
        body_vel_term = float(np.dot(wrist_linear_vel, wrist_linear_vel))
        body_qvel_term = float(np.dot(wrist_angular_vel, wrist_angular_vel))
        action_rate_term = float(np.mean((action - self.previous_action) ** 2))
        motion_regularization_scale = 0.1 if self.step_count < self.config.pre_grasp_steps else 1.0

        weights = self.config.reward
        reward_site = weights["site_reward"] * site_term
        reward_pose = weights["pose_reward"] * pose_term
        reward_contact = weights["contact_reward"] * contact_term
        reward_impulse = weights["impulse_reward"] * impulse_term
        reward_falling = weights["falling_reward"] * falling_term
        reward_obj = weights.get("obj_reward", 0.0) * obj_displacement_term
        reward_rel_obj = weights["rel_obj_reward"] * rel_obj_term
        reward_body_vel = weights["body_vel_reward"] * body_vel_term * motion_regularization_scale
        reward_body_qvel = weights["body_qvel_reward"] * body_qvel_term * motion_regularization_scale
        reward_wrist_pose = weights.get("wrist_pose_reward", 0.0) * wrist_pose_term
        reward_wrist_align = weights["wrist_align_reward"] * wrist_align_term
        reward_penetration = weights.get("penetration_reward", 0.0) * penetration_term
        reward_action_rate = weights["action_rate"] * action_rate_term * motion_regularization_scale

        reward_preclip = (
            reward_site
            + reward_pose
            + reward_contact
            + reward_impulse
            + reward_falling
            + reward_obj
            + reward_rel_obj
            + reward_body_vel
            + reward_body_qvel
            + reward_wrist_pose
            + reward_wrist_align
            + reward_penetration
            + reward_action_rate
        )
        reward = float(max(reward_preclip, self.config.reward_clip))

        reward_info = {
            "wrist_translation_error_m": wrist_translation_error,
            "wrist_rotation_error_deg": wrist_rotation_error_deg,
            "site_rmse_m": semantic_site_rmse,
            "hand_pose_error": hand_pose_error,
            "wrist_command_translation_error_m": wrist_command_translation_error,
            "wrist_command_rotation_error_deg": wrist_command_rotation_error_deg,
            "hand_command_error": hand_command_error,
            "term_site": site_term,
            "term_pose": pose_term,
            "term_wrist_pose": wrist_pose_term,
            "term_wrist_align": wrist_align_term,
            "contact_term": contact_term,
            "contact_bit_match": contact_bit_match,
            "matched_positive_contacts": matched_positive,
            "false_positive_contacts": false_positive,
            "hard_contact_bit_match": hard_contact_bit_match,
            "hard_matched_positive_contacts": hard_matched_positive,
            "hard_false_positive_contacts": hard_false_positive,
            "proximity_contact_bit_match": proximity_contact_bit_match,
            "proximity_matched_positive_contacts": proximity_matched_positive,
            "proximity_false_positive_contacts": proximity_false_positive,
            "target_contact_count": float(target_contact_mask.sum()),
            "hybrid_contact_count": float(actual_contact_mask.sum()),
            "hard_contact_count": float(hard_contact_mask.sum()),
            "proximity_contact_count": float(proximity_contact_mask.sum()),
            "target_contact_distance_mean_m": float(np.mean(target_contact_distances)) if target_contact_distances.size else 0.0,
            "target_contact_distance_min_m": float(np.min(target_contact_distances)) if target_contact_distances.size else 0.0,
            "all_contact_distance_mean_m": float(np.mean(contact_distances)),
            "target_contact_proximity_score_mean": float(np.mean(target_contact_scores)) if target_contact_scores.size else 0.0,
            "impulse_term": impulse_term,
            "target_contact_force_mean_n": mean_target_contact_force,
            "target_contact_force_max_n": max_target_contact_force,
            "term_falling": falling_term,
            "term_obj": obj_displacement_term,
            "term_rel_obj": rel_obj_term,
            "term_body_vel": body_vel_term,
            "term_body_qvel": body_qvel_term,
            "term_penetration": penetration_term,
            "term_action_rate": action_rate_term,
            "motion_regularization_scale": motion_regularization_scale,
            "reward_site": reward_site,
            "reward_pose": reward_pose,
            "reward_wrist_pose": reward_wrist_pose,
            "reward_wrist_align": reward_wrist_align,
            "reward_contact": reward_contact,
            "reward_impulse": reward_impulse,
            "reward_falling": reward_falling,
            "reward_obj": reward_obj,
            "reward_rel_obj": reward_rel_obj,
            "reward_body_vel": reward_body_vel,
            "reward_body_qvel": reward_body_qvel,
            "reward_penetration": reward_penetration,
            "reward_action_rate": reward_action_rate,
            "reward_preclip": reward_preclip,
            "reward_total": reward,
            "reward_clip_delta": reward - reward_preclip,
            "penetration_m": float(contact_diag["total_penetration_m"]),
            "max_penetration_m": float(contact_diag["max_penetration_m"]),
            "falling_term": falling_term,
            "object_height_m": object_height,
            "object_displacement_m": obj_displacement_term,
            "table_clearance_m": float(table_clearance_wrist[2]),
            "table_contact": float(table_contact),
            "slipped": float(self.slipped),
        }
        return reward, reward_info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.runtime.reset()
        forced_goal_index = None if options is None else options.get("goal_index")
        self.current_goal_index = self._choose_goal_index() if forced_goal_index is None else int(forced_goal_index)
        self.current_goal = self._goal_from_index(self.current_goal_index)

        object_pose = np.asarray(self.current_goal.object_pose_init, dtype=np.float64)
        self.runtime.set_object_pose(object_pose)

        pregrasp_wrist_pose, pregrasp_hand_qpos = self._compute_pregrasp_targets(object_pose)
        pregrasp_ctrl = self._solve_actuated_targets(
            wrist_pose_world=pregrasp_wrist_pose,
            hand_qpos=pregrasp_hand_qpos,
            initial_arm_qpos=self.runtime.home_actuated[:6],
        )
        joint_noise = self.np_random.normal(0.0, self.config.joint_noise_std, size=pregrasp_ctrl.shape)
        joint_noise[6:] *= 0.5
        self.current_ctrl = self.runtime.clamp_actuated(pregrasp_ctrl + joint_noise)
        self.runtime.set_robot_actuated_qpos(self.current_ctrl)

        self.pregrasp_wrist_pose_world = pregrasp_wrist_pose.copy()
        self.pregrasp_hand_qpos = pregrasp_hand_qpos.copy()
        self.current_wrist_target_pose = wrist_pose_from_semantic_sites(self.runtime.get_semantic_sites_world())
        self.current_hand_target_qpos = self.current_ctrl[6:].copy()
        self.step_count = 0
        self.table_dropped = False
        self.slipped = False
        self.first_slip_step = None
        self.previous_action = np.zeros(12, dtype=np.float64)
        self.metrics = EpisodeMetrics()
        self.last_object_pos = self.runtime.get_object_pose()[:3].copy()

        obs = self._observation()
        info = {
            "goal_index": self.current_goal_index,
            "label_idx": self.current_goal.label_idx,
            "goal_object_id": self.current_goal.object_id,
            "valid_execution": float(self.current_goal.valid_execution),
        }
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        wrist_translation_delta = action[:3] * self.config.action_delta_wrist_pos_m
        wrist_rotation_delta = action[3:6] * self.config.action_delta_wrist_rot_rad
        hand_delta = action[6:] * self.config.action_delta_hand

        object_pose = self.runtime.get_object_pose()
        guided_wrist_target_pose, guided_hand_target_qpos = self._guided_targets(object_pose)

        unclamped_hand_target_qpos = guided_hand_target_qpos + hand_delta
        self.current_wrist_target_pose = apply_local_pose_delta(
            guided_wrist_target_pose,
            wrist_translation_delta,
            wrist_rotation_delta,
        )
        self.current_hand_target_qpos = self.runtime.clamp_hand(unclamped_hand_target_qpos)

        self.current_ctrl = self._solve_actuated_targets(
            wrist_pose_world=self.current_wrist_target_pose,
            hand_qpos=self.current_hand_target_qpos,
            initial_arm_qpos=self.current_ctrl[:6],
        )
        self.runtime.step(self.current_ctrl, self.config.frame_skip)
        self.step_count += 1

        if self.drop_table_after_pregrasp and not self.table_dropped and self.step_count >= self.config.pre_grasp_steps:
            self.runtime.set_table_height(0.0)
            self.table_dropped = True

        current_object_pos = self.runtime.get_object_pose()[:3].copy()
        displacement = float(np.linalg.norm(current_object_pos - self.last_object_pos))
        self.last_object_pos = current_object_pos
        if not self.slipped:
            self.metrics.displacement_trace.append(displacement)

        reward, reward_info = self._compute_reward(action)
        self.metrics.reward_sum += reward
        obs = self._observation()

        terminated = False
        # Keep the episode length fixed like the original D-Grasp setup.
        # Slip only affects reward/success metrics and no longer ends the episode early.
        truncated = self.step_count >= self.config.total_episode_steps
        info = {
            **reward_info,
            "goal_index": self.current_goal_index,
            "label_idx": self.current_goal.label_idx if self.current_goal is not None else -1,
            "step_displacement": displacement,
            "table_dropped": float(self.table_dropped),
            "goal_valid_execution": float(self.current_goal.valid_execution) if self.current_goal is not None else 0.0,
            "hand_target_clip_fraction": float(
                np.mean(np.abs(self.current_hand_target_qpos - unclamped_hand_target_qpos) > 1e-9)
            ),
            "arm_joint_limit_fraction": float(
                np.mean(
                    (self.current_ctrl[:6] <= self.runtime.arm_lower + 1e-4)
                    | (self.current_ctrl[:6] >= self.runtime.arm_upper - 1e-4)
                )
            ),
            "hand_joint_limit_fraction": float(
                np.mean(
                    (self.current_ctrl[6:] <= self.runtime.hand_lower + 1e-4)
                    | (self.current_ctrl[6:] >= self.runtime.hand_upper - 1e-4)
                )
            ),
        }
        if truncated:
            trace = np.asarray(self.metrics.displacement_trace, dtype=np.float64)
            goal_attained = float(
                reward_info["wrist_translation_error_m"] <= self.config.conversion.wrist_error_threshold_m
                and reward_info["site_rmse_m"] <= self.config.conversion.site_rmse_threshold_m
                and reward_info["matched_positive_contacts"] >= 0.5
            )
            episode_success = float(not self.slipped) if self.drop_table_after_pregrasp else goal_attained
            info.update(
                {
                    "episode_reward": float(self.metrics.reward_sum),
                    "episode_success": episode_success,
                    "episode_goal_attained": goal_attained,
                    "displacement_mean": float(trace.mean()) if trace.size else 0.0,
                    "displacement_std": float(trace.std()) if trace.size else 0.0,
                }
            )

        self.previous_action = action.copy()
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        return None

    def get_qpos(self) -> np.ndarray:
        return self.runtime.data.qpos.copy()


def create_env(
    config: TaskConfig,
    drop_table_after_pregrasp: bool = True,
    deterministic_goal_index: int | None = None,
) -> FR5LowLevelGraspEnv:
    return FR5LowLevelGraspEnv(
        config=config,
        drop_table_after_pregrasp=drop_table_after_pregrasp,
        deterministic_goal_index=deterministic_goal_index,
    )


def make_env(config: TaskConfig, drop_table_after_pregrasp: bool = True, deterministic_goal_index: int | None = None):
    return partial(
        create_env,
        config=config,
        drop_table_after_pregrasp=drop_table_after_pregrasp,
        deterministic_goal_index=deterministic_goal_index,
    )
