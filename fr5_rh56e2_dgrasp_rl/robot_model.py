from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import mujoco
import numpy as np

from .scene_builder import build_training_scene
from .semantics import (
    GENHAND_CONTACT_FAMILY_NAMES,
    GENHAND_ROBOT_CONTACT_PADSET_FILE,
    SEMANTIC_CONTACT_DISTANCE_THRESHOLDS_M,
    SEMANTIC_CONTACT_NAMES,
)
from .task_config import TaskConfig


class RobotSceneModel:
    def __init__(self, config: TaskConfig, scene_xml: Path | None = None, metadata_path: Path | None = None) -> None:
        if scene_xml is None or metadata_path is None:
            scene_xml, metadata_path = build_training_scene(config)

        self.config = config
        self.scene_xml = Path(scene_xml)
        self.metadata_path = Path(metadata_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)

        self.movable_joints = list(self.metadata["movable_joints"])
        self.actuated_joints = list(self.metadata["actuated_joints"])
        self.arm_joints = list(self.metadata["arm_joints"])
        self.hand_joints = list(self.metadata["hand_joints"])
        self.semantic_sites = list(self.metadata["semantic_sites"])
        self.contact_group_bodies = {
            key: list(value) for key, value in self.metadata["contact_group_bodies"].items()
        }
        self.contact_group_bodies_12 = {
            "palm": ["j6_Link"],
            "thumb_proximal": ["rh56e2_right_thumb_2"],
            "thumb_middle": ["rh56e2_right_thumb_3"],
            "thumb_distal": ["rh56e2_right_thumb_4"],
            "index_proximal": ["rh56e2_right_index_1"],
            "index_distal": ["rh56e2_right_index_2"],
            "middle_proximal": ["rh56e2_right_middle_1"],
            "middle_distal": ["rh56e2_right_middle_2"],
            "ring_proximal": ["rh56e2_right_ring_1"],
            "ring_distal": ["rh56e2_right_ring_2"],
            "little_proximal": ["rh56e2_right_little_1"],
            "little_distal": ["rh56e2_right_little_2"],
        }
        self.arm_link_names = ["j1_Link", "j2_Link", "j3_Link", "j4_Link", "j5_Link", "j6_Link"]
        self.arm_sweep_link_pairs = [("j2_Link", "j3_Link"), ("j3_Link", "j4_Link"), ("j4_Link", "j5_Link"), ("j5_Link", "j6_Link")]
        self.object_body_name = str(self.metadata["object_body_name"])
        self.object_joint_name = str(self.metadata["object_joint_name"])
        self.object_center_site = str(self.metadata["object_center_site"])
        self.table_geom_name = str(self.metadata["table_geom_name"])
        self.default_object_pose = np.asarray(self.metadata["default_object_pose"], dtype=np.float64)
        self.home_qpos = np.asarray(self.metadata["home_qpos"], dtype=np.float64)
        self.home_ctrl = np.asarray(self.metadata["home_ctrl"], dtype=np.float64)
        self.joint_limits: Dict[str, tuple[float, float]] = {
            name: (float(bounds[0]), float(bounds[1]))
            for name, bounds in self.metadata["joint_limits"].items()
        }
        self.mimics = {
            item["joint"]: {
                "master": item["master"],
                "multiplier": float(item["multiplier"]),
                "offset": float(item["offset"]),
            }
            for item in self.metadata["mimics"]
        }

        self.ctrl_index_by_joint = {
            joint_name: index for index, joint_name in enumerate(self.actuated_joints)
        }
        self.qpos_index_by_joint = {}
        self.qvel_index_by_joint = {}
        for joint_name in self.movable_joints:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.qpos_index_by_joint[joint_name] = int(self.model.jnt_qposadr[joint_id])
            self.qvel_index_by_joint[joint_name] = int(self.model.jnt_dofadr[joint_id])

        self.site_id_by_name = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in self.semantic_sites + [self.object_center_site]
        }
        self.body_id_by_name = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in {
                self.object_body_name,
                *self.arm_link_names,
                "j6_Link",
                *self.contact_group_bodies["thumb"],
                *self.contact_group_bodies["index"],
                *self.contact_group_bodies["middle"],
                *self.contact_group_bodies["ring"],
                *self.contact_group_bodies["little"],
                *self.contact_group_bodies_12["thumb_middle"],
                *self.contact_group_bodies_12["thumb_distal"],
            }
        }
        self.contact_group_body_ids = {
            group: {self.body_id_by_name[name] for name in body_names}
            for group, body_names in self.contact_group_bodies.items()
        }
        self.contact_group_body_ids_12 = {
            group: {self.body_id_by_name[name] for name in body_names}
            for group, body_names in self.contact_group_bodies_12.items()
        }
        self.body_geom_ids: dict[int, list[int]] = {}
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            self.body_geom_ids.setdefault(body_id, []).append(geom_id)
        self.object_body_id = self.body_id_by_name[self.object_body_name]
        self.table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.table_geom_name)
        self.object_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.object_joint_name)
        self.object_qpos_adr = int(self.model.jnt_qposadr[self.object_joint_id])
        self.object_qvel_adr = int(self.model.jnt_dofadr[self.object_joint_id])
        self.original_table_center_z = float(self.model.geom_pos[self.table_geom_id, 2])
        self.object_contact_geom_ids = [
            geom_id
            for geom_id in self.body_geom_ids.get(self.object_body_id, [])
            if int(self.model.geom_contype[geom_id]) != 0 or int(self.model.geom_conaffinity[geom_id]) != 0
        ]
        self.contact_group_geom_ids_12 = {
            group: [
                geom_id
                for body_id in body_ids
                for geom_id in self.body_geom_ids.get(body_id, [])
                if int(self.model.geom_contype[geom_id]) != 0 or int(self.model.geom_conaffinity[geom_id]) != 0
            ]
            for group, body_ids in self.contact_group_body_ids_12.items()
        }
        self.contact_candidate_names: list[str] = []
        self.contact_candidate_family_indices: list[int] = []
        self.contact_candidate_patch_channels: list[list[dict[str, np.ndarray | int | str]]] = []
        family_index_by_name = {
            family_name: family_index for family_index, family_name in enumerate(GENHAND_CONTACT_FAMILY_NAMES)
        }
        for family_name, patch_specs in self._load_contact_padset().items():
            patches: list[dict[str, np.ndarray | int | str]] = []
            for patch_spec in patch_specs:
                body_name = str(patch_spec["body_name"])
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_id == -1:
                    continue
                local_points = np.asarray(patch_spec["contact_points"], dtype=np.float64).reshape(-1, 3)
                local_normals = np.asarray(patch_spec["contact_normals"], dtype=np.float64).reshape(-1, 3)
                patches.append(
                    {
                        "body_id": int(body_id),
                        "body_name": body_name,
                        "sensor_mesh": str(patch_spec["sensor_mesh"]),
                        "local_points": local_points,
                        "local_normals": local_normals,
                    }
                )
            if not patches:
                continue
            self.contact_candidate_names.append(family_name)
            self.contact_candidate_family_indices.append(int(family_index_by_name[str(family_name)]))
            self.contact_candidate_patch_channels.append(patches)

        self.actuated_lower = np.array([self.joint_limits[name][0] for name in self.actuated_joints], dtype=np.float64)
        self.actuated_upper = np.array([self.joint_limits[name][1] for name in self.actuated_joints], dtype=np.float64)
        self.arm_lower = np.array([self.joint_limits[name][0] for name in self.arm_joints], dtype=np.float64)
        self.arm_upper = np.array([self.joint_limits[name][1] for name in self.arm_joints], dtype=np.float64)
        self.hand_lower = np.array([self.joint_limits[name][0] for name in self.hand_joints], dtype=np.float64)
        self.hand_upper = np.array([self.joint_limits[name][1] for name in self.hand_joints], dtype=np.float64)
        self.home_joint_map = {
            joint_name: float(self.home_qpos[index])
            for index, joint_name in enumerate(self.movable_joints)
        }
        self.home_actuated = self.home_ctrl.copy()
        self.default_actuator_kp = self.model.actuator_gainprm[:, 0].copy()
        self.default_actuator_bias_kp = (-self.model.actuator_biasprm[:, 1]).copy()
        self.reset()

    def reset(self) -> None:
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            self.data.qpos[: len(self.home_qpos)] = self.home_qpos
            self.data.ctrl[:] = self.home_ctrl
        self.set_object_pose(self.default_object_pose)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.home_ctrl
        self.data.qfrc_applied[:] = 0.0
        self.set_position_control_stiffness(1.0, 1.0)
        self.set_table_height(self.original_table_center_z)
        mujoco.mj_forward(self.model, self.data)

    def set_position_control_stiffness(self, arm_kp_scale: float = 1.0, hand_kp_scale: float = 1.0) -> None:
        arm_count = len(self.arm_joints)
        actuator_count = len(self.actuated_joints)
        scales = np.ones(actuator_count, dtype=np.float64)
        scales[:arm_count] = float(arm_kp_scale)
        scales[arm_count:] = float(hand_kp_scale)
        self.model.actuator_gainprm[:actuator_count, 0] = self.default_actuator_kp[:actuator_count] * scales
        self.model.actuator_biasprm[:actuator_count, 1] = -self.default_actuator_bias_kp[:actuator_count] * scales

    def set_arm_hold_mode(self, enabled: bool, arm_kp_scale: float = 1.0) -> None:
        scale = float(arm_kp_scale if enabled else 1.0)
        self.set_position_control_stiffness(arm_kp_scale=scale, hand_kp_scale=1.0)

    def _arm_support_torque(self, force_world: np.ndarray) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self.model,
            self.data,
            jacp,
            jacr,
            self.site_id_by_name["wrist_mount"],
        )
        tau = jacp.T @ np.asarray(force_world, dtype=np.float64)
        qfrc = np.zeros(self.model.nv, dtype=np.float64)
        arm_dof_indices = [self.qvel_index_by_joint[name] for name in self.arm_joints]
        qfrc[arm_dof_indices] = tau[arm_dof_indices]
        return qfrc

    def clamp_actuated(self, values: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(values, self.actuated_lower), self.actuated_upper)

    def clamp_arm(self, values: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(values, self.arm_lower), self.arm_upper)

    def clamp_hand(self, values: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(values, self.hand_lower), self.hand_upper)

    def _resolve_joint_targets(self, joint_targets: Dict[str, float], base_joint_values: Dict[str, float] | None = None) -> Dict[str, float]:
        explicit_targets = {
            joint_name: float(np.clip(value, *self.joint_limits[joint_name]))
            for joint_name, value in joint_targets.items()
        }
        base_values = dict(self.home_joint_map)
        if base_joint_values is not None:
            base_values.update(base_joint_values)

        resolved: Dict[str, float] = {}

        def resolve(joint_name: str) -> float:
            if joint_name in resolved:
                return resolved[joint_name]
            if joint_name in explicit_targets:
                value = explicit_targets[joint_name]
            elif joint_name in self.mimics:
                mimic = self.mimics[joint_name]
                value = mimic["offset"] + mimic["multiplier"] * resolve(mimic["master"])
            else:
                value = float(base_values.get(joint_name, 0.0))
            value = float(np.clip(value, *self.joint_limits[joint_name]))
            resolved[joint_name] = value
            return value

        for joint_name in self.movable_joints:
            resolve(joint_name)
        return resolved

    def actuated_to_full_joint_map(self, actuated_qpos: np.ndarray) -> Dict[str, float]:
        base_joint_values = {
            name: float(value) for name, value in zip(self.actuated_joints, np.asarray(actuated_qpos, dtype=np.float64))
        }
        return self._resolve_joint_targets(base_joint_values)

    def set_robot_actuated_qpos(self, actuated_qpos: np.ndarray, update_ctrl: bool = True) -> None:
        actuated_qpos = self.clamp_actuated(np.asarray(actuated_qpos, dtype=np.float64))
        resolved = self.actuated_to_full_joint_map(actuated_qpos)
        for joint_name, value in resolved.items():
            self.data.qpos[self.qpos_index_by_joint[joint_name]] = value
        self.data.qvel[:] = 0.0
        if update_ctrl:
            self.data.ctrl[:] = actuated_qpos
        mujoco.mj_forward(self.model, self.data)

    def get_actuated_qpos(self) -> np.ndarray:
        return np.array(
            [self.data.qpos[self.qpos_index_by_joint[name]] for name in self.actuated_joints],
            dtype=np.float64,
        )

    def get_actuated_qvel(self) -> np.ndarray:
        return np.array(
            [self.data.qvel[self.qvel_index_by_joint[name]] for name in self.actuated_joints],
            dtype=np.float64,
        )

    def set_object_pose(self, pose: np.ndarray) -> None:
        pose = np.asarray(pose, dtype=np.float64)
        self.data.qpos[self.object_qpos_adr : self.object_qpos_adr + 7] = pose
        self.data.qvel[self.object_qvel_adr : self.object_qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def pin_object_pose(self, pose: np.ndarray, forward: bool = True) -> None:
        pose = np.asarray(pose, dtype=np.float64)
        self.data.qpos[self.object_qpos_adr : self.object_qpos_adr + 7] = pose
        self.data.qvel[self.object_qvel_adr : self.object_qvel_adr + 6] = 0.0
        if forward:
            mujoco.mj_forward(self.model, self.data)

    def get_object_pose(self) -> np.ndarray:
        return self.data.qpos[self.object_qpos_adr : self.object_qpos_adr + 7].copy()

    def get_object_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.object_body_id,
            velocity,
            0,
        )
        angular = velocity[:3].copy()
        linear = velocity[3:].copy()
        return linear, angular

    def get_semantic_sites_world(self) -> np.ndarray:
        return np.vstack([self.data.site_xpos[self.site_id_by_name[name]].copy() for name in self.semantic_sites])

    def get_contact_proxy_points_world_12(self) -> np.ndarray:
        points = []
        for group_name in SEMANTIC_CONTACT_NAMES:
            geom_ids = self.contact_group_geom_ids_12.get(group_name, [])
            if geom_ids:
                geom_points = [self.data.geom_xpos[int(geom_id)].copy() for geom_id in geom_ids]
                points.append(np.mean(np.asarray(geom_points, dtype=np.float64), axis=0))
            else:
                fallback_body_ids = list(self.contact_group_body_ids_12.get(group_name, []))
                if fallback_body_ids:
                    points.append(self.data.xpos[int(fallback_body_ids[0])].copy())
                else:
                    points.append(np.zeros(3, dtype=np.float64))
        return np.asarray(points, dtype=np.float64)

    def _load_contact_padset(self) -> dict[str, list[dict[str, object]]]:
        padset_path = Path(__file__).resolve().parent.parent / "assets" / "base_scene" / GENHAND_ROBOT_CONTACT_PADSET_FILE
        payload = json.loads(padset_path.read_text(encoding="utf-8"))
        grouped: dict[str, list[dict[str, object]]] = {family: [] for family in GENHAND_CONTACT_FAMILY_NAMES}
        for item in payload:
            family_name = str(item["family_name"])
            if family_name not in grouped:
                continue
            grouped[family_name].append(dict(item))
        return grouped

    def get_contact_candidate_point_sets_world(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        point_sets: list[np.ndarray] = []
        normal_sets: list[np.ndarray] = []
        family_indices: list[int] = []
        for family_index, patch_specs in zip(
            self.contact_candidate_family_indices,
            self.contact_candidate_patch_channels,
        ):
            family_points: list[np.ndarray] = []
            family_normals: list[np.ndarray] = []
            for patch in patch_specs:
                body_id = int(patch["body_id"])
                local_points = np.asarray(patch["local_points"], dtype=np.float64)
                local_normals = np.asarray(patch["local_normals"], dtype=np.float64)
                rotation = self.data.xmat[body_id].reshape(3, 3).copy()
                translation = self.data.xpos[body_id].copy()
                family_points.append((rotation @ local_points.T).T + translation)
                family_normals.append((rotation @ local_normals.T).T)
            if not family_points:
                continue
            points = np.concatenate(family_points, axis=0)
            normals = np.concatenate(family_normals, axis=0)
            point_sets.append(points)
            normal_sets.append(normals)
            family_indices.append(int(family_index))
        if not point_sets:
            return (
                np.zeros((0, 0, 3), dtype=np.float64),
                np.zeros((0, 0, 3), dtype=np.float64),
                np.zeros(0, dtype=np.int64),
            )
        return (
            np.asarray(point_sets, dtype=np.float64),
            np.asarray(normal_sets, dtype=np.float64),
            np.asarray(family_indices, dtype=np.int64),
        )

    def get_contact_candidate_points_world(self) -> tuple[np.ndarray, np.ndarray]:
        point_sets, _normal_sets, family_indices = self.get_contact_candidate_point_sets_world()
        if point_sets.size == 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.int64)
        flat_points = point_sets.reshape(-1, 3)
        repeated_families = np.repeat(family_indices, point_sets.shape[1])
        return (
            np.asarray(flat_points, dtype=np.float64),
            np.asarray(repeated_families, dtype=np.int64),
        )

    def get_finger_directions_world(self) -> np.ndarray:
        point_sets, _normal_sets, family_indices = self.get_contact_candidate_point_sets_world()
        semantic_sites = self.get_semantic_sites_world()
        channel_centers = np.zeros((len(GENHAND_CONTACT_FAMILY_NAMES), 3), dtype=np.float64)
        for point_set, family_index in zip(point_sets, family_indices):
            channel_centers[int(family_index)] = np.mean(np.asarray(point_set, dtype=np.float64), axis=0)
        finger_dirs = [
            semantic_sites[2] - channel_centers[0],  # thumb
            semantic_sites[3] - channel_centers[1],  # index
            semantic_sites[4] - channel_centers[2],  # middle
            semantic_sites[5] - channel_centers[3],  # ring
            semantic_sites[6] - channel_centers[4],  # little
        ]
        norms = np.linalg.norm(finger_dirs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return np.asarray(finger_dirs, dtype=np.float64) / norms

    def get_table_top_z(self) -> float:
        return float(self.model.geom_pos[self.table_geom_id, 2] + self.model.geom_size[self.table_geom_id, 2])

    def get_proxy_table_clearance_12(self) -> np.ndarray:
        top_z = self.get_table_top_z()
        points = self.get_contact_proxy_points_world_12()
        return np.asarray(points[:, 2] - top_z, dtype=np.float64)

    def get_arm_link_positions_world(self, link_names: list[str] | None = None) -> np.ndarray:
        names = self.arm_link_names if link_names is None else link_names
        return np.vstack([self.data.xpos[self.body_id_by_name[name]].copy() for name in names])

    def get_arm_table_clearance(self, link_names: list[str] | None = None) -> np.ndarray:
        positions = self.get_arm_link_positions_world(link_names)
        return np.asarray(positions[:, 2] - self.get_table_top_z(), dtype=np.float64)

    def get_arm_sweep_proxy_points_world(self) -> np.ndarray:
        points: list[np.ndarray] = []
        for start_name, end_name in self.arm_sweep_link_pairs:
            start = self.data.xpos[self.body_id_by_name[start_name]].copy()
            end = self.data.xpos[self.body_id_by_name[end_name]].copy()
            for alpha in (0.0, 0.33, 0.66, 1.0):
                points.append((1.0 - alpha) * start + alpha * end)
        return np.asarray(points, dtype=np.float64)

    def get_arm_sweep_table_clearance(self) -> np.ndarray:
        points = self.get_arm_sweep_proxy_points_world()
        return np.asarray(points[:, 2] - self.get_table_top_z(), dtype=np.float64)

    def get_site_velocity(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        site_id = self.site_id_by_name[site_name]
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        linear = jacp @ self.data.qvel
        angular = jacr @ self.data.qvel
        return linear, angular

    def set_table_height(self, center_z: float) -> None:
        self.model.geom_pos[self.table_geom_id, 2] = center_z
        mujoco.mj_forward(self.model, self.data)

    def step(
        self,
        ctrl: np.ndarray,
        frame_skip: int,
        arm_support_force_world: np.ndarray | None = None,
        fixed_object_pose: np.ndarray | None = None,
    ) -> None:
        self.data.ctrl[:] = self.clamp_actuated(np.asarray(ctrl, dtype=np.float64))
        support_torque = None
        if arm_support_force_world is not None:
            support_torque = self._arm_support_torque(np.asarray(arm_support_force_world, dtype=np.float64))
        for _ in range(frame_skip):
            self.data.qfrc_applied[:] = 0.0
            if support_torque is not None:
                self.data.qfrc_applied[:] = support_torque
            mujoco.mj_step(self.model, self.data)
            if fixed_object_pose is not None:
                self.pin_object_pose(fixed_object_pose, forward=True)
        self.data.qfrc_applied[:] = 0.0

    def settle_actuated_pose(
        self,
        actuated_qpos: np.ndarray,
        settle_steps: int,
        fixed_object_pose: np.ndarray | None = None,
        arm_kp_scale: float = 1.0,
        hand_kp_scale: float = 1.0,
    ) -> None:
        actuated_qpos = self.clamp_actuated(np.asarray(actuated_qpos, dtype=np.float64))
        self.set_position_control_stiffness(
            arm_kp_scale=float(arm_kp_scale),
            hand_kp_scale=float(hand_kp_scale),
        )
        try:
            self.set_robot_actuated_qpos(actuated_qpos, update_ctrl=True)
            if fixed_object_pose is not None:
                self.pin_object_pose(fixed_object_pose, forward=True)
            for _ in range(max(0, int(settle_steps))):
                self.step(actuated_qpos, 1, fixed_object_pose=fixed_object_pose)
        finally:
            self.set_position_control_stiffness(1.0, 1.0)

    def _collect_object_contacts(
        self,
        group_body_ids: dict[str, set[int]],
        group_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, bool, float, float]:
        mask = np.zeros(len(group_names), dtype=np.float64)
        forces = np.zeros(len(group_names), dtype=np.float64)
        table_contact = False
        total_penetration = 0.0
        max_penetration = 0.0

        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            involves_object = body1 == self.object_body_id or body2 == self.object_body_id
            if not involves_object:
                continue

            if geom1 == self.table_geom_id or geom2 == self.table_geom_id:
                table_contact = True

            other_body = body2 if body1 == self.object_body_id else body1
            wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_index, wrench)
            contact_force = float(np.linalg.norm(wrench[:3]))
            penetration = max(0.0, float(-contact.dist))
            total_penetration += penetration
            max_penetration = max(max_penetration, penetration)

            for group_index, group_name in enumerate(group_names):
                if other_body in group_body_ids[group_name]:
                    mask[group_index] = 1.0
                    forces[group_index] += contact_force
                    break

        return mask, forces, table_contact, total_penetration, max_penetration

    def _collect_object_proximity(
        self,
        group_geom_ids: dict[str, list[int]],
        group_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        thresholds = np.asarray(SEMANTIC_CONTACT_DISTANCE_THRESHOLDS_M, dtype=np.float64)
        distmax = float(np.max(thresholds) + 0.02)
        distances = np.full(len(group_names), distmax, dtype=np.float64)
        proximity_scores = np.zeros(len(group_names), dtype=np.float64)
        proximity_mask = np.zeros(len(group_names), dtype=np.float64)

        if not self.object_contact_geom_ids:
            return distances, proximity_scores, proximity_mask

        for group_index, group_name in enumerate(group_names):
            min_distance = distmax
            for object_geom_id in self.object_contact_geom_ids:
                for group_geom_id in group_geom_ids.get(group_name, []):
                    distance = float(
                        mujoco.mj_geomDistance(
                            self.model,
                            self.data,
                            object_geom_id,
                            group_geom_id,
                            distmax,
                            None,
                        )
                    )
                    min_distance = min(min_distance, distance)
            threshold = float(thresholds[group_index])
            distances[group_index] = min_distance
            proximity_scores[group_index] = float(np.clip((threshold - min_distance) / max(threshold, 1e-8), 0.0, 1.0))
            proximity_mask[group_index] = float(min_distance <= threshold)

        return distances, proximity_scores, proximity_mask

    def get_contact_state(self) -> tuple[np.ndarray, np.ndarray, bool]:
        group_names = list(self.contact_group_body_ids.keys())
        mask, forces, table_contact, _, _ = self._collect_object_contacts(
            self.contact_group_body_ids,
            group_names,
        )
        return mask, forces, table_contact

    def get_contact_state_12(self) -> tuple[np.ndarray, np.ndarray, bool]:
        mask, forces, table_contact, _, _ = self._collect_object_contacts(
            self.contact_group_body_ids_12,
            list(SEMANTIC_CONTACT_NAMES),
        )
        return mask, forces, table_contact

    def get_contact_diagnostics_12(self) -> dict[str, object]:
        mask, forces, table_contact, total_penetration, max_penetration = self._collect_object_contacts(
            self.contact_group_body_ids_12,
            list(SEMANTIC_CONTACT_NAMES),
        )
        distances, proximity_scores, proximity_mask = self._collect_object_proximity(
            self.contact_group_geom_ids_12,
            list(SEMANTIC_CONTACT_NAMES),
        )
        hybrid_mask = np.maximum(mask, proximity_mask)
        return {
            "mask": mask,
            "hard_mask": mask,
            "forces": forces,
            "distances_m": distances,
            "proximity_scores": proximity_scores,
            "proximity_mask": proximity_mask,
            "hybrid_mask": hybrid_mask,
            "table_contact": table_contact,
            "total_penetration_m": total_penetration,
            "max_penetration_m": max_penetration,
        }
