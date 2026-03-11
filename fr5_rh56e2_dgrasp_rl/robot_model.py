from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import mujoco
import numpy as np

from .scene_builder import build_training_scene
from .semantics import SEMANTIC_CONTACT_NAMES
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
        self.object_body_id = self.body_id_by_name[self.object_body_name]
        self.table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.table_geom_name)
        self.object_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.object_joint_name)
        self.object_qpos_adr = int(self.model.jnt_qposadr[self.object_joint_id])
        self.object_qvel_adr = int(self.model.jnt_dofadr[self.object_joint_id])
        self.original_table_center_z = float(self.model.geom_pos[self.table_geom_id, 2])

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
        self.set_table_height(self.original_table_center_z)
        mujoco.mj_forward(self.model, self.data)

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

    def step(self, ctrl: np.ndarray, frame_skip: int) -> None:
        self.data.ctrl[:] = self.clamp_actuated(np.asarray(ctrl, dtype=np.float64))
        for _ in range(frame_skip):
            mujoco.mj_step(self.model, self.data)

    def settle_actuated_pose(self, actuated_qpos: np.ndarray, settle_steps: int) -> None:
        actuated_qpos = self.clamp_actuated(np.asarray(actuated_qpos, dtype=np.float64))
        self.set_robot_actuated_qpos(actuated_qpos, update_ctrl=True)
        for _ in range(max(0, int(settle_steps))):
            mujoco.mj_step(self.model, self.data)

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
        return {
            "mask": mask,
            "forces": forces,
            "table_contact": table_contact,
            "total_penetration_m": total_penetration,
            "max_penetration_m": max_penetration,
        }
