from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .paths import PROJECT_DIR


@dataclass
class PPOConfig:
    hidden_sizes: list[int]
    gamma: float
    gae_lambda: float
    clip_ratio: float
    learning_rate: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    desired_kl: float
    epochs: int
    minibatches: int
    train_updates: int
    save_every: int


@dataclass
class EvalConfig:
    num_episodes: int
    drop_table_after_pregrasp: bool
    freeze_control_after_pregrasp: bool
    success_height_threshold_m: float
    save_trajectories: bool


@dataclass
class ConversionConfig:
    wrist_error_threshold_m: float
    wrist_rot_threshold_deg: float
    site_rmse_threshold_m: float
    min_valid_goals: int
    arm_ik_iterations: int
    arm_ik_damping: float
    finger_opt_max_nfev: int
    joint_opt_max_nfev: int
    candidate_screening_mode: str
    saved_target_mode: str


@dataclass
class TaskConfig:
    task_name: str
    object_id: int
    object_name: str
    object_dims_m: list[float]
    object_geom_type: str
    object_mass_kg: float
    table_center: list[float]
    table_size: list[float]
    table_friction: list[float]
    default_object_pose: list[float]
    workspace_translation: list[float]
    control_dt: float
    frame_skip: int
    pre_grasp_steps: int
    hold_steps: int
    num_envs: int
    action_delta_arm: float
    action_delta_wrist_pos_m: float
    action_delta_wrist_rot_rad: float
    action_delta_hand: float
    joint_noise_std: float
    pregrasp_backoff_m: float
    pregrasp_lift_m: float
    reward_clip: float
    reward: dict[str, float] = field(default_factory=dict)
    ppo: PPOConfig | None = None
    eval: EvalConfig | None = None
    conversion: ConversionConfig | None = None

    @property
    def total_episode_steps(self) -> int:
        return self.pre_grasp_steps + self.hold_steps

    @property
    def timestep(self) -> float:
        return self.control_dt / float(self.frame_skip)

    @property
    def project_dir(self) -> Path:
        return PROJECT_DIR

    @property
    def default_scene_xml(self) -> Path:
        return self.project_dir / "build" / f"fr5_rh56e2_{self.object_name}_scene.xml"

    @property
    def default_scene_metadata(self) -> Path:
        return self.project_dir / "build" / f"fr5_rh56e2_{self.object_name}_metadata.json"

    @property
    def converted_goals_path(self) -> Path:
        return self.project_dir / "data" / f"{self.object_name}_converted_goals.json"

    @property
    def manual_goals_path(self) -> Path:
        return self.project_dir / "data" / f"{self.object_name}_manual_goals.json"

    @classmethod
    def from_json(cls, path: Path) -> "TaskConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("object_geom_type", "box")
        payload.setdefault("conversion", {})
        payload["conversion"].setdefault("candidate_screening_mode", "settle")
        payload["conversion"].setdefault("saved_target_mode", "settled_anchored")
        payload["ppo"] = PPOConfig(**payload["ppo"])
        payload["eval"] = EvalConfig(**payload["eval"])
        payload["conversion"] = ConversionConfig(**payload["conversion"])
        return cls(**payload)

    @classmethod
    def default(cls) -> "TaskConfig":
        return cls.from_json(PROJECT_DIR / "config" / "default_task.json")
