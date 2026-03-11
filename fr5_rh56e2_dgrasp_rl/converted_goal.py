from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ConvertedGoal:
    object_id: int
    label_idx: int
    object_pose_init: list[float]
    object_pose_goal: list[float]
    target_qpos_12: list[float]
    target_sites_obj_21: list[float]
    target_contact_mask_6: list[float]
    fit_error: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "label_idx": self.label_idx,
            "object_pose_init": self.object_pose_init,
            "object_pose_goal": self.object_pose_goal,
            "target_qpos_12": self.target_qpos_12,
            "target_sites_obj_21": self.target_sites_obj_21,
            "target_contact_mask_6": self.target_contact_mask_6,
            "fit_error": self.fit_error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConvertedGoal":
        return cls(
            object_id=int(payload["object_id"]),
            label_idx=int(payload["label_idx"]),
            object_pose_init=[float(v) for v in payload["object_pose_init"]],
            object_pose_goal=[float(v) for v in payload["object_pose_goal"]],
            target_qpos_12=[float(v) for v in payload["target_qpos_12"]],
            target_sites_obj_21=[float(v) for v in payload["target_sites_obj_21"]],
            target_contact_mask_6=[float(v) for v in payload["target_contact_mask_6"]],
            fit_error={str(k): float(v) for k, v in payload["fit_error"].items()},
        )


def save_converted_goals(path: Path, goals: list[ConvertedGoal]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [goal.to_dict() for goal in goals]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_converted_goals(path: Path) -> list[ConvertedGoal]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ConvertedGoal.from_dict(item) for item in payload]
