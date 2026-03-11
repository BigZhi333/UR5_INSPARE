from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer as mujoco_viewer
import numpy as np
import torch

from .scene_builder import build_training_scene
from .task_config import TaskConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a saved FR5+RH56E2 trajectory.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "default_task.json")
    parser.add_argument("--trajectory-file", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser


def replay_main(args: argparse.Namespace | None = None) -> None:
    parsed = build_arg_parser().parse_args() if args is None else args

    config = TaskConfig.from_json(parsed.config)
    scene_xml, _ = build_training_scene(config)
    trajectory_data = np.load(parsed.trajectory_file)
    qpos = trajectory_data["qpos"][parsed.episode_index]
    length = int(trajectory_data["lengths"][parsed.episode_index])

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)

    with mujoco_viewer.launch_passive(model, data) as viewer:
        for frame_idx in range(length):
            data.qpos[:] = qpos[frame_idx]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / parsed.fps)
