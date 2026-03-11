from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer as mujoco_viewer
import numpy as np

from fr5_rh56e2_dgrasp_rl.kinematics import solve_arm_wrist_palm_ik
from fr5_rh56e2_dgrasp_rl.paths import configure_local_runtime_env
from fr5_rh56e2_dgrasp_rl.pose_driven_data import (
    PROJECTION_SETTLE_STEPS,
    PoseDrivenSample,
    load_pose_driven_samples,
    pose_driven_samples_path,
    prepare_pose_driven_samples,
    wrist_pose_from_semantic_sites,
    wrist_pose_to_target_sites,
)
from fr5_rh56e2_dgrasp_rl.robot_model import RobotSceneModel
from fr5_rh56e2_dgrasp_rl.scene_builder import build_training_scene
from fr5_rh56e2_dgrasp_rl.semantics import SEMANTIC_CONTACT_NAMES
from fr5_rh56e2_dgrasp_rl.task_config import TaskConfig
from fr5_rh56e2_dgrasp_rl.utils import normalize


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View one pose-driven D-Grasp sample on FR5+RH56E2.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config" / "default_task.json")
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--label-idx", type=int, default=None)
    parser.add_argument("--best-fit", action="store_true")
    parser.add_argument("--best-grasp", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "build" / "pose_driven_sample_preview.png",
    )
    parser.add_argument("--animate-steps", type=int, default=180)
    parser.add_argument("--hold-steps", type=int, default=120)
    parser.add_argument("--sim-substeps", type=int, default=4)
    parser.add_argument("--kinematic", action="store_true")
    return parser


def configure_camera(camera: mujoco.MjvCamera) -> None:
    camera.lookat[:] = np.array([0.42, 0.02, 0.63], dtype=np.float64)
    camera.distance = 1.15
    camera.azimuth = 138.0
    camera.elevation = -22.0


def render_preview(model: mujoco.MjModel, data: mujoco.MjData, width: int, height: int, output: Path) -> Path:
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    configure_camera(camera)
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()

    try:
        from PIL import Image

        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(output)
    except ImportError:
        ppm_path = output.with_suffix(".ppm")
        ppm_path.parent.mkdir(parents=True, exist_ok=True)
        with ppm_path.open("wb") as handle:
            handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            handle.write(pixels[:, :, :3].tobytes())
        return ppm_path
    return output


def quaternion_slerp(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    start = normalize(np.asarray(start, dtype=np.float64))
    end = normalize(np.asarray(end, dtype=np.float64))
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    if dot > 0.9995:
        return normalize((1.0 - alpha) * start + alpha * end)
    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * alpha
    sin_theta = float(np.sin(theta))
    scale_start = float(np.sin(theta_0 - theta) / sin_theta_0)
    scale_end = float(sin_theta / sin_theta_0)
    return scale_start * start + scale_end * end


def interpolate_pose(start_pose: np.ndarray, end_pose: np.ndarray, alpha: float) -> np.ndarray:
    pose = np.zeros(7, dtype=np.float64)
    pose[:3] = (1.0 - alpha) * start_pose[:3] + alpha * end_pose[:3]
    pose[3:] = quaternion_slerp(start_pose[3:], end_pose[3:], alpha)
    return pose


def select_sample(samples: list[PoseDrivenSample], args: argparse.Namespace) -> tuple[int, PoseDrivenSample]:
    if not samples:
        raise RuntimeError("No pose-driven samples are available.")
    if args.label_idx is not None:
        for sample_index, sample in enumerate(samples):
            if sample.label_idx == args.label_idx:
                return sample_index, sample
        raise ValueError(f"Label index {args.label_idx} was not found.")
    if args.best_grasp:
        def grasp_rank(idx: int) -> tuple[float, ...]:
            sample = samples[idx]
            fit = sample.fit_error
            num_contacts = float(sum(value > 0.5 for value in sample.contact_mask_12))
            return (
                float(not sample.valid_execution),
                float(-(num_contacts > 0.0)),
                -num_contacts,
                float(fit.get("projected_max_penetration_m", 0.0)),
                float(fit.get("source_contact_hamming", 999.0)),
                float(fit.get("source_tip_rmse_m", fit.get("tip_rmse_m", 999.0))),
                -float(sample.object_pose_goal[2]),
            )

        sample_index = min(range(len(samples)), key=grasp_rank)
        return sample_index, samples[sample_index]
    if args.best_fit:
        sample_index = min(
            range(len(samples)),
            key=lambda idx: samples[idx].fit_error.get(
                "source_tip_rmse_m",
                samples[idx].fit_error.get("tip_rmse_m", 0.0),
            ),
        )
        return sample_index, samples[sample_index]
    sample_index = 0 if args.sample_index is None else max(0, min(args.sample_index, len(samples) - 1))
    return sample_index, samples[sample_index]


def print_sample_summary(sample_index: int, sample: PoseDrivenSample, samples_path: Path) -> None:
    fit = sample.fit_error
    print(f"Samples file: {samples_path}")
    print(f"Sample index: {sample_index}")
    print(f"Label idx: {sample.label_idx}")
    print(f"Execution valid: {sample.valid_execution}")
    print(
        "Projection stats: "
        f"source_tip_rmse={fit.get('source_tip_rmse_m', fit.get('tip_rmse_m', 0.0)):.4f} m, "
        f"max_penetration={fit.get('projected_max_penetration_m', 0.0):.4f} m, "
        f"contact_hamming={fit.get('source_contact_hamming', 0.0):.0f}, "
        f"retreat={fit.get('projected_retreat_m', 0.0):.4f} m"
    )
    print(
        "Object goal pose: "
        + ", ".join(f"{value:.4f}" for value in sample.object_pose_goal[:3])
    )
    print(
        "Hand qpos 6: "
        + ", ".join(f"{value:.4f}" for value in sample.hand_qpos_6)
    )
    print(
        "Projected contact mask 12: "
        + ", ".join(
            f"{name}={int(round(value))}" for name, value in zip(SEMANTIC_CONTACT_NAMES, sample.contact_mask_12)
        )
    )
    print(
        "Source contact mask 12: "
        + ", ".join(
            f"{name}={int(round(value))}" for name, value in zip(SEMANTIC_CONTACT_NAMES, sample.source_contact_mask_12)
        )
    )


def solve_final_arm_qpos(runtime: RobotSceneModel, config: TaskConfig, sample: PoseDrivenSample) -> np.ndarray:
    return solve_arm_wrist_palm_ik(
        runtime=runtime,
        target_sites_world=wrist_pose_to_target_sites(np.asarray(sample.wrist_pose_goal_world, dtype=np.float64)),
        initial_arm_qpos=runtime.get_actuated_qpos()[:6],
        hand_qpos=np.asarray(sample.hand_qpos_6, dtype=np.float64),
        iterations=config.conversion.arm_ik_iterations,
        damping=config.conversion.arm_ik_damping,
    )


def set_final_target_pose(runtime: RobotSceneModel, config: TaskConfig, sample: PoseDrivenSample) -> None:
    runtime.reset()
    runtime.set_object_pose(np.asarray(sample.object_pose_goal, dtype=np.float64))
    arm_qpos = solve_final_arm_qpos(runtime, config, sample)
    runtime.settle_actuated_pose(
        np.concatenate([arm_qpos, np.asarray(sample.hand_qpos_6, dtype=np.float64)]),
        PROJECTION_SETTLE_STEPS,
    )


def play_sample_kinematic(
    runtime: RobotSceneModel,
    config: TaskConfig,
    sample: PoseDrivenSample,
    animate_steps: int,
    hold_steps: int,
) -> None:
    runtime.reset()
    runtime.set_object_pose(np.asarray(sample.object_pose_goal, dtype=np.float64))

    start_actuated = runtime.get_actuated_qpos()
    start_wrist_pose = wrist_pose_from_semantic_sites(runtime.get_semantic_sites_world())
    target_wrist_pose = np.asarray(sample.wrist_pose_goal_world, dtype=np.float64)
    target_hand_qpos = np.asarray(sample.hand_qpos_6, dtype=np.float64)
    arm_qpos = start_actuated[:6].copy()

    with mujoco_viewer.launch_passive(runtime.model, runtime.data) as viewer:
        configure_camera(viewer.cam)
        for step in range(max(1, animate_steps)):
            alpha = float(step + 1) / float(max(1, animate_steps))
            wrist_pose = interpolate_pose(start_wrist_pose, target_wrist_pose, alpha)
            hand_qpos = (1.0 - alpha) * start_actuated[6:] + alpha * target_hand_qpos
            arm_qpos = solve_arm_wrist_palm_ik(
                runtime=runtime,
                target_sites_world=wrist_pose_to_target_sites(wrist_pose),
                initial_arm_qpos=arm_qpos,
                hand_qpos=hand_qpos,
                iterations=config.conversion.arm_ik_iterations,
                damping=config.conversion.arm_ik_damping,
            )
            runtime.set_robot_actuated_qpos(np.concatenate([arm_qpos, hand_qpos]))
            viewer.sync()
            time.sleep(1.0 / 60.0)

        final_actuated = np.concatenate([arm_qpos, target_hand_qpos])
        runtime.settle_actuated_pose(final_actuated, PROJECTION_SETTLE_STEPS)
        for _ in range(max(1, hold_steps)):
            viewer.sync()
            time.sleep(1.0 / 60.0)


def play_sample_dynamic(
    runtime: RobotSceneModel,
    config: TaskConfig,
    sample: PoseDrivenSample,
    animate_steps: int,
    hold_steps: int,
    sim_substeps: int,
) -> None:
    runtime.reset()
    runtime.set_object_pose(np.asarray(sample.object_pose_goal, dtype=np.float64))

    start_actuated = runtime.get_actuated_qpos()
    start_wrist_pose = wrist_pose_from_semantic_sites(runtime.get_semantic_sites_world())
    target_wrist_pose = np.asarray(sample.wrist_pose_goal_world, dtype=np.float64)
    target_hand_qpos = np.asarray(sample.hand_qpos_6, dtype=np.float64)
    ctrl_target = start_actuated.copy()
    arm_qpos = start_actuated[:6].copy()

    with mujoco_viewer.launch_passive(runtime.model, runtime.data) as viewer:
        configure_camera(viewer.cam)
        for step in range(max(1, animate_steps)):
            alpha = float(step + 1) / float(max(1, animate_steps))
            wrist_pose = interpolate_pose(start_wrist_pose, target_wrist_pose, alpha)
            hand_qpos = (1.0 - alpha) * start_actuated[6:] + alpha * target_hand_qpos
            arm_qpos = solve_arm_wrist_palm_ik(
                runtime=runtime,
                target_sites_world=wrist_pose_to_target_sites(wrist_pose),
                initial_arm_qpos=arm_qpos,
                hand_qpos=hand_qpos,
                iterations=config.conversion.arm_ik_iterations,
                damping=config.conversion.arm_ik_damping,
            )
            ctrl_target = runtime.clamp_actuated(np.concatenate([arm_qpos, hand_qpos]))
            runtime.step(ctrl_target, max(1, sim_substeps))
            viewer.sync()
            time.sleep(1.0 / 60.0)

        final_ctrl = runtime.clamp_actuated(np.concatenate([arm_qpos, target_hand_qpos]))
        for _ in range(max(1, hold_steps)):
            runtime.step(final_ctrl, max(1, sim_substeps))
            viewer.sync()
            time.sleep(1.0 / 60.0)


def main() -> None:
    configure_local_runtime_env()
    args = build_arg_parser().parse_args()
    config = TaskConfig.from_json(args.config)
    if args.force_prepare:
        samples = prepare_pose_driven_samples(config, force_rebuild=True)
    else:
        samples_path = pose_driven_samples_path(config)
        samples = load_pose_driven_samples(samples_path) if samples_path.exists() else prepare_pose_driven_samples(config)
    samples_path = pose_driven_samples_path(config)
    sample_index, sample = select_sample(samples, args)
    print_sample_summary(sample_index, sample, samples_path)

    scene_xml, metadata_path = build_training_scene(config)
    runtime = RobotSceneModel(config, scene_xml=scene_xml, metadata_path=metadata_path)

    if args.preview:
        set_final_target_pose(runtime, config, sample)
        output = render_preview(runtime.model, runtime.data, args.width, args.height, args.output)
        print(f"Preview written to: {output}")
        return

    if args.kinematic:
        play_sample_kinematic(runtime, config, sample, args.animate_steps, args.hold_steps)
        return

    play_sample_dynamic(runtime, config, sample, args.animate_steps, args.hold_steps, args.sim_substeps)


if __name__ == "__main__":
    main()
