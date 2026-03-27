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
    PROJECTION_HOLD_ARM_KP_SCALE,
    PROJECTION_HOLD_PRELOAD_STEPS,
    PROJECTION_HOLD_SUPPORT_FORCE_SCALE,
    PROJECTION_SETTLE_STEPS,
    PoseDrivenSample,
    load_pose_driven_samples,
    pose_driven_samples_path,
    prepare_pose_driven_samples,
    wrist_pose_from_semantic_sites,
)
from fr5_rh56e2_dgrasp_rl.robot_model import RobotSceneModel
from fr5_rh56e2_dgrasp_rl.scene_builder import build_training_scene
from fr5_rh56e2_dgrasp_rl.semantics import SEMANTIC_CONTACT_NAMES
from fr5_rh56e2_dgrasp_rl.task_config import TaskConfig
from fr5_rh56e2_dgrasp_rl.utils import normalize

VIEWER_DEFAULT_HOLD_ARM_KP_SCALE = max(4.0, PROJECTION_HOLD_ARM_KP_SCALE)
VIEWER_DEFAULT_HOLD_HAND_KP_SCALE = 2.5


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View one pose-driven D-Grasp sample on FR5+RH56E2.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config" / "default_task.json")
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--label-idx", type=int, default=None)
    parser.add_argument("--all", action="store_true")
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
    parser.add_argument("--pause-steps", type=int, default=30)
    parser.add_argument("--sim-substeps", type=int, default=4)
    parser.add_argument("--kinematic", action="store_true")
    parser.add_argument("--physics-approach", action="store_true")
    parser.add_argument("--hold-arm-kp-scale", type=float, default=VIEWER_DEFAULT_HOLD_ARM_KP_SCALE)
    parser.add_argument("--hold-hand-kp-scale", type=float, default=VIEWER_DEFAULT_HOLD_HAND_KP_SCALE)
    parser.add_argument("--hold-support-scale", type=float, default=PROJECTION_HOLD_SUPPORT_FORCE_SCALE)
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
            hold_tested = bool(fit.get("projected_hold_tested", True))
            if not hold_tested:
                return (
                    float(not sample.valid_execution),
                    -float(fit.get("projected_contact_group_count", 0.0)),
                    -float(fit.get("projected_hard_contact_group_count", 0.0)),
                    -float(fit.get("projected_has_thumb_opposition", 0.0)),
                    float(fit.get("projected_anchor_rmse_m", 999.0)),
                    float(fit.get("projected_max_penetration_m", 999.0)),
                    -float(fit.get("projected_matched_target_contacts", 0.0)),
                    float(fit.get("source_tip_rmse_m", fit.get("tip_rmse_m", 999.0))),
                    float(fit.get("projected_reach_base_facing_cos", 999.0)),
                    float(-fit.get("projected_reach_object_facing_cos", -999.0)),
                    float(-(num_contacts > 0.0)),
                    -num_contacts,
                    -float(sample.object_pose_goal[2]),
                )
            return (
                float(not sample.valid_execution),
                -float(fit.get("projected_hold_hybrid_contact_group_count", 0.0)),
                -float(fit.get("projected_hold_hard_contact_group_count", 0.0)),
                -float(fit.get("projected_hold_has_thumb_opposition", 0.0)),
                float(fit.get("projected_hold_object_drop_m", 999.0)),
                float(fit.get("projected_anchor_rmse_m", 999.0)),
                float(fit.get("projected_hold_object_translation_m", 999.0)),
                float(fit.get("projected_hold_object_rotation_deg", 999.0)),
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


def select_samples(samples: list[PoseDrivenSample], args: argparse.Namespace) -> list[tuple[int, PoseDrivenSample]]:
    if args.all:
        return list(enumerate(samples))
    sample_index, sample = select_sample(samples, args)
    return [(sample_index, sample)]


def print_sample_summary(sample_index: int, sample: PoseDrivenSample, samples_path: Path) -> None:
    fit = sample.fit_error
    print(f"Samples file: {samples_path}")
    print(f"Sample index: {sample_index}")
    print(f"Label idx: {sample.label_idx}")
    print(f"Execution valid: {sample.valid_execution}")
    method_flag = fit.get("retarget_method", "genhand_direct")
    if isinstance(method_flag, str):
        method_name = method_flag
    else:
        method_name = "genhand_direct" if float(method_flag) >= 0.5 else "legacy_projection"
    print(f"Retarget method: {method_name}")
    print(
        "Projection stats: "
        f"source_tip_rmse={fit.get('source_tip_rmse_m', fit.get('tip_rmse_m', 0.0)):.4f} m, "
        f"anchor_rmse={fit.get('projected_anchor_rmse_m', 0.0):.4f} m, "
        f"assign_cost={fit.get('projected_anchor_assignment_norm_cost', 0.0):.4f}, "
        f"max_penetration={fit.get('projected_max_penetration_m', 0.0):.4f} m, "
        f"contact_hamming={fit.get('source_contact_hamming', 0.0):.0f}, "
        f"retreat={fit.get('projected_retreat_m', 0.0):.4f} m"
    )
    print(
        "GenHand stats: "
        f"target_anchor_rmse={fit.get('genhand_target_anchor_rmse_m', 0.0):.4f} m, "
        f"cluster_score={fit.get('genhand_cluster_score_mean', 0.0):.4f}, "
        f"fc_loss={fit.get('genhand_fc_loss', 0.0):.4f}, "
        f"fc_wrench={fit.get('genhand_fc_net_wrench', 0.0):.4f}, "
        f"teacher_cost={fit.get('teacher_cost', 0.0):.4f}"
    )
    print(
        "Hold stats: "
        + (
            "disabled"
            if not bool(fit.get("projected_hold_tested", True))
            else
            (
                f"drop={fit.get('projected_hold_object_drop_m', 0.0):.4f} m, "
                f"groups(hybrid/hard)={fit.get('projected_hold_hybrid_contact_group_count', 0.0):.1f}/"
                f"{fit.get('projected_hold_hard_contact_group_count', 0.0):.1f}, "
                f"thumb_opposition={int(round(fit.get('projected_hold_has_thumb_opposition', 0.0)))}"
            )
        )
    )
    if "projected_cylinder_palm_clearance_error_m" in fit:
        print(
            "Cylinder metrics: "
            f"settled_clear={fit.get('projected_cylinder_palm_clearance_error_m', 0.0):.4f} m, "
            f"settled_opp={fit.get('projected_cylinder_opposition_cos', 0.0):.3f}, "
            f"hold_clear={fit.get('projected_hold_cylinder_palm_clearance_error_m', 0.0):.4f} m, "
            f"hold_opp={fit.get('projected_hold_cylinder_opposition_cos', 0.0):.3f}"
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
    if hasattr(sample, "physical_contact_mask_12"):
        print(
            "Projected hard contact mask 12: "
            + ", ".join(
                f"{name}={int(round(value))}"
                for name, value in zip(SEMANTIC_CONTACT_NAMES, sample.physical_contact_mask_12)
            )
        )
    if hasattr(sample, "proximity_contact_mask_12"):
        print(
            "Projected proximity mask 12: "
            + ", ".join(
                f"{name}={int(round(value))}"
                for name, value in zip(SEMANTIC_CONTACT_NAMES, sample.proximity_contact_mask_12)
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
        target_wrist_pose_world=np.asarray(sample.wrist_pose_goal_world, dtype=np.float64),
        initial_arm_qpos=runtime.get_actuated_qpos()[:6],
        hand_qpos=np.asarray(sample.hand_qpos_6, dtype=np.float64),
        iterations=config.conversion.arm_ik_iterations,
        damping=config.conversion.arm_ik_damping,
    )


def set_final_target_pose(runtime: RobotSceneModel, config: TaskConfig, sample: PoseDrivenSample) -> None:
    runtime.reset()
    runtime.set_object_pose(np.asarray(sample.object_pose_goal, dtype=np.float64))
    arm_qpos = solve_final_arm_qpos(runtime, config, sample)
    runtime.set_robot_actuated_qpos(
        np.concatenate([arm_qpos, np.asarray(sample.hand_qpos_6, dtype=np.float64)]),
        update_ctrl=True,
    )


def _zero_runtime_velocities(runtime: RobotSceneModel) -> None:
    runtime.data.qvel[:] = 0.0
    runtime.data.qfrc_applied[:] = 0.0
    if hasattr(runtime.data, "qacc_warmstart"):
        runtime.data.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(runtime.model, runtime.data)


def _run_hold_validation_preview(
    runtime: RobotSceneModel,
    config: TaskConfig,
    final_ctrl: np.ndarray,
    hold_steps: int,
    sim_substeps: int,
    arm_kp_scale: float,
    hand_kp_scale: float,
    support_scale: float,
    viewer: mujoco_viewer.Handle,
) -> None:
    final_ctrl = runtime.clamp_actuated(np.asarray(final_ctrl, dtype=np.float64))
    runtime.set_robot_actuated_qpos_kinematic(final_ctrl, update_ctrl=True)
    _zero_runtime_velocities(runtime)
    runtime.set_arm_hold_mode(True, arm_kp_scale=arm_kp_scale, hand_kp_scale=hand_kp_scale)
    support_force_world = np.array(
        [0.0, 0.0, float(config.object_mass_kg) * float(support_scale)],
        dtype=np.float64,
    )
    preload_steps = max(PROJECTION_HOLD_PRELOAD_STEPS, 1)
    try:
        for _ in range(preload_steps):
            runtime.set_robot_actuated_qpos_kinematic(final_ctrl, update_ctrl=True)
            runtime.step(final_ctrl, max(1, sim_substeps), arm_support_force_world=support_force_world)
            runtime.set_robot_actuated_qpos_kinematic(final_ctrl, update_ctrl=True)
            viewer.sync()
            time.sleep(1.0 / 60.0)
        runtime.set_robot_actuated_qpos_kinematic(final_ctrl, update_ctrl=True)
        runtime.set_table_height(0.0)
        for _ in range(max(1, hold_steps)):
            runtime.set_robot_actuated_qpos_kinematic(final_ctrl, update_ctrl=True)
            runtime.step(final_ctrl, max(1, sim_substeps), arm_support_force_world=support_force_world)
            runtime.set_robot_actuated_qpos_kinematic(final_ctrl, update_ctrl=True)
            viewer.sync()
            time.sleep(1.0 / 60.0)
    finally:
        runtime.set_arm_hold_mode(False)


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
                target_wrist_pose_world=wrist_pose,
                initial_arm_qpos=arm_qpos,
                hand_qpos=hand_qpos,
                iterations=config.conversion.arm_ik_iterations,
                damping=config.conversion.arm_ik_damping,
            )
            runtime.set_robot_actuated_qpos(np.concatenate([arm_qpos, hand_qpos]))
            viewer.sync()
            time.sleep(1.0 / 60.0)

        final_actuated = np.concatenate([arm_qpos, target_hand_qpos])
        runtime.set_robot_actuated_qpos(final_actuated, update_ctrl=True)
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
    hold_arm_kp_scale: float,
    hold_hand_kp_scale: float,
    hold_support_scale: float,
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
                target_wrist_pose_world=wrist_pose,
                initial_arm_qpos=arm_qpos,
                hand_qpos=hand_qpos,
                iterations=config.conversion.arm_ik_iterations,
                damping=config.conversion.arm_ik_damping,
            )
            ctrl_target = runtime.clamp_actuated(np.concatenate([arm_qpos, hand_qpos]))
            runtime.set_robot_actuated_qpos(ctrl_target, update_ctrl=True)
            viewer.sync()
            time.sleep(1.0 / 60.0)

        final_ctrl = runtime.clamp_actuated(np.concatenate([arm_qpos, target_hand_qpos]))
        _run_hold_validation_preview(
            runtime,
            config,
            final_ctrl,
            hold_steps,
            sim_substeps,
            hold_arm_kp_scale,
            hold_hand_kp_scale,
            hold_support_scale,
            viewer,
        )


def play_sample_physics_approach(
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
                target_wrist_pose_world=wrist_pose,
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


def play_samples_kinematic(
    runtime: RobotSceneModel,
    config: TaskConfig,
    selected_samples: list[tuple[int, PoseDrivenSample]],
    samples_path: Path,
    animate_steps: int,
    hold_steps: int,
    pause_steps: int,
) -> None:
    with mujoco_viewer.launch_passive(runtime.model, runtime.data) as viewer:
        configure_camera(viewer.cam)
        for sample_index, sample in selected_samples:
            print_sample_summary(sample_index, sample, samples_path)
            runtime.reset()
            runtime.set_object_pose(np.asarray(sample.object_pose_goal, dtype=np.float64))

            start_actuated = runtime.get_actuated_qpos()
            start_wrist_pose = wrist_pose_from_semantic_sites(runtime.get_semantic_sites_world())
            target_wrist_pose = np.asarray(sample.wrist_pose_goal_world, dtype=np.float64)
            target_hand_qpos = np.asarray(sample.hand_qpos_6, dtype=np.float64)
            arm_qpos = start_actuated[:6].copy()

            for step in range(max(1, animate_steps)):
                alpha = float(step + 1) / float(max(1, animate_steps))
                wrist_pose = interpolate_pose(start_wrist_pose, target_wrist_pose, alpha)
                hand_qpos = (1.0 - alpha) * start_actuated[6:] + alpha * target_hand_qpos
                arm_qpos = solve_arm_wrist_palm_ik(
                    runtime=runtime,
                    target_wrist_pose_world=wrist_pose,
                    initial_arm_qpos=arm_qpos,
                    hand_qpos=hand_qpos,
                    iterations=config.conversion.arm_ik_iterations,
                    damping=config.conversion.arm_ik_damping,
                )
                runtime.set_robot_actuated_qpos(np.concatenate([arm_qpos, hand_qpos]))
                viewer.sync()
                time.sleep(1.0 / 60.0)

            final_actuated = np.concatenate([arm_qpos, target_hand_qpos])
            runtime.set_robot_actuated_qpos(final_actuated, update_ctrl=True)
            for _ in range(max(1, hold_steps)):
                viewer.sync()
                time.sleep(1.0 / 60.0)
            for _ in range(max(0, pause_steps)):
                viewer.sync()
                time.sleep(1.0 / 60.0)


def play_samples_dynamic(
    runtime: RobotSceneModel,
    config: TaskConfig,
    selected_samples: list[tuple[int, PoseDrivenSample]],
    samples_path: Path,
    animate_steps: int,
    hold_steps: int,
    pause_steps: int,
    sim_substeps: int,
    hold_arm_kp_scale: float,
    hold_hand_kp_scale: float,
    hold_support_scale: float,
) -> None:
    with mujoco_viewer.launch_passive(runtime.model, runtime.data) as viewer:
        configure_camera(viewer.cam)
        for sample_index, sample in selected_samples:
            print_sample_summary(sample_index, sample, samples_path)
            runtime.reset()
            runtime.set_object_pose(np.asarray(sample.object_pose_goal, dtype=np.float64))

            start_actuated = runtime.get_actuated_qpos()
            start_wrist_pose = wrist_pose_from_semantic_sites(runtime.get_semantic_sites_world())
            target_wrist_pose = np.asarray(sample.wrist_pose_goal_world, dtype=np.float64)
            target_hand_qpos = np.asarray(sample.hand_qpos_6, dtype=np.float64)
            ctrl_target = start_actuated.copy()
            arm_qpos = start_actuated[:6].copy()

            for step in range(max(1, animate_steps)):
                alpha = float(step + 1) / float(max(1, animate_steps))
                wrist_pose = interpolate_pose(start_wrist_pose, target_wrist_pose, alpha)
                hand_qpos = (1.0 - alpha) * start_actuated[6:] + alpha * target_hand_qpos
                arm_qpos = solve_arm_wrist_palm_ik(
                    runtime=runtime,
                    target_wrist_pose_world=wrist_pose,
                    initial_arm_qpos=arm_qpos,
                    hand_qpos=hand_qpos,
                    iterations=config.conversion.arm_ik_iterations,
                    damping=config.conversion.arm_ik_damping,
                )
                ctrl_target = runtime.clamp_actuated(np.concatenate([arm_qpos, hand_qpos]))
                runtime.set_robot_actuated_qpos(ctrl_target, update_ctrl=True)
                viewer.sync()
                time.sleep(1.0 / 60.0)

            final_ctrl = runtime.clamp_actuated(np.concatenate([arm_qpos, target_hand_qpos]))
            _run_hold_validation_preview(
                runtime,
                config,
                final_ctrl,
                hold_steps,
                sim_substeps,
                hold_arm_kp_scale,
                hold_hand_kp_scale,
                hold_support_scale,
                viewer,
            )
            for _ in range(max(0, pause_steps)):
                runtime.set_robot_actuated_qpos(final_ctrl, update_ctrl=True)
                viewer.sync()
                time.sleep(1.0 / 60.0)


def play_samples_physics_approach(
    runtime: RobotSceneModel,
    config: TaskConfig,
    selected_samples: list[tuple[int, PoseDrivenSample]],
    samples_path: Path,
    animate_steps: int,
    hold_steps: int,
    pause_steps: int,
    sim_substeps: int,
) -> None:
    with mujoco_viewer.launch_passive(runtime.model, runtime.data) as viewer:
        configure_camera(viewer.cam)
        for sample_index, sample in selected_samples:
            print_sample_summary(sample_index, sample, samples_path)
            runtime.reset()
            runtime.set_object_pose(np.asarray(sample.object_pose_goal, dtype=np.float64))

            start_actuated = runtime.get_actuated_qpos()
            start_wrist_pose = wrist_pose_from_semantic_sites(runtime.get_semantic_sites_world())
            target_wrist_pose = np.asarray(sample.wrist_pose_goal_world, dtype=np.float64)
            target_hand_qpos = np.asarray(sample.hand_qpos_6, dtype=np.float64)
            ctrl_target = start_actuated.copy()
            arm_qpos = start_actuated[:6].copy()

            for step in range(max(1, animate_steps)):
                alpha = float(step + 1) / float(max(1, animate_steps))
                wrist_pose = interpolate_pose(start_wrist_pose, target_wrist_pose, alpha)
                hand_qpos = (1.0 - alpha) * start_actuated[6:] + alpha * target_hand_qpos
                arm_qpos = solve_arm_wrist_palm_ik(
                    runtime=runtime,
                    target_wrist_pose_world=wrist_pose,
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
            for _ in range(max(0, pause_steps)):
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
    selected_samples = select_samples(samples, args)
    sample_index, sample = selected_samples[0]
    if not args.all:
        print_sample_summary(sample_index, sample, samples_path)

    scene_xml, metadata_path = build_training_scene(config)
    runtime = RobotSceneModel(config, scene_xml=scene_xml, metadata_path=metadata_path)

    if args.preview:
        if args.all:
            raise ValueError("--preview does not support --all. Please preview one sample at a time.")
        set_final_target_pose(runtime, config, sample)
        output = render_preview(runtime.model, runtime.data, args.width, args.height, args.output)
        print(f"Preview written to: {output}")
        return

    if args.all and args.kinematic:
        play_samples_kinematic(
            runtime,
            config,
            selected_samples,
            samples_path,
            args.animate_steps,
            args.hold_steps,
            args.pause_steps,
        )
        return

    if args.all and args.physics_approach:
        play_samples_physics_approach(
            runtime,
            config,
            selected_samples,
            samples_path,
            args.animate_steps,
            args.hold_steps,
            args.pause_steps,
            args.sim_substeps,
        )
        return

    if args.all:
        play_samples_dynamic(
            runtime,
            config,
            selected_samples,
            samples_path,
            args.animate_steps,
            args.hold_steps,
            args.pause_steps,
            args.sim_substeps,
            args.hold_arm_kp_scale,
            args.hold_hand_kp_scale,
            args.hold_support_scale,
        )
        return

    if args.kinematic:
        play_sample_kinematic(runtime, config, sample, args.animate_steps, args.hold_steps)
        return

    if args.physics_approach:
        play_sample_physics_approach(
            runtime,
            config,
            sample,
            args.animate_steps,
            args.hold_steps,
            args.sim_substeps,
        )
        return

    play_sample_dynamic(
        runtime,
        config,
        sample,
        args.animate_steps,
        args.hold_steps,
        args.sim_substeps,
        args.hold_arm_kp_scale,
        args.hold_hand_kp_scale,
        args.hold_support_scale,
    )


if __name__ == "__main__":
    main()
