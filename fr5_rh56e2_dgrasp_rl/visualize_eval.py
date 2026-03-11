from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer as mujoco_viewer
import numpy as np
import torch

from .env import FR5LowLevelGraspEnv
from .ppo import ActorCritic
from .task_config import TaskConfig
from .train_loop import resolve_torch_device


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize FR5+RH56E2 policy evaluation with live physics.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "default_task.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--start-episode", type=int, default=0, help="Sequential eval episode index to start from.")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--goal-index", type=int, default=None, help="Force a specific goal/sample index.")
    parser.add_argument("--realtime-scale", type=float, default=1.0, help="1.0 means sleep for control_dt after each env.step.")
    parser.add_argument("--pause-after-reset", type=float, default=0.75)
    parser.add_argument("--pause-after-episode", type=float, default=1.5)
    return parser


def configure_camera(camera: mujoco.MjvCamera) -> None:
    camera.lookat[:] = np.array([0.42, 0.02, 0.63], dtype=np.float64)
    camera.distance = 1.15
    camera.azimuth = 138.0
    camera.elevation = -22.0


def _sleep_for_control_step(control_dt: float, realtime_scale: float) -> None:
    if realtime_scale <= 0.0:
        return
    time.sleep(control_dt / realtime_scale)


def _phase_name(env: FR5LowLevelGraspEnv) -> str:
    return "pregrasp" if env.step_count < env.config.pre_grasp_steps else "hold"


def _end_reason(env: FR5LowLevelGraspEnv, done: bool) -> str:
    if not done:
        return "running"
    if env.step_count >= env.config.total_episode_steps:
        if env.slipped:
            return "time_limit_slipped"
        return "time_limit"
    if env.table_dropped and env.slipped:
        return "slipped_after_drop"
    return "terminated"


def _set_overlay(
    viewer: mujoco.viewer.Handle,
    env: FR5LowLevelGraspEnv,
    episode_idx: int,
    goal_index: int,
    label_idx: int,
    last_reward: float,
    done: bool,
    episode_reward: float | None = None,
) -> None:
    step_text = f"{env.step_count}/{env.config.total_episode_steps}"
    left = "\n".join(
        [
            "Episode",
            "Goal",
            "Label",
            "Step",
            "Phase",
            "Table Dropped",
            "Slipped",
            "Reward",
            "End Reason",
        ]
    )
    right = "\n".join(
        [
            str(episode_idx),
            str(goal_index),
            str(label_idx),
            step_text,
            _phase_name(env),
            "yes" if env.table_dropped else "no",
            "yes" if env.slipped else "no",
            f"{(env.metrics.reward_sum if done and episode_reward is None else (episode_reward if episode_reward is not None else last_reward)):.3f}"
            if done
            else f"{last_reward:.3f}",
            _end_reason(env, done),
        ]
    )
    viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right))


def visualize_eval_main(args: argparse.Namespace | None = None) -> None:
    parsed = build_arg_parser().parse_args() if args is None else args

    config = TaskConfig.from_json(parsed.config)
    device = resolve_torch_device(parsed.device)
    env = FR5LowLevelGraspEnv(config=config, drop_table_after_pregrasp=config.eval.drop_table_after_pregrasp)

    checkpoint = torch.load(parsed.checkpoint, map_location=device)
    model = ActorCritic(checkpoint["obs_dim"], checkpoint["act_dim"], config.ppo.hidden_sizes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with mujoco_viewer.launch_passive(env.runtime.model, env.runtime.data) as viewer:
        configure_camera(viewer.cam)
        viewer.sync()

        for episode_offset in range(parsed.num_episodes):
            if not viewer.is_running():
                break

            eval_episode_index = parsed.start_episode + episode_offset
            goal_index = parsed.goal_index if parsed.goal_index is not None else eval_episode_index % len(env.goals)
            obs, info = env.reset(
                seed=parsed.seed + eval_episode_index,
                options={"goal_index": int(goal_index)},
            )
            _set_overlay(
                viewer,
                env,
                episode_idx=eval_episode_index,
                goal_index=int(goal_index),
                label_idx=int(info.get("label_idx", -1)),
                last_reward=0.0,
                done=False,
            )
            viewer.sync()
            if parsed.pause_after_reset > 0.0:
                time.sleep(parsed.pause_after_reset)

            print(
                f"[eval] episode={eval_episode_index} goal_index={goal_index} "
                f"label_idx={info.get('label_idx', -1)} valid_execution={int(info.get('valid_execution', 0.0))}"
            )

            done = False
            step_count = 0
            last_info = dict(info)
            last_reward = 0.0

            while viewer.is_running() and not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = model.step(obs_tensor, deterministic=True)[0].squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, last_info = env.step(action)
                last_reward = float(reward)
                step_count += 1
                _set_overlay(
                    viewer,
                    env,
                    episode_idx=eval_episode_index,
                    goal_index=int(goal_index),
                    label_idx=int(last_info.get("label_idx", info.get("label_idx", -1))),
                    last_reward=last_reward,
                    done=bool(terminated or truncated),
                    episode_reward=float(last_info.get("episode_reward", 0.0)) if bool(terminated or truncated) else None,
                )
                viewer.sync()
                _sleep_for_control_step(config.control_dt, parsed.realtime_scale)
                done = bool(terminated or truncated)

            print(
                f"[eval] done episode={eval_episode_index} steps={step_count} "
                f"reward={last_info.get('episode_reward', last_reward):.3f} "
                f"success={last_info.get('episode_success', 0.0):.0f} "
                f"disp_mean={last_info.get('displacement_mean', 0.0):.6f} "
                f"slipped={int(last_info.get('slipped', 0.0))}"
            )

            _set_overlay(
                viewer,
                env,
                episode_idx=eval_episode_index,
                goal_index=int(goal_index),
                label_idx=int(last_info.get("label_idx", info.get("label_idx", -1))),
                last_reward=last_reward,
                done=True,
                episode_reward=float(last_info.get("episode_reward", last_reward)),
            )
            viewer.sync()
            if parsed.pause_after_episode > 0.0 and viewer.is_running() and episode_offset < parsed.num_episodes - 1:
                time.sleep(parsed.pause_after_episode)

    env.close()
