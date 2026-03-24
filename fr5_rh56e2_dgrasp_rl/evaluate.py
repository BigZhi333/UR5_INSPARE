from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import FR5LowLevelGraspEnv
from .ppo import ActorCritic
from .task_config import TaskConfig
from .train_loop import resolve_torch_device


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate FR5+RH56E2 grasp policy.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "default_task.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--run-name", type=str, default="eval")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    return parser


def evaluate_main(args: argparse.Namespace | None = None) -> Path:
    parsed = build_arg_parser().parse_args() if args is None else args

    config = TaskConfig.from_json(parsed.config)
    device = resolve_torch_device(parsed.device)
    env = FR5LowLevelGraspEnv(
        config=config,
        drop_table_after_pregrasp=config.eval.drop_table_after_pregrasp,
        freeze_control_after_pregrasp=config.eval.freeze_control_after_pregrasp,
    )
    checkpoint = torch.load(parsed.checkpoint, map_location=device)
    model = ActorCritic(checkpoint["obs_dim"], checkpoint["act_dim"], config.ppo.hidden_sizes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    num_episodes = parsed.num_episodes or config.eval.num_episodes
    trajectories = []
    summary_rows = []
    success_values = []
    displacement_values = []

    for episode_idx in range(num_episodes):
        goal_index = episode_idx % len(env.goals)
        obs, reset_info = env.reset(seed=parsed.seed + episode_idx, options={"goal_index": goal_index})
        episode_traj = [env.get_qpos()]
        done = False
        info = dict(reset_info)
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = model.step(obs_tensor, deterministic=True)[0].squeeze(0).cpu().numpy()
            obs, _, terminated, truncated, info = env.step(action)
            episode_traj.append(env.get_qpos())
            done = bool(terminated or truncated)

        success = float(info.get("episode_success", 0.0))
        displacement_mean = float(info.get("displacement_mean", 0.0))
        success_values.append(success)
        displacement_values.append(displacement_mean)
        trajectories.append(np.asarray(episode_traj, dtype=np.float64))
        summary_rows.append(
            {
                "episode_idx": episode_idx,
                "goal_index": goal_index,
                "label_idx": int(info.get("label_idx", -1)),
                "success": success,
                "displacement_mean": displacement_mean,
                "displacement_std": float(info.get("displacement_std", 0.0)),
            }
        )

    eval_dir = config.project_dir / "evals" / parsed.run_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(parsed.checkpoint),
        "device": str(device),
        "success_rate": float(np.mean(success_values)) if success_values else 0.0,
        "displacement_mean": float(np.mean(displacement_values)) if displacement_values else 0.0,
        "displacement_std": float(np.std(displacement_values)) if displacement_values else 0.0,
        "episodes": summary_rows,
    }
    (eval_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if config.eval.save_trajectories:
        max_len = max(traj.shape[0] for traj in trajectories)
        nq = trajectories[0].shape[1]
        padded = np.zeros((len(trajectories), max_len, nq), dtype=np.float64)
        lengths = np.zeros(len(trajectories), dtype=np.int32)
        for idx, traj in enumerate(trajectories):
            padded[idx, : traj.shape[0]] = traj
            lengths[idx] = traj.shape[0]
        np.savez(eval_dir / "trajectories.npz", qpos=padded, lengths=lengths)

    env.close()
    return eval_dir
