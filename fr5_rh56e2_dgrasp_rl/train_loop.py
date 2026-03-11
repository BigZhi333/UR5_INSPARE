from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv
from torch.utils.tensorboard import SummaryWriter

from .env import make_env
from .paths import configure_local_runtime_env, ensure_runtime_dirs
from .pose_driven_data import prepare_pose_driven_samples
from .ppo import ActorCritic, PPOTrainer
from .ppo.storage import RolloutBuffer
from .task_config import TaskConfig


def _info_array(infos: dict, key: str, num_envs: int) -> np.ndarray:
    if key in infos:
        return np.asarray(infos[key], dtype=np.float64).reshape(num_envs)
    return np.zeros(num_envs, dtype=np.float64)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FR5+RH56E2 low-level grasp PPO with pose-driven labels.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "default_task.json")
    parser.add_argument("--run-name", type=str, default="sugar_box_ppo")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--force-convert", action="store_true")
    return parser


def resolve_torch_device(device_arg: str) -> torch.device:
    normalized = device_arg.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch build does not have CUDA available.")
    return torch.device(normalized)


def train_main(args: argparse.Namespace | None = None) -> Path:
    configure_local_runtime_env()
    ensure_runtime_dirs()
    parsed = build_arg_parser().parse_args() if args is None else args
    config = TaskConfig.from_json(parsed.config)
    prepare_pose_driven_samples(config, force_rebuild=parsed.force_convert)

    num_envs = parsed.num_envs or config.num_envs
    run_dir = config.project_dir / "checkpoints" / parsed.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(config.project_dir / "logs" / parsed.run_name))

    vector_env = AsyncVectorEnv(
        [make_env(config, drop_table_after_pregrasp=True) for _ in range(num_envs)]
    )
    obs, _ = vector_env.reset(seed=parsed.seed)

    obs_dim = obs.shape[-1]
    act_dim = int(vector_env.single_action_space.shape[0])
    device = resolve_torch_device(parsed.device)
    actor_critic = ActorCritic(obs_dim, act_dim, config.ppo.hidden_sizes).to(device)
    trainer = PPOTrainer(
        actor_critic=actor_critic,
        learning_rate=config.ppo.learning_rate,
        clip_ratio=config.ppo.clip_ratio,
        value_coef=config.ppo.value_coef,
        entropy_coef=config.ppo.entropy_coef,
        max_grad_norm=config.ppo.max_grad_norm,
        desired_kl=config.ppo.desired_kl,
    )
    updates = parsed.updates or config.ppo.train_updates

    for update in range(updates):
        buffer = RolloutBuffer(config.total_episode_steps, num_envs, obs_dim, act_dim, device=device)
        rollout_rewards = np.zeros(num_envs, dtype=np.float64)
        final_infos: dict = {}

        for _ in range(config.total_episode_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                actions_tensor, log_probs_tensor, values_tensor = actor_critic.step(obs_tensor, deterministic=False)
            actions = actions_tensor.cpu().numpy()
            next_obs, rewards, terminated, truncated, infos = vector_env.step(actions)
            dones = np.logical_or(terminated, truncated).astype(np.float32)
            buffer.add(
                obs=obs_tensor,
                actions=actions_tensor,
                log_probs=log_probs_tensor,
                rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device),
                dones=torch.as_tensor(dones, dtype=torch.float32, device=device),
                values=values_tensor,
            )
            rollout_rewards += rewards
            obs = next_obs
            final_infos = infos

        with torch.no_grad():
            last_values = actor_critic.value(torch.as_tensor(obs, dtype=torch.float32, device=device)).squeeze(-1)
        buffer.compute_returns_and_advantages(last_values, config.ppo.gamma, config.ppo.gae_lambda)
        stats = trainer.update(buffer, config.ppo.epochs, config.ppo.minibatches)

        success = _info_array(final_infos, "episode_success", num_envs).mean()
        displacement_mean = _info_array(final_infos, "displacement_mean", num_envs).mean()
        writer.add_scalar("train/rollout_reward", float(rollout_rewards.mean()), update)
        writer.add_scalar("train/success_rate", float(success), update)
        writer.add_scalar("train/displacement_mean", float(displacement_mean), update)
        writer.add_scalar("ppo/policy_loss", stats.policy_loss, update)
        writer.add_scalar("ppo/value_loss", stats.value_loss, update)
        writer.add_scalar("ppo/entropy", stats.entropy, update)
        writer.add_scalar("ppo/approx_kl", stats.approx_kl, update)

        if update % config.ppo.save_every == 0 or update == updates - 1:
            checkpoint_path = run_dir / f"checkpoint_{update:04d}.pt"
            torch.save(
                {
                    "model_state_dict": actor_critic.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "config_path": str(parsed.config),
                    "update": update,
                    "device": str(device),
                },
                checkpoint_path,
            )

    summary_path = run_dir / "train_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "updates": updates,
                "run_name": parsed.run_name,
                "config": str(parsed.config),
                "dataset": str(config.project_dir / "data" / f"{config.object_name}_pose_driven_samples.json"),
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    vector_env.close()
    writer.close()
    return run_dir
