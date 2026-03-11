from __future__ import annotations

import argparse
import json
import time
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


def _maybe_info_array(infos: dict, key: str, num_envs: int) -> np.ndarray | None:
    if key not in infos:
        return None
    return np.asarray(infos[key], dtype=np.float64).reshape(num_envs)


def _format_metric(value: float, precision: int = 4) -> str:
    return f"{value:.{precision}f}"


def _print_update_summary(
    update: int,
    updates: int,
    total_env_steps: int,
    fps: float,
    update_seconds: float,
    rollout_reward_mean: float,
    rollout_reward_std: float,
    episode_reward_mean: float,
    episode_reward_std: float,
    success: float,
    displacement_mean: float,
    displacement_std: float,
    step_means: dict[str, float],
    stats,
) -> None:
    print("=" * 110)
    print(
        f"[UPDATE {update + 1:04d}/{updates:04d}] "
        f"env_steps={total_env_steps}  fps={fps:7.1f}  sec={update_seconds:7.2f}"
    )
    print(
        "  reward: "
        f"rollout={_format_metric(rollout_reward_mean)} +/- {_format_metric(rollout_reward_std)}  "
        f"episode={_format_metric(episode_reward_mean)} +/- {_format_metric(episode_reward_std)}"
    )
    print(
        "  task:   "
        f"success={_format_metric(success, 3)}  "
        f"disp={_format_metric(displacement_mean, 6)} +/- {_format_metric(displacement_std, 6)}  "
        f"slip={_format_metric(step_means['slipped'], 3)}  "
        f"table_contact={_format_metric(step_means['table_contact'], 3)}"
    )
    print(
        "  error:  "
        f"site_rmse={_format_metric(step_means['site_rmse_m'])}  "
        f"wrist_t={_format_metric(step_means['wrist_translation_error_m'])}  "
        f"wrist_r={_format_metric(step_means['wrist_rotation_error_deg'], 2)}  "
        f"hand={_format_metric(step_means['hand_pose_error'])}"
    )
    print(
        "  contact:"
        f" match={_format_metric(step_means['contact_bit_match'], 3)}  "
        f"target_hit={_format_metric(step_means['matched_positive_contacts'], 3)}  "
        f"false_pos={_format_metric(step_means['false_positive_contacts'], 3)}  "
        f"impulse={_format_metric(step_means['impulse_term'], 3)}"
    )
    print(
        "  reward terms(raw): "
        f"site={_format_metric(step_means['term_site'])}  "
        f"pose={_format_metric(step_means['term_pose'])}  "
        f"w_pose={_format_metric(step_means['term_wrist_pose'])}  "
        f"w_align={_format_metric(step_means['term_wrist_align'])}  "
        f"contact={_format_metric(step_means['contact_term'])}  "
        f"penetration={_format_metric(step_means['term_penetration'])}"
    )
    print(
        "  reward contrib:    "
        f"site={_format_metric(step_means['reward_site'])}  "
        f"pose={_format_metric(step_means['reward_pose'])}  "
        f"w_pose={_format_metric(step_means['reward_wrist_pose'])}  "
        f"w_align={_format_metric(step_means['reward_wrist_align'])}  "
        f"contact={_format_metric(step_means['reward_contact'])}  "
        f"impulse={_format_metric(step_means['reward_impulse'])}"
    )
    print(
        "  reward contrib:    "
        f"falling={_format_metric(step_means['reward_falling'])}  "
        f"rel_obj={_format_metric(step_means['reward_rel_obj'])}  "
        f"body_vel={_format_metric(step_means['reward_body_vel'])}  "
        f"body_qvel={_format_metric(step_means['reward_body_qvel'])}  "
        f"penetration={_format_metric(step_means['reward_penetration'])}  "
        f"act_rate={_format_metric(step_means['reward_action_rate'])}"
    )
    print(
        "  reward sum: "
        f"preclip={_format_metric(step_means['reward_preclip'])}  "
        f"total={_format_metric(step_means['reward_total'])}  "
        f"clip_delta={_format_metric(step_means['reward_clip_delta'])}"
    )
    print(
        "  ppo:    "
        f"policy_loss={_format_metric(stats.policy_loss)}  "
        f"value_loss={_format_metric(stats.value_loss)}  "
        f"entropy={_format_metric(stats.entropy)}  "
        f"approx_kl={_format_metric(stats.approx_kl)}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FR5+RH56E2 low-level grasp PPO with pose-driven labels.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "default_task.json")
    parser.add_argument("--run-name", type=str, default="sugar_box_ppo")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--log-every", type=int, default=1, help="Print one training summary every N PPO updates.")
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
    log_dir = config.project_dir / "logs" / parsed.run_name
    writer = SummaryWriter(log_dir=str(log_dir))

    vector_env = AsyncVectorEnv(
        # Match the original D-Grasp training protocol:
        # training rollouts keep the table in place, while evaluation performs the drop test.
        [make_env(config, drop_table_after_pregrasp=False) for _ in range(num_envs)]
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
    rollout_info_keys = [
        "wrist_translation_error_m",
        "wrist_rotation_error_deg",
        "site_rmse_m",
        "hand_pose_error",
        "term_site",
        "term_pose",
        "term_wrist_pose",
        "term_wrist_align",
        "contact_term",
        "contact_bit_match",
        "matched_positive_contacts",
        "false_positive_contacts",
        "impulse_term",
        "term_falling",
        "term_rel_obj",
        "term_body_vel",
        "term_body_qvel",
        "term_penetration",
        "term_action_rate",
        "reward_site",
        "reward_pose",
        "reward_wrist_pose",
        "reward_wrist_align",
        "reward_contact",
        "reward_impulse",
        "reward_falling",
        "reward_rel_obj",
        "reward_body_vel",
        "reward_body_qvel",
        "reward_penetration",
        "reward_action_rate",
        "reward_preclip",
        "reward_total",
        "reward_clip_delta",
        "penetration_m",
        "max_penetration_m",
        "falling_term",
        "object_height_m",
        "table_contact",
        "slipped",
        "table_dropped",
        "step_displacement",
    ]
    total_env_steps = 0

    print(f"[train] run_name={parsed.run_name}")
    print(f"[train] log_dir={log_dir}")
    print(f"[train] checkpoints={run_dir}")
    print(
        "[train] tensorboard: "
        f"tensorboard --logdir \"{config.project_dir / 'logs'}\" --port 6006"
    )
    print(
        "[train] config: "
        f"updates={updates} num_envs={num_envs} episode_steps={config.total_episode_steps} "
        f"device={device} drop_table_train=False drop_table_eval={config.eval.drop_table_after_pregrasp}"
    )

    for update in range(updates):
        update_start_time = time.perf_counter()
        buffer = RolloutBuffer(config.total_episode_steps, num_envs, obs_dim, act_dim, device=device)
        rollout_rewards = np.zeros(num_envs, dtype=np.float64)
        step_info_sums = {key: 0.0 for key in rollout_info_keys}
        step_info_counts = {key: 0 for key in rollout_info_keys}
        action_abs_sum = 0.0
        wrist_pos_abs_sum = 0.0
        wrist_rot_abs_sum = 0.0
        hand_abs_sum = 0.0
        action_samples = 0
        final_infos: dict = {}

        for _ in range(config.total_episode_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                actions_tensor, log_probs_tensor, values_tensor = actor_critic.step(obs_tensor, deterministic=False)
            actions = actions_tensor.cpu().numpy()
            action_abs_sum += float(np.abs(actions).mean())
            wrist_pos_abs_sum += float(np.abs(actions[:, :3]).mean())
            wrist_rot_abs_sum += float(np.abs(actions[:, 3:6]).mean())
            hand_abs_sum += float(np.abs(actions[:, 6:]).mean())
            action_samples += 1
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
            total_env_steps += num_envs
            for key in rollout_info_keys:
                values = _maybe_info_array(infos, key, num_envs)
                if values is None:
                    continue
                step_info_sums[key] += float(values.mean())
                step_info_counts[key] += 1
            obs = next_obs
            final_infos = infos

        with torch.no_grad():
            last_values = actor_critic.value(torch.as_tensor(obs, dtype=torch.float32, device=device)).squeeze(-1)
        buffer.compute_returns_and_advantages(last_values, config.ppo.gamma, config.ppo.gae_lambda)
        stats = trainer.update(buffer, config.ppo.epochs, config.ppo.minibatches)

        update_seconds = time.perf_counter() - update_start_time
        fps = float(config.total_episode_steps * num_envs / max(update_seconds, 1e-8))
        success = float(_info_array(final_infos, "episode_success", num_envs).mean())
        displacement_mean = float(_info_array(final_infos, "displacement_mean", num_envs).mean())
        displacement_std = float(_info_array(final_infos, "displacement_std", num_envs).mean())
        episode_reward_mean = float(_info_array(final_infos, "episode_reward", num_envs).mean())
        episode_reward_std = float(_info_array(final_infos, "episode_reward", num_envs).std())
        rollout_reward_mean = float(rollout_rewards.mean())
        rollout_reward_std = float(rollout_rewards.std())
        step_means = {
            key: step_info_sums[key] / max(step_info_counts[key], 1)
            for key in rollout_info_keys
        }
        action_abs_mean = action_abs_sum / max(action_samples, 1)
        wrist_pos_abs_mean = wrist_pos_abs_sum / max(action_samples, 1)
        wrist_rot_abs_mean = wrist_rot_abs_sum / max(action_samples, 1)
        hand_abs_mean = hand_abs_sum / max(action_samples, 1)

        writer.add_scalar("train/rollout_reward", rollout_reward_mean, update)
        writer.add_scalar("train/rollout_reward_mean", rollout_reward_mean, update)
        writer.add_scalar("train/rollout_reward_std", rollout_reward_std, update)
        writer.add_scalar("train/episode_reward_mean", episode_reward_mean, update)
        writer.add_scalar("train/episode_reward_std", episode_reward_std, update)
        writer.add_scalar("train/success_rate", success, update)
        writer.add_scalar("train/displacement_mean", displacement_mean, update)
        writer.add_scalar("train/displacement_std", displacement_std, update)
        writer.add_scalar("train/total_env_steps", float(total_env_steps), update)
        writer.add_scalar("ppo/policy_loss", stats.policy_loss, update)
        writer.add_scalar("ppo/value_loss", stats.value_loss, update)
        writer.add_scalar("ppo/entropy", stats.entropy, update)
        writer.add_scalar("ppo/approx_kl", stats.approx_kl, update)
        writer.add_scalar("perf/update_seconds", update_seconds, update)
        writer.add_scalar("perf/fps_env_steps", fps, update)
        writer.add_scalar("policy/action_abs_mean", action_abs_mean, update)
        writer.add_scalar("policy/wrist_pos_action_abs_mean", wrist_pos_abs_mean, update)
        writer.add_scalar("policy/wrist_rot_action_abs_mean", wrist_rot_abs_mean, update)
        writer.add_scalar("policy/hand_action_abs_mean", hand_abs_mean, update)

        for key, value in step_means.items():
            writer.add_scalar(f"rollout/{key}", value, update)

        progress_payload = {
            "update": update,
            "updates_total": updates,
            "run_name": parsed.run_name,
            "device": str(device),
            "num_envs": num_envs,
            "episode_steps": config.total_episode_steps,
            "total_env_steps": total_env_steps,
            "rollout_reward_mean": rollout_reward_mean,
            "rollout_reward_std": rollout_reward_std,
            "episode_reward_mean": episode_reward_mean,
            "episode_reward_std": episode_reward_std,
            "success_rate": success,
            "displacement_mean": displacement_mean,
            "displacement_std": displacement_std,
            "fps_env_steps": fps,
            "update_seconds": update_seconds,
            "policy_loss": stats.policy_loss,
            "value_loss": stats.value_loss,
            "entropy": stats.entropy,
            "approx_kl": stats.approx_kl,
            "action_abs_mean": action_abs_mean,
            "wrist_pos_action_abs_mean": wrist_pos_abs_mean,
            "wrist_rot_action_abs_mean": wrist_rot_abs_mean,
            "hand_action_abs_mean": hand_abs_mean,
            "rollout_metrics": step_means,
        }
        (log_dir / "latest_progress.json").write_text(
            json.dumps(progress_payload, indent=2),
            encoding="utf-8",
        )
        writer.flush()

        if update % parsed.log_every == 0 or update == updates - 1:
            _print_update_summary(
                update=update,
                updates=updates,
                total_env_steps=total_env_steps,
                fps=fps,
                update_seconds=update_seconds,
                rollout_reward_mean=rollout_reward_mean,
                rollout_reward_std=rollout_reward_std,
                episode_reward_mean=episode_reward_mean,
                episode_reward_std=episode_reward_std,
                success=success,
                displacement_mean=displacement_mean,
                displacement_std=displacement_std,
                step_means=step_means,
                stats=stats,
            )

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
            print(f"[checkpoint] saved {checkpoint_path}")
            print("-" * 110)

    summary_path = run_dir / "train_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "updates": updates,
                "run_name": parsed.run_name,
                "config": str(parsed.config),
                "dataset": str(config.project_dir / "data" / f"{config.object_name}_pose_driven_samples.json"),
                "device": str(device),
                "log_dir": str(log_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    vector_env.close()
    writer.close()
    return run_dir
