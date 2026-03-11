from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .networks import ActorCritic
from .storage import RolloutBuffer


@dataclass
class PPOStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float


class PPOTrainer:
    def __init__(
        self,
        actor_critic: ActorCritic,
        learning_rate: float,
        clip_ratio: float,
        value_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
        desired_kl: float,
    ) -> None:
        self.actor_critic = actor_critic
        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate)
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.desired_kl = desired_kl

    def update(
        self,
        buffer: RolloutBuffer,
        epochs: int,
        minibatches: int,
    ) -> PPOStats:
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        updates = 0

        for _ in range(epochs):
            for obs, actions, old_log_probs, returns, advantages, old_values in buffer.minibatches(minibatches):
                new_log_probs, entropy, values = self.actor_critic.evaluate_actions(obs, actions)
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                policy_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()

                value_pred_clipped = old_values + (values - old_values).clamp(-self.clip_ratio, self.clip_ratio)
                value_loss = torch.max(
                    F.mse_loss(values, returns, reduction="none"),
                    F.mse_loss(value_pred_clipped, returns, reduction="none"),
                ).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean().abs().item()
                if self.desired_kl > 0.0 and approx_kl > self.desired_kl * 2.0:
                    for group in self.optimizer.param_groups:
                        group["lr"] = max(1e-5, group["lr"] / 1.2)
                elif self.desired_kl > 0.0 and 0.0 < approx_kl < self.desired_kl / 2.0:
                    for group in self.optimizer.param_groups:
                        group["lr"] = min(1e-2, group["lr"] * 1.2)

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.mean().item())
                total_kl += approx_kl
                updates += 1

        return PPOStats(
            policy_loss=total_policy_loss / max(updates, 1),
            value_loss=total_value_loss / max(updates, 1),
            entropy=total_entropy / max(updates, 1),
            approx_kl=total_kl / max(updates, 1),
        )
