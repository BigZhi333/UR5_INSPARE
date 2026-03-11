from __future__ import annotations

import torch


class RolloutBuffer:
    def __init__(self, horizon: int, num_envs: int, obs_dim: int, act_dim: int, device: torch.device) -> None:
        self.horizon = horizon
        self.num_envs = num_envs
        self.device = device
        self.obs = torch.zeros((horizon, num_envs, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((horizon, num_envs, act_dim), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.values[self.ptr] = values
        self.ptr += 1

    def compute_returns_and_advantages(self, last_values: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        advantage = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for step in reversed(range(self.horizon)):
            if step == self.horizon - 1:
                next_value = last_values
            else:
                next_value = self.values[step + 1]
            next_non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            advantage = delta + gamma * gae_lambda * next_non_terminal * advantage
            self.advantages[step] = advantage
        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def minibatches(self, minibatches: int):
        batch_size = self.horizon * self.num_envs
        mini_batch_size = batch_size // minibatches
        indices = torch.randperm(batch_size, device=self.device)
        flat_obs = self.obs.reshape(batch_size, -1)
        flat_actions = self.actions.reshape(batch_size, -1)
        flat_log_probs = self.log_probs.reshape(batch_size)
        flat_returns = self.returns.reshape(batch_size)
        flat_advantages = self.advantages.reshape(batch_size)
        flat_values = self.values.reshape(batch_size)
        for start in range(0, batch_size, mini_batch_size):
            batch_ids = indices[start : start + mini_batch_size]
            yield (
                flat_obs[batch_ids],
                flat_actions[batch_ids],
                flat_log_probs[batch_ids],
                flat_returns[batch_ids],
                flat_advantages[batch_ids],
                flat_values[batch_ids],
            )
