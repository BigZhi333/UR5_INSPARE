from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
TANH_EPS = 1e-6


def build_mlp(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_sizes:
        linear = nn.Linear(previous, hidden)
        nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(linear.bias)
        layers.extend((linear, nn.Tanh()))
        previous = hidden
    output = nn.Linear(previous, output_dim)
    nn.init.orthogonal_(output.weight, gain=0.01 if output_dim > 1 else 1.0)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        self.policy = build_mlp(obs_dim, hidden_sizes, act_dim)
        self.value = build_mlp(obs_dim, hidden_sizes, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def _distribution_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.policy(obs)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mean)
        return mean, std

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean, std = self._distribution_params(obs)
        return Normal(mean, std)

    def _atanh(self, action: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(action, -1.0 + TANH_EPS, 1.0 - TANH_EPS)
        return 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))

    def _squashed_log_prob(
        self,
        dist: Normal,
        pre_tanh_action: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        correction = torch.log(1.0 - action.pow(2) + TANH_EPS).sum(dim=-1)
        return dist.log_prob(pre_tanh_action).sum(dim=-1) - correction

    def step(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        pre_tanh_action = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(pre_tanh_action)
        log_prob = self._squashed_log_prob(dist, pre_tanh_action, action)
        value = self.value(obs).squeeze(-1)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        pre_tanh_actions = self._atanh(actions)
        log_prob = self._squashed_log_prob(dist, pre_tanh_actions, actions)
        # Use the base Gaussian entropy as a stable approximation for the squashed policy entropy term.
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs).squeeze(-1)
        return log_prob, entropy, value
