from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


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

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.policy(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def step(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.value(obs).squeeze(-1)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs).squeeze(-1)
        return log_prob, entropy, value
