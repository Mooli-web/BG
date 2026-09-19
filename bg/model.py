"""Neural policy/value model and action-masked sampling helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from .constants import ACTION_SIZE, OBSERVATION_SIZE


class ResidualBlock(nn.Module):
    """A small pre-activation residual block for the low-dimensional board state."""

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width * 2, width)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return residual + x


class BackgammonValueNetwork(nn.Module):
    """Value-only network used by the TD-Gammon-style trainer.

    The network estimates the expected game outcome from the current player's
    perspective.  A value-only model is intentionally paired with legal-move
    lookahead at action time; this is the classic TD-Gammon pattern and avoids
    forcing PPO to learn a large policy under a very sparse game reward.
    """

    def __init__(
        self,
        observation_size: int = OBSERVATION_SIZE,
        hidden_size: int = 256,
        residual_blocks: int = 3,
    ):
        super().__init__()
        if residual_blocks < 1:
            raise ValueError("residual_blocks must be at least 1")
        self.observation_size = observation_size
        self.action_size = 0
        self.hidden_size = hidden_size
        self.residual_blocks = residual_blocks
        self.input_layer = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.trunk = nn.Sequential(*(ResidualBlock(hidden_size) for _ in range(residual_blocks)))
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        hidden = self.input_layer(observation)
        hidden = self.trunk(hidden)
        return self.value_head(hidden).squeeze(-1)


class BackgammonActorCritic(nn.Module):
    """Shared MLP with separate masked policy and value heads.

    Backgammon's observation is a compact structured board rather than an
    image.  A residual MLP is therefore more appropriate and much lighter
    than a CNN or transformer, while still giving the value head enough depth
    to model contact, bearing-off, and hitting positions.
    """

    def __init__(
        self,
        observation_size: int = OBSERVATION_SIZE,
        action_size: int = ACTION_SIZE,
        hidden_size: int = 256,
        residual_blocks: int = 3,
    ):
        super().__init__()
        if residual_blocks < 1:
            raise ValueError("residual_blocks must be at least 1")
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.residual_blocks = residual_blocks

        self.input_layer = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.trunk = nn.Sequential(*(ResidualBlock(hidden_size) for _ in range(residual_blocks)))
        self.policy_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, action_size),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        hidden = self.input_layer(observation)
        hidden = self.trunk(hidden)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value

    def distribution(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        """Return a categorical policy with illegal moves removed."""
        logits, value = self(observation)
        if action_mask.dtype != torch.bool:
            action_mask = action_mask.bool()
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        if action_mask.shape != logits.shape:
            raise ValueError(
                f"action mask shape {tuple(action_mask.shape)} does not match logits {tuple(logits.shape)}"
            )
        if not torch.all(action_mask.any(dim=-1)):
            raise ValueError("each observation must have at least one legal action")
        masked_logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return Categorical(logits=masked_logits), value

    @torch.no_grad()
    def choose_action(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose actions and return action, log-probability, and value."""
        distribution, value = self.distribution(observation, action_mask)
        if deterministic:
            action = distribution.logits.argmax(dim=-1)
        else:
            action = distribution.sample()
        return action, distribution.log_prob(action), value


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` into a torch device."""
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a saved model and its metadata."""
    device = torch.device(device)
    checkpoint = torch.load(Path(path), map_location=device)
    config = checkpoint.get("model_config", {})
    algorithm = checkpoint.get("algorithm", "ppo")
    if algorithm == "td_lambda":
        model = BackgammonValueNetwork(
            observation_size=int(config.get("observation_size", OBSERVATION_SIZE)),
            hidden_size=int(config.get("hidden_size", 256)),
            residual_blocks=int(config.get("residual_blocks", 3)),
        ).to(device)
    else:
        model = BackgammonActorCritic(
            observation_size=int(config.get("observation_size", OBSERVATION_SIZE)),
            action_size=int(config.get("action_size", ACTION_SIZE)),
            hidden_size=int(config.get("hidden_size", 256)),
            residual_blocks=int(config.get("residual_blocks", 3)),
        ).to(device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint
