"""TD(lambda) / TD-Gammon-style self-play training.

Backgammon is a long-horizon stochastic game.  A value network trained with
bootstrapped temporal-difference targets is a better fit than a policy-gradient
method when only the final win/loss is available.  The trainer below uses
self-play, epsilon-greedy value-guided move selection, and sign-aware TD(lambda)
targets when turns change hands.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .constants import ACTION_SIZE, OBSERVATION_SIZE
from .env import BackgammonEnv
from .inference import choose_action
from .model import BackgammonValueNetwork, resolve_device
from .training import (
    _atomic_torch_save,
    _optimizer_to_device,
    _prune_periodic_checkpoints,
    _seed_everything,
)


@dataclass
class TDTrainConfig:
    """Defaults for a practical TD-Gammon-style run."""

    total_steps: int = 20_000_000
    num_envs: int = 8
    rollout_steps: int = 128
    learning_rate: float = 1e-4
    gamma: float = 0.995
    td_lambda: float = 0.70
    update_epochs: int = 4
    minibatch_size: int = 1024
    hidden_size: int = 256
    residual_blocks: int = 3
    reward_shaping: float = 0.0005
    epsilon_start: float = 0.25
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 10_000_000
    search_samples: int = 1
    seed: int = 7
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 5
    log_interval: int = 10
    device: str = "auto"
    resume: str | None = None


def _td_payload(
    model: BackgammonValueNetwork,
    optimizer: torch.optim.Optimizer,
    config: TDTrainConfig,
    global_steps: int,
    update: int,
) -> dict[str, Any]:
    return {
        "algorithm": "td_lambda",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_steps": global_steps,
        "update": update,
        "model_config": {
            "observation_size": OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "hidden_size": config.hidden_size,
            "residual_blocks": config.residual_blocks,
        },
        "train_config": asdict(config),
    }


def _save_td_checkpoint(
    directory: Path,
    model: BackgammonValueNetwork,
    optimizer: torch.optim.Optimizer,
    config: TDTrainConfig,
    global_steps: int,
    update: int,
    periodic: bool = False,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = _td_payload(model, optimizer, config, global_steps, update)
    latest = directory / "latest.pt"
    _atomic_torch_save(payload, latest)
    if periodic:
        periodic_path = directory / f"checkpoint_{global_steps:012d}.pt"
        _atomic_torch_save(payload, periodic_path)
        _prune_periodic_checkpoints(directory)
    return latest


def _append_log(directory: Path, row: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "training.csv"
    fields = list(row.keys())
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _value_batch(model: BackgammonValueNetwork, observations: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    with torch.no_grad():
        values = model(tensor)
    return values.cpu().numpy()


def _epsilon(config: TDTrainConfig, global_steps: int) -> float:
    progress = min(1.0, global_steps / max(1, config.epsilon_decay_steps))
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def _select_action(
    model: BackgammonValueNetwork,
    env: BackgammonEnv,
    device: torch.device,
    epsilon: float,
    rng: np.random.Generator,
    search_samples: int,
) -> int:
    legal = np.flatnonzero(env.action_mask())
    if len(legal) == 1 or rng.random() < epsilon:
        return int(rng.choice(legal))
    return choose_action(
        model,
        env,
        device,
        deterministic=True,
        search_samples=max(1, search_samples),
    )


def _td_update(
    model: BackgammonValueNetwork,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    observations: np.ndarray,
    targets: np.ndarray,
    config: TDTrainConfig,
) -> float:
    flat_observations = torch.as_tensor(
        observations.reshape(-1, OBSERVATION_SIZE), dtype=torch.float32, device=device
    )
    flat_targets = torch.as_tensor(targets.reshape(-1), dtype=torch.float32, device=device)
    total_items = flat_observations.shape[0]
    minibatch_size = min(config.minibatch_size, total_items)
    losses: list[float] = []

    model.train()
    for _ in range(config.update_epochs):
        permutation = torch.randperm(total_items, device=device)
        for start in range(0, total_items, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            predictions = model(flat_observations[indices])
            # Smooth L1 is more stable than squared error while the value
            # network is still learning from mostly random games.
            loss = nn.functional.smooth_l1_loss(predictions, flat_targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
    return float(np.mean(losses))


def train_td(config: TDTrainConfig) -> Path:
    """Train a value network with sign-aware TD(lambda) self-play."""
    if config.total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if config.num_envs <= 0 or config.rollout_steps <= 0:
        raise ValueError("num_envs and rollout_steps must be positive")
    if not 0.0 <= config.td_lambda <= 1.0:
        raise ValueError("td_lambda must be between 0 and 1")
    if config.save_interval <= 0 or config.log_interval <= 0:
        raise ValueError("save_interval and log_interval must be positive")
    if config.epsilon_end < 0 or config.epsilon_start < config.epsilon_end:
        raise ValueError("epsilon_start must be >= epsilon_end >= 0")

    _seed_everything(config.seed)
    device = resolve_device(config.device)
    checkpoint_dir = Path(config.checkpoint_dir)
    model = BackgammonValueNetwork(
        observation_size=OBSERVATION_SIZE,
        hidden_size=config.hidden_size,
        residual_blocks=config.residual_blocks,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    global_steps = 0
    update = 0

    if config.resume:
        checkpoint = torch.load(config.resume, map_location=device)
        checkpoint_algorithm = checkpoint.get("algorithm", "unknown")
        if checkpoint_algorithm != "td_lambda":
            raise ValueError(
                "--resume must point to a TD(lambda) checkpoint; "
                f"found algorithm={checkpoint_algorithm!r}. Start TD(lambda) in a new checkpoint directory."
            )
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            _optimizer_to_device(optimizer, device)
        global_steps = int(checkpoint.get("global_steps", 0))
        update = int(checkpoint.get("update", 0))
        print(f"Resumed TD(lambda) checkpoint at {global_steps:,} steps.")
    else:
        initial = _save_td_checkpoint(
            checkpoint_dir, model, optimizer, config, global_steps, update, periodic=False
        )
        print(f"Saved initial TD(lambda) checkpoint: {initial}")

    rng = np.random.default_rng(config.seed + 91_337)
    envs = [
        BackgammonEnv(seed=config.seed + 20_000 + index, reward_shaping=config.reward_shaping)
        for index in range(config.num_envs)
    ]
    observations = np.stack([env.reset() for env in envs]).astype(np.float32)
    completed_lengths: list[int] = []
    completed_wins = [0, 0]
    total_games = 0
    rollout_size = config.rollout_steps * config.num_envs

    print(
        f"Training TD(lambda) self-play on {device} | target {config.total_steps:,} steps | "
        f"epsilon {config.epsilon_start:.3f}->{config.epsilon_end:.3f}"
    )

    while global_steps < config.total_steps:
        rollout_observations = np.zeros(
            (config.rollout_steps, config.num_envs, OBSERVATION_SIZE), dtype=np.float32
        )
        rollout_values = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        rollout_rewards = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        rollout_dones = np.zeros((config.rollout_steps, config.num_envs), dtype=np.bool_)
        rollout_players = np.zeros((config.rollout_steps, config.num_envs), dtype=np.int8)
        rollout_next_players = np.zeros((config.rollout_steps, config.num_envs), dtype=np.int8)

        model.eval()
        for time_index in range(config.rollout_steps):
            players = np.asarray([env.current_player for env in envs], dtype=np.int8)
            rollout_observations[time_index] = observations
            rollout_players[time_index] = players
            rollout_values[time_index] = _value_batch(model, observations, device)
            epsilon = _epsilon(config, global_steps + time_index * config.num_envs)

            next_observations = np.empty_like(observations)
            for env_index, env in enumerate(envs):
                action = _select_action(
                    model, env, device, epsilon, rng, config.search_samples
                )
                next_observation, reward, done, info = env.step(action)
                rollout_rewards[time_index, env_index] = reward
                rollout_dones[time_index, env_index] = done
                if done:
                    total_games += 1
                    winner = info.get("winner")
                    if winner in (0, 1):
                        completed_wins[int(winner)] += 1
                    completed_lengths.append(int(info["episode_steps"]))
                    next_observation = env.reset()
                next_observations[env_index] = next_observation
                rollout_next_players[time_index, env_index] = env.current_player
            observations = next_observations

        next_values = _value_batch(model, observations, device)
        advantages = np.zeros_like(rollout_rewards, dtype=np.float32)
        targets = np.zeros_like(rollout_rewards, dtype=np.float32)
        last_advantage = np.zeros(config.num_envs, dtype=np.float32)
        for time_index in reversed(range(config.rollout_steps)):
            bootstrap_values = next_values if time_index == config.rollout_steps - 1 else rollout_values[time_index + 1]
            signs = np.where(
                rollout_players[time_index] == rollout_next_players[time_index],
                1.0,
                -1.0,
            ).astype(np.float32)
            nonterminal = (~rollout_dones[time_index]).astype(np.float32)
            delta = (
                rollout_rewards[time_index]
                + config.gamma * nonterminal * signs * bootstrap_values
                - rollout_values[time_index]
            )
            last_advantage = delta + (
                config.gamma
                * config.td_lambda
                * nonterminal
                * signs
                * last_advantage
            )
            advantages[time_index] = last_advantage
            targets[time_index] = np.clip(
                rollout_values[time_index] + last_advantage,
                -1.0,
                1.0,
            )

        value_loss = _td_update(
            model, optimizer, device, rollout_observations, targets, config
        )
        global_steps += rollout_size
        update += 1
        if update % config.log_interval == 0 or global_steps >= config.total_steps:
            games_total = sum(completed_wins)
            white_rate = completed_wins[0] / games_total if games_total else 0.0
            mean_length = float(np.mean(completed_lengths[-100:])) if completed_lengths else 0.0
            current_epsilon = _epsilon(config, global_steps)
            print(
                f"update {update:>5} | steps {global_steps:>10,} | games {total_games:>6} | "
                f"game_len {mean_length:>6.1f} | white {white_rate:.3f} | "
                f"epsilon {current_epsilon:.3f} | value_loss {value_loss:.5f}"
            )
            _append_log(
                checkpoint_dir,
                {
                    "algorithm": "td_lambda",
                    "update": update,
                    "steps": global_steps,
                    "games": total_games,
                    "mean_game_length": mean_length,
                    "white_win_rate": white_rate,
                    "epsilon": current_epsilon,
                    "value_loss": value_loss,
                },
            )

        if update % config.save_interval == 0:
            path = _save_td_checkpoint(
                checkpoint_dir, model, optimizer, config, global_steps, update, periodic=True
            )
            print(f"Saved {path}")

    final_path = _save_td_checkpoint(
        checkpoint_dir, model, optimizer, config, global_steps, update, periodic=False
    )
    print(f"TD(lambda) training complete. Final model: {final_path}")
    return final_path
