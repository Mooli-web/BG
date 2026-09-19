"""Self-play PPO training for the backgammon environment."""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .constants import ACTION_SIZE, OBSERVATION_SIZE
from .env import BackgammonEnv
from .model import BackgammonActorCritic, resolve_device


@dataclass
class TrainConfig:
    """Training defaults suitable for a first several-million-step run."""

    total_steps: int = 5_000_000
    num_envs: int = 16
    rollout_steps: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 1024
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 256
    residual_blocks: int = 3
    reward_shaping: float = 0.0005
    seed: int = 7
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 50
    log_interval: int = 10
    device: str = "auto"
    resume: str | None = None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _checkpoint_payload(
    model: BackgammonActorCritic,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    global_steps: int,
    update: int,
) -> dict[str, Any]:
    return {
        "algorithm": "ppo",
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


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint atomically so a Colab disconnect cannot ruin latest.pt."""
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _save_checkpoint(
    directory: Path,
    model: BackgammonActorCritic,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    global_steps: int,
    update: int,
    periodic: bool = False,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(model, optimizer, config, global_steps, update)
    latest = directory / "latest.pt"
    _atomic_torch_save(payload, latest)
    if periodic:
        periodic_path = directory / f"checkpoint_{global_steps:012d}.pt"
        _atomic_torch_save(payload, periodic_path)
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


def _ppo_update(
    model: BackgammonActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    observations: np.ndarray,
    masks: np.ndarray,
    actions: np.ndarray,
    old_log_probs: np.ndarray,
    old_values: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
    config: TrainConfig,
) -> dict[str, float]:
    """Perform the clipped PPO update over one rollout."""
    flat_obs = torch.as_tensor(observations.reshape(-1, OBSERVATION_SIZE), dtype=torch.float32, device=device)
    flat_masks = torch.as_tensor(masks.reshape(-1, ACTION_SIZE), dtype=torch.bool, device=device)
    flat_actions = torch.as_tensor(actions.reshape(-1), dtype=torch.long, device=device)
    flat_old_log_probs = torch.as_tensor(old_log_probs.reshape(-1), dtype=torch.float32, device=device)
    flat_old_values = torch.as_tensor(old_values.reshape(-1), dtype=torch.float32, device=device)
    flat_advantages = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=device)
    flat_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device)

    flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
    total_items = flat_obs.shape[0]
    minibatch_size = min(config.minibatch_size, total_items)

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    clip_fractions: list[float] = []
    approx_kls: list[float] = []

    model.train()
    for _ in range(config.update_epochs):
        permutation = torch.randperm(total_items, device=device)
        for start in range(0, total_items, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            distribution, values = model.distribution(flat_obs[indices], flat_masks[indices])
            log_probs = distribution.log_prob(flat_actions[indices])
            entropy = distribution.entropy().mean()

            ratios = (log_probs - flat_old_log_probs[indices]).exp()
            unclipped = ratios * flat_advantages[indices]
            clipped = torch.clamp(
                ratios,
                1.0 - config.clip_coef,
                1.0 + config.clip_coef,
            ) * flat_advantages[indices]
            policy_loss = -torch.minimum(unclipped, clipped).mean()

            value_prediction = values
            value_clipped = flat_old_values[indices] + torch.clamp(
                value_prediction - flat_old_values[indices],
                -config.clip_coef,
                config.clip_coef,
            )
            value_loss_unclipped = (value_prediction - flat_returns[indices]).pow(2)
            value_loss_clipped = (value_clipped - flat_returns[indices]).pow(2)
            value_loss = 0.5 * torch.maximum(value_loss_unclipped, value_loss_clipped).mean()

            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (flat_old_log_probs[indices] - log_probs).mean().item()
                clip_fraction = ((ratios - 1.0).abs() > config.clip_coef).float().mean().item()
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))
            approx_kls.append(approx_kl)
            clip_fractions.append(clip_fraction)

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "approx_kl": float(np.mean(approx_kls)),
        "clip_fraction": float(np.mean(clip_fractions)),
    }


def train(config: TrainConfig) -> Path:
    """Run self-play PPO until ``config.total_steps`` is reached."""
    if config.total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if config.num_envs <= 0 or config.rollout_steps <= 0:
        raise ValueError("num_envs and rollout_steps must be positive")
    if config.save_interval <= 0 or config.log_interval <= 0:
        raise ValueError("save_interval and log_interval must be positive")

    _seed_everything(config.seed)
    device = resolve_device(config.device)
    checkpoint_dir = Path(config.checkpoint_dir)

    model = BackgammonActorCritic(
        observation_size=OBSERVATION_SIZE,
        action_size=ACTION_SIZE,
        hidden_size=config.hidden_size,
        residual_blocks=config.residual_blocks,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    global_steps = 0
    update = 0

    if config.resume:
        checkpoint = torch.load(config.resume, map_location=device)
        model.load_state_dict(checkpoint.get("model", checkpoint))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            _optimizer_to_device(optimizer, device)
        global_steps = int(checkpoint.get("global_steps", checkpoint.get("total_steps", 0)))
        update = int(checkpoint.get("update", 0))
        print(f"Resumed {config.resume} at {global_steps:,} environment steps.")
    else:
        # Create a valid restart point immediately. This is especially useful
        # when a Colab runtime disconnects during the first rollout.
        initial_path = _save_checkpoint(
            checkpoint_dir,
            model,
            optimizer,
            config,
            global_steps,
            update,
            periodic=False,
        )
        print(f"Saved initial checkpoint: {initial_path}")

    envs = [
        BackgammonEnv(seed=config.seed + 10_000 + index, reward_shaping=config.reward_shaping)
        for index in range(config.num_envs)
    ]
    observations = np.stack([env.reset() for env in envs]).astype(np.float32)
    completed_lengths: list[int] = []
    completed_wins = [0, 0]
    total_games = 0

    rollout_size = config.rollout_steps * config.num_envs
    print(
        f"Training self-play PPO on {device} | target {config.total_steps:,} steps | "
        f"{config.num_envs} environments x {config.rollout_steps} rollout steps"
    )

    while global_steps < config.total_steps:
        rollout_observations = np.zeros(
            (config.rollout_steps, config.num_envs, OBSERVATION_SIZE), dtype=np.float32
        )
        rollout_masks = np.zeros(
            (config.rollout_steps, config.num_envs, ACTION_SIZE), dtype=np.bool_
        )
        rollout_actions = np.zeros((config.rollout_steps, config.num_envs), dtype=np.int64)
        rollout_log_probs = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        rollout_values = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        rollout_rewards = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        rollout_dones = np.zeros((config.rollout_steps, config.num_envs), dtype=np.bool_)
        rollout_players = np.zeros((config.rollout_steps, config.num_envs), dtype=np.int8)
        rollout_next_players = np.zeros((config.rollout_steps, config.num_envs), dtype=np.int8)

        for time_index in range(config.rollout_steps):
            masks = np.stack([env.action_mask() for env in envs])
            players = np.asarray([env.current_player for env in envs], dtype=np.int8)
            rollout_observations[time_index] = observations
            rollout_masks[time_index] = masks
            rollout_players[time_index] = players

            obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
            mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=device)
            with torch.no_grad():
                distribution, values = model.distribution(obs_tensor, mask_tensor)
                actions_tensor = distribution.sample()
                log_probs_tensor = distribution.log_prob(actions_tensor)
            actions = actions_tensor.cpu().numpy()
            rollout_actions[time_index] = actions
            rollout_log_probs[time_index] = log_probs_tensor.cpu().numpy()
            rollout_values[time_index] = values.cpu().numpy()

            next_observations = np.empty_like(observations)
            for env_index, env in enumerate(envs):
                next_observation, reward, done, info = env.step(int(actions[env_index]))
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

        with torch.no_grad():
            next_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
            _, next_values_tensor = model(next_tensor)
        next_values = next_values_tensor.cpu().numpy()

        advantages = np.zeros_like(rollout_rewards, dtype=np.float32)
        last_advantage = np.zeros(config.num_envs, dtype=np.float32)
        for time_index in reversed(range(config.rollout_steps)):
            if time_index == config.rollout_steps - 1:
                bootstrap_values = next_values
            else:
                bootstrap_values = rollout_values[time_index + 1]

            signs = np.where(
                rollout_players[time_index] == rollout_next_players[time_index],
                1.0,
                -1.0,
            ).astype(np.float32)
            nonterminal = (~rollout_dones[time_index]).astype(np.float32)
            deltas = (
                rollout_rewards[time_index]
                + config.gamma * nonterminal * signs * bootstrap_values
                - rollout_values[time_index]
            )
            advantages[time_index] = deltas + (
                config.gamma
                * config.gae_lambda
                * nonterminal
                * signs
                * last_advantage
            )
            last_advantage = advantages[time_index]

        returns = advantages + rollout_values
        metrics = _ppo_update(
            model=model,
            optimizer=optimizer,
            device=device,
            observations=rollout_observations,
            masks=rollout_masks,
            actions=rollout_actions,
            old_log_probs=rollout_log_probs,
            old_values=rollout_values,
            advantages=advantages,
            returns=returns,
            config=config,
        )

        global_steps += rollout_size
        update += 1
        if update % config.log_interval == 0 or global_steps >= config.total_steps:
            mean_length = float(np.mean(completed_lengths[-100:])) if completed_lengths else 0.0
            wins_total = sum(completed_wins)
            white_rate = completed_wins[0] / wins_total if wins_total else 0.0
            print(
                f"update {update:>5} | steps {global_steps:>10,} | games {total_games:>6} | "
                f"game_len {mean_length:>6.1f} | white {white_rate:.3f} | "
                f"policy {metrics['policy_loss']:+.4f} | value {metrics['value_loss']:.4f} | "
                f"entropy {metrics['entropy']:.3f}"
            )
            _append_log(
                checkpoint_dir,
                {
                    "update": update,
                    "steps": global_steps,
                    "games": total_games,
                    "mean_game_length": mean_length,
                    "white_win_rate": white_rate,
                    **metrics,
                },
            )

        if update % config.save_interval == 0:
            path = _save_checkpoint(
                checkpoint_dir,
                model,
                optimizer,
                config,
                global_steps,
                update,
                periodic=True,
            )
            print(f"Saved {path}")

    final_path = _save_checkpoint(
        checkpoint_dir,
        model,
        optimizer,
        config,
        global_steps,
        update,
        periodic=False,
    )
    print(f"Training complete. Final model: {final_path}")
    return final_path
