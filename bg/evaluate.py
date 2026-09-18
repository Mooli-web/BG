"""Command-line evaluation utilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from .env import BackgammonEnv
from .model import BackgammonActorCritic, load_checkpoint, resolve_device


@dataclass
class MatchResult:
    games: int
    ai_wins: int
    random_wins: int
    average_game_length: float

    @property
    def ai_win_rate(self) -> float:
        return self.ai_wins / self.games if self.games else 0.0


def _model_action(
    model: BackgammonActorCritic,
    observation: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    deterministic: bool,
) -> int:
    observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
    action, _, _ = model.choose_action(observation_tensor, mask_tensor, deterministic=deterministic)
    return int(action.item())


def _random_action(mask: np.ndarray, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal))


def play_match(
    model: BackgammonActorCritic,
    games: int = 100,
    device: torch.device | str = "cpu",
    deterministic: bool = True,
    seed: int = 123,
    self_play: bool = False,
) -> MatchResult:
    """Play the model against a random player, or against itself."""
    if games <= 0:
        raise ValueError("games must be positive")
    device = torch.device(device)
    rng = np.random.default_rng(seed)
    ai_wins = 0
    random_wins = 0
    lengths: list[int] = []
    model.eval()

    for game_index in range(games):
        env = BackgammonEnv(seed=seed + game_index + 1)
        ai_player = game_index % 2
        observation = env.reset()
        done = False
        while not done:
            mask = env.action_mask()
            if self_play or env.current_player == ai_player:
                action = _model_action(model, observation, mask, device, deterministic)
            else:
                action = _random_action(mask, rng)
            observation, _, done, info = env.step(action)
        winner = int(info["winner"])
        if self_play:
            # In self-play both sides use the same policy; report the winner as
            # an AI win so the total still describes completed games.
            ai_wins += 1
        elif winner == ai_player:
            ai_wins += 1
        else:
            random_wins += 1
        lengths.append(int(info["episode_steps"]))

    return MatchResult(
        games=games,
        ai_wins=ai_wins,
        random_wins=random_wins,
        average_game_length=float(np.mean(lengths)),
    )


def evaluate_checkpoint(
    checkpoint_path: str,
    games: int = 100,
    device_name: str = "auto",
    deterministic: bool = True,
    self_play: bool = False,
) -> MatchResult:
    device = resolve_device(device_name)
    model, _ = load_checkpoint(checkpoint_path, device=device)
    return play_match(
        model,
        games=games,
        device=device,
        deterministic=deterministic,
        self_play=self_play,
    )


def add_evaluate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True, help="path to checkpoints/latest.pt")
    parser.add_argument("--games", type=int, default=100, help="number of games to play")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample the policy instead of taking its best legal move",
    )
    parser.add_argument(
        "--self-play",
        action="store_true",
        help="use the checkpoint for both players instead of one random opponent",
    )


def evaluation_from_args(args: argparse.Namespace) -> int:
    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        games=args.games,
        device_name=args.device,
        deterministic=not args.stochastic,
        self_play=args.self_play,
    )
    print(
        f"games={result.games} | ai_wins={result.ai_wins} | "
        f"random_wins={result.random_wins} | ai_win_rate={result.ai_win_rate:.1%} | "
        f"average_game_length={result.average_game_length:.1f} decisions"
    )
    return 0
