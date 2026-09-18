"""A lightweight, vectorisation-friendly backgammon environment.

The policy chooses one checker move at a time.  A complete dice roll is
expanded into a sequence of such decisions, so the policy only needs 151
logits: 24 board sources plus the bar for each die face, and a forced-pass
action.  The environment masks every action that is not part of a legal
maximal dice sequence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .constants import (
    ACTION_SIZE,
    BAR,
    BLACK,
    BOARD_POINTS,
    DICE_SIDES,
    OBSERVATION_SIZE,
    PASS_ACTION,
    WHITE,
    action_index,
    decode_action,
)
from .game import GameState, Move, enumerate_turn_sequences, opponent, position_advantage


class BackgammonEnv:
    """Self-play environment with a Gym-like ``reset``/``step`` interface.

    Rewards are from the point of view of the player who made the action.  A
    win is +1 and a loss is -1.  ``reward_shaping`` can add a small progress
    reward based on pips; leave it at zero for a strictly sparse game reward.
    """

    observation_size = OBSERVATION_SIZE
    action_size = ACTION_SIZE

    def __init__(self, seed: int | None = None, reward_shaping: float = 0.0):
        self.rng = np.random.default_rng(seed)
        self.reward_shaping = float(reward_shaping)
        self.state = GameState.initial()
        self.done = False
        self.winner: int | None = None
        self.episode_steps = 0
        self._turn_sequences: list[tuple[Move, ...]] = [()]
        self._episode_return = [0.0, 0.0]

    @property
    def current_player(self) -> int:
        return self.state.current_player

    @property
    def episode_return(self) -> tuple[float, float]:
        return tuple(self._episode_return)

    def reset(self, starting_player: int | None = None) -> np.ndarray:
        """Start a new game and return the canonical observation.

        ``starting_player`` is useful for the GUI and tests.  Training leaves
        it as ``None`` so each environment gets an unbiased random starter.
        """
        if starting_player is None:
            starting_player = int(self.rng.integers(0, 2))
        if starting_player not in (WHITE, BLACK):
            raise ValueError("starting_player must be 0 (white) or 1 (black)")
        self.state = GameState.initial(starting_player=starting_player)
        self.done = False
        self.winner = None
        self.episode_steps = 0
        self._episode_return = [0.0, 0.0]
        self._roll_for_current_player()
        return self.observation()

    def _roll_for_current_player(self) -> None:
        first = int(self.rng.integers(1, DICE_SIDES + 1))
        second = int(self.rng.integers(1, DICE_SIDES + 1))
        dice = (first,) * 4 if first == second else (first, second)
        self.state.dice = tuple(sorted(dice))
        self._turn_sequences = enumerate_turn_sequences(self.state, self.state.dice)

    def observation(self) -> np.ndarray:
        """Return a player-relative observation for the current player."""
        return observation_from_state(self.state)

    def action_mask(self) -> np.ndarray:
        """Return a boolean mask for the current state's policy actions."""
        mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
        if self.done:
            mask[PASS_ACTION] = True
            return mask
        for sequence in self._turn_sequences:
            if sequence:
                source, die = sequence[0]
                mask[action_index(source, die)] = True
            else:
                mask[PASS_ACTION] = True
        # There is always at least one maximal sequence, but keeping this
        # fallback makes corrupted/custom positions fail safely.
        if not mask.any():
            mask[PASS_ACTION] = True
        return mask

    def legal_actions(self) -> list[int]:
        return np.flatnonzero(self.action_mask()).astype(int).tolist()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Apply one masked move and return ``(obs, reward, done, info)``."""
        if self.done:
            raise RuntimeError("cannot step a finished game; call reset()")
        action = int(action)
        mask = self.action_mask()
        if not 0 <= action < ACTION_SIZE or not mask[action]:
            raise ValueError(f"action {action} is illegal in the current state")

        actor = self.state.current_player
        before_advantage = position_advantage(self.state, actor)
        decoded = decode_action(action)

        if decoded is None:
            # A pass is legal only when no die can be used.  _turn_sequences
            # contains exactly the empty sequence in that case.
            if not any(self._turn_sequences):
                self._turn_sequences = []
            else:  # defensive; the action mask should have rejected this
                raise ValueError("pass is only legal when no checker can move")
        else:
            source, die = decoded
            move = (source, die)
            matching = [sequence for sequence in self._turn_sequences if sequence and sequence[0] == move]
            if not matching:
                raise ValueError(f"move {move} is not a legal sequence prefix")
            from .game import apply_move

            apply_move(self.state, actor, move)
            remaining_dice = list(self.state.dice)
            remaining_dice.remove(die)
            self.state.dice = tuple(remaining_dice)
            self._turn_sequences = _unique_suffixes(sequence[1:] for sequence in matching)

        self.episode_steps += 1
        reward = 0.0
        if self.reward_shaping:
            reward += self.reward_shaping * (
                position_advantage(self.state, actor) - before_advantage
            )

        if self.state.off[actor] >= 15:
            self.done = True
            self.winner = actor
            reward += 1.0
        elif not self._turn_sequences or all(not sequence for sequence in self._turn_sequences):
            # The maximal sequence has been completed, or no die was playable.
            self.state.current_player = opponent(actor)
            self._roll_for_current_player()

        self._episode_return[actor] += reward
        info: dict[str, Any] = {
            "player": actor,
            "next_player": self.state.current_player,
            "winner": self.winner,
            "episode_steps": self.episode_steps,
        }
        if self.done:
            # The player who did not make the final action sees the opposite
            # terminal reward when the trainer converts it to its perspective.
            info["loser"] = opponent(actor)
        return self.observation(), float(reward), self.done, info


def _unique_suffixes(sequences: Any) -> list[tuple[Move, ...]]:
    seen: set[tuple[Move, ...]] = set()
    result: list[tuple[Move, ...]] = []
    for sequence in sequences:
        sequence = tuple(sequence)
        if sequence not in seen:
            seen.add(sequence)
            result.append(sequence)
    return result


def observation_from_state(state: GameState) -> np.ndarray:
    """Create a current-player-relative feature vector.

    Reversing the board for black means one shared network can learn one set of
    movement patterns.  Each point uses two normalised counts, followed by own
    and opponent bar/off counts and a six-value remaining-dice histogram.
    """
    player = state.current_player
    sign = 1 if player == WHITE else -1
    features = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    offset = 0

    for canonical_point in range(BOARD_POINTS):
        global_point = canonical_point if player == WHITE else BOARD_POINTS - 1 - canonical_point
        value = state.board[global_point] * sign
        features[offset] = max(value, 0) / 15.0
        features[offset + 1] = max(-value, 0) / 15.0
        offset += 2

    features[offset] = state.bar[player] / 15.0
    features[offset + 1] = state.bar[opponent(player)] / 15.0
    features[offset + 2] = state.off[player] / 15.0
    features[offset + 3] = state.off[opponent(player)] / 15.0
    offset += 4

    for die in range(1, DICE_SIDES + 1):
        features[offset + die - 1] = state.dice.count(die) / 4.0
    return features
