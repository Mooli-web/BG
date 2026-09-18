"""Backgammon rules and move generation.

The internal point numbering is deliberately explicit:

* White (player 0) moves from point 0 towards point 23.
* Black (player 1) moves from point 23 towards point 0.
* ``board[p] > 0`` contains white checkers and ``board[p] < 0`` contains
  black checkers.
* A checker on the bar enters at the opponent's home board.

The environment turns a complete dice roll into one or more policy decisions.
This module is independent of PyTorch and is consequently easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .constants import BAR, BLACK, BOARD_POINTS, MAX_CHECKERS, WHITE

Move = tuple[int, int]  # (source point, die)


def _sign(player: int) -> int:
    return 1 if player == WHITE else -1


def opponent(player: int) -> int:
    return WHITE if player == BLACK else BLACK


@dataclass
class GameState:
    """A complete position, including the player and dice currently in use."""

    board: list[int] = field(default_factory=lambda: [0] * BOARD_POINTS)
    bar: list[int] = field(default_factory=lambda: [0, 0])
    off: list[int] = field(default_factory=lambda: [0, 0])
    current_player: int = WHITE
    # ``dice`` is the remaining dice for the current turn. ``turn_dice``
    # keeps the original roll so a UI can show which dice have been used.
    dice: tuple[int, ...] = ()
    turn_dice: tuple[int, ...] = ()

    @classmethod
    def initial(cls, starting_player: int = WHITE) -> "GameState":
        """Return the standard starting position.

        The position is mirrored for the two colours.  Who starts is supplied
        by the environment; using a random starter avoids a first-player bias
        during self-play.
        """
        board = [0] * BOARD_POINTS
        board[0] = 2
        board[11] = 5
        board[16] = 3
        board[18] = 5

        board[23] = -2
        board[12] = -5
        board[7] = -3
        board[5] = -5
        return cls(board=board, current_player=starting_player)

    def copy(self) -> "GameState":
        return GameState(
            board=self.board.copy(),
            bar=self.bar.copy(),
            off=self.off.copy(),
            current_player=self.current_player,
            dice=tuple(self.dice),
            turn_dice=tuple(self.turn_dice),
        )

    def validate(self) -> None:
        """Raise ``ValueError`` if checker accounting is inconsistent."""
        if len(self.board) != BOARD_POINTS:
            raise ValueError("board must have 24 points")
        if any(abs(value) > MAX_CHECKERS for value in self.board):
            raise ValueError("a point contains too many checkers")
        for player in (WHITE, BLACK):
            on_board = sum(max(value * _sign(player), 0) for value in self.board)
            total = on_board + self.bar[player] + self.off[player]
            if total != MAX_CHECKERS:
                raise ValueError(
                    f"player {player} has {total} checkers instead of {MAX_CHECKERS}"
                )
        if self.current_player not in (WHITE, BLACK):
            raise ValueError("current_player must be WHITE or BLACK")


def pip_count(state: GameState, player: int) -> int:
    """Return the number of pips still needed by ``player``.

    Bar checkers are counted as 25 pips.  This is mainly used for optional
    reward shaping and for displaying useful diagnostics, not as a rule.
    """
    sign = _sign(player)
    if player == WHITE:
        points = sum(max(state.board[p] * sign, 0) * (BOARD_POINTS - p) for p in range(BOARD_POINTS))
    else:
        points = sum(max(state.board[p] * sign, 0) * (p + 1) for p in range(BOARD_POINTS))
    return points + state.bar[player] * (BOARD_POINTS + 1)


def position_advantage(state: GameState, player: int) -> float:
    """A small, symmetric progress heuristic for optional dense rewards."""
    other = opponent(player)

    def score(side: int) -> float:
        # Higher is better: borne-off pieces help, while remaining pips and
        # pieces on the bar hurt.
        return state.off[side] * 24.0 - pip_count(state, side) - state.bar[side] * 12.0

    return score(player) - score(other)


def _can_land(board_value: int, player: int) -> bool:
    """A point is open if empty, friendly, or occupied by one opponent."""
    sign = _sign(player)
    return board_value * sign >= -1


def _has_checker_farther_from_exit(state: GameState, player: int, source: int) -> bool:
    sign = _sign(player)
    # White exits beyond point 23, so smaller home-board points are farther
    # from the exit.  Black exits beyond point 0, so larger points are farther.
    if player == WHITE:
        return any(state.board[p] * sign > 0 for p in range(18, source))
    return any(state.board[p] * sign > 0 for p in range(source + 1, 6))


def destination_for(source: int, die: int, player: int) -> int | None:
    """Return a board point or ``None`` when the move bears a checker off."""
    if source == BAR:
        return die - 1 if player == WHITE else BOARD_POINTS - die
    destination = source + die if player == WHITE else source - die
    return destination if 0 <= destination < BOARD_POINTS else None


def legal_moves_for_die(state: GameState, player: int, die: int) -> list[Move]:
    """List every legal single-checker move for one die.

    Bar priority and the exact/overshoot bearing-off rule are handled here.
    The caller is responsible for enforcing the fact that a checker on the bar
    must enter before any checker on the board is moved.
    """
    if not 1 <= die <= 6:
        raise ValueError("die must be between 1 and 6")
    sign = _sign(player)

    if state.bar[player] > 0:
        destination = destination_for(BAR, die, player)
        assert destination is not None
        if _can_land(state.board[destination], player):
            return [(BAR, die)]
        return []

    moves: list[Move] = []
    for source, value in enumerate(state.board):
        if value * sign <= 0:
            continue

        destination = destination_for(source, die, player)
        if destination is not None:
            if _can_land(state.board[destination], player):
                moves.append((source, die))
            continue

        # A move beyond the edge can bear off only from the home board.
        in_home = source >= 18 if player == WHITE else source <= 5
        if not in_home:
            continue
        distance = BOARD_POINTS - source if player == WHITE else source + 1
        if die < distance:
            continue
        # An oversized die may be used only by the furthest checker from the
        # exit.  An exact die does not have this restriction.
        if die > distance and _has_checker_farther_from_exit(state, player, source):
            continue
        moves.append((source, die))
    return moves


def apply_move(state: GameState, player: int, move: Move) -> None:
    """Apply a previously validated single move in place."""
    source, die = move
    legal = legal_moves_for_die(state, player, die)
    if move not in legal:
        raise ValueError(f"illegal move {move} for player {player}")

    sign = _sign(player)
    destination = destination_for(source, die, player)
    if source == BAR:
        state.bar[player] -= 1
    else:
        state.board[source] -= sign

    if destination is None:
        state.off[player] += 1
        return

    if state.board[destination] * sign == -1:
        # A legal destination can contain at most one opponent checker.
        state.board[destination] = 0
        state.bar[opponent(player)] += 1
    state.board[destination] += sign


def _deduplicate(sequences: Iterable[tuple[Move, ...]]) -> list[tuple[Move, ...]]:
    seen: set[tuple[Move, ...]] = set()
    result: list[tuple[Move, ...]] = []
    for sequence in sequences:
        if sequence not in seen:
            seen.add(sequence)
            result.append(sequence)
    return result


def enumerate_turn_sequences(state: GameState, dice: Iterable[int]) -> list[tuple[Move, ...]]:
    """Enumerate legal maximal sequences for a dice roll.

    A backgammon turn is represented as a sequence rather than a single
    action.  We keep only sequences that use the maximum number of dice.  This
    implements the important rule that both dice must be used when possible,
    and naturally handles doubles (four copies of the same die).  The list is
    normally tiny, even in positions with many legal moves.
    """
    dice_tuple = tuple(int(die) for die in dice)
    if any(not 1 <= die <= 6 for die in dice_tuple):
        raise ValueError("dice must contain values between 1 and 6")

    def search(position: GameState, remaining: tuple[int, ...], prefix: tuple[Move, ...]) -> list[tuple[Move, ...]]:
        choices: list[tuple[int, Move]] = []
        for die in sorted(set(remaining), reverse=True):
            choices.extend((die, move) for move in legal_moves_for_die(position, position.current_player, die))

        if not choices:
            return [prefix]

        candidates: list[tuple[Move, ...]] = []
        for die, move in choices:
            next_position = position.copy()
            apply_move(next_position, position.current_player, move)
            remaining_list = list(remaining)
            remaining_list.remove(die)
            next_remaining = tuple(remaining_list)
            candidates.extend(search(next_position, next_remaining, prefix + (move,)))

        maximum = max(len(sequence) for sequence in candidates)
        best = [sequence for sequence in candidates if len(sequence) == maximum]

        # Official rule: if both dice are individually playable but they
        # cannot both be used in the turn, only the higher die may be played.
        # This matters only for a non-double two-die roll whose best sequence
        # has length one.  If the higher die has no legal move, the lower one
        # remains the only available choice.
        if len(remaining) == 2 and remaining[0] != remaining[1] and maximum == 1:
            higher_die = max(remaining)
            higher_sequences = [sequence for sequence in best if sequence[0][1] == higher_die]
            if higher_sequences:
                best = higher_sequences

        return _deduplicate(best)

    return search(state.copy(), dice_tuple, ())
