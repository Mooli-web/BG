"""Shared dimensions and action encoding for the backgammon environment."""

BOARD_POINTS = 24
DICE_SIDES = 6
MAX_CHECKERS = 15
BAR = BOARD_POINTS

# A move is (source, die), where source is one of points 0..23 or BAR.
# 6 * 25 leaves one separate action for a forced pass.
SOURCES_PER_DIE = BOARD_POINTS + 1
ACTION_SIZE = DICE_SIDES * SOURCES_PER_DIE + 1
PASS_ACTION = ACTION_SIZE - 1

# 24 points * (own count, opponent count), bar/off for both sides,
# and a count for each remaining die face.
OBSERVATION_SIZE = BOARD_POINTS * 2 + 4 + DICE_SIDES

WHITE = 0
BLACK = 1
PLAYERS = (WHITE, BLACK)


def action_index(source: int, die: int) -> int:
    """Encode a (source, die) move into the policy action space."""
    if not 0 <= source <= BAR:
        raise ValueError(f"source must be in [0, {BAR}], got {source}")
    if not 1 <= die <= DICE_SIDES:
        raise ValueError(f"die must be in [1, {DICE_SIDES}], got {die}")
    return (die - 1) * SOURCES_PER_DIE + source


def decode_action(action: int) -> tuple[int, int] | None:
    """Decode an action; return None for the forced-pass action."""
    if action == PASS_ACTION:
        return None
    if not 0 <= action < PASS_ACTION:
        raise ValueError(f"action must be in [0, {PASS_ACTION}], got {action}")
    die, source = divmod(action, SOURCES_PER_DIE)
    return source, die + 1
