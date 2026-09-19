from bg.constants import BAR, BLACK, WHITE
from bg.game import (
    GameState,
    apply_move,
    enumerate_turn_sequences,
    legal_moves_for_die,
)


def test_initial_position_has_fifteen_checkers_per_side():
    state = GameState.initial()
    state.validate()
    assert sum(max(value, 0) for value in state.board) == 15
    assert sum(max(-value, 0) for value in state.board) == 15


def test_bar_has_priority_over_board_moves():
    state = GameState(
        board=[0] * 24,
        bar=[1, 0],
        off=[0, 15],
        current_player=WHITE,
    )
    state.board[18] = 14
    state.validate()

    assert legal_moves_for_die(state, WHITE, 1) == [(BAR, 1)]
    apply_move(state, WHITE, (BAR, 1))
    assert state.bar[WHITE] == 0
    assert state.board[0] == 1


def test_single_checker_hit_goes_to_opponents_bar():
    state = GameState(board=[0] * 24, off=[14, 14], current_player=WHITE)
    state.board[0] = 1
    state.board[1] = -1
    state.validate()

    apply_move(state, WHITE, (0, 1))
    assert state.board[0] == 0
    assert state.board[1] == 1
    assert state.bar[BLACK] == 1


def test_bearing_off_is_forbidden_until_all_checkers_reach_home():
    state = GameState(board=[0] * 24, off=[13, 15], current_player=WHITE)
    state.board[17] = 1  # one checker is still outside White's home board
    state.board[18] = 1
    state.validate()

    assert (18, 6) not in legal_moves_for_die(state, WHITE, 6)
    assert (17, 6) in legal_moves_for_die(state, WHITE, 6)

    # Once the outside checker enters the home board, bearing off becomes
    # legal on a subsequent decision in the same turn.
    apply_move(state, WHITE, (17, 6))
    assert (18, 6) in legal_moves_for_die(state, WHITE, 6)


def test_oversized_bearing_off_uses_furthest_checker_rule():
    state = GameState(board=[0] * 24, off=[13, 15], current_player=WHITE)
    state.board[18] = 1
    state.board[20] = 1
    state.validate()

    # The checker on point 20 cannot use an oversized six while point 18 is
    # still farther from the exit.
    assert (20, 6) not in legal_moves_for_die(state, WHITE, 6)
    assert (18, 6) in legal_moves_for_die(state, WHITE, 6)

    state = GameState(board=[0] * 24, off=[14, 15], current_player=WHITE)
    state.board[20] = 1
    state.validate()
    assert (20, 6) in legal_moves_for_die(state, WHITE, 6)


def test_only_playable_die_is_kept_when_other_die_is_blocked():
    state = GameState(board=[0] * 24, off=[14, 13], current_player=WHITE)
    state.board[18] = 1
    state.board[19] = -2
    state.validate()

    sequences = enumerate_turn_sequences(state, (1, 6))
    assert sequences == [((18, 6),)]


def test_if_only_one_of_two_dice_can_be_used_the_higher_is_forced():
    state = GameState(board=[0] * 24, off=[14, 13], current_player=WHITE)
    state.board[0] = 1
    state.board[3] = -2
    state.validate()

    # Both 1 and 2 can move point 0 individually, but after either move the
    # other move would land on the blocked point 3.
    assert enumerate_turn_sequences(state, (1, 2)) == [((0, 2),)]


def test_a_normal_roll_uses_both_dice_when_possible():
    state = GameState.initial(starting_player=WHITE)
    sequences = enumerate_turn_sequences(state, (1, 2))
    assert sequences
    assert {len(sequence) for sequence in sequences} == {2}
