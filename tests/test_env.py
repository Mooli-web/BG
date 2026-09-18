import numpy as np

from bg.constants import ACTION_SIZE, OBSERVATION_SIZE, PASS_ACTION
from bg.env import BackgammonEnv


def test_environment_shapes_and_masks_are_valid():
    env = BackgammonEnv(seed=2)
    observation = env.reset(starting_player=0)
    mask = env.action_mask()
    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert mask.shape == (ACTION_SIZE,)
    assert mask.dtype == np.bool_
    assert mask.any()


def test_environment_can_play_many_random_games_without_invalid_state():
    env = BackgammonEnv(seed=10)
    for game in range(5):
        env.reset(starting_player=game % 2)
        for _ in range(500):
            action = int(np.flatnonzero(env.action_mask())[0])
            _, reward, done, info = env.step(action)
            assert np.isfinite(reward)
            assert info["player"] in (0, 1)
            if done:
                assert info["winner"] in (0, 1)
                break
        else:
            raise AssertionError("deterministic legal games should finish within 500 decisions")


def test_pass_is_available_when_all_entry_points_are_blocked():
    env = BackgammonEnv(seed=1)
    env.reset(starting_player=0)
    # Make a valid position with White on the bar and both entry points closed
    # for the currently rolled dice by replacing the private test state.  The
    # next line uses the public mask/step path, not an invalid action.
    env.state.bar[0] = 1
    env.state.board[0] = -2
    env.state.board[1] = -2
    env.state.board[2] = -2
    env.state.dice = (1, 2)
    from bg.game import enumerate_turn_sequences

    env._turn_sequences = enumerate_turn_sequences(env.state, env.state.dice)
    assert env.action_mask()[PASS_ACTION]
    _, reward, done, _ = env.step(PASS_ACTION)
    assert reward == 0.0
    assert not done
