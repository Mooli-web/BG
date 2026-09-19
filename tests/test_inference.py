import numpy as np
import torch

from bg.env import BackgammonEnv
from bg.inference import choose_action
from bg.model import BackgammonActorCritic, BackgammonValueNetwork


def test_value_guided_search_returns_a_legal_action():
    env = BackgammonEnv(seed=4)
    env.reset(starting_player=0)
    model = BackgammonActorCritic()
    action = choose_action(model, env, torch.device("cpu"), search_samples=1)
    assert action in np.flatnonzero(env.action_mask())


def test_value_network_uses_search_even_without_a_policy_head():
    env = BackgammonEnv(seed=5)
    env.reset(starting_player=1)
    model = BackgammonValueNetwork()
    action = choose_action(model, env, torch.device("cpu"), search_samples=0)
    assert action in np.flatnonzero(env.action_mask())
