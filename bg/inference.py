"""Inference helpers for the policy and a lightweight value-guided search.

The PPO policy is useful on its own, but a raw one-step policy can still make
obviously poor choices in a long-horizon game such as backgammon.  At play
time we optionally score every legal move with the value head after applying
that move.  This is not used during training, so training remains fast; it is
a small expectiminimax-style improvement for the human-facing player.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from .env import BackgammonEnv
from .model import BackgammonActorCritic, BackgammonValueNetwork

ModelLike = BackgammonActorCritic | BackgammonValueNetwork


SEARCH_GAMMA = 0.995


def _policy_action(
    model: BackgammonActorCritic,
    env: BackgammonEnv,
    device: torch.device,
    deterministic: bool,
) -> int:
    observation = torch.as_tensor(env.observation(), dtype=torch.float32, device=device)
    mask = torch.as_tensor(env.action_mask(), dtype=torch.bool, device=device)
    action, _, _ = model.choose_action(observation, mask, deterministic=deterministic)
    return int(action.item())


def value_of_observation(
    model: ModelLike,
    observation: np.ndarray | torch.Tensor,
    device: torch.device | str,
) -> float:
    """Return the model value for one observation."""
    device = torch.device(device)
    tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
    with torch.no_grad():
        if isinstance(model, BackgammonValueNetwork):
            value = model(tensor)
        else:
            _, value = model(tensor)
    return float(value.reshape(-1)[0].item())


def _batched_values(
    model: ModelLike,
    observations: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    if not observations:
        return np.empty(0, dtype=np.float32)
    batch = torch.as_tensor(np.stack(observations), dtype=torch.float32, device=device)
    with torch.no_grad():
        if isinstance(model, BackgammonValueNetwork):
            values = model(batch)
        else:
            _, values = model(batch)
    return values.detach().cpu().numpy()


def choose_action(
    model: ModelLike,
    env: BackgammonEnv,
    device: torch.device | str,
    deterministic: bool = True,
    search_samples: int = 0,
) -> int:
    """Choose an action, optionally with one-ply value-guided search.

    ``search_samples=0`` exactly uses the learned policy.  With a positive
    value, every legal action is applied on a copied environment and the value
    of the resulting position is averaged over several possible dice rolls.
    The policy logit remains a small prior, so search does not discard the
    policy's learned preferences when values are close.
    """
    device = torch.device(device)
    if isinstance(model, BackgammonValueNetwork):
        # A value-only model has no policy head; one-ply value search is its
        # action selector even when the caller passes search_samples=0.
        search_samples = max(1, search_samples)
    elif search_samples <= 0:
        return _policy_action(model, env, device, deterministic)

    legal_actions = np.flatnonzero(env.action_mask())
    if len(legal_actions) <= 1:
        return int(legal_actions[0])

    model.eval()
    actor = env.current_player
    if isinstance(model, BackgammonValueNetwork):
        # There is no learned action prior for TD-Gammon-style value models.
        legal_logits = np.zeros(len(legal_actions), dtype=np.float32)
    else:
        base_observation = torch.as_tensor(env.observation(), dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, _ = model(base_observation)
        legal_logits = logits[0, torch.as_tensor(legal_actions, device=device)].detach().cpu().numpy()
    prior_scale = 0.04
    legal_logits = (legal_logits - legal_logits.mean()) / (legal_logits.std() + 1e-6)

    pending: list[tuple[int, float, bool, int, np.ndarray]] = []
    # A local deterministic seed makes repeated UI redraws reproducible while
    # still giving different chance outcomes to different samples/actions.
    base_seed = (env.episode_steps + 1) * 1_000_003 + actor * 97
    for action_position, action in enumerate(legal_actions):
        for sample in range(search_samples):
            candidate = deepcopy(env)
            candidate.rng = np.random.default_rng(
                base_seed + int(action) * 101 + sample * 10_007
            )
            next_observation, reward, done, info = candidate.step(int(action))
            pending.append(
                (
                    int(action),
                    float(reward),
                    bool(done),
                    int(info["next_player"]),
                    next_observation,
                )
            )

    nonterminal_observations = [item[4] for item in pending if not item[2]]
    nonterminal_values = _batched_values(model, nonterminal_observations, device)
    value_index = 0
    scores: list[float] = []
    for action in legal_actions:
        action_scores: list[float] = []
        for item in pending:
            if item[0] != int(action):
                continue
            _, reward, done, next_player, _ = item
            if done:
                score = reward
            else:
                next_value = float(nonterminal_values[value_index])
                value_index += 1
                # The value head is from the player-to-move perspective. Flip
                # it after a completed turn because the opponent is to move.
                if next_player != actor:
                    next_value = -next_value
                score = reward + SEARCH_GAMMA * next_value
            action_scores.append(score)
        scores.append(float(np.mean(action_scores)))

    scores_array = np.asarray(scores, dtype=np.float32) + prior_scale * legal_logits
    if deterministic:
        return int(legal_actions[int(np.argmax(scores_array))])

    distribution = torch.distributions.Categorical(
        logits=torch.as_tensor(scores_array, dtype=torch.float32, device=device)
    )
    return int(legal_actions[int(distribution.sample().item())])
