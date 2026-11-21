import numpy as np

from cql_minigrid.env import make_env
from cql_minigrid.shaping import CollectorReward


def test_distance_terms_telescope_and_idle_steps_have_a_cost():
    env = CollectorReward(make_env(max_steps=3, n_obstacles=0), scale=0.2, step_cost=0.005)
    obs, _ = env.reset(seed=12)
    initial = env.potential()
    difference = 0.0
    for step in range(3):
        obs, reward, terminated, truncated, info = env.step(0)
        difference += reward - info['original_reward']
    np.testing.assert_allclose(difference, 0.2 * (env.potential() - initial) - 3 * 0.005)
    assert terminated and not truncated and info['original_truncated']
    assert obs[-1] == 1.0
    env.close()


def test_forward_progress_does_not_hide_original_reward():
    env = CollectorReward(make_env(n_obstacles=0), scale=0.2, step_cost=0.005)
    env.reset(seed=12)
    _, reward, _, _, info = env.step(2)
    np.testing.assert_allclose(reward, 0.195)
    assert info['original_reward'] == 0.0
    env.close()
