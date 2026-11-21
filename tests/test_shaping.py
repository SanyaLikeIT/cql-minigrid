import numpy as np

from cql_minigrid.env import make_env
from cql_minigrid.shaping import CollectorReward


def test_discounted_potential_terms_telescope_at_timeout():
    env = CollectorReward(make_env(max_steps=3, n_obstacles=0), scale=0.5, gamma=0.99)
    obs, _ = env.reset(seed=12)
    initial = env.potential()
    difference = 0.0
    for step in range(3):
        obs, reward, terminated, truncated, info = env.step(0)
        difference += 0.99 ** step * (reward - info['original_reward'])
    np.testing.assert_allclose(difference, -0.5 * initial)
    assert terminated and not truncated and info['original_truncated']
    assert obs[-1] == 1.0
    env.close()
