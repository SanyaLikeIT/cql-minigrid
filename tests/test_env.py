import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from cql_minigrid.env import make_env


def test_repeatable_full_observations():
    a, b = make_env(), make_env()
    x, _ = a.reset(seed=120)
    y, _ = b.reset(seed=120)
    assert a.observation_space.contains(x)
    np.testing.assert_array_equal(x, y)
    for action in [0, 1, 2, 0, 2]:
        x, r, t, u, i = a.step(action)
        y, s, v, w, j = b.step(action)
        np.testing.assert_array_equal(x, y)
        assert (r, t, u, i) == (s, v, w, j)
        if t or u:
            break
    a.close(); b.close()


def test_endings_and_priority_at_time_limit():
    env = make_env(max_steps=1, n_obstacles=0)
    env.reset(seed=0)
    _, r, t, u, info = env.step(0)
    assert (r, t, u, info['outcome']) == (0, False, True, 'timeout')
    env.reset(seed=0)
    env.unwrapped.agent_dir = 2  # face the wall at x=0
    _, r, t, u, info = env.step(2)
    assert (r, t, u, info['outcome']) == (-1, True, True, 'collision')
    env.reset(seed=0)
    env.unwrapped.agent_pos = (5, 6)
    env.unwrapped.agent_dir = 0
    _, r, t, u, info = env.step(2)
    assert r > 0 and t and u and info['outcome'] == 'success'
    env.close()


def test_action_contract_and_elapsed_time():
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert env.action_space.n == 3 and obs[-1] == 0
    obs, *_ = env.step(0)
    assert obs[-1] == 1 / 256
    with pytest.raises(ValueError):
        env.step(3)
    env.close()
