import numpy as np
import pytest
from cql_minigrid.env import compatible_observation_spec, make_env, observation_spec


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


def test_named_elapsed_steps_and_proposal_timeout():
    env = make_env(n_obstacles=0)
    obs, info = env.reset(seed=12)
    component = observation_spec()['components']['elapsed_steps']
    assert component['index'] == 836 and component['normalizer'] == 256
    assert env.elapsed_steps == info['elapsed_steps'] == 0
    for step in range(1, 257):
        obs, reward, terminated, truncated, info = env.step(0)
        assert reward == 0 and not terminated
        assert env.elapsed_steps == info['elapsed_steps'] == step
        assert obs[component['index']] == step / 256
        assert truncated == (step == 256)
    assert info['outcome'] == 'timeout'
    env.close()


@pytest.mark.parametrize('prior_steps', [0, 255])
def test_proposal_goal_reward_and_ending_at_256(prior_steps):
    env = make_env(n_obstacles=0)
    env.reset(seed=12)
    env.unwrapped.agent_pos = (5, 6)
    env.unwrapped.agent_dir = 0
    env.unwrapped.step_count = prior_steps
    _, reward, terminated, truncated, info = env.step(2)
    assert reward == pytest.approx(1 - 0.9 * (prior_steps + 1) / 256)
    assert terminated and truncated == (prior_steps == 255)
    assert info['outcome'] == 'success' and info['elapsed_steps'] == prior_steps + 1
    env.close()


def test_proposal_wall_collision_at_256():
    env = make_env(n_obstacles=0)
    env.reset(seed=12)
    env.unwrapped.agent_dir = 2
    env.unwrapped.step_count = 255
    _, reward, terminated, truncated, info = env.step(2)
    assert reward == -1 and terminated and truncated
    assert info['outcome'] == 'collision' and info['elapsed_steps'] == 256
    env.close()


def test_legacy_pilot_observation_schema_is_compatible():
    legacy = {'env_id': 'MiniGrid-Dynamic-Obstacles-8x8-v0',
              'version': 'full-symbolic-direction-time-v1', 'shape': [837],
              'max_steps': 256, 'actions': ['left', 'right', 'forward'],
              'encoding': 'object one-hot, color/5, state/3, direction one-hot, elapsed/max_steps'}
    assert compatible_observation_spec(legacy)
    assert not compatible_observation_spec({**legacy, 'max_steps': 128})
