import numpy as np
import pytest
import torch

from cql_minigrid.collector import collect_dataset
from cql_minigrid.offline import NetworkPolicy, td_targets, train_offline
from cql_minigrid.rollout import random_policy


def test_finite_horizon_and_continuing_time_limit_targets():
    reward = torch.tensor([1., 2., 3., 4.])
    next_value = torch.tensor([10., 20., 30., 40.])
    terminated = torch.tensor([False, True, False, True])
    truncated = torch.tensor([False, False, True, True])
    torch.testing.assert_close(td_targets(reward, next_value, terminated, truncated, .9),
                               torch.tensor([10., 2., 3., 4.]))
    torch.testing.assert_close(td_targets(reward, next_value, terminated, truncated, .9, True),
                               torch.tensor([10., 2., 30., 4.]))


@pytest.mark.parametrize('method', ['dqn', 'cql', 'bc'])
def test_resume_matches_uninterrupted_updates(tmp_path, monkeypatch, method):
    dataset = tmp_path / 'random.npz'
    collect_dataset(dataset, {'random': random_policy}, {'random': 1.}, target=32,
                    episode_seed_start=1000000, policy_seed_start=120,
                    source_hashes={'random': 'uniform-v1'}, source_audit={})
    # Updating saved transitions must never reset or step an environment.
    import cql_minigrid.env as env_module
    monkeypatch.setattr(env_module, 'make_env', lambda **kw: (_ for _ in ()).throw(AssertionError('offline environment access')))
    config = {'hidden': [16,16], 'updates': 4, 'batch_size': 8, 'learning_rate': .001,
              'gamma': .99, 'bootstrap_on_truncation': False, 'target_interval': 2,
              'checkpoint_interval': 2, 'log_interval': 1, 'alpha': 1.0}
    uninterrupted, resumed = tmp_path / 'full.pt', tmp_path / 'resume.pt'
    train_offline(dataset, uninterrupted, config, method=method, seed=12)
    train_offline(dataset, resumed, config, method=method, seed=12, stop_after=2)
    train_offline(dataset, resumed, config, method=method, seed=12, resume=True)
    a = torch.load(uninterrupted, weights_only=False)
    b = torch.load(resumed, weights_only=False)
    for name in a['model']:
        assert torch.equal(a['model'][name], b['model'][name])
    assert a['history'] == b['history']
    policy = NetworkPolicy(resumed)
    obs = np.zeros(a['observation']['shape'], dtype=np.float32)
    assert policy(obs, np.random.default_rng(12)) in [0,1,2]
