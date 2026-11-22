import numpy as np
import pytest

from cql_minigrid.collector import collect_dataset, save_npz
from cql_minigrid.common import sha256, write_json
from cql_minigrid.data import load_dataset
from cql_minigrid.rollout import random_policy


def test_complete_episode_dataset_can_resume_without_changes(tmp_path):
    path = tmp_path / "random.npz"
    kwargs = dict(policies={"random": random_policy}, weights={"random": 1.0}, target=32,
                  episode_seed_start=1000000, policy_seed_start=120,
                  source_hashes={"random": "uniform-v1"}, source_audit={}, shard_episodes=2)
    meta = collect_dataset(path, **kwargs)
    arrays, _ = load_dataset(path)
    assert meta["transitions"] >= 32
    assert arrays["terminated"][-1] or arrays["truncated"][-1]
    with np.load(path) as stored:
        assert stored['elapsed_steps'].dtype == np.int32
        assert stored['next_elapsed_steps'].dtype == np.int32
    np.testing.assert_array_equal(arrays['elapsed_steps'], arrays['observation'][:, -1] * 256)
    np.testing.assert_array_equal(arrays['next_elapsed_steps'], arrays['elapsed_steps'] + 1)
    resumed = collect_dataset(path, **kwargs)
    assert meta == resumed
    for eid in np.unique(arrays["episode_id"]):
        rows = arrays["episode_id"] == eid
        assert np.unique(arrays["source"][rows]).size == 1
        assert np.unique(arrays["seed"][rows]).size == 1
    with pytest.raises(ValueError, match="changed"):
        collect_dataset(path, **{**kwargs, "source_hashes": {"random": "different"}})
    with path.open("ab") as file:
        file.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(path)


def test_legacy_pilot_counter_derivation_and_counter_validation(tmp_path):
    path = tmp_path / 'legacy.npz'
    meta = collect_dataset(path, {'random': random_policy}, {'random': 1.0}, target=16,
                           episode_seed_start=1000000, policy_seed_start=120,
                           source_hashes={'random': 'uniform-v1'}, source_audit={})
    with np.load(path) as stored:
        arrays = {key: stored[key] for key in stored.files if key not in ['elapsed_steps', 'next_elapsed_steps']}
    save_npz(path, arrays)
    meta['sha256'] = sha256(path)
    meta['observation'].pop('components')
    meta.pop('recorded_fields')
    write_json(path.with_suffix('.json'), meta)
    original = sha256(path)
    loaded, _ = load_dataset(path)
    assert sha256(path) == original
    np.testing.assert_array_equal(loaded['next_elapsed_steps'], loaded['elapsed_steps'] + 1)
    loaded['next_elapsed_steps'][0] = 255
    save_npz(path, loaded)
    meta['sha256'] = sha256(path)
    write_json(path.with_suffix('.json'), meta)
    with pytest.raises(ValueError, match='next_elapsed_steps'):
        load_dataset(path)


def test_transition_quotas_mix_different_episode_lengths_and_resume(tmp_path, monkeypatch):
    from cql_minigrid import collector
    class DifferentLengths:
        elapsed_steps = 0
        def reset(self, seed):
            self.elapsed_steps = 0
            return np.zeros(1), {}
        def step(self, action):
            self.elapsed_steps += 1
            done = self.elapsed_steps >= (31 if action == 0 else 2)
            return np.zeros(1), 0., done, False, {'elapsed_steps':self.elapsed_steps,'outcome':'collision'}
        def close(self):
            pass
    monkeypatch.setattr(collector, 'make_env', DifferentLengths)
    kwargs = dict(policies={'long':lambda obs,rng:0, 'short':lambda obs,rng:1},
                  weights={'long':.5,'short':.5}, target=2000, episode_seed_start=1000000,
                  policy_seed_start=120, source_hashes={},source_audit={},shard_episodes=7)
    quota = collect_dataset(tmp_path/'quota.npz', **kwargs, sampling_unit='transitions')
    assert 2000 <= quota['transitions'] < 2031
    assert abs(quota['sources']['long']['transitions']/quota['transitions']-.5) < .02
    assert quota == collect_dataset(tmp_path/'quota.npz', **kwargs, sampling_unit='transitions')
    # Reconstruct a genuinely interrupted collection from its first saved shard.
    from pathlib import Path
    import json, shutil
    interrupted = tmp_path/'interrupted.npz'
    parts = interrupted.with_suffix('.parts');parts.mkdir()
    manifest = json.loads((tmp_path/'quota.parts/progress.json').read_text())
    manifest['parts'] = manifest['parts'][:1]
    manifest['episodes'] = manifest['episodes'][:7]
    manifest['transitions'] = sum(e['steps'] for e in manifest['episodes'])
    shutil.copy2(tmp_path/'quota.parts/part-00000.npz',parts/'part-00000.npz')
    write_json(parts/'progress.json',manifest)
    resumed = collect_dataset(interrupted, **kwargs, sampling_unit='transitions')
    assert resumed['sha256'] == quota['sha256']
    episode = collect_dataset(tmp_path/'episode.npz', **kwargs)
    assert episode['sources']['long']['transitions']/episode['transitions'] > .85
    with pytest.raises(ValueError,match='changed'):
        collect_dataset(tmp_path/'quota.npz', **kwargs, sampling_unit='episodes')
