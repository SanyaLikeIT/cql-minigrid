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
