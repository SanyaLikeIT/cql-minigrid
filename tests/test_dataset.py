import numpy as np
import pytest

from cql_minigrid.collector import collect_dataset
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
