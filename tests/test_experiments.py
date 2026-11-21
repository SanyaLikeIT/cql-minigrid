import json
import pytest
from cql_minigrid.collector import collect_dataset
from cql_minigrid.experiments import run_matrix
from cql_minigrid.rollout import random_policy


def test_final_matrix_rejects_a_pilot_dataset_budget(tmp_path):
    collect_dataset(tmp_path/'datasets/strong.npz',{'random':random_policy},{'random':1.0},
                    target=16,episode_seed_start=1000000,policy_seed_start=120,
                    source_hashes={'random':'uniform-v1'},source_audit={})
    config=json.load(open('configs/main.json'))
    assert config['datasets']['transitions']==200000
    with pytest.raises(ValueError,match='transition budget'):
        run_matrix(config,tmp_path)
