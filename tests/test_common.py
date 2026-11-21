import numpy as np
import torch

from cql_minigrid.common import seed_everything, sha256, write_json


def test_seed_repeats_numpy_and_torch():
    seed_everything(12)
    expected = (np.random.random(), torch.rand(3))
    seed_everything(12)
    assert np.random.random() == expected[0]
    assert torch.equal(torch.rand(3), expected[1])


def test_report_is_json_and_has_stable_hash(tmp_path):
    path = tmp_path / "report.json"
    write_json(path, {"seed": 12})
    first = sha256(path)
    write_json(path, {"seed": 12})
    assert sha256(path) == first
