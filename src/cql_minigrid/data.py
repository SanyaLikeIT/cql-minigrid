import json
from pathlib import Path

import numpy as np

from .common import sha256
from .env import compatible_observation_spec, observation_spec


def load_dataset(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["sha256"] != sha256(path):
        raise ValueError("Dataset checksum mismatch")
    if not compatible_observation_spec(metadata["observation"]):
        raise ValueError("Dataset observation interface differs")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    required = {"observation", "action", "reward", "next_observation", "terminated", "truncated",
                "episode_id", "source", "seed", "policy_seed"}
    if not required.issubset(arrays):
        raise ValueError("Missing transition fields")
    n = len(arrays["action"])
    if n != metadata["transitions"] or any(len(value) != n for value in arrays.values()):
        raise ValueError("Transition counts differ")
    for key in ["observation", "next_observation"]:
        if arrays[key].shape != (n, *observation_spec()["shape"]) or not np.isfinite(arrays[key]).all():
            raise ValueError("Invalid observation array")
    if not np.isin(arrays["action"], [0, 1, 2]).all():
        raise ValueError("Invalid recorded action")
    if any(arrays[key].dtype != np.bool_ for key in ["terminated", "truncated"]):
        raise ValueError("Ending flags must be Boolean")
    return arrays, metadata
