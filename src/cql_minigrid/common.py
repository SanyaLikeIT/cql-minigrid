import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def resource_path(path, config_path=None) -> Path:
    """Resolve an input without changing the configuration stored in reports."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    if config_path is None and path.exists():
        return path
    bases = []
    if config_path is not None:
        config = Path(config_path).resolve()
        bases.append(config.parent)
        for parent in config.parents:
            if (parent / 'pyproject.toml').is_file() and (parent / 'src/cql_minigrid').is_dir():
                bases.append(parent)
                break
    package = Path(__file__).resolve().parent
    bases.append(package.parent.parent if package.parent.name == 'src' else package.parent)
    for base in bases:
        candidate = base / path
        if candidate.exists():
            return path if path.exists() and path.resolve() == candidate.resolve() else candidate
    return path


def seed_everything(seed: int, threads: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def write_json(path, value) -> None:
    """Replace a report atomically so interrupted writes cannot corrupt it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")
        tmp = out.name
    os.replace(tmp, path)


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
