"""Offline updates. Environment steps are confined to collection and evaluation."""
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import seed_everything, write_json
from .data import load_dataset


class ActionNetwork(nn.Module):
    def __init__(self, observation_dim, hidden=(128, 128), actions=3):
        super().__init__()
        layers = []
        widths = [observation_dim, *hidden, actions]
        for index, (before, after) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
            layers.append(nn.Linear(before, after))
            if index < len(widths) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        return self.layers(obs)


def td_targets(reward, next_values, terminated, truncated, gamma, bootstrap_on_truncation=False):
    done = terminated if bootstrap_on_truncation else (terminated | truncated)
    return reward + gamma * (~done).float() * next_values


def dqn_loss(q_values, action, target):
    return F.smooth_l1_loss(q_values.gather(1, action[:, None]).squeeze(1), target)


def save_checkpoint(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def train_offline(dataset, output, config, *, seed=0, resume=False, stop_after=None):
    seed_everything(seed)
    data, metadata = load_dataset(dataset)
    output = Path(output)
    identity = {'method': 'dqn', 'seed': seed, 'config': config, 'dataset_sha256': metadata['sha256']}
    rng = np.random.default_rng(seed)
    model = ActionNetwork(data['observation'].shape[1], config['hidden'])
    target_model = ActionNetwork(data['observation'].shape[1], config['hidden'])
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    update, history, seconds_before = 0, [], 0.0
    if output.exists():
        if not resume:
            raise FileExistsError('Checkpoint exists; use --resume or another output path')
        saved = torch.load(output, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Checkpoint dataset, seed, method, or config differs')
        model.load_state_dict(saved['model'])
        target_model.load_state_dict(saved['target_model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['batch_rng']
        torch.set_rng_state(saved['torch_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        update, history, seconds_before = saved['updates'], saved['history'], saved['seconds']
    tensors = {key: torch.from_numpy(data[key]) for key in
               ['observation', 'next_observation', 'action', 'reward', 'terminated', 'truncated']}
    started = time.perf_counter()
    end = min(config['updates'], stop_after) if stop_after is not None else config['updates']
    for update in range(update + 1, end + 1):
        idx = rng.integers(len(data['action']), size=config['batch_size'])
        batch = {key: value[idx] for key, value in tensors.items()}
        q = model(batch['observation'])
        with torch.no_grad():
            next_values = target_model(batch['next_observation']).max(dim=1).values
            target = td_targets(batch['reward'], next_values, batch['terminated'], batch['truncated'],
                                config['gamma'], config['bootstrap_on_truncation'])
        loss = dqn_loss(q, batch['action'], target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if update % config['target_interval'] == 0:
            target_model.load_state_dict(model.state_dict())
        if update == 1 or update % config['log_interval'] == 0 or update == end:
            history.append({'update': update, 'loss': float(loss.detach()), 'q_mean': float(q.detach().mean())})
        if update % config['checkpoint_interval'] == 0 or update == end:
            checkpoint = {'identity': identity, 'model': model.state_dict(),
                          'target_model': target_model.state_dict(), 'optimizer': optimizer.state_dict(),
                          'observation': metadata['observation'], 'updates': update, 'history': history,
                          'batch_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                          'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
                          'seconds': seconds_before + time.perf_counter() - started}
            save_checkpoint(output, checkpoint)
            write_json(output.with_suffix('.train.json'), {key: checkpoint[key] for key in
                       ['identity', 'observation', 'updates', 'history', 'seconds']})
            print(json.dumps({'method': identity['method'], 'seed': seed, 'updates': update,
                              'seconds': round(checkpoint['seconds'], 2)}), flush=True)
    return output


class NetworkPolicy:
    def __init__(self, checkpoint):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.model = ActionNetwork(saved['observation']['shape'][0], saved['identity']['config']['hidden'])
        self.model.load_state_dict(saved['model'])
        self.model.eval()

    def __call__(self, obs, rng):
        with torch.no_grad():
            return int(self.model(torch.as_tensor(obs, dtype=torch.float32)[None]).argmax(dim=1).item())
