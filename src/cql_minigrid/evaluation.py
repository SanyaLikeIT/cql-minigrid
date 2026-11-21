"""Held-out metrics and short replay GIFs using the original environment rewards."""
from pathlib import Path
import math
import os

import numpy as np
from PIL import Image
from stable_baselines3 import PPO

from .common import seed_everything, sha256, write_json
from .env import make_env, observation_spec
from .offline import NetworkPolicy
from .rollout import ppo_policy, random_policy, run_episodes


def load_policy(method, checkpoint=None):
    if method == 'random':
        return random_policy, {'method': 'random'}
    if checkpoint is None:
        raise ValueError('A checkpoint is required for this policy')
    if method == 'ppo':
        model = PPO.load(checkpoint, device='cpu')
        return ppo_policy(model), {'method': 'ppo', 'timesteps': int(model.num_timesteps)}
    policy = NetworkPolicy(checkpoint)
    if policy.identity['method'] != method:
        raise ValueError('Checkpoint method does not match the requested policy')
    return policy, policy.identity


def evaluate(method, checkpoint, seeds, output):
    seed_everything(0)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Evaluation seeds must be nonempty and unique')
    policy, identity = load_policy(method, checkpoint)
    report = {**run_episodes(policy, seeds), 'seeds': seeds, 'policy': identity,
              'checkpoint': str(checkpoint) if checkpoint else None,
              'checkpoint_sha256': sha256(checkpoint) if checkpoint else None,
              'observation': observation_spec()}
    write_json(output, report)
    return report


def export_gif(policy, path, seed, *, fps=8, max_frames=80):
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    env = make_env(render_mode='rgb_array')
    obs, _ = env.reset(seed=int(seed))
    rng = np.random.default_rng(int(seed))
    frames = [Image.fromarray(env.render())]
    total = 0.0
    for step in range(1, 257):
        obs, reward, terminated, truncated, info = env.step(policy(obs, rng))
        total += float(reward)
        frames.append(Image.fromarray(env.render()))
        if terminated or truncated:
            break
    env.close()
    stride = max(1, math.ceil(len(frames) / max_frames))
    selected = frames[::stride]
    if (len(frames) - 1) % stride:
        selected.append(frames[-1])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected[0].save(path, save_all=True, append_images=selected[1:],
                     duration=round(1000 / fps), loop=0, optimize=True)
    metadata = {'seed': int(seed), 'steps': step, 'return': total, 'outcome': info['outcome'],
                'terminated': bool(terminated), 'truncated': bool(truncated),
                'frame_stride': stride, 'fps': fps, 'gif_sha256': sha256(path)}
    write_json(path.with_suffix('.json'), metadata)
    return metadata
