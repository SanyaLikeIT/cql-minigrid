"""Held-out metrics and reproducible GIF/MP4 gameplay with original rewards."""
from pathlib import Path
import os

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageSequence
from stable_baselines3 import PPO

from .common import seed_everything, sha256, write_json
from .env import make_env, observation_spec
from .offline import NetworkPolicy
from .rollout import ppo_policy, random_policy, run_episodes

EVALUATION_VERSION = 'proposal-metrics-v1'
REPLAY_VERSION = 'gif-mp4-v1'


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
    report = {'evaluation_version': EVALUATION_VERSION, **run_episodes(policy, seeds), 'seeds': seeds, 'policy': identity,
              'checkpoint': str(checkpoint) if checkpoint else None,
              'checkpoint_sha256': sha256(checkpoint) if checkpoint else None,
              'observation': observation_spec()}
    write_json(output, report)
    return report


def export_replay(policy, path, seed, *, fps=8, max_frames=80):
    path = Path(path)
    kind = path.suffix.lower().lstrip('.')
    if kind not in ['gif', 'mp4'] or fps <= 0 or max_frames < 2:
        raise ValueError('Replay needs .gif/.mp4, positive fps, and at least two frames')
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    env = make_env(render_mode='rgb_array')
    obs, _ = env.reset(seed=int(seed))
    rng = np.random.default_rng(int(seed))
    frames = [Image.fromarray(env.render())]
    total = 0.0
    try:
        for step in range(1, 257):
            obs, reward, terminated, truncated, info = env.step(policy(obs, rng))
            total += float(reward)
            frames.append(Image.fromarray(env.render()))
            if terminated or truncated:
                break
    finally:
        env.close()
    indices = np.linspace(0, len(frames)-1, min(len(frames), max_frames), dtype=int).tolist()
    selected = [frames[index] for index in indices]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp' + path.suffix)
    try:
        if kind == 'gif':
            selected[0].save(temporary, save_all=True, append_images=selected[1:],
                             duration=round(1000/fps), loop=0, optimize=True)
            with Image.open(temporary) as gif:
                encoded_frames = gif.n_frames
                duration = sum(frame.info.get('duration', 0) for frame in ImageSequence.Iterator(gif))/1000
        else:
            writer = imageio_ffmpeg.write_frames(str(temporary), selected[0].size,
                fps=fps, codec='libx264', pix_fmt_out='yuv420p', quality=8,
                output_params=['-movflags', '+faststart', '-threads', '1'], ffmpeg_log_level='error', ffmpeg_timeout=30)
            try:
                writer.send(None)
                for frame in selected:
                    writer.send(np.asarray(frame.convert('RGB')).tobytes())
            finally:
                writer.close()
            encoded_frames, duration = imageio_ffmpeg.count_frames_and_secs(str(temporary))
            if encoded_frames != len(selected):
                raise RuntimeError('MP4 frame count differs from the recorded replay')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256(path)
    metadata = {'replay_version': REPLAY_VERSION, 'seed': int(seed), 'steps': step, 'elapsed_steps': info['elapsed_steps'],
                'return': total, 'outcome': info['outcome'], 'terminated': bool(terminated),
                'truncated': bool(truncated), 'format': kind, 'fps': fps, 'frame_indices': indices,
                'sampled_frames': len(selected), 'encoded_frames': int(encoded_frames),
                'duration_seconds': float(duration), 'file_sha256': digest, kind+'_sha256': digest}
    if kind == 'mp4':
        metadata.update(codec='h264', pixel_format='yuv420p', ffmpeg_version=imageio_ffmpeg.get_ffmpeg_version())
    write_json(path.with_suffix(path.suffix+'.json'), metadata)
    return metadata


def export_gif(policy, path, seed, **kwargs):
    if Path(path).suffix.lower() != '.gif':raise ValueError('GIF output must end in .gif')
    return export_replay(policy, path, seed, **kwargs)


def export_mp4(policy, path, seed, **kwargs):
    if Path(path).suffix.lower() != '.mp4':raise ValueError('MP4 output must end in .mp4')
    return export_replay(policy, path, seed, **kwargs)
