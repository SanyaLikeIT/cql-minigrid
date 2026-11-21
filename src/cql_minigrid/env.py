"""One observation and outcome contract for collection, training, and evaluation."""
import gymnasium as gym
from gymnasium import spaces
import minigrid  # noqa: F401 -- registers the environments
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.wrappers import FullyObsWrapper
import numpy as np

ENV_ID = 'MiniGrid-Dynamic-Obstacles-8x8-v0'
OBSERVATION_VERSION = 'full-symbolic-direction-time-v1'


class Navigation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(FullyObsWrapper(env))
        self.n_objects = len(OBJECT_TO_IDX)
        size = self.unwrapped.width * self.unwrapped.height * (self.n_objects + 2) + 5
        self.observation_space = spaces.Box(0.0, 1.0, (size,), np.float32)
        self.action_space = spaces.Discrete(3)
        self.eye = np.eye(self.n_objects, dtype=np.float32)

    @property
    def elapsed_steps(self):
        return int(self.unwrapped.step_count)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return obs, {**info, 'elapsed_steps': self.elapsed_steps}

    def observation(self, obs):
        image = obs['image']
        cells = np.concatenate((self.eye[image[..., 0]],
                                image[..., 1:2].astype(np.float32) / 5.0,
                                image[..., 2:3].astype(np.float32) / 3.0), axis=-1)
        direction = np.eye(4, dtype=np.float32)[self.unwrapped.agent_dir]
        elapsed_steps = np.array([self.elapsed_steps / self.unwrapped.max_steps], np.float32)
        return np.concatenate((cells.ravel(), direction, elapsed_steps)).astype(np.float32)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid navigation action: {action}')
        obs, reward, terminated, truncated, info = self.env.step(action)
        outcome = None
        if terminated and reward < 0:
            outcome = 'collision'
        elif terminated and reward > 0:
            outcome = 'success'
        elif truncated:
            outcome = 'timeout'
        elif terminated:
            raise RuntimeError('Unclassified terminal transition')
        return self.observation(obs), reward, terminated, truncated, {
            **info, 'outcome': outcome, 'elapsed_steps': self.elapsed_steps}


def make_env(*, max_steps=256, render_mode=None, **kwargs):
    return Navigation(gym.make(ENV_ID, max_steps=max_steps, render_mode=render_mode, **kwargs))


def observation_spec():
    result = {'env_id': ENV_ID, 'version': OBSERVATION_VERSION,
              'shape': [8 * 8 * (len(OBJECT_TO_IDX) + 2) + 5], 'max_steps': 256,
              'actions': ['left', 'right', 'forward'],
              'encoding': 'object one-hot, color/5, state/3, direction one-hot, elapsed/max_steps',
              'components': {
                  'grid': {'start': 0, 'stop': 832, 'shape': [8, 8, 13]},
                  'direction': {'start': 832, 'stop': 836, 'encoding': 'one-hot'},
                  'elapsed_steps': {'index': 836, 'raw_dtype': 'int32', 'raw_range': [0, 256],
                                    'encoded_dtype': 'float32', 'normalizer': 256}}}
    return result


def compatible_observation_spec(saved):
    """The original pilot already encoded elapsed_steps / 256 at index 836."""
    current = observation_spec()
    core = ['env_id', 'version', 'shape', 'max_steps', 'actions', 'encoding']
    return (all(saved.get(key) == current[key] for key in core)
            and ('components' not in saved or saved['components'] == current['components']))
