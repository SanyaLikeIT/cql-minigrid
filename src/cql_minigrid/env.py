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

    def observation(self, obs):
        image = obs['image']
        cells = np.concatenate((self.eye[image[..., 0]],
                                image[..., 1:2].astype(np.float32) / 5.0,
                                image[..., 2:3].astype(np.float32) / 3.0), axis=-1)
        direction = np.eye(4, dtype=np.float32)[self.unwrapped.agent_dir]
        elapsed = np.array([self.unwrapped.step_count / self.unwrapped.max_steps], np.float32)
        return np.concatenate((cells.ravel(), direction, elapsed)).astype(np.float32)

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
        return self.observation(obs), reward, terminated, truncated, {**info, 'outcome': outcome}


def make_env(*, max_steps=256, render_mode=None, **kwargs):
    return Navigation(gym.make(ENV_ID, max_steps=max_steps, render_mode=render_mode, **kwargs))


def observation_spec():
    env = make_env()
    result = {'env_id': ENV_ID, 'version': OBSERVATION_VERSION,
              'shape': list(env.observation_space.shape), 'max_steps': 256,
              'actions': ['left', 'right', 'forward'],
              'encoding': 'object one-hot, color/5, state/3, direction one-hot, elapsed/max_steps'}
    env.close()
    return result
