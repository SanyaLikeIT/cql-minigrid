"""Collector-only potential shaping; recorded/evaluation rewards stay original."""
import gymnasium as gym


class CollectorReward(gym.Wrapper):
    def __init__(self, env, *, scale, gamma):
        super().__init__(env)
        self.scale, self.gamma = scale, gamma
        self.previous = None

    def potential(self):
        env = self.unwrapped
        x, y = env.agent_pos
        distance = abs(env.width - 2 - x) + abs(env.height - 2 - y)
        return -distance / (env.width + env.height - 6)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous = self.potential()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        following = 0.0 if terminated or truncated else self.potential()
        shaped = reward + self.scale * (self.gamma * following - self.previous)
        self.previous = following
        # Elapsed time is observed: this is a finite-horizon terminal state.
        # Avoid SB3's automatic value bootstrap at a TimeLimit truncation.
        return obs, shaped, terminated or truncated, False, {
            **info, "original_reward": reward, "original_terminated": terminated,
            "original_truncated": truncated,
        }
