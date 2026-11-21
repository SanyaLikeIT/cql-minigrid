"""Collector-only distance rewards; recorded/evaluation rewards stay original."""
import gymnasium as gym


class CollectorReward(gym.Wrapper):
    def __init__(self, env, *, scale, step_cost):
        super().__init__(env)
        self.scale, self.step_cost = scale, step_cost
        self.previous = None

    def potential(self):
        env = self.unwrapped
        x, y = env.agent_pos
        distance = abs(env.width - 2 - x) + abs(env.height - 2 - y)
        return -distance

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous = self.potential()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        following = self.potential()
        shaped = reward + self.scale * (following - self.previous) - self.step_cost
        self.previous = following
        # Elapsed time is observed: this is a finite-horizon terminal state.
        # Avoid SB3's automatic value bootstrap at a TimeLimit truncation.
        return obs, shaped, terminated or truncated, False, {
            **info, "original_reward": reward, "original_terminated": terminated,
            "original_truncated": truncated,
        }
