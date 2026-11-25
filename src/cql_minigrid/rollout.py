"""Seeded episodes shared by source audits and final evaluation."""
import numpy as np

from .env import make_env

STALL_THRESHOLD = 16
NAVIGATION_METRICS = ['stall_rate', 'max_stationary_steps_mean', 'turn_fraction_mean', 'steps_mean']


class NavigationStats:
    """Count stationary actions, including legal turns and obstacle waits."""
    def __init__(self):
        self.steps = self.turn_steps = self.stationary_steps = 0
        self.current_stationary = self.max_stationary_steps = 0

    def update(self, action, before, after):
        self.steps += 1
        self.turn_steps += int(action != 2)
        stationary = tuple(before) == tuple(after)
        self.stationary_steps += int(stationary)
        self.current_stationary = self.current_stationary + 1 if stationary else 0
        self.max_stationary_steps = max(self.max_stationary_steps, self.current_stationary)

    def report(self):
        return {'turn_steps': self.turn_steps, 'stationary_steps': self.stationary_steps,
                'max_stationary_steps': self.max_stationary_steps}


def random_policy(obs, rng):
    return int(rng.integers(3))


def ppo_policy(model):
    def act(obs, rng):
        return int(model.predict(obs, deterministic=True)[0])
    act.training_timesteps = int(model.num_timesteps)
    return act


def summarize(episodes):
    lengths = [e["steps"] for e in episodes if e["outcome"] == "success"]
    summary = {
        "episodes": len(episodes),
        **{f"{outcome}_rate": float(np.mean([e["outcome"] == outcome for e in episodes]))
           for outcome in ["success", "collision", "timeout"]},
        "return_mean": float(np.mean([e["return"] for e in episodes])),
        "successful_steps_mean": float(np.mean(lengths)) if lengths else None,
    }
    if episodes and all('max_stationary_steps' in e for e in episodes):
        summary.update(
            stall_rate=float(np.mean([e['max_stationary_steps'] >= STALL_THRESHOLD for e in episodes])),
            max_stationary_steps_mean=float(np.mean([e['max_stationary_steps'] for e in episodes])),
            turn_fraction_mean=float(np.mean([e['turn_steps'] / e['steps'] for e in episodes])),
            steps_mean=float(np.mean([e['steps'] for e in episodes])))
    return summary


def run_episodes(policy, seeds, *, max_steps=256):
    env = make_env(max_steps=max_steps)
    episodes = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        rng = np.random.default_rng(int(seed))
        total = 0.0
        stats = NavigationStats()
        for step in range(1, max_steps + 1):
            action = policy(obs, rng)
            before = tuple(env.unwrapped.agent_pos)
            obs, reward, terminated, truncated, info = env.step(action)
            stats.update(action, before, env.unwrapped.agent_pos)
            total += float(reward)
            if terminated or truncated:
                episodes.append({"seed": int(seed), "steps": step, "return": total,
                                 "elapsed_steps": info["elapsed_steps"],
                                 "outcome": info["outcome"], "terminated": bool(terminated),
                                 "truncated": bool(truncated), **stats.report()})
                break
    env.close()
    return {"summary": summarize(episodes), "episodes": episodes}
