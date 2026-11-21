"""Seeded episodes shared by source audits and final evaluation."""
import numpy as np

from .env import make_env


def random_policy(obs, rng):
    return int(rng.integers(3))


def ppo_policy(model):
    def act(obs, rng):
        return int(model.predict(obs, deterministic=True)[0])
    act.training_timesteps = int(model.num_timesteps)
    return act


def summarize(episodes):
    lengths = [e["steps"] for e in episodes if e["outcome"] == "success"]
    return {
        "episodes": len(episodes),
        **{f"{outcome}_rate": float(np.mean([e["outcome"] == outcome for e in episodes]))
           for outcome in ["success", "collision", "timeout"]},
        "return_mean": float(np.mean([e["return"] for e in episodes])),
        "successful_steps_mean": float(np.mean(lengths)) if lengths else None,
    }


def run_episodes(policy, seeds, *, max_steps=256):
    env = make_env(max_steps=max_steps)
    episodes = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        rng = np.random.default_rng(int(seed))
        total = 0.0
        for step in range(1, max_steps + 1):
            obs, reward, terminated, truncated, info = env.step(policy(obs, rng))
            total += float(reward)
            if terminated or truncated:
                episodes.append({"seed": int(seed), "steps": step, "return": total,
                                 "outcome": info["outcome"], "terminated": bool(terminated),
                                 "truncated": bool(truncated)})
                break
    env.close()
    return {"summary": summarize(episodes), "episodes": episodes}
