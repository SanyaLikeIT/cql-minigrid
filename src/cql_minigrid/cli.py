import argparse
import json
import platform


def main():
    parser = argparse.ArgumentParser(description="Team 12 offline MiniGrid experiments")
    parser.add_argument("command", choices=["doctor", "env-smoke"])
    args = parser.parse_args()
    if args.command == "doctor":
        import gymnasium
        import minigrid
        import numpy
        import stable_baselines3
        import torch

        print(json.dumps({
            "python": platform.python_version(), "torch": torch.__version__,
            "gymnasium": gymnasium.__version__, "minigrid": minigrid.__version__,
            "numpy": numpy.__version__, "stable_baselines3": stable_baselines3.__version__,
            "device": "cpu",
        }, indent=2))
    elif args.command == "env-smoke":
        from .env import make_env, observation_spec
        env = make_env()
        env.reset(seed=12)
        total = 0.0
        for _ in range(256):
            _, reward, terminated, truncated, info = env.step(2)
            total += reward
            if terminated or truncated:
                break
        env.close()
        print(json.dumps({"observation": observation_spec(), "return": total,
                          "outcome": info["outcome"]}, indent=2))
