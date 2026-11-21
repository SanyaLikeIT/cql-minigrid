import argparse
import json
import platform


def main():
    parser = argparse.ArgumentParser(description="Team 12 offline MiniGrid experiments")
    parser.add_argument("command", choices=["doctor"])
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
