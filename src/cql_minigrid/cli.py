import argparse
import json
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Team 12 offline MiniGrid experiments")
    parser.add_argument("command", choices=["doctor", "env-smoke", "ppo", "collect", "train"])
    parser.add_argument("--config", default="configs/main.json")
    parser.add_argument("--root", default="artifacts/main")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--transitions", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--method", choices=["dqn"], default="dqn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
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
    elif args.command in ["ppo", "collect"]:
        from .collector import collect, train_ppo
        config = json.loads(Path(args.config).read_text())
        if args.command == "ppo":
            train_ppo(config["ppo"], Path(args.root) / "ppo", resume=args.resume)
        else:
            results = collect(config, args.root, target=args.transitions)
            print(json.dumps({name: {"transitions": value["transitions"], "episodes": value["episodes"]}
                              for name, value in results.items()}, indent=2))
    elif args.command == "train":
        from .offline import train_offline
        config = json.loads(Path(args.config).read_text())
        dataset = Path(args.dataset) if args.dataset else Path(args.root) / "datasets/strong.npz"
        output = Path(args.output) if args.output else Path(args.root) / "runs" / dataset.stem / args.method / f"seed-{args.seed}.pt"
        train_offline(dataset, output, config["offline"], seed=args.seed, resume=args.resume)
