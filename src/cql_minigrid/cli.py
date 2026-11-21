import argparse
import json
import os
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Team 12 offline MiniGrid experiments")
    parser.add_argument("command", choices=["doctor", "env-smoke", "ppo", "collect", "train", "evaluate", "replay", "experiments", "pipeline"])
    parser.add_argument("--config", default="configs/main.json")
    parser.add_argument("--root", default="artifacts/main")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--transitions", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--method", choices=["dqn", "cql", "bc", "ppo", "random"], default="dqn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--checkpoint")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reports")
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(Path(args.root).resolve() / ".matplotlib"))
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
        train_offline(dataset, output, config["offline"], method=args.method, seed=args.seed, resume=args.resume)
    elif args.command in ["evaluate", "replay"]:
        from .evaluation import evaluate, export_gif, load_policy
        from .common import seed_everything
        seed_everything(0)
        config = json.loads(Path(args.config).read_text())
        seeds = json.loads(Path(config["seeds_file"]).read_text())["evaluation"]
        checkpoint = args.checkpoint
        if checkpoint is None and args.method != "random":
            checkpoint = Path(args.root) / "ppo/final.zip" if args.method == "ppo" else Path(args.root) / "runs/strong" / args.method / f"seed-{args.seed}.pt"
        if args.command == "evaluate":
            output = args.output or Path(args.root) / "evaluation" / f"{args.method}-seed-{args.seed}.json"
            report = evaluate(args.method, checkpoint, seeds, output)
            print(json.dumps(report["summary"], indent=2))
        else:
            output = args.output or Path(args.root) / "replays" / f"{args.method}-seed-{args.seed}.gif"
            policy, _ = load_policy(args.method, checkpoint)
            print(json.dumps(export_gif(policy, output, seeds[0]), indent=2))
    elif args.command in ["experiments", "pipeline"]:
        from .experiments import pipeline, run_matrix
        config = json.loads(Path(args.config).read_text())
        run = pipeline if args.command == "pipeline" else run_matrix
        result = run(config, args.root, workers=args.workers, reports=args.reports)
        print(json.dumps({"runs": len(result["runs"]), "seconds": result["matrix_session_seconds"],
                          "aggregate": result["aggregate"]}, indent=2))
