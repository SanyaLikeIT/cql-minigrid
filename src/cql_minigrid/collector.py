"""PPO checkpoints, source audits, and resumable complete-episode datasets."""
import json
import os
from pathlib import Path
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .common import resource_path, seed_everything, sha256, write_json
from .env import make_env, observation_spec
from .rollout import ppo_policy, random_policy, run_episodes, summarize
from .shaping import CollectorReward

TRANSITION_FIELDS = ['observation', 'action', 'reward', 'next_observation',
                     'terminated', 'truncated', 'episode_id', 'source', 'seed',
                     'policy_seed', 'elapsed_steps', 'next_elapsed_steps']


def save_ppo(model, path):
    path = Path(path)
    temporary = path.with_name(path.stem + ".tmp.zip")
    model.save(temporary)
    os.replace(temporary, path)


def train_ppo(config, root, *, resume=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"])
    started = time.perf_counter()
    progress_path = root / "progress.json"
    previous = json.loads(progress_path.read_text()) if resume and progress_path.exists() else None
    if previous and previous["config"] != config:
        raise ValueError("PPO resume config differs from the saved config")
    if (previous and previous['timesteps'] >= config['total_timesteps']
            and (root/'final.zip').exists() and (root/'early.zip').exists()):
        return previous
    if not resume and any((root / name).exists() for name in ['latest.zip','final.zip']):
        raise FileExistsError("PPO output exists; use --resume or another output directory")
    env = DummyVecEnv([lambda: Monitor(CollectorReward(make_env(),
        scale=config["shaping_scale"], step_cost=config["step_cost"])) for _ in range(config["n_envs"])])
    env.seed(config["seed"])
    if previous:
        model = PPO.load(root / "latest.zip", env=env, device="cpu")
    else:
        model = PPO("MlpPolicy", env, seed=config["seed"], device="cpu", verbose=0,
                    learning_rate=config["learning_rate"], n_steps=config["n_steps"],
                    batch_size=config["batch_size"], n_epochs=config["n_epochs"],
                    gamma=config["gamma"], ent_coef=config["ent_coef"],
                    policy_kwargs={"net_arch": dict(pi=[128, 128], vf=[128, 128])})
    elapsed_before = previous["seconds"] if previous else 0
    report = previous
    if previous and model.num_timesteps >= config["total_timesteps"] and (root / "final.zip").exists():
        env.close()
        return previous
    while model.num_timesteps < config["total_timesteps"]:
        if not (root / "early.zip").exists():
            target = config["early_timesteps"]
        else:
            target = min(config["total_timesteps"], model.num_timesteps + config["checkpoint_interval"])
        model.learn(total_timesteps=max(1, target - model.num_timesteps), reset_num_timesteps=False)
        save_ppo(model, root / "latest.zip")
        if not (root / "early.zip").exists():
            save_ppo(model, root / "early.zip")
        history = (report or {}).get("history", [])
        history.append({"timesteps": int(model.num_timesteps),
                        "training_return_mean": float(np.mean([ep["r"] for ep in model.ep_info_buffer])),
                        "episode_length_mean": float(np.mean([ep["l"] for ep in model.ep_info_buffer]))})
        report = {"config": config, "timesteps": model.num_timesteps,
                  "seconds": elapsed_before + time.perf_counter() - started,
                  "observation": observation_spec(), "resumed": bool(previous), "history": history,
                  "resume_note": "SB3 restores optimizer; environment episodes restart on resume."}
        write_json(progress_path, report)
        print(json.dumps({"ppo_steps": model.num_timesteps, "seconds": round(report["seconds"], 2)}), flush=True)
    save_ppo(model, root / "final.zip")
    env.close()
    return report


def load_sources(root):
    root = Path(root)
    policies = {"strong": ppo_policy(PPO.load(root / "final.zip", device="cpu")),
                "early": ppo_policy(PPO.load(root / "early.zip", device="cpu")),
                "random": random_policy}
    hashes = {name: sha256(root / f"{file}.zip")
              for name, file in [("strong", "final"), ("early", "early")]}
    hashes["random"] = "numpy-generator-uniform-three-actions-v1"
    return policies, hashes


def audit_sources(policies, hashes, seeds, output):
    report = {"checkpoint_sha256": hashes, "validation_seeds": seeds,
              "sources": {name: {**run_episodes(policy, seeds),
                                 "training_timesteps": getattr(policy, "training_timesteps", 0)}
                          for name, policy in policies.items()}}
    write_json(output, report)
    return report


def save_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    os.replace(temporary, path)


def collect_dataset(path, policies, weights, *, target, episode_seed_start,
                    policy_seed_start, source_hashes, source_audit, shard_episodes=128,
                    sampling_unit='episodes'):
    """Mix complete episodes, optionally controlling source transition quotas."""
    if sampling_unit not in ['episodes', 'transitions']:
        raise ValueError('sampling_unit must be episodes or transitions')
    if target < 1 or shard_episodes < 1:
        raise ValueError('target and shard_episodes must be positive')
    path = Path(path)
    parts = path.with_suffix(".parts")
    manifest_path = parts / "progress.json"
    identity = {"weights": weights, "target": target, "episode_seed_start": episode_seed_start,
                "policy_seed_start": policy_seed_start, "source_sha256": source_hashes,
                "observation": observation_spec()}
    # Keep the existing episode-sampled manifests compatible and unchanged.
    if sampling_unit != 'episodes':
        identity['sampling_unit'] = sampling_unit
    if not manifest_path.exists() and (path.exists() or path.with_suffix('.json').exists()):
        if not path.exists() or not path.with_suffix('.json').exists():
            raise ValueError('Frozen dataset needs both NPZ and JSON metadata')
        previous = json.loads(path.with_suffix('.json').read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError('Dataset configuration or source checkpoint changed')
        if previous['sha256'] != sha256(path):
            raise ValueError('Completed dataset checksum mismatch')
        return previous
    parts.mkdir(parents=True, exist_ok=True)
    manifest = {"identity": identity, "parts": [], "episodes": [], "transitions": 0}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("Dataset configuration or source checkpoint changed")
    names = list(weights)
    probabilities = np.array(list(weights.values()), dtype=float)
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
        raise ValueError("Mixture weights must be nonnegative and sum to one")
    counts = {name: sum(e['steps'] for e in manifest['episodes'] if e['source'] == name)
              for name in names}
    quotas = {name: target * weights[name] for name in names}
    env = make_env()
    while manifest["transitions"] < target:
        rows = {key: [] for key in TRANSITION_FIELDS}
        episode_records = []
        for _ in range(shard_episodes):
            eid = len(manifest["episodes"]) + len(episode_records)
            seed, policy_seed = episode_seed_start + eid, policy_seed_start + eid
            rng = np.random.default_rng(policy_seed)
            if sampling_unit == 'transitions':
                # Fill the largest remaining quota; never split an episode.
                name = max(names, key=lambda name: quotas[name] - counts[name])
            else:
                name = str(rng.choice(names, p=probabilities))
            obs, _ = env.reset(seed=seed)
            total = 0.0
            for step in range(1, 257):
                elapsed_steps = env.elapsed_steps
                action = policies[name](obs, rng)
                next_obs, reward, terminated, truncated, info = env.step(action)
                values = [obs, action, reward, next_obs, terminated, truncated, eid, name, seed,
                          policy_seed, elapsed_steps, info['elapsed_steps']]
                for key, value in zip(rows, values, strict=True):
                    rows[key].append(value)
                obs = next_obs
                total += float(reward)
                if terminated or truncated:
                    counts[name] += step
                    episode_records.append({"episode_id": eid, "source": name, "seed": seed,
                                            "policy_seed": policy_seed, "steps": step,
                                            "return": total, "outcome": info["outcome"]})
                    break
            if manifest["transitions"] + len(rows["action"]) >= target:
                break
        index = len(manifest["parts"])
        part_path = parts / f"part-{index:05d}.npz"
        arrays = {key: np.asarray(value) for key, value in rows.items()}
        arrays["reward"] = arrays["reward"].astype(np.float32)
        for key in ['elapsed_steps', 'next_elapsed_steps']:
            arrays[key] = arrays[key].astype(np.int32)
        save_npz(part_path, arrays)
        manifest["parts"].append({"file": part_path.name, "sha256": sha256(part_path)})
        manifest["episodes"].extend(episode_records)
        manifest["transitions"] += len(rows["action"])
        write_json(manifest_path, manifest)
        print(json.dumps({"dataset": path.stem, "transitions": manifest["transitions"]}), flush=True)
    env.close()
    metadata_path = path.with_suffix(".json")
    if path.exists() and metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous["sha256"] != sha256(path):
            raise ValueError("Completed dataset checksum mismatch")
    else:
        shard_paths = []
        for part in manifest["parts"]:
            part_path = parts / part["file"]
            if sha256(part_path) != part["sha256"]:
                raise ValueError("Dataset shard checksum mismatch")
            shard_paths.append(part_path)
        merged = {}
        # Assemble one field at a time: 200k full-grid transitions are large.
        for key in TRANSITION_FIELDS:
            chunks = []
            for part_path in shard_paths:
                with np.load(part_path, allow_pickle=False) as data:
                    chunks.append(data[key])
            merged[key] = np.concatenate(chunks)
            del chunks
        save_npz(path, merged)
    sources = {}
    for name in names:
        records = [e for e in manifest["episodes"] if e["source"] == name]
        if records:
            sources[name] = {**summarize(records), "transitions": sum(e["steps"] for e in records)}
    metadata = {**identity, "sha256": sha256(path), "transitions": manifest["transitions"],
                "episodes": len(manifest["episodes"]), "sources": sources,
                "recorded_fields": TRANSITION_FIELDS,
                "source_audit": source_audit, "episode_records": manifest["episodes"]}
    write_json(path.with_suffix(".json"), metadata)
    return metadata


def collect(config, root, *, target=None, config_path=None):
    seed_everything(config["ppo"]["seed"])
    root = Path(root)
    policies, hashes = load_sources(root / "ppo")
    seeds = json.loads(resource_path(config['seeds_file'], config_path).read_text())
    audit = audit_sources(policies, hashes, seeds["validation"], root / "source_audit.json")
    strong = audit["sources"]["strong"]["summary"]["success_rate"]
    if strong < config["datasets"]["minimum_collector_success"]:
        raise ValueError(f"Collector success {strong:.3f} is below configured minimum")
    results = {}
    for i, (name, weights) in enumerate(config["datasets"]["mixtures"].items()):
        results[name] = collect_dataset(root / "datasets" / f"{name}.npz", policies, weights,
            target=target or config["datasets"]["transitions"],
            episode_seed_start=seeds["collection_episode_start"] + i * 100000,
            policy_seed_start=seeds["collection_rng"] + i * 100000,
            source_hashes=hashes, source_audit=audit,
            sampling_unit=config['datasets'].get('sampling_unit', 'episodes'))
    return results
