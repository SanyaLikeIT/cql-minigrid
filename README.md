# cql-minigrid

Team 12 — Conservative Q-Learning for offline navigation in
`MiniGrid-Dynamic-Obstacles-8x8-v0`.

The project compares discrete CQL, offline DQN, and behavioral cloning on
fixed PPO datasets. Implementation and experiment results are tracked in
GitHub Issues and pull requests.

Use Python 3.12. CPU installation (bash or fish):

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m cql_minigrid doctor
.venv/bin/python -m pytest -q
```

Datasets, weights, and full logs belong under the ignored `artifacts/` directory.
GitHub Actions runs short CPU checks. Full experiments run locally.

The environment uses its original rewards and 256-step limit. Every policy
receives the full symbolic grid (object one-hot, color, state), direction,
and elapsed time, as specified in `configs/observation.json`. Actions are
left, right, and forward. Success has positive terminal reward; collision
has reward -1 and includes a forward move blocked by a wall. A time limit
is recorded separately as `truncated`. If both flags are set, the terminal
success or collision takes precedence over timeout in outcome counts.

`configs/seeds.json` separates PPO, data collection, validation, and final
evaluation seeds. Offline training seeds initialize the networks and batch
samplers; all final policies use the same reserved evaluation seeds.

Train the collector, then audit its sources and collect complete episodes:

```sh
.venv/bin/python -m cql_minigrid ppo --config configs/main.json --root artifacts/main
.venv/bin/python -m cql_minigrid collect --config configs/main.json --root artifacts/main
```

Add `--resume` to the PPO command after interruption. PPO saves its optimizer
and early/latest/final checkpoints; resuming restarts environment episodes,
so it is not bitwise equivalent to uninterrupted PPO. Dataset collection
resumes automatically at the last complete shard, with deterministic
episode and policy seeds. Checksums reject changed source checkpoints.

The main datasets target 50,000 transitions each, following the current
assignment (the earlier proposal suggested 200,000). Complete final episodes
may exceed the target by up to 255 transitions. The mixed dataset selects
sources per episode with weights 0.5 final PPO, 0.25 early PPO, and 0.25
random. Actual episode and transition counts and source outcomes are stored
in the dataset JSON files. Validation source audits precede collection;
final PPO must reach at least 60% validation success to be labeled strong.

PPO alone uses potential shaping: `r + 0.5 * (gamma * Phi(next) - Phi(now))`,
where Phi is negative normalized Manhattan distance to the goal and is zero
at either episode ending. Its finite-horizon time limit is terminal for
the PPO value target. Source audits, saved transitions, and final evaluation
use the original rewards and both ending flags. An initial unshaped PPO
pilot reached 0/64 validation successes after 401,408 steps; its local logs
are retained under `artifacts/pilots/unshaped/`.

For a small CPU collector pilot, use `configs/smoke.json` and a separate
`--root artifacts/smoke`. This pilot disables the collector quality gate.
