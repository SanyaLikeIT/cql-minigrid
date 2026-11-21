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
