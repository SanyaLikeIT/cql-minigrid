# CQL MiniGrid — Team 12 Moodle submission

This project compares offline DQN, discrete Conservative Q-Learning (CQL), and
behavioral cloning in `MiniGrid-Dynamic-Obstacles-8x8-v0`. PPO supplies frozen
experience. No Git, GitHub account, authentication, or development-machine paths
are required. Training and evaluation use CPU.

Read `SUBMISSION_MANIFEST.json` in the ZIP: `mode: draft` and
`submission_complete: false` mean the author's final presentation and demo are
still missing. **A draft is not the complete Moodle deliverable.**
`docs/Presentation.pdf` is the earlier project proposal, not the final presentation.

## Install on Linux

Use **Python 3.12**. From the extracted `cql-minigrid` directory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[dev,notebook]'
.venv/bin/python -m cql_minigrid doctor
```

An internet connection is needed only to install dependencies. The notebook,
tests, evaluation, and package builder then work offline. Existing RL dependency
pins are retained. The `notebook` extra adds `nbformat`, `nbclient`, and
`ipykernel`; a web notebook frontend is optional, not required for execution.

## Execute the required notebook

The notebook is **`notebooks/CQL_value_update_demo.ipynb`**, with verified outputs
already saved. Open it in a Jupyter-compatible editor using the `.venv` Python
kernel and choose **Run All**, or execute it in a fresh kernel without a frontend:

```sh
.venv/bin/python scripts/execute_notebook.py notebooks/CQL_value_update_demo.ipynb
```

The command uses the interpreter that launches it, and leaves the original file
unchanged. To save fresh outputs, add
`--output notebooks/CQL_value_update_demo.ipynb`.
It performs one illustrative direct-Q-value SGD update using the project's
actual Huber TD loss and discrete CQL penalty, compares DQN/CQL, checks values
and gradients, and reads real results from compact JSON reports. It does not
load the large datasets or retrain an agent. Real agent training uses a neural
network and Adam; the notebook states this distinction explicitly.

## Tests and a saved-agent evaluation

```sh
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m pytest -q
.venv/bin/python -m cql_minigrid evaluate --config configs/main.json --method cql --checkpoint artifacts/final/runs/mixed/cql/seed-0.pt --output artifacts/submission-evaluation.json
```

This evaluation uses the original 100 held-out test seeds. Optional Git-specific
tests skip when Git or checkout metadata is unavailable; the core tests and
numerical example still run. With explicit absolute `--config`, `--checkpoint`,
and `--output` paths, evaluation also works from another working directory.

## Contents and recorded results

| Path | Contents |
|---|---|
| `src/cql_minigrid/` | Environment, collectors, offline losses/trainers, evaluation, reporting, alpha study |
| `configs/` | Original observation schema, hyperparameters, explicit seeds, study settings |
| `notebooks/CQL_value_update_demo.ipynb` | Executed educational CQL update, not a full training run |
| `reports/summary.json`, `reports/per_run.csv` | Original 18 final runs on two approximately 200k-transition files |
| `reports/ablation/` | 39-run follow-up, all five metrics, plots, hashes, runtime/session scope |
| `reports/pilot-50k/` | Small historical report files only; raw pilot data/models are omitted |
| `reports/demos/`, `reports/ablation/demos/` | Supporting MP4 gameplay and sidecars; these are not the author's final demo |
| `artifacts/final/datasets/` | Frozen strong and episode-sampled mixed NPZ files and metadata |
| `artifacts/ablation/datasets/` | Transition-balanced mixed NPZ file and metadata |
| `artifacts/final/runs/`, `artifacts/ablation/runs/` | All 57 reported final offline checkpoints, evaluation JSON, training histories |
| `artifacts/final/ppo/` | Final and early PPO checkpoints plus training progress |
| `scripts/` | Standard-library ZIP builder and fresh-kernel notebook executor |
| `SUBMISSION_MANIFEST.json` | Explicit per-file sizes, SHA-256 hashes, categories, dataset counts, media status |

The frozen files contain **200,035 strong**, **200,019 original mixed**, and
**200,025 balanced mixed** transitions. Balanced source counts are
100,065 final PPO / 49,980 early PPO / 49,980 random. All reported final offline
agents can be evaluated from the included checkpoints. All offline comparisons
can be retrained from the three included frozen datasets and configurations.
There is no need to recollect data or retrain PPO for evaluation or the notebook.

At alpha 1, CQL success is 61.7 ± 1.5% on strong data and 60.7 ± 6.1% on original
mixed data. On balanced mixed data, alpha 0/0.1/1/5 gives 61.0 ± 51.1%,
48.0 ± 25.2%, 2.7 ± 2.3%, and 0.0 ± 0.0% success, respectively. Balanced-data
DQN seeds give 91%, 90%, and 2%; BC has 0% success. Means/sample SD use three
training seeds and common 100-episode test sets. These results do not establish
statistical significance or general superiority. Source quality, coverage, and
dataset realization vary together. See `docs/ablation.md` for limitations.

Historical result files retain the implementation provenance measured when
those experiments ran. The new submission manifest hashes the actual packaged
files; it does not rewrite old results or invent a Git revision. Runtime
provenance uses source SHA-256 hashes and a null Git revision when no matching
checkout is available. No integrity check requires `git show` or Git history.

## Optional training reproduction

Use a new output location to preserve the supplied evidence. For a single run:

```sh
.venv/bin/python -m cql_minigrid train --config configs/main.json --method cql --dataset artifacts/final/datasets/strong.npz --seed 0 --output artifacts/reproduced/strong-cql-0.pt
```

Repeat methods/seeds/files as described in `README.md`, or use `experiments`
with a new root containing copies of the supplied `datasets/`, `ppo/`, and
`source_audit.json`. Run the full alpha comparison into a separate location:

```sh
.venv/bin/python -m cql_minigrid ablation --config configs/ablation.json --root artifacts/reproduced-ablation --reports artifacts/reproduced-ablation/reports
```

The source also supports fresh PPO collection and the complete original pipeline.
Intermediate shards, obsolete pilots, duplicate `latest.zip`, duplicate PPO
copies, and GIF copies of supporting MP4s are excluded only from the submission
ZIP. Original workspace data is preserved. Completed datasets work without
shards; completed PPO can be reused without `latest.zip`. Partial historical
collection cannot be resumed from omitted intermediate files.

## Add your media and create the final Moodle ZIP

No final presentation or final demo has been generated by this preparation.
Put **your actual final files** at these exact locations:

1. `submission/presentation/FINAL_PRESENTATION.pdf`
2. `submission/demo/DEMO.mp4`

```sh
mkdir -p submission/presentation submission/demo
# Copy your completed PDF and MP4 into the two locations above.
.venv/bin/python scripts/build_submission.py --mode final
```

The final command refuses missing/empty/structurally invalid media and rejects
a byte-for-byte copy of the earlier proposal as the final presentation. It
creates `dist/cql-minigrid-moodle-final.zip` only after both files pass checks.
These checks verify file structure, not the academic content of your slides or
video; review both before uploading.

To regenerate the code/notebook draft without final media:

```sh
.venv/bin/python scripts/build_submission.py --mode draft
```

Output: `dist/cql-minigrid-moodle-draft.zip`. The builder uses only Python's
standard library, never calls Git, verifies dataset/checkpoint hashes and saved
notebook execution counts, and uses stable ordering/timestamps/permissions for
reproducible ZIP bytes. No Moodle upload-size limit is assumed. The manifest
lists every included file and deliberate exclusion. After extracting the draft,
you can add media there and run the same final command.
