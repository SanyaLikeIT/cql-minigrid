# Ten-seed reliability extension

The current study uses training seeds 0–9. Original three-seed results and the
navigation test on seeds 30000–30199 remain historical evidence under `reports/`.
Current results are written separately to `reports/reliability/`.

Run from the project root with the Python 3.12 environment:

```sh
.venv/bin/python -m cql_minigrid.reliability
.venv/bin/python scripts/render_reliability.py
```

The runner checks all four frozen NPZ SHA-256 values and checkpoint identities,
skips compatible completed models, resumes partial optimizer/RNG states, and
writes progress and result JSON atomically. Historical numerical changes were
reviewed: observation encoding, rewards, TD loss, CQL penalty, optimizer and
network architecture for the original baseline remain equivalent. Later code
adds optional geometry, initialization and a default-zero step penalty; it does
not change the old default learning objective. Historical weights are checked
against their recorded hashes. Evaluation is refreshed with the current
navigation instrumentation without overwriting historical episode files.

The original baseline has 60 runs: two datasets, three methods, ten training
seeds, 100 common test environments each. The alpha study has 120 CQL runs and
ten balanced-data BC runs. For new seeds, strong/mixed CQL alpha 0 reuses the
corresponding DQN weights: its loss and optimization are identical. Alpha 1
reuses the corresponding baseline CQL weights. The recorded checkpoint method
may therefore be DQN for an alpha-0 experimental condition; the loader always
uses the actual stored method. Existing historical duplicates are preserved.

Navigation retains all eight investigated candidates: three original-mixed
penalties, three guided penalties without geometry, and two with geometry.
Every candidate initializes from the same-seed original mixed CQL checkpoint.
All candidates use the original validation seeds 10000–10063. Selection uses
only these validation outcomes and the original success/collision quality
gates. The selected configuration and all validation aggregates are written to
`artifacts/reliability/navigation/test_lock.json` before evaluating the new
common seeds **40000–40199**. The runner checks disjointness against collection,
training, PPO, validation, original test, and historical navigation test seeds.
Ten original and ten improved models produce 4,000 fresh-test episodes. The
BFS teacher remains solely a dataset collector; inference uses the network.

All reported uncertainty treats the trained model as the experimental unit.
For n valid model outcomes, the two-sided 95% Student t interval is
`mean ± t.ppf(0.975, n-1) * sample_SD / sqrt(n)`; paired intervals apply this
formula to per-training-seed differences. Intervals are not clipped to rate
bounds and are unadjusted exploratory intervals across multiple comparisons.
Sample SD, median, range and individual outcomes are reported separately.
Conditional successful lengths exclude only models with no successes, with
valid sample counts explicit. Failures remain in every unconditional metric.
The fixed datasets, one PPO source seed and shared environment sets limit the
scope: these intervals describe training-seed variability conditional on this
protocol, not uncertainty over independently collected datasets or all maps.

SciPy Student t reference:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html

A single worker preserves the historical one-thread CPU protocol and avoids
holding multiple 1.3 GB expanded transition arrays. Dataset collection and
hyperparameters remain unchanged. Logs and `*_progress.json` expose progress;
re-run the same command after interruption. The optional `--archive` accepts
a previous cleanup archive to recover historical control weights on the first
run; recovered controls remain inside the reliability artifacts thereafter.
