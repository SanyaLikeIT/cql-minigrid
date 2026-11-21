# Project requirements

Source: [Presentation.pdf](Presentation.pdf), Team 12 project proposal.
The slides define project requirements; code and reports below implement them.

| Slide | Requirement | Implementation / evidence |
|---|---|---|
| 4 | MiniGrid-Dynamic-Obstacles-8x8-v0; left, right, forward; full symbolic grid, direction, elapsed time | `env.py`, `configs/observation.json`, Issue #2 |
| 5 | Goal reward `1 - 0.9 * steps / 256`; collision reward -1, including walls; ordinary reward 0; timeout at 256 steps | `env.py`, exact reward/ending tests in `tests/test_env.py`, Issue #2 |
| 6 | PPO checkpoints; stronger-only and stronger/early/random datasets; approximately 200,000 transitions per final file | `collector.py`, `configs/main.json`, Issue #3 |
| 7 | Matched DQN/CQL, BC, three training seeds, reserved common test seeds, collector PPO reference | `offline.py`, `experiments.py`, Issues #4–#8 |
| 7 | Success/collision/timeout rates, mean return, mean steps on successful episodes | `rollout.py`, `reporting.py`, Issues #7–#8 |
| 8 | Frozen data, plots, gameplay videos | NPZ/JSON metadata, reports, MP4/GIF replay export, Issues #3, #7–#8 |

`elapsed_steps` is the integer environment counter, starting at zero and
reaching 256 at the time limit. Its named observation component is encoded
as `elapsed_steps / 256` at index 836 of the shared float32 vector. This is
the same numerical encoding used by the existing PPO and pilot datasets.
The raw integer counter is also exposed in reset/step info.

Final datasets target approximately 200,000 transitions each. Existing
50,000-transition datasets and their measurements are retained as a pilot.
The separate small CPU pilot remains available through `configs/smoke.json`.
MP4 gameplay is exported alongside GIF demonstrations.
