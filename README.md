# MEC DualBP Robust Task-Offloading Experiments

This repository preserves the implementation, frozen configurations, tests and paper-level evidence used for the Chapter 5 experiments of the MEC robust task-offloading study.

The codebase includes the verified physics and uncertainty layers, dataset and label pipelines, DualBP and Improved Physics-Residual DualBP models, formal experiment runners, online interval evaluation, OE-RVEA/multi-algorithm comparison, postprocessing and related tests.

## Project layout

- `src/mec_dualbp/`: physics, uncertainty, datasets, models, training, optimization and experiment implementations.
- `scripts/`: selected formal finalization and paper-level postprocessing scripts.
- `configs/`: frozen formal scenario and resource configurations.
- `tests/`: test coverage associated with the archived implementation.
- `docs/`: frozen scientific protocols, traceability notes and the Chapter 5 evidence archive.
- `artifacts/frozen_models/`: the two formally bound `seed=31005` Improved DualBP candidate models.

## Reproducibility snapshot

- This repository stores the source code, configurations and paper-level result snapshot corresponding to the final Chapter 5 experiments.
- The formal experiments were primarily executed on the original older computer.
- Complete raw `outputs/` trees, caches and historical checkpoints are not uploaded to GitHub.
- GitHub retains only traceable core code, frozen configurations, the two uniquely bound formal models and paper evidence.
- Chapter 5 results entry point: [`docs/experiment_archive/chapter5_final_20260930/`](docs/experiment_archive/chapter5_final_20260930/)
- Final experiment Gate: `NO_REQUIRED_EXPERIMENT_RERUN_V2`

Runtime values belong to the original experiment environment. This repository does not claim that a different device will reproduce identical wall-clock time.

## Integrity and interpretation

Use the following files together when citing Chapter 5 results:

- `docs/experiment_archive/chapter5_final_20260930/PAPER_KEY_NUMBERS.md`
- `docs/experiment_archive/chapter5_final_20260930/PAPER_CLAIM_BOUNDARIES.md`
- `docs/experiment_archive/chapter5_final_20260930/SOURCE_MAPPING.csv`

Historical Formal/Frozen results must not be overwritten. Any new execution on another computer should use a new version and output directory.
