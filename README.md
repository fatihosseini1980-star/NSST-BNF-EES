# Adaptive Nonstationary Skew-t Bayesian Neural Field

Computational supplement for the 50-replication simulations, independent benchmark comparisons, sensitivity analysis, small-data VI/HMC study, and retrospective Tehran PM2.5 AQI analysis. The code and recorded numerical results share one `nsst_bnf/` model implementation. No fresh fitting is needed to inspect the included outputs.

## Setup

Python 3.12 is recommended. Install packages with `python -m pip install -r requirements.txt`. Run commands from the repository root and limit parallel BLAS threads (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`). A full refit is computationally intensive; the saved results include model checkpoints and predictive draws.

## Verify included results

```
python -m unittest discover -s tests
python verify_repository.py
```

## Reproduce all simulations

```
python run_simulations.py --replications 50 --workers 4
```

The workflow executes the S1/S2 adaptive, stationary and Gaussian fits; independent Matérn GP and quantile histogram boosting benchmarks; the S2 sensitivity grid; and the four-chain small-data VI/HMC diagnostics. Simulation output is under `results/simulation_algorithm2/`, `results/independent_benchmarks/`, `results/sensitivity_algorithm2/`, and `results/mcmc_long_trajectory/`. `figures/simulation/` contains summary plots generated from the saved summaries.

## Reproduce Tehran analysis

```
python prepare_tehran.py
python prepare_tehran.py --retain-500
python run_tehran.py --all
```

The analysis uses the screened `data/monthly_qa_screened.csv`, with 36 stations. Included results are in `results/adaptive15/`, `results/stationary15/`, `results/gaussian15/`, `results/adaptive30/`, `results/independent/`, and `results/sensitivity_retaining500/`. Figures are in `figures/tehran/`. The 15-neighbor adaptive fit is primary; the 30-neighbor fit is a prespecified approximation sensitivity check, not a test-set-selected replacement. The retain-500 scenario documents sensitivity to a potentially genuine AQI extreme. All model comparisons are evaluated on the same 315 test observations.

## Recorded main Tehran results

| Model | RMSE | MAE | CRPS | 90% coverage | 90% width |
|---|---:|---:|---:|---:|---:|
| Adaptive VI | 21.822 | 17.617 | 12.525 | 0.806 | 57.402 |
| Adaptive TSAVP | 21.818 | 17.630 | 12.441 | 0.835 | 63.813 |
| Stationary skew-t VI | 22.077 | 17.451 | 12.585 | 0.794 | 56.431 |
| Gaussian VI | 24.605 | 20.244 | 14.563 | 0.724 | 56.279 |
| Matérn GP | 22.022 | 17.828 | 12.467 | 0.883 | 68.222 |
| Quantile HGB | 21.909 | 17.577 | 12.671 | 0.740 | 50.518 |

Coverage remains below its nominal target for TSAVP. These are retrospective results, not a claim of universally superior performance. For preprocessing choices and the separate 500-retention analysis, see `DATA_PROVENANCE.md`.

## Repository layout

- `nsst_bnf/`: field, ordered Vecchia inference, structured VI/HMC and TSAVP.
- `run_studies.py`, `run_simulations.py`, `run_sensitivity.py`, `run_benchmarks.py`, `additional_diagnostics.py`: simulation and inferential studies.
- `prepare_tehran.py`, `run_tehran.py`, `run_benchmarks_clean.py`, `tehran_figures.py`, `tehran_regional_map.py`, `tehran_interpolated_map.py`: real-data reproduction.
- `data/`: canonical English-header daily/monthly inputs, station metadata and boundaries.
- `results/`: saved configurations, model weights, predictive samples, numerical metrics and diagnostics.
- `figures/`: generated figures.
- `tests/`: numerical unit tests.
