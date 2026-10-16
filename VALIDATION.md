# Verification record

- Eight numerical unit tests passed.
- `verify_repository.py` independently recomputes daily-to-monthly means, all saved Tehran scores from predictive draws, simulation group means and standard errors, HMC array dimensions, and 22-by-12 regional prediction counts.
- S1 and S2 simulation comparisons each contain 50 replications for every method. Replications 48 and 49 of the S2 independent comparators were recovered using the original deterministic seed formula; recomputed full-replication summary values match the previously stored summary to floating-point precision. The paired-difference file was recalculated from all 50 pairs.
- The saved Tehran analysis uses 36 retained monitoring stations, a training/validation/test split of 2633/319/315, and 200 draws per fitted predictive law. The 30-neighbor and retain-500 outputs are designated sensitivity runs.
- `figures/paper/` contains the eleven PNGs used in the current manuscript; `figures/simulation/` also contains regenerated summaries from the stored replication-level results.
- Reproducing the full fits may require substantial CPU time. Included checkpoints and predictive draws allow inspecting the recorded results without rerunning model fitting.
