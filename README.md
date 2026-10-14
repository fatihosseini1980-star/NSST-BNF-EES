# Reproducible computational supplement

Install requirements.txt. Run commands from the supplement root.

- python -m unittest discover -s tests
- python run_studies.py simulation --replications 50 --workers 4 --out results/simulation_algorithm2
- python run_studies.py tehran --csv data/monthly_supplied.csv --neighbors 15 --steps 1500 --draws 200 --out results/tehran_vecchia15
- python run_benchmarks.py --study simulation --workers 4
- python run_benchmarks.py --study tehran
- python additional_diagnostics.py

Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 before execution to control parallelism. Saved configurations record the exact settings for each experiment, including the 30-neighbor fit, sensitivity and HMC. Results contain replication scores, chains, diagnostics and predictive draws. Read run_sensitivity.py and run_studies.py --help for the remaining options.

The pm25 column is the PM2.5 AQI sub-index, expressed in dimensionless index points. The monthly input is used directly; no concentration conversion is performed. The 2024 evaluation is a retrospective temporal holdout.


