"""Paired replication-level contrasts between adaptive VI and independent comparators."""
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
sim=pd.read_csv(P/'results/simulation_algorithm2/simulation_replicates.csv')
base=pd.read_csv(P/'results/independent_benchmarks/simulation_replicates.csv')
rows=[]
for scenario in ['S1','S2']:
    vi=sim[(sim.scenario==scenario)&(sim.variant=='adaptive')&(sim.method=='VI')].set_index('replication')
    for method in ['Matern_GP','Quantile_HGB']:
        benchmark=base[(base.scenario==scenario)&(base.method==method)].set_index('replication')
        assert len(vi)==len(benchmark)==50 and vi.index.is_unique and benchmark.index.is_unique
        assert set(vi.index)==set(benchmark.index)==set(range(50))
        for metric in ['RMSE','CRPS','Coverage90','AIW90']:
            d=(vi[metric]-benchmark[metric]).sort_index().to_numpy()
            mcse=d.std(ddof=1)/np.sqrt(len(d));mean=d.mean()
            rows.append(dict(scenario=scenario,contrast=f'Adaptive VI minus {method}',metric=metric,
                             mean_difference=mean,paired_MCSE=mcse,normal_low=mean-1.96*mcse,
                             normal_high=mean+1.96*mcse,n=len(d)))
out=P/'results/independent_benchmarks/paired_differences.csv'
pd.DataFrame(rows).to_csv(out,index=False)
print(f'Saved {len(rows)} paired comparisons to {out.relative_to(P)}')
