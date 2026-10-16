"""Verify canonical source data, stored predictions, simulations and figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from nsst_bnf.calibration import metrics

P = Path(__file__).resolve().parent
source = pd.read_csv(P/'data/source_daily_aqi.csv')
daily = pd.read_csv(P/'data/daily_quality_screened.csv')
data = pd.read_csv(P/'data/monthly_qa_screened.csv')
assert len(source) == 90023 and len(daily) == 83478 and len(data) == 3348
assert data.station.nunique() == 36
assert not data.station.isin(['Sharif','Shahrdari-2']).any()
assert not data.duplicated(['station','year','month']).any()
assert (len(data[data.year <= 2022]), len(data[data.year == 2023]), len(data[data.year == 2024])) == (2633,319,315)
assert len(data[data.year <= 2024]) == 3267
agg = daily.groupby(['Stations','year','month'],as_index=False)['PM2.5'].mean().rename(columns={'Stations':'station','PM2.5':'recomputed'})
check = data.merge(agg,on=['station','year','month'],validate='one_to_one')
assert len(check) == len(data) and np.max(np.abs(check.pm25-check.recomputed)) < 1e-7

frame = data[data.year <= 2024].sort_values(['year','month','station']).reset_index(drop=True)
for name in ['adaptive15','stationary15','gaussian15','adaptive30','sensitivity_retaining500']:
    result = P/'results'/name
    z = np.load(result/'tehran_predictive_draws.npz')
    pred = pd.read_csv(result/'tehran_predictions.csv')
    assert z['vi'].shape == (200,315) and z['tsavp'].shape == (200,315)
    assert len(pred) == 315
    assert np.allclose(pred.TSAVP_mean,z['tsavp'].mean(0),atol=1e-10)
    assert np.allclose(pred.VI_mean,z['vi'].mean(0),atol=1e-10)
    saved = json.loads((result/'tehran_metrics.json').read_text())['scores']
    for field, method in [('vi','VI'),('tsavp','TSAVP')]:
        score = metrics(pred.pm25.to_numpy(),z[field])
        for key in ['RMSE','MAE','CRPS','Coverage90','AIW90','Bias']:
            assert np.isclose(saved[method][key],score[key],atol=1e-8),(name,method,key)

independent = pd.read_csv(P/'results/independent/tehran_metrics.csv')
z = np.load(P/'results/independent/tehran_predictive_draws.npz')
assert len(independent) == 2
for item in independent.itertuples():
    scored = metrics(frame.loc[z['test_rows'],'pm25'].to_numpy(),z[item.method])
    for key in ['RMSE','MAE','CRPS','Coverage90','AIW90','Bias']:
        assert np.isclose(scored[key],getattr(item,key),atol=1e-8),(item.method,key)

sim = pd.read_csv(P/'results/simulation_algorithm2/simulation_replicates.csv')
bench = pd.read_csv(P/'results/independent_benchmarks/simulation_replicates.csv')
assert len(sim) == 400 and len(bench) == 200
assert all(sim.groupby(['scenario','variant','method']).size() == 50)
assert all(bench.groupby(['scenario','method']).size() == 50)
summary = pd.read_csv(P/'results/simulation_algorithm2/simulation_summary.csv')
for r in summary.itertuples():
    values = sim[(sim.scenario==r.scenario)&(sim.variant==r.variant)&(sim.method==r.method)][r.metric]
    assert len(values)==50 and np.isclose(values.mean(),r.mean,atol=1e-10)
    assert np.isclose(values.std(ddof=1)/np.sqrt(50),r.MCSE,atol=1e-10)

chains = np.load(P/'results/mcmc_long_trajectory/hmc_chains.npz')['chains']
assert chains.shape[0]==4 and chains.shape[1]==3000
regions = pd.read_csv(P/'results/tehran_regional_map/regional_point_predictions.csv')
assert len(regions)==264 and set(regions.region)==set(range(1,23))
assert all(regions.groupby('month').size()==22)
inter = json.loads((P/'results/tehran_interpolation/interpolation_diagnostics.json').read_text())
assert inter['max_absolute_error_at_22_anchor_points'] < 1e-9
assert len(list((P/'figures/tehran').glob('*.png'))) == 10
assert len(list((P/'figures/simulation').glob('*.png'))) == 4
assert len(list((P/'figures/paper').glob('*.png'))) == 11
print('PASS: 36 stations, source-to-monthly values, complete stored model scores, 50-replication simulation, four-chain HMC, maps and figures.')
