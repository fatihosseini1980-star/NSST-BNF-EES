"""Descriptive multilevel calibration for a dependent retrospective test panel.
No iid PIT p-values or independent-observation binomial confidence intervals.
"""
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    rows=[];pitrows=[]
    for folder in Path('results').glob('tehran_*'):
        path=folder/'tehran_predictive_draws.npz'
        if not path.exists():continue
        z=np.load(path);pred=pd.read_csv(folder/'tehran_predictions.csv');y=pred.pm25.to_numpy()
        for name in ['vi','tsavp']:
            samples=z[name]
            pit=(np.sum(samples<y,axis=0)+.5*np.sum(samples==y,axis=0))/len(samples)
            for nominal in [.5,.8,.9,.95]:
                lo,hi=np.quantile(samples,[(1-nominal)/2,(1+nominal)/2],axis=0)
                rows.append(dict(run=folder.name,method=name,nominal=nominal,
                                 coverage=np.mean((y>=lo)&(y<=hi)),width=np.mean(hi-lo),n=len(y)))
            for i in range(len(y)):
                pitrows.append(dict(run=folder.name,method=name,station=pred.station.iloc[i],month=int(pred.month.iloc[i]),PIT=pit[i]))
    pd.DataFrame(rows).to_csv('results/tehran_multilevel_calibration.csv',index=False)
    pd.DataFrame(pitrows).to_csv('results/tehran_pit_values.csv',index=False)


if __name__=='__main__':main()
