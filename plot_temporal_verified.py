"""No manipulation of model predictions: uses 200 saved joint TSAVP draws."""
from pathlib import Path
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;out=P/'results/tehran_figures';out.mkdir(exist_ok=True)
t=pd.read_csv(P/'results/adaptive15/tehran_predictions.csv')
z=np.load(P/'results/adaptive15/tehran_predictive_draws.npz');assert z['tsavp'].shape==(200,315);assert np.max(np.abs(t.TSAVP_mean-z['tsavp'].mean(axis=0)))<1e-10
month=t.groupby('month').agg(n=('station','size'),observed_mean=('pm25','mean'),observed_median=('pm25','median'),tsavp_mean=('TSAVP_mean','mean')).reset_index()
month.to_csv(out/'tehran_temporal_monthly_verified.csv',index=False)
fig,ax=plt.subplots(figsize=(9,4.1))
ax.plot(month.month,month.observed_mean,'o-',label='Observed mean',lw=2.1,ms=6)
ax.plot(month.month,month.observed_median,'o-',label='Observed median',lw=2.1,ms=6)
ax.plot(month.month,month.tsavp_mean,'o-',label='TSAVP mean',lw=2.1,ms=6)
ax.set_xticks(range(1,13));ax.set_xlabel('Month (2024)');ax.set_ylabel('PM$_{2.5}$ AQI (index points)');ax.legend(ncol=3,fontsize=9,loc='upper left');ax.grid(alpha=.2)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'tehran_temporal_verified.{ext}',dpi=320,bbox_inches='tight')
print(month.round(4).to_string(index=False))
