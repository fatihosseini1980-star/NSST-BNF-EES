"""Generate simulation coverage and error comparison directly from saved summaries."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
out=ROOT/'figures/simulation';out.mkdir(parents=True,exist_ok=True)
a=pd.read_csv(ROOT/'results/simulation_algorithm2/simulation_summary.csv')
b=pd.read_csv(ROOT/'results/independent_benchmarks/simulation_summary.csv')
for metric in ['Coverage90','RMSE','CRPS']:
    fig,ax=plt.subplots(figsize=(8,3.5))
    for j,scenario in enumerate(['S1','S2']):
        x=a[(a.scenario==scenario)&(a.metric==metric)].copy()
        x['label']=x['variant']+' '+x['method']
        y=b[(b.scenario==scenario)&(b.metric==metric)].copy()
        vals=list(zip(x.label,x['mean'],x.MCSE))+list(zip(y.method,y['mean'],y.MCSE))
        for i,(name,value,se) in enumerate(vals):
            ax.errorbar(j*10+i,value,yerr=1.96*se,fmt='o',capsize=2,label=str(name) if j==0 else None)
    if metric=='Coverage90':ax.axhline(.9,ls='--',lw=1,label='Nominal 90%')
    ax.set_xticks([3,13],['S1','S2']);ax.set_ylabel(metric);ax.set_xlim(-1,18);ax.grid(alpha=.17)
    ax.legend(fontsize=7,ncol=3,loc='best');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/(metric.lower()+'.'+ext),dpi=300)
    plt.close(fig)
