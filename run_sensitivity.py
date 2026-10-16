"""Small, explicitly exploratory paired sensitivity study; does not tune on test."""
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from run_studies import simulation_data,fit_evaluate,dump
from nsst_bnf.model import Config

SETTINGS={
    'reference':dict(hidden=4,prior_sd=.7,rank=2,steps=1200,jitter=1e-6),
    'hidden2':dict(hidden=2,prior_sd=.7,rank=2,steps=1200,jitter=1e-6),
    'hidden8':dict(hidden=8,prior_sd=.7,rank=2,steps=1200,jitter=1e-6),
    'prior035':dict(hidden=4,prior_sd=.35,rank=2,steps=1200,jitter=1e-6),
    'prior140':dict(hidden=4,prior_sd=1.4,rank=2,steps=1200,jitter=1e-6),
    'diagonal_vi':dict(hidden=4,prior_sd=.7,rank=0,steps=1200,jitter=1e-6),
    'double_steps':dict(hidden=4,prior_sd=.7,rank=2,steps=2400,jitter=1e-6),
    'jitter1e5':dict(hidden=4,prior_sd=.7,rank=2,steps=1200,jitter=1e-5),
}


def one(task):
    name,rep=task;s=SETTINGS[name];seed=200000+41*rep+10000
    u,y,truth=simulation_data(seed,'S2');config=Config(hidden=s['hidden'],prior_sd=s['prior_sd'],jitter=s['jitter'])
    result=fit_evaluate(u,u,y,np.arange(120),np.arange(120,150),np.arange(150,200),config,s['steps'],400,s['rank'],seed+1,truth)
    model,fit,w,scores,rec,pars,grid,scalers,sv,st=result
    return [dict(setting=name,replication=rep,method=k,**v,**s) for k,v in scores.items()]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replications',type=int,default=5);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--out',default='results/sensitivity');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for r in pool.map(one,[(name,rep) for name in SETTINGS for rep in range(a.replications)]):
            rows+=r;pd.DataFrame(rows).to_csv(out/'sensitivity_replicates.csv',index=False)
    d=pd.DataFrame(rows);d.groupby(['setting','method'])[['RMSE','CRPS','Coverage90','AIW90']].mean().to_csv(out/'sensitivity_means.csv')
    dump(out/'configuration.json',dict(settings=SETTINGS,replications=a.replications,scenario='S2',purpose='exploratory sensitivity, not test-set model selection'))
