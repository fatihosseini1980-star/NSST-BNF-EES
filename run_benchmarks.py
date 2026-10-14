"""Independent GP and quantile boosting; validation-only model selection.
QHGB uses seven rearranged quantiles and endpoint-clipped inverse-CDF interpolation.
"""
import argparse,time,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,WhiteKernel
from sklearn.ensemble import HistGradientBoostingRegressor
from run_studies import simulation_data,dump
from nsst_bnf.calibration import metrics
PROBS=np.array([.01,.05,.25,.5,.75,.95,.99])
def fit_benchmarks(u,x,y,tr,va,te,seed):
 ym=y[tr].mean();ys=y[tr].std(ddof=1);yy=(y-ym)/ys;xm=x[tr].mean(0);xs=x[tr].std(0);xs[xs<1e-10]=1;xx=(x-xm)/xs
 idx=np.r_[va,te];nv=len(va);q=(np.arange(400)+.5)/400;rows=[];grids=[];preds={}
 for method in ['Matern_GP','Quantile_HGB']:
  start=time.perf_counter();best=None
  if method=='Matern_GP':
   mean=Ridge(alpha=1.).fit(xx[tr],yy[tr]);res=yy[tr]-mean.predict(xx[tr]);settings=[dict(length=l,noise_variance=n) for l in [.2,.5,1.] for n in [.05,.2]]
  else:settings=[dict(max_leaf_nodes=n) for n in [7,15]]
  for pars in settings:
   if method=='Matern_GP':
    gp=GaussianProcessRegressor(kernel=Matern(length_scale=pars['length'],nu=1.5,length_scale_bounds='fixed')+WhiteKernel(noise_level=pars['noise_variance'],noise_level_bounds='fixed'),optimizer=None,alpha=1e-8,normalize_y=False)
    gp.fit(u[tr],res);mu,sd=gp.predict(u[idx],return_std=True);mu+=mean.predict(xx[idx]);samples=mu[None,:]+norm.ppf(q)[:,None]*sd[None,:]
   else:
    quantiles=[]
    for prob in PROBS:
     model=HistGradientBoostingRegressor(loss='quantile',quantile=float(prob),max_iter=150,learning_rate=.05,max_leaf_nodes=pars['max_leaf_nodes'],min_samples_leaf=10,l2_regularization=1.,early_stopping=False,random_state=seed)
     model.fit(xx[tr],yy[tr]);quantiles.append(model.predict(xx[idx]))
    quantiles=np.sort(np.array(quantiles),axis=0);samples=np.column_stack([np.interp(q,PROBS,quantiles[:,j]) for j in range(len(idx))])
   score=metrics(yy[va],samples[:,:nv])['CRPS'];grids.append(dict(method=method,**pars,validation_CRPS_standardized=score))
   if best is None or score<best[0]:best=(score,samples,pars)
  samples=best[1][:,nv:]*ys+ym;preds[method]=samples;rows.append(dict(method=method,seconds=time.perf_counter()-start,parameters=json.dumps(best[2]),**metrics(y[te],samples)))
 return rows,grids,preds

def one(task):
 s,r=task;seed=200000+41*r+(10000 if s=='S2' else 0);u,y,_=simulation_data(seed,s)
 rows,grid,_=fit_benchmarks(u,u,y,np.arange(120),np.arange(120,150),np.arange(150,200),seed)
 return [dict(scenario=s,replication=r,**a) for a in rows],[dict(scenario=s,replication=r,**a) for a in grid]
def main():
 p=argparse.ArgumentParser();p.add_argument('--study',choices=['simulation','tehran'],required=True);p.add_argument('--replications',type=int,default=50);p.add_argument('--workers',type=int,default=4);p.add_argument('--out',default='results/independent_benchmarks');args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 dump(out/(args.study+'_configuration.json'),dict(args=vars(args),GP='Ridge(alpha=1) mean; unit-variance Matern3/2 plus white noise; full training set; six validation-selected kernels; no likelihood optimization',QHGB='7 quantiles; 150 iterations; learning_rate=.05; leaves7/15; minleaf10; L2=1; no early stop; monotone rearrangement; endpoint-clipped inverse CDF',selection='minimum validation CRPS in training-standardized units',predictive_grid_size=400,probabilities=PROBS,test_selection=False))
 if args.study=='simulation':
  rows=[];grids=[]
  with ProcessPoolExecutor(max_workers=args.workers) as pool:
   for i,(a,b) in enumerate(pool.map(one,[(s,r) for s in ['S1','S2'] for r in range(args.replications)])):
    rows+=a;grids+=b;pd.DataFrame(rows).to_csv(out/'simulation_replicates.csv',index=False);pd.DataFrame(grids).to_csv(out/'simulation_validation_grid.csv',index=False);print('completed',i+1,flush=True)
  frame=pd.DataFrame(rows);summary=[]
  for (s,m),d in frame.groupby(['scenario','method']):
   for metric in ['RMSE','MAE','CRPS','Coverage90','AIW90']:summary.append(dict(scenario=s,method=m,metric=metric,mean=d[metric].mean(),MCSE=d[metric].std(ddof=1)/np.sqrt(len(d)),n=len(d)))
  pd.DataFrame(summary).to_csv(out/'simulation_summary.csv',index=False)
 else:
  df=pd.read_csv(Path(__file__).parent/'data/monthly_supplied.csv',encoding='cp1256');df=df[df.year<=2024].sort_values(['year','month','station']).reset_index(drop=True);tr=np.flatnonzero(df.year<=2022);va=np.flatnonzero(df.year==2023);te=np.flatnonzero(df.year==2024);raw=np.c_[df.lon,df.lat,12*(df.year-2014)+df.month-1];u=(raw-raw[tr].min(0))/np.ptp(raw[tr],axis=0);x=np.c_[u,df.m_sin,df.m_cos]
  rows,grid,preds=fit_benchmarks(u,x,df.pm25.to_numpy(),tr,va,te,4401);pd.DataFrame(rows).to_csv(out/'tehran_metrics.csv',index=False);pd.DataFrame(grid).to_csv(out/'tehran_validation_grid.csv',index=False);np.savez_compressed(out/'tehran_predictive_draws.npz',**preds,test_rows=te);print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()
