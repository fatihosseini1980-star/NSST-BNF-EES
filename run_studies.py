"""Reproducible reconstruction studies. All outputs are NEW, never sn4 originals.

Examples (set OPENBLAS_NUM_THREADS=1 before Python):
 python run_studies.py simulation --replications 50 --workers 4
 python run_studies.py mcmc --draws 1000 --warmup 1000
 python run_studies.py tehran --csv /path/to/monthly.csv
"""
import argparse, json, time, platform, sys
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from nsst_bnf.model import Model, Config, kernel, CENTER, softplus
from nsst_bnf.inference import fit_vi, draw_vi, hmc
from nsst_bnf.calibration import metrics, calibrate, tsavp


def dump(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else str(a)))


def save_fit(path, fit):
    np.savez_compressed(path,loc=fit['loc'],logsd=fit['logsd'],**{f'factor_{i}':f for i,f in enumerate(fit['factors'])})


def simulation_data(seed,scenario,ntrain=120,nval=30,ntest=50):
    rng=np.random.default_rng(seed);n=ntrain+nval+ntest
    u=rng.uniform(size=(n,3));u=u[np.argsort(u[:,2])]
    mean=1.5*np.sin(2*np.pi*u[:,0])+np.cos(2*np.pi*u[:,1])+.8*np.sin(2*np.pi*u[:,2])+.5*u[:,0]*u[:,1]
    alpha=1.2*np.sin(2*np.pi*u[:,0])-.8*np.cos(2*np.pi*u[:,1])+.5*u[:,2]
    delta=np.full(n,.45) if scenario=='S1' else .65*alpha/np.sqrt(1+alpha**2)
    phi=np.full(n,.24) if scenario=='S1' else softplus(-1.35+.45*np.sin(2*np.pi*u[:,0])+.25*u[:,2])
    s2=.25;nu=5.;uu=abs(rng.normal());ww=rng.gamma(nu/2,2/nu)
    eps=np.linalg.cholesky(s2*(kernel(u,u,phi,phi)+1e-6*np.eye(n)))@rng.normal(size=n)
    y=mean+(delta*(uu-CENTER)+eps)/np.sqrt(ww)
    return u,y,dict(mean=mean,delta=delta,phi=phi,sigma2=s2,nu=nu,U=uu,W=ww)


def fit_evaluate(u,x,y,train,val,test,config,steps,draws,rank,seed,truth=None):
    # Both response and feature scalers use training only. u already encodes the declared metric.
    ym=float(y[train].mean());ys=float(y[train].std(ddof=1))
    xm=x[train].mean(0);xs=x[train].std(0);xs=np.where(xs>1e-10,xs,1.)
    xx=(x-xm)/xs;yy=(y-ym)/ys
    model=Model(u[train],xx[train],yy[train],config)
    fit=fit_vi(model,steps=steps,rank=rank,seed=seed)
    weights=draw_vi(fit,model.blocks,draws,seed+1)
    idx=np.r_[val,test]
    comps=model.predictive_components(weights,u[idx],xx[idx],seed+2)
    cv={k:v[:,:len(val)] for k,v in comps.items()};ct={k:v[:,len(val):] for k,v in comps.items()}
    pars,grid=calibrate(yy[val],cv,seed=seed+3)
    sv=ct['vi']*ys+ym;st=tsavp(ct,**pars,seed=seed+4)*ys+ym
    scores={'VI':metrics(y[test],sv),'TSAVP':metrics(y[test],st)}
    recover={}
    if truth:
        fields=[model.fields(w,u[test],xx[test])[:3] for w in weights]
        for k,name in enumerate(['mean','delta','phi']):
            samples=np.array([f[k] for f in fields])
            if name=='mean':samples=samples*ys+ym
            elif name=='delta':samples=samples*ys
            target=truth[name][test];lo,hi=np.quantile(samples,[.05,.95],axis=0)
            recover[name+'_RMSE']=float(np.sqrt(np.mean((samples.mean(0)-target)**2)))
            recover[name+'_Coverage90']=float(np.mean((lo<=target)&(hi>=target)))
        for name,samples,target in [('sigma2',np.exp(weights[:,model.scalar])*ys**2,truth['sigma2']),
                                    ('nu',2+np.exp(weights[:,model.scalar+1]),truth['nu'])]:
            lo,hi=np.quantile(samples,[.05,.95]);recover[name+'_mean']=float(samples.mean())
            recover[name+'_Coverage90']=bool(lo<=target<=hi)
    scalers=dict(response_mean=ym,response_sd=ys,feature_mean=xm,feature_sd=xs)
    return model,fit,weights,scores,recover,pars,grid,scalers,sv,st


def simulation_one(task):
    scenario,rep,steps,draws,hidden,rank=task
    seed=200000+rep*41+(10000 if scenario=='S2' else 0)
    u,y,truth=simulation_data(seed,scenario)
    rows=[];recovery=[];cal=[];hist=[]
    for variant in ['adaptive','stationary','gaussian']:
        config=Config(hidden=hidden,variant=variant)
        result=fit_evaluate(u,u,y,np.arange(120),np.arange(120,150),np.arange(150,200),
                            config,steps,draws,rank,seed+1,truth)
        model,fit,w,scores,rec,pars,grid,scalers,sv,st=result
        for method,score in scores.items():
            if variant!='adaptive' and method!='VI':continue
            rows.append(dict(scenario=scenario,replication=rep,variant=variant,method=method,
                             seconds=fit['elapsed_seconds'],**score))
        if variant=='adaptive':
            recovery.append(dict(scenario=scenario,replication=rep,**rec))
            cal.append(dict(scenario=scenario,replication=rep,**pars))
            for h in fit['history']:hist.append(dict(scenario=scenario,replication=rep,**h))
    return rows,recovery,cal,hist


def run_simulation(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tasks=[(s,r,args.steps,args.draws,args.hidden,args.rank) for s in ['S1','S2'] for r in range(args.replications)]
    rows=[];rec=[];cal=[];hist=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(a,b,c,d) in enumerate(pool.map(simulation_one,tasks)):
            rows+=a;rec+=b;cal+=c;hist+=d
            pd.DataFrame(rows).to_csv(out/'simulation_replicates.csv',index=False)
            pd.DataFrame(rec).to_csv(out/'function_recovery.csv',index=False)
            pd.DataFrame(cal).to_csv(out/'calibration_parameters.csv',index=False)
            pd.DataFrame(hist).to_csv(out/'optimization_history.csv',index=False)
            print(f"completed {i+1}/{len(tasks)}",flush=True)
    raw=pd.DataFrame(rows);summary=[]
    for key,d in raw.groupby(['scenario','variant','method']):
        for metric in ['RMSE','MAE','CRPS','Coverage90','AIW90']:
            mean=d[metric].mean();se=d[metric].std(ddof=1)/np.sqrt(len(d))
            summary.append(dict(scenario=key[0],variant=key[1],method=key[2],metric=metric,
                                mean=mean,replication_sd=d[metric].std(ddof=1),MCSE=se,
                                normal_MC_low=mean-1.96*se,normal_MC_high=mean+1.96*se,n=len(d)))
    pd.DataFrame(summary).to_csv(out/'simulation_summary.csv',index=False)
    dump(out/'simulation_configuration.json',dict(args=vars(args),model=asdict(Config(hidden=args.hidden)),
         seed_formula='200000+41*replication+(10000 for S2)',ntrain=120,nval=30,ntest=50,
         status='new reconstruction experiment, not reproduction of historical sn4 table',
         python=sys.version,numpy=np.__version__,platform=platform.platform()))


def split_rhat(chains):
    """Rank-normalized split Rhat, maximum with folded split Rhat."""
    c,n,p=chains.shape;half=n//2
    split=np.concatenate([chains[:,:half],chains[:,-half:]],axis=0)
    def ordinary(z):
        within=np.var(z,axis=1,ddof=1).mean(0);between=half*np.var(z.mean(1),axis=0,ddof=1)
        return np.sqrt(((half-1)/half*within+between/half)/np.maximum(within,1e-30))
    def rank(z):
        flat=z.reshape(-1,p);r=np.column_stack([rankdata(flat[:,j]) for j in range(p)])
        return norm.ppf((r-.375)/(len(flat)+.25)).reshape(z.shape)
    r=ordinary(rank(split));fold=np.abs(split-np.median(split,axis=(0,1)))
    return np.maximum(r,ordinary(rank(fold)))


def mcmc_one(task):
    u,y,config,initial,draws,warmup,seed,target,leapfrog=task
    model=Model(u,u,y,config)
    return hmc(model,initial,draws=draws,warmup=warmup,seed=seed,target_accept=target,leapfrog=leapfrog)


def run_mcmc(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    u,y,truth=simulation_data(8041,'S2',ntrain=24,nval=6,ntest=6)
    # SAME neural model for VI and HMC; small-data check has hidden=2 by default.
    yy=(y[:24]-y[:24].mean())/y[:24].std(ddof=1)
    config=Config(hidden=args.hidden);model=Model(u[:24],u[:24],yy,config)
    fit=fit_vi(model,steps=args.steps,rank=args.rank,seed=801)
    vi=draw_vi(fit,model.blocks,4000,802);starts=draw_vi(fit,model.blocks,4,803)
    tasks=[(u[:24],yy,config,starts[i],args.draws,args.warmup,900+i,args.target_accept,args.leapfrog) for i in range(4)]
    with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(mcmc_one,tasks))
    chains=np.stack([r[0] for r in results]);np.savez_compressed(out/'hmc_chains.npz',chains=chains,vi=vi)
    def summaries(w):
        f=model.fields(w,u[24:],u[24:]);s=model.scalar
        return np.r_[np.exp(w[s]),2+np.exp(w[s+1]),softplus(w[s+2]),np.exp(w[s+3]),f[0],f[1],f[2]]
    fs=np.array([[summaries(w) for w in chain] for chain in chains]);vs=np.array([summaries(w) for w in vi])
    names=['sigma2','nu','U','W']+[f'{a}_{j}' for a in ['mean','delta','phi'] for j in range(12)]
    rhat=split_rhat(fs);records=[]
    for j,name in enumerate(names):
        hm=fs[:,:,j].ravel();vv=vs[:,j]
        records.append(dict(quantity=name,HMC_mean=hm.mean(),HMC_sd=hm.std(ddof=1),
                            VI_mean=vv.mean(),VI_sd=vv.std(ddof=1),split_rank_Rhat=rhat[j],
                            VI_to_HMC_sd=vv.std(ddof=1)/hm.std(ddof=1)))
    pd.DataFrame(records).to_csv(out/'vi_hmc_comparison.csv',index=False)
    dump(out/'hmc_diagnostics.json',dict(chains=[r[1] for r in results],configuration=asdict(config),
         maximum_function_scalar_rhat=float(rhat.max()),all_below_1_01=bool((rhat<1.01).all()),
         warning='Do not interpret HMC as a reference if mixing diagnostics fail. ESS not implemented.',args=vars(args)))
    pd.DataFrame(fit['history']).to_csv(out/'mcmc_vi_optimization.csv',index=False)
    print('HMC maximum Rhat:',rhat.max(),flush=True)


def run_tehran(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.csv,encoding='cp1256');df=df[df.year<=2024].copy().sort_values(['year','month','station']).reset_index(drop=True)
    train=np.flatnonzero(df.year.to_numpy()<=2022);val=np.flatnonzero(df.year.to_numpy()==2023);test=np.flatnonzero(df.year.to_numpy()==2024)
    # All metric scales come ONLY from training. No municipal labels enter the spatial metric.
    raw_u=np.c_[df.lon,df.lat,12*(df.year-2014)+df.month-1]
    origin=raw_u[train].min(0);scale=np.ptp(raw_u[train],axis=0);u=(raw_u-origin)/scale
    x=np.c_[u,df.m_sin,df.m_cos]
    config=Config(hidden=args.hidden,neighbors=args.neighbors,variant=args.variant)
    dump(out/'tehran_run_started.json',dict(args=vars(args),configuration=asdict(config),
         ntrain=len(train),nval=len(val),ntest=len(test),coordinate_origin=origin,coordinate_scale=scale,
         target_unit='Author-directed PM2.5 AQI sub-index; dimensionless; source definition provisional; no numeric conversion',
         test_status='2024 historically inspected; retrospective temporal holdout, not a new untouched test'))
    result=fit_evaluate(u,x,df.pm25.to_numpy(),train,val,test,config,args.steps,args.draws,args.rank,4401)
    model,fit,w,scores,rec,pars,grid,scalers,sv,st=result
    save_fit(out/'tehran_vi_fit.npz',fit);np.savez_compressed(out/'tehran_predictive_draws.npz',vi=sv,tsavp=st,test_rows=test)
    dump(out/'tehran_scalers.json',scalers);dump(out/'tehran_metrics.json',dict(scores=scores,calibration=pars))
    pd.DataFrame(fit['history']).to_csv(out/'tehran_optimization.csv',index=False)
    pd.DataFrame(grid).to_csv(out/'tehran_validation_grid.csv',index=False)
    pred=df.loc[test,['station','year','month','pm25']].copy()
    for name,samples in [('VI',sv),('TSAVP',st)]:
        pred[name+'_mean']=samples.mean(0);pred[name+'_lower']=np.quantile(samples,.05,axis=0);pred[name+'_upper']=np.quantile(samples,.95,axis=0)
    pred.to_csv(out/'tehran_predictions.csv',index=False)
    print(json.dumps(scores),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('study',choices=['simulation','mcmc','tehran'])
    p.add_argument('--steps',type=int,default=1200);p.add_argument('--draws',type=int,default=400)
    p.add_argument('--rank',type=int,default=2);p.add_argument('--hidden',type=int,default=4)
    p.add_argument('--replications',type=int,default=50);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--warmup',type=int,default=1000);p.add_argument('--csv');p.add_argument('--out',default='results')
    p.add_argument('--neighbors',type=int,default=0,help='Tehran: 0 exact dense, positive ordered Vecchia approximation')
    p.add_argument('--target-accept',type=float,default=.9);p.add_argument('--leapfrog',type=int,default=10)
    p.add_argument('--variant',choices=['adaptive','stationary','gaussian'],default='adaptive')
    args=p.parse_args()
    if args.study=='simulation':run_simulation(args)
    elif args.study=='mcmc':run_mcmc(args)
    else:
        if not args.csv:p.error('--csv is required for tehran')
        run_tehran(args)


if __name__=='__main__':main()
