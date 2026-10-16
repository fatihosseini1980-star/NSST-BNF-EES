"""Split-chain ESS with Geyer initial-positive/monotone paired autocorrelations.
Algorithm reference: https://mc-stan.org/docs/reference-manual/analysis.html
No external diagnostics package required. Includes independent IID/AR(1) checks.
"""
import numpy as np,pandas as pd
from pathlib import Path
from scipy.stats import norm,rankdata
from run_studies import simulation_data,split_rhat,dump
from nsst_bnf.model import Model,Config,softplus

def split(a):
 h=a.shape[1]//2
 return np.concatenate([a[:,:h],a[:,-h:]],axis=0)
def ess(a):
 a=np.asarray(a,dtype=float);m,n=a.shape
 if np.ptp(a)<1e-14:return float(a.size)
 z=a-a.mean(1,keepdims=True);f=np.fft.rfft(z,n=2*n,axis=1);ac=np.fft.irfft(f*f.conj(),n=2*n,axis=1)[:,:n]/n
 within=np.var(a,axis=1,ddof=1).mean();v=(n-1)/n*within+np.var(a.mean(1),ddof=1)
 rho=1-(within-ac.mean(0))/v;rho[0]=1
 pairs=rho[:2*(n//2)].reshape(-1,2).sum(1);bad=np.flatnonzero(pairs<=0)
 if len(bad):pairs=pairs[:bad[0]]
 pairs=np.minimum.accumulate(pairs)
 tau=max(-1+2*pairs.sum(),1/np.log10(m*n))
 return float(m*n/tau)
def diagnostics(a):
 s=split(a);r=rankdata(s.ravel());z=norm.ppf((r-.375)/(len(r)+.25)).reshape(s.shape)
 bulk=ess(z);tail=min(ess(split(a<=np.quantile(a,q))) for q in [.05,.95]);raw=ess(s)
 return dict(ESS_bulk=bulk,ESS_tail=tail,MCSE_mean=float(a.std(ddof=1)/np.sqrt(raw)))
def main():
 out=Path('results/mcmc_long_trajectory');data=np.load(out/'hmc_chains.npz');chains=data['chains'];u,y,_=simulation_data(8041,'S2',ntrain=24,nval=6,ntest=6);yy=(y[:24]-y[:24].mean())/y[:24].std(ddof=1);model=Model(u[:24],u[:24],yy,Config(hidden=2));s=model.scalar
 def summarize(w):
  f=model.fields(w,u[24:],u[24:]);return np.r_[np.exp(w[s]),2+np.exp(w[s+1]),softplus(w[s+2]),np.exp(w[s+3]),f[0],f[1],f[2],model.log_joint(w,gradient=False)]
 values=np.array([[summarize(w) for w in chain] for chain in chains]);names=['sigma2','nu','U','W']+[f'{a}_{j}' for a in ['mean','delta','phi'] for j in range(12)]+['log_joint'];rh=split_rhat(values)
 rows=[dict(quantity=name,split_rank_Rhat=rh[j],**diagnostics(values[:,:,j])) for j,name in enumerate(names)]
 pd.DataFrame(rows).to_csv(out/'extended_diagnostics.csv',index=False)
 wr=split_rhat(chains);pd.DataFrame([dict(parameter=j,split_rank_Rhat=wr[j],**diagnostics(chains[:,:,j])) for j in range(chains.shape[-1])]).to_csv(out/'weight_diagnostics.csv',index=False)
 rng=np.random.default_rng(901);iid=rng.normal(size=(4,12000));ar=np.empty_like(iid);ar[:,0]=iid[:,0]
 for j in range(1,ar.shape[1]):ar[:,j]=.8*ar[:,j-1]+.6*iid[:,j]
 check=dict(iid_ESS=ess(split(iid)),iid_expected=iid.size,AR08_ESS=ess(split(ar)),AR08_asymptotic=iid.size/9)
 assert .65<iid.size/check['iid_ESS']<1.5 and .65<check['AR08_ESS']/(iid.size/9)<1.5
 dump(out/'extended_diagnostics_summary.json',dict(minimum_bulk_ESS=min(r['ESS_bulk'] for r in rows),minimum_tail_ESS=min(r['ESS_tail'] for r in rows),maximum_summary_Rhat=float(rh.max()),maximum_weight_Rhat=float(wr.max()),validation=check,scope='one small-data fit; diagnostics do not establish identifiability or global exploration'))
 print(pd.DataFrame(rows).iloc[[0,1,2,3,-1]].to_string(index=False),flush=True)
if __name__=='__main__':main()
