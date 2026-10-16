"""Validation-only finite-grid calibration; no truth-function access."""
import itertools
import numpy as np
from scipy.special import gammaincinv
from .model import CENTER


def metrics(y, samples):
    y=np.asarray(y);m=len(samples);ordered=np.sort(samples,axis=0)
    crps=np.mean(np.abs(samples-y),axis=0)-np.sum((2*np.arange(1,m+1)-m-1)[:,None]*ordered,axis=0)/m**2
    pred=samples.mean(0);lo,hi=np.quantile(samples,[.05,.95],axis=0)
    return dict(RMSE=float(np.sqrt(np.mean((pred-y)**2))),MAE=float(np.mean(np.abs(pred-y))),
                CRPS=float(crps.mean()),Coverage90=float(np.mean((y>=lo)&(y<=hi))),
                AIW90=float(np.mean(hi-lo)),Bias=float(np.mean(pred-y)))


def tsavp(components,nu=5.,a_delta=1.,a_r=1.,seed=4):
    """Algorithm 2 TSAVP in sn4 WITH the conditional mean correction.
    This remains validation calibration, NOT the fitted model's posterior predictive.
    Common random numbers across calibration candidates reduce Monte Carlo noise.
    """
    rng=np.random.default_rng(seed);m=len(components['mean'])
    u=np.abs(rng.normal(size=(m,1)))-CENTER
    w=gammaincinv(nu/2,rng.uniform(1e-10,1-1e-10,size=(m,1)))/(nu/2)
    return components['calibration_base']+(a_delta*components['calibration_delta']*u+a_r*components['innovation'])/np.sqrt(w)


def calibrate(yval,components,grid=None,seed=4,coverage_weight=2.,width_weight=.01):
    # Values and weights refer to TRAINING-standardized response units.
    grid=grid or dict(nu=[3.,5.,10.,30.],a_delta=[0.,.5,1.],a_r=[.5,1.,1.5,2.])
    records=[]
    for nu,ad,ar in itertools.product(grid['nu'],grid['a_delta'],grid['a_r']):
        pars=dict(nu=nu,a_delta=ad,a_r=ar)
        met=metrics(yval,tsavp(components,**pars,seed=seed))
        score=met['CRPS']+coverage_weight*abs(met['Coverage90']-.9)+width_weight*met['AIW90']
        records.append(dict(**pars,score=float(score),**met))
    best=min(records,key=lambda x:x['score'])
    return {k:best[k] for k in ['nu','a_delta','a_r']},records
