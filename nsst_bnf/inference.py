"""Block Gaussian VI with diagonal + low-rank network blocks; fixed-step Adam.
Dense covariance likelihood and exact transformed-prior gradients are retained.
"""
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .model import LOG2PI


def fit_vi(model, steps=1200, samples=2, rank=2, lr=0.005, seed=1, initial=None):
    rng = np.random.default_rng(seed)
    loc = model.initial(seed) if initial is None else initial.copy()
    logsd = np.full(model.size, np.log(0.04))
    factors = [rng.normal(0,0.003,(s.stop-s.start,rank if i<3 else 0)) for i,s in enumerate(model.blocks)]
    params = [loc,logsd]+factors
    first = [np.zeros_like(p) for p in params]; second = [p.copy() for p in first]
    history, start = [],time.perf_counter()
    for step in range(1,steps+1):
        sd = np.exp(logsd)
        entropy = 0.; eg_sd=np.empty(model.size); eg_f=[]
        for s,f in zip(model.blocks,factors):
            ss=sd[s]**2
            cov=np.diag(ss)+f@f.T
            cf=cho_factor(cov,lower=True,check_finite=False)
            inv=cho_solve(cf,np.eye(len(ss)),check_finite=False)
            entropy += 0.5*(len(ss)*(1+LOG2PI)+2*np.log(np.diag(cf[0])).sum())
            eg_sd[s]=np.diag(inv)*ss;eg_f.append(inv@f)
        gs=[np.zeros_like(p) for p in params];elbo=entropy
        for _ in range(samples):
            e=rng.normal(size=model.size);z=[rng.normal(size=f.shape[1]) for f in factors]
            w=loc+sd*e
            for s,f,zz in zip(model.blocks,factors,z):w[s]+=f@zz
            lp,g=model.log_joint(w)
            if not np.isfinite(lp) or not np.isfinite(g).all():
                raise FloatingPointError(f"Nonfinite objective at step {step}; do not use this fit")
            elbo+=lp/samples;gs[0]+=g/samples;gs[1]+=g*sd*e/samples
            for i,(s,zz) in enumerate(zip(model.blocks,z)):gs[i+2]+=np.outer(g[s],zz)/samples
        gs[1]+=eg_sd
        for i,g in enumerate(eg_f):gs[i+2]+=g
        norm=np.sqrt(sum(np.sum(g*g) for g in gs));clip=min(1.,100./max(norm,1e-30))
        rate=lr*(1.-0.8*(step-1)/steps)
        for i,(p,g) in enumerate(zip(params,gs)):
            g=g*clip;first[i]*=.9;first[i]+=.1*g;second[i]*=.999;second[i]+=.001*g*g
            p+=rate*(first[i]/(1-.9**step))/(np.sqrt(second[i]/(1-.999**step))+1e-8)
        if step==1 or step%20==0 or step==steps:
            history.append(dict(step=step,elbo=float(elbo),gradient_norm=float(norm),seconds=time.perf_counter()-start))
    return dict(loc=loc,logsd=logsd,factors=factors,history=history,steps=steps,rank=rank,seed=seed,
                elapsed_seconds=time.perf_counter()-start,mc_samples=samples,learning_rate=lr)


def draw_vi(fit, blocks, draws=400, seed=2):
    rng=np.random.default_rng(seed)
    w=fit['loc']+np.exp(fit['logsd'])*rng.normal(size=(draws,len(fit['loc'])))
    for s,f in zip(blocks,fit['factors']):w[:,s]+=rng.normal(size=(draws,f.shape[1]))@f.T
    return w


def hmc(model, initial, draws=1000, warmup=1000, leapfrog=10, seed=3, target_accept=.9):
    """Exact-target Metropolis-corrected HMC, identity mass; warmup-only step tuning.
    No claim of convergence is made; chain diagnostics are required separately.
    """
    rng=np.random.default_rng(seed);q=initial.copy();lp,g=model.log_joint(q)
    eps=0.008;out=[];accepted=[];errors=[]
    for t in range(warmup+draws):
        p=rng.normal(size=len(q));p0=p.copy();qt=q.copy();gt=g.copy();lpt=lp
        p+=.5*eps*gt;valid=True
        try:
            for j in range(leapfrog):
                qt+=eps*p;lpt,gt=model.log_joint(qt)
                if not np.isfinite(lpt):valid=False;break
                if j<leapfrog-1:p+=eps*gt
            p+=.5*eps*gt
            error=(lp-.5*np.dot(p0,p0))-(lpt-.5*np.dot(p,p))
            prob=min(1.,np.exp(min(0.,-error))) if valid else 0.
        except (np.linalg.LinAlgError,FloatingPointError):
            prob=0.;error=np.inf
        accept=rng.uniform()<prob
        if accept:q,lp,g=qt,lpt,gt
        if t<warmup:
            eps=float(np.clip(eps*np.exp((prob-target_accept)/np.sqrt(t+10)),1e-5,.1))
        else:
            out.append(q.copy());accepted.append(accept);errors.append(float(error))
    return np.asarray(out),dict(acceptance=float(np.mean(accepted)),step_size=eps,
                              energy_errors_above_100=int(np.sum(np.abs(errors)>100)),warmup=warmup,draws=draws,
                              warmup_acceptance_target=target_accept,leapfrog_steps=leapfrog)

