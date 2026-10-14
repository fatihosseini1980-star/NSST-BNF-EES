"""Ordered nearest-previous-neighbor Gaussian conditionals inside sn4 augmentation.

This is a declared approximation, NOT the exact dense Gaussian likelihood.
Order is fixed by input rows (the Tehran runner sorts by time and station).
Prediction appends unobserved points to training, with joint recursive sampling.
References: Katzfuss & Guinness (2021), doi:10.1214/19-STS755;
Katzfuss et al. (2020), doi:10.1007/s13253-020-00401-7.
"""
import numpy as np
from .model import CENTER, softplus, kernel


def parents(u,k,start=0):
    out=[]
    for i in range(start,len(u)):
        count=min(k,i)
        if not count:out.append(np.array([],dtype=int));continue
        d=np.sum((u[:i]-u[i])**2,axis=1)
        ids=np.argpartition(d,count-1)[:count]
        out.append(ids[np.lexsort((ids,d[ids]))])
    return out


def groups(u,k):
    par=parents(u,k);by={}
    for i,p in enumerate(par):by.setdefault(len(p),[]).append(np.r_[i,p])
    # Precompute coordinate distances on neighborhoods only.
    return [(np.asarray(v),np.sum((u[np.asarray(v)][:,:,None,:]-u[np.asarray(v)][:,None,:,:])**2,axis=3))
            for v in by.values()]


def likelihood_statistics(group,u,phi,r,precision,gradient,jitter):
    n=len(u);vm=np.zeros(n);gp=np.zeros(n);logdet=0.;quad=0.
    for ids,d2 in group:
        pp=phi[ids];ss=pp[:,:,None]**2+pp[:,None,:]**2
        cc=np.exp(1.5*np.log(2*pp[:,:,None]*pp[:,None,:]/ss)-d2/ss)
        c=cc+jitter*np.eye(ids.shape[1])[None,:,:]
        # Inverse-based log-density differences are used only on small neighborhoods.
        inv=np.linalg.inv(c);sign,ld=np.linalg.slogdet(c)
        if np.any(sign<=0):raise np.linalg.LinAlgError('Nonpositive neighborhood covariance')
        rr=r[ids];v=np.einsum('bij,bj->bi',inv,rr)
        q=np.sum(rr*v,axis=1)
        g=precision*v
        gc=.5*(precision*v[:,:,None]*v[:,None,:]-inv) if gradient else None
        if ids.shape[1]>1:
            cp=c[:,1:,1:];ip=np.linalg.inv(cp);vp=np.einsum('bij,bj->bi',ip,rr[:,1:])
            sp,ldp=np.linalg.slogdet(cp)
            if np.any(sp<=0):raise np.linalg.LinAlgError('Nonpositive parent covariance')
            ld-=ldp;q-=np.sum(rr[:,1:]*vp,axis=1);g[:,1:]-=precision*vp
            if gradient:gc[:,1:,1:]-=.5*(precision*vp[:,:,None]*vp[:,None,:]-ip)
        logdet+=ld.sum();quad+=precision*q.sum()
        if gradient:
            np.add.at(vm,ids.ravel(),g.ravel())
            deriv=1.5/pp[:,:,None]-3*pp[:,:,None]/ss+2*d2*pp[:,:,None]/ss**2
            local=2*np.sum(gc*cc*deriv,axis=2)
            np.add.at(gp,ids.ravel(),local.ravel())
    return float(logdet),float(quad),vm,gp


def predictive_components(model,weights,ustar,xstar,seed):
    rng=np.random.default_rng(seed);n=len(model.y);r=len(ustar)
    u=np.vstack([model.u,ustar]);xx=np.vstack([model.x,xstar])
    par=parents(u,model.config.neighbors,start=n)
    vi=[];means=[];deltas=[];innovations=[];cms=[];bases=[];residual_deltas=[]
    for w in weights:
        m,d,p,*_=model.fields(w,u,xx)
        ls,ln,ru,lw=w[model.scalar:];s2=np.exp(ls)
        ww=1. if model.config.variant=='gaussian' else np.exp(lw)
        a=0. if model.config.variant=='gaussian' else (softplus(ru)-CENTER)/np.sqrt(ww)
        # Work in centered residual coordinates.
        sim=np.zeros(n+r);cm=np.zeros(n+r);eps=np.zeros(n+r)
        base=np.zeros(n+r);projected_delta=np.zeros(n+r)
        base[:n]=model.y-m[:n];projected_delta[:n]=d[:n]
        sim[:n]=model.y-m[:n]-d[:n]*a;cm[:n]=sim[:n]
        for j,ids in enumerate(par):
            i=n+j
            c=kernel(u[ids],u[ids],p[ids],p[ids])+model.config.jitter*np.eye(len(ids))
            cross=kernel(u[ids],u[i:i+1],p[ids],p[i:i+1]).ravel()
            b=np.linalg.solve(c,cross)
            variance=1.+model.config.jitter-np.dot(b,cross)
            if variance<=0:raise np.linalg.LinAlgError('Nonpositive Vecchia conditional variance')
            innovation=np.sqrt(s2*variance)*rng.normal()
            sim[i]=np.dot(b,sim[ids])+innovation/np.sqrt(ww)
            cm[i]=np.dot(b,cm[ids])
            eps[i]=np.dot(b,eps[ids])+innovation
            base[i]=np.dot(b,base[ids]);projected_delta[i]=np.dot(b,projected_delta[ids])
        vi.append(m[n:]+d[n:]*a+sim[n:]);cms.append(m[n:]+d[n:]*a+cm[n:])
        means.append(m[n:]);deltas.append(d[n:]);innovations.append(eps[n:])
        bases.append(m[n:]+base[n:]);residual_deltas.append(d[n:]-projected_delta[n:])
    return dict(vi=np.asarray(vi),mean=np.asarray(means),delta=np.asarray(deltas),
                innovation=np.asarray(innovations),conditional_mean=np.asarray(cms),
                calibration_base=np.asarray(bases),calibration_delta=np.asarray(residual_deltas))
