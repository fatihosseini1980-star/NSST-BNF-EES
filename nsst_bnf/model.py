"""Dense centered common-latent skew-t neural field; explicit analytic derivatives.

The likelihood is multivariate, never a sum of independent minibatch likelihoods.
Coordinates must have three columns; input scaling is part of the experiment.
The fixed diagonal jitter is recorded, and is not a fitted spatial range.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit, gammaln, digamma
from scipy.spatial.distance import cdist

CENTER = np.sqrt(2 / np.pi)
LOG2PI = np.log(2 * np.pi)


def softplus(x):
    return np.logaddexp(0, x)


@dataclass
class Config:
    hidden: int = 4
    prior_sd: float = 0.7
    variance_shape: float = 2.0
    variance_scale: float = 0.5
    df_shape: float = 2.0
    df_rate: float = 0.5
    jitter: float = 1e-6
    variant: str = "adaptive"  # adaptive, stationary, gaussian
    neighbors: int = 0  # 0=dense exact; positive=ordered Vecchia likelihood approximation


class Network:
    """One tanh hidden layer and a scalar linear output; hidden=0 is linear."""
    def __init__(self, inputs, hidden):
        self.inputs, self.hidden = inputs, hidden
        self.size = inputs + 1 if hidden == 0 else inputs * hidden + 2 * hidden + 1

    def initial(self, rng):
        w = rng.normal(0, 0.15, self.size)
        w[-1] = 0
        return w

    def forward(self, w, x):
        if self.hidden == 0:
            return x @ w[:-1] + w[-1], x
        p, h = self.inputs * self.hidden, self.hidden
        z = np.tanh(x @ w[:p].reshape(self.inputs, h) + w[p:p+h])
        return z @ w[p+h:p+2*h] + w[-1], (x, z)

    def backward(self, w, cache, g):
        if self.hidden == 0:
            return np.r_[cache.T @ g, g.sum()]
        x, z = cache
        p, h = self.inputs * self.hidden, self.hidden
        dz = g[:, None] * w[p+h:p+2*h] * (1 - z*z)
        return np.r_[(x.T @ dz).ravel(), dz.sum(0), z.T @ g, g.sum()]


def kernel(u, v, phi, psi):
    d2 = cdist(u, v, "sqeuclidean")
    s = phi[:, None]**2 + psi[None, :]**2
    return np.exp(1.5 * np.log(2 * phi[:, None] * psi[None, :] / s) - d2 / s)


class Model:
    def __init__(self, coordinates, features, y, config=None):
        self.u, self.x, self.y = map(lambda a: np.asarray(a, dtype=float), (coordinates, features, y))
        if self.u.shape != (len(self.y), 3) or self.x.shape[0] != len(self.y):
            raise ValueError("Expected n x 3 coordinates, n x p features and n responses")
        if not all(np.isfinite(a).all() for a in (self.u, self.x, self.y)):
            raise ValueError("Nonfinite inputs")
        self.config = config or Config()
        if self.config.variant not in ("adaptive", "stationary", "gaussian"):
            raise ValueError("Unknown variant")
        # A stationary comparator has intercept-only range/skewness functions.
        constant = self.config.variant != "adaptive"
        self.nets = [Network(self.x.shape[1], self.config.hidden),
                     Network(0 if constant else 3, 0 if constant else self.config.hidden),
                     Network(0 if constant else 3, 0 if constant else self.config.hidden)]
        self.slices, start = [], 0
        for net in self.nets:
            self.slices.append(slice(start, start + net.size)); start += net.size
        self.scalar = start
        self.size = start + 4  # log(sigma^2), log(nu-2), raw(U), log(W)
        self.blocks = self.slices + [slice(start, start+4)]
        self.d2 = None if self.config.neighbors else cdist(self.u, self.u, "sqeuclidean")
        self.eye = None if self.config.neighbors else np.eye(len(self.y))
        if self.config.neighbors:
            from .vecchia import groups
            self.neighbor_groups = groups(self.u,self.config.neighbors)

    def initial(self, seed=0):
        rng = np.random.default_rng(seed)
        w = np.concatenate([n.initial(rng) for n in self.nets] +
                           [np.array([np.log(0.4), np.log(3), np.log(np.expm1(CENTER)), 0.])])
        w[self.slices[2].stop-1] = np.log(np.expm1(0.25))
        return w

    def fields(self, w, u, x):
        z = u if self.config.variant == "adaptive" else np.empty((len(u), 0))
        outputs, caches = zip(*(n.forward(w[s], inp) for n,s,inp in
                               zip(self.nets,self.slices,(x,z,z))))
        mean, alpha, rawphi = outputs
        delta = alpha / np.sqrt(1 + alpha*alpha)
        if self.config.variant == "gaussian":
            delta = np.zeros_like(delta)
        return mean, delta, softplus(rawphi), alpha, rawphi, caches

    def log_joint(self, w, gradient=True):
        """Density on unconstrained coordinates INCLUDING all Jacobians."""
        c, n, j = self.config, len(self.y), self.scalar
        ls, ln, ru, lw = w[j:]
        if max(abs(ls),abs(ln),abs(lw),abs(ru)) > 60:
            return (-np.inf, np.zeros_like(w)) if gradient else -np.inf
        s2, vnu, uu, ww = np.exp(ls), np.exp(ln), softplus(ru), np.exp(lw)
        nu = 2 + vnu
        gaussian = c.variant == "gaussian"
        we = 1. if gaussian else ww
        a = 0. if gaussian else (uu-CENTER)/np.sqrt(we)
        m, delta, phi, alpha, rp, cache = self.fields(w,self.u,self.x)
        r = self.y-m-delta*a
        if c.neighbors:
            from .vecchia import likelihood_statistics
            logdet,quad,vm,gp=likelihood_statistics(self.neighbor_groups,self.u,phi,r,we/s2,gradient,c.jitter)
        else:
            s = phi[:,None]**2 + phi[None,:]**2
            corr = np.exp(1.5*np.log(2*phi[:,None]*phi[None,:]/s)-self.d2/s)
            cc = corr + c.jitter*self.eye
            cf = cho_factor(cc,lower=True,check_finite=False)
            cr = cho_solve(cf,r,check_finite=False)
            quad = (we/s2)*np.dot(r,cr)
            logdet=2*np.log(np.diag(cf[0])).sum()
        lp = -0.5*(n*LOG2PI+n*(ls-np.log(we))+logdet+quad)
        # All network weights and biases have the declared common normal prior.
        lp += -0.5*np.sum((w[:j]/c.prior_sd)**2)-j*(np.log(c.prior_sd)+0.5*LOG2PI)
        lp += c.variance_shape*np.log(c.variance_scale)-gammaln(c.variance_shape)
        lp += -c.variance_shape*ls-c.variance_scale/s2
        lp += c.df_shape*np.log(c.df_rate)-gammaln(c.df_shape)+c.df_shape*ln-c.df_rate*vnu
        lp += 0.5*np.log(2/np.pi)-0.5*uu*uu-softplus(-ru)
        k = nu/2
        lp += k*np.log(k)-gammaln(k)+k*lw-k*ww
        if not gradient:
            return float(lp)
        g = np.zeros_like(w)
        if not c.neighbors: vm = we/s2*cr
        g[self.slices[0]] = self.nets[0].backward(w[self.slices[0]],cache[0],vm)
        gd = vm*a/(1+alpha*alpha)**1.5
        if gaussian: gd *= 0
        g[self.slices[1]] = self.nets[1].backward(w[self.slices[1]],cache[1],gd)
        if not c.neighbors:
            invc = cho_solve(cf,self.eye,check_finite=False)
            gc = 0.5*((we/s2)*np.outer(cr,cr)-invc)
            derivative = 1.5/phi[:,None]-3*phi[:,None]/s+2*self.d2*phi[:,None]/s**2
            gp = 2*np.sum(gc*corr*derivative,axis=1)
        gp *= expit(rp)
        g[self.slices[2]] = self.nets[2].backward(w[self.slices[2]],cache[2],gp)
        g[:j] -= w[:j]/c.prior_sd**2
        g[j] = 0.5*(quad-n)-c.variance_shape+c.variance_scale/s2
        g[j+1] = c.df_shape-c.df_rate*vnu + 0.5*vnu*(np.log(k)+1-digamma(k)+lw-ww)
        g[j+2] = -uu*expit(ru)+expit(-ru)
        g[j+3] = k*(1-ww)
        if not gaussian:
            g[j+2] += np.dot(vm,delta)*expit(ru)/np.sqrt(ww)
            g[j+3] += 0.5*(n-quad)-0.5*a*np.dot(vm,delta)
        return float(lp),g

    def predictive_components(self, weights, ustar, xstar, seed=1):
        """Joint draws. One U/W per field draw. Cholesky conditions on ALL training y.

        TSAVP follows Algorithm 2 in sn4, including its conditional mean correction.
        The shorter residual-only description is NOT used to discard this term.
        """
        rng = np.random.default_rng(seed)
        if self.config.neighbors:
            from .vecchia import predictive_components
            return predictive_components(self,weights,ustar,xstar,seed)
        nstar = len(ustar)
        vi, means, deltas, innovations, conditional_means = [],[],[],[],[]
        bases, residual_deltas = [],[]
        for w in weights:
            m,d,p,*_ = self.fields(w,self.u,self.x)
            ms,ds,ps,*_ = self.fields(w,ustar,xstar)
            ls,ln,ru,lw = w[self.scalar:]
            ww = 1. if self.config.variant == "gaussian" else np.exp(lw)
            a = 0. if self.config.variant == "gaussian" else (softplus(ru)-CENTER)/np.sqrt(ww)
            c = kernel(self.u,self.u,p,p)+self.config.jitter*self.eye
            cross = kernel(self.u,ustar,p,ps)
            cf = cho_factor(c,lower=True,check_finite=False)
            solve = cho_solve(cf,cross,check_finite=False)
            schur = kernel(ustar,ustar,ps,ps)+self.config.jitter*np.eye(nstar)-cross.T@solve
            schur = 0.5*(schur+schur.T)
            # No eigenvalue clipping: non-SPD results must fail visibly.
            lc = np.linalg.cholesky(schur)
            eps = np.exp(0.5*ls)*(lc@rng.normal(size=nstar))
            cm = ms+ds*a+cross.T@cho_solve(cf,self.y-m-d*a,check_finite=False)
            bases.append(ms+cross.T@cho_solve(cf,self.y-m,check_finite=False))
            residual_deltas.append(ds-cross.T@cho_solve(cf,d,check_finite=False))
            vi.append(cm+eps/np.sqrt(ww)); means.append(ms); deltas.append(ds)
            innovations.append(eps); conditional_means.append(cm)
        return dict(vi=np.asarray(vi),mean=np.asarray(means),delta=np.asarray(deltas),
                    innovation=np.asarray(innovations),conditional_mean=np.asarray(conditional_means),
                    calibration_base=np.asarray(bases),calibration_delta=np.asarray(residual_deltas))

