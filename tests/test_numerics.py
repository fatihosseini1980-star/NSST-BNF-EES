import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from nsst_bnf.model import Model, Config, kernel, CENTER
from nsst_bnf.inference import fit_vi, draw_vi
from nsst_bnf.calibration import calibrate


class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(90)
        self.u=self.rng.uniform(size=(12,3));self.y=self.rng.normal(size=12)

    def test_all_transformed_gradients(self):
        for variant in ['adaptive','stationary','gaussian']:
            for hidden in [0,4]:
                mod=Model(self.u,self.u,self.y,Config(hidden=hidden,variant=variant))
                w=mod.initial(6);w+=self.rng.normal(0,.07,len(w))
                lp,g=mod.log_joint(w)
                numeric=approx_derivative(lambda z:mod.log_joint(z,False),w,method='3-point').ravel()
                np.testing.assert_allclose(g,numeric,atol=2e-5,rtol=2e-5)

    def test_kernel_consistency_and_stationarity(self):
        u=self.rng.uniform(size=(30,3));phi=np.exp(self.rng.normal(-1,.4,30))
        c=kernel(u,u,phi,phi)
        self.assertGreater(np.linalg.eigvalsh(c).min(),-1e-12)
        np.testing.assert_allclose(np.diag(c),1,atol=1e-14)
        select=np.array([20,1,7,4])
        np.testing.assert_allclose(kernel(u[select],u[select],phi[select],phi[select]),c[np.ix_(select,select)])
        constant=np.full(30,.4)
        exact=np.exp(-np.sum((u[:,None,:]-u[None,:,:])**2,axis=2)/(2*.4**2))
        np.testing.assert_allclose(kernel(u,u,constant,constant),exact)

    def test_vecchia_gradient_and_dense_limit(self):
        for neighbors in [3,11]:
            mod=Model(self.u,self.u,self.y,Config(hidden=2,neighbors=neighbors))
            w=mod.initial(11);lp,g=mod.log_joint(w)
            numeric=approx_derivative(lambda z:mod.log_joint(z,False),w,method='3-point').ravel()
            np.testing.assert_allclose(g,numeric,atol=3e-5,rtol=3e-5)
            if neighbors==11:
                dense=Model(self.u,self.u,self.y,Config(hidden=2))
                ld,gd=dense.log_joint(w)
                np.testing.assert_allclose(lp,ld,atol=1e-8)
                np.testing.assert_allclose(g,gd,atol=1e-7)

    def test_vecchia_prediction_dense_limit(self):
        dense=Model(self.u,self.u,self.y,Config(hidden=2));w=dense.initial(21)
        approx=Model(self.u,self.u,self.y,Config(hidden=2,neighbors=20))
        us=self.rng.uniform(size=(4,3));ws=np.tile(w,(5,1))
        a=dense.predictive_components(ws,us,us);b=approx.predictive_components(ws,us,us)
        np.testing.assert_allclose(a['conditional_mean'],b['conditional_mean'],atol=1e-8)
        np.testing.assert_allclose(a['calibration_base'],b['calibration_base'],atol=1e-8)
        np.testing.assert_allclose(a['calibration_delta'],b['calibration_delta'],atol=1e-8)

    def test_algorithm2_calibration_mean_identity(self):
        from nsst_bnf.model import softplus
        mod=Model(self.u,self.u,self.y,Config(hidden=2));w=mod.initial(14)
        us=self.rng.uniform(size=(5,3));out=mod.predictive_components(np.tile(w,(5,1)),us,us)
        a=(softplus(w[mod.scalar+2])-CENTER)/np.sqrt(np.exp(w[mod.scalar+3]))
        np.testing.assert_allclose(out['calibration_base']+out['calibration_delta']*a,out['conditional_mean'],atol=1e-10)

    def test_common_latent_moments(self):
        u=self.u[:3];p=np.full(3,.4);d=np.array([-.7,.1,.8]);s2=.3;nu=7
        c=kernel(u,u,p,p);n=120000
        z=(self.rng.normal(size=(n,3))@np.linalg.cholesky(s2*c).T+
           (np.abs(self.rng.normal(size=(n,1)))-CENTER)*d)/np.sqrt(self.rng.gamma(nu/2,2/nu,(n,1)))
        np.testing.assert_allclose(z.mean(0),0,atol=.013)
        theory=nu/(nu-2)*(s2*c+(1-2/np.pi)*np.outer(d,d))
        np.testing.assert_allclose(np.cov(z,rowvar=False),theory,atol=.02,rtol=.035)

    def test_prediction_conditions_on_training(self):
        mod=Model(self.u,self.u,self.y,Config(hidden=0));w=mod.initial(10)
        result=mod.predictive_components(np.tile(w,(20,1)),self.u,self.u)
        np.testing.assert_allclose(result['conditional_mean'],np.tile(self.y,(20,1)),atol=1e-3)
        self.assertLess(np.sqrt(np.mean((result['vi']-self.y)**2)),.01)

    def test_vi_and_calibration_interface(self):
        mod=Model(self.u,self.u,self.y,Config(hidden=0))
        fit=fit_vi(mod,steps=40,seed=7)
        w=draw_vi(fit,mod.blocks,draws=50)
        out=mod.predictive_components(w,self.u[:4]+.01,self.u[:4]+.01)
        pars,records=calibrate(self.y[:4],out)
        self.assertEqual(len(records),48)
        self.assertTrue(all(np.isfinite(out['vi'].ravel())))
        # All low-rank network covariance blocks are positive definite.
        for s,f in zip(mod.blocks,fit['factors']):
            self.assertGreater(np.linalg.eigvalsh(np.diag(np.exp(2*fit['logsd'][s]))+f@f.T).min(),0)


if __name__=='__main__':unittest.main()
