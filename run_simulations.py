"""Reproduce simulation, independent benchmarks, sensitivity and HMC diagnostics."""
from pathlib import Path
import argparse,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
def run(*args):
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
def main():
    p=argparse.ArgumentParser();p.add_argument('--replications',type=int,default=50);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--skip-hmc',action='store_true');p.add_argument('--hmc-draws',type=int,default=3000);p.add_argument('--hmc-warmup',type=int,default=2500)
    a=p.parse_args()
    run('run_studies.py','simulation','--replications',str(a.replications),'--workers',str(a.workers),'--steps','1200','--draws','400','--hidden','4','--rank','2','--out','results/simulation_algorithm2')
    run('run_benchmarks.py','--study','simulation','--replications',str(a.replications),'--workers',str(a.workers),'--out','results/independent_benchmarks')
    run('paired_comparisons.py')
    run('run_sensitivity.py','--replications','5','--workers',str(a.workers),'--out','results/sensitivity_algorithm2')
    if not a.skip_hmc:
        run('run_studies.py','mcmc','--steps','2000','--rank','2','--hidden','2','--draws',str(a.hmc_draws),'--warmup',str(a.hmc_warmup),'--target-accept','0.99','--leapfrog','50','--out','results/mcmc_long_trajectory')
        run('additional_diagnostics.py')
    run('plot_simulations.py')
if __name__=='__main__':main()
