"""Reproduce the specified Tehran analysis with the fixed 2022/2023/2024 split."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
MODELS=[('adaptive15','adaptive',15),('stationary15','stationary',15),('gaussian15','gaussian',15),('adaptive30','adaptive',30)]
def run(*a):
    subprocess.run([sys.executable,*a],cwd=ROOT,check=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
def main():
    p=argparse.ArgumentParser();p.add_argument('--models',action='store_true');p.add_argument('--benchmarks',action='store_true');p.add_argument('--figures',action='store_true');p.add_argument('--retain-500-sensitivity',action='store_true');p.add_argument('--all',action='store_true');a=p.parse_args()
    selected=not any([a.models,a.benchmarks,a.figures,a.retain_500_sensitivity,a.all])
    if a.all or a.models or selected:
        for name,variant,neighbors in MODELS:
            run('run_studies.py','tehran','--csv','data/monthly_qa_screened.csv','--variant',variant,'--neighbors',str(neighbors),'--steps','1500','--draws','200','--rank','2','--hidden','4','--out',f'results/{name}')
    if a.all or a.benchmarks or selected:run('run_benchmarks_clean.py')
    if a.all or a.retain_500_sensitivity:
        run('run_studies.py','tehran','--csv','data/monthly_exclude_two_keep500.csv','--variant','adaptive','--neighbors','15','--steps','1500','--draws','200','--rank','2','--hidden','4','--out','results/sensitivity_retaining500')
    if a.all or a.figures or selected:
        for script in ['tehran_figures.py','plot_temporal_verified.py','tehran_regional_map.py','tehran_interpolated_map.py']:
            run(script)
        import shutil
        source=ROOT/'results/tehran_figures';target=ROOT/'figures/tehran';target.mkdir(parents=True,exist_ok=True)
        for name in ['tehran_stations','tehran_distribution','tehran_observed_spatial','tehran_seasonal','tehran_pit','tehran_december','tehran_fields','tehran_temporal_verified','tehran_monthly_predictions']:
            shutil.copy2(source/(name+'.png'),target/(name+'.png'))
        shutil.copy2(ROOT/'results/tehran_interpolation/Tehran_2024_interpolated_TSAVP_12_panels.png',target/'Tehran_2024_interpolated_TSAVP_12_panels.png')
if __name__=='__main__':main()
