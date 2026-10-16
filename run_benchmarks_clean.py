"""Independent comparator refit on the conservative QA dataset, no 2024 tuning."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from run_benchmarks import fit_benchmarks
out=ROOT/'results/independent';out.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(ROOT/'data/monthly_qa_screened.csv');df=df[df.year<=2024].sort_values(['year','month','station']).reset_index(drop=True)
tr=np.flatnonzero(df.year<=2022);va=np.flatnonzero(df.year==2023);te=np.flatnonzero(df.year==2024)
raw=np.c_[df.lon,df.lat,12*(df.year-2014)+df.month-1];u=(raw-raw[tr].min(0))/np.ptp(raw[tr],axis=0);x=np.c_[u,df.m_sin,df.m_cos]
rows,grid,preds=fit_benchmarks(u,x,df.pm25.to_numpy(),tr,va,te,4401)
pd.DataFrame(rows).to_csv(out/'tehran_metrics.csv',index=False);pd.DataFrame(grid).to_csv(out/'tehran_validation_grid.csv',index=False)
np.savez_compressed(out/'tehran_predictive_draws.npz',**preds,test_rows=te)
(out/'configuration.json').write_text(json.dumps({'train':len(tr),'validation':len(va),'test':len(te),'data':'data/monthly_qa_screened.csv','selection':'validation 2023; not test 2024','method':'The fixed fit_benchmarks implementation in run_benchmarks.py','seed':4401},indent=2))
print(pd.DataFrame(rows).to_string(index=False),flush=True)
