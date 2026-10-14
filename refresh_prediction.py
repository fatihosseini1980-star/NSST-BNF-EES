"""Regenerate joint predictions from saved VI checkpoints, without refitting.
Used after the Algorithm-2 conditional-mean implementation was reconciled.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from run_studies import dump
from nsst_bnf.model import Model,Config
from nsst_bnf.inference import draw_vi
from nsst_bnf.calibration import calibrate,tsavp,metrics


def refresh(folder,csv):
    out=Path(folder);meta=json.loads((out/'tehran_run_started.json').read_text())
    scalers=json.loads((out/'tehran_scalers.json').read_text());config=Config(**meta['configuration'])
    df=pd.read_csv(csv,encoding='cp1256');df=df[df.year<=2024].sort_values(['year','month','station']).reset_index(drop=True)
    train=np.flatnonzero(df.year.to_numpy()<=2022);val=np.flatnonzero(df.year.to_numpy()==2023);test=np.flatnonzero(df.year.to_numpy()==2024)
    raw=np.c_[df.lon,df.lat,12*(df.year-2014)+df.month-1]
    u=(raw-np.array(meta['coordinate_origin']))/np.array(meta['coordinate_scale'])
    x=np.c_[u,df.m_sin,df.m_cos];x=(x-np.array(scalers['feature_mean']))/np.array(scalers['feature_sd'])
    ym,ys=scalers['response_mean'],scalers['response_sd'];y=df.pm25.to_numpy();yy=(y-ym)/ys
    model=Model(u[train],x[train],yy[train],config)
    z=np.load(out/'tehran_vi_fit.npz');fit=dict(loc=z['loc'],logsd=z['logsd'],factors=[z[f'factor_{i}'] for i in range(4)])
    weights=draw_vi(fit,model.blocks,meta['args']['draws'],4402);idx=np.r_[val,test]
    comps=model.predictive_components(weights,u[idx],x[idx],4403)
    cv={k:v[:,:len(val)] for k,v in comps.items()};ct={k:v[:,len(val):] for k,v in comps.items()}
    pars,grid=calibrate(yy[val],cv,seed=4404)
    sv=ct['vi']*ys+ym;st=tsavp(ct,**pars,seed=4405)*ys+ym
    scores={'VI':metrics(y[test],sv),'TSAVP':metrics(y[test],st)}
    dump(out/'tehran_metrics.json',dict(scores=scores,calibration=pars,implementation='sn4 Algorithm 2 including conditional mean correction'))
    np.savez_compressed(out/'tehran_predictive_draws.npz',vi=sv,tsavp=st,test_rows=test)
    pd.DataFrame(grid).to_csv(out/'tehran_validation_grid.csv',index=False)
    pred=df.loc[test,['station','year','month','pm25']].copy()
    for name,samples in [('VI',sv),('TSAVP',st)]:
        pred[name+'_mean']=samples.mean(0);pred[name+'_lower']=np.quantile(samples,.05,axis=0);pred[name+'_upper']=np.quantile(samples,.95,axis=0)
    pred.to_csv(out/'tehran_predictions.csv',index=False)
    print(folder,json.dumps(scores),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--csv',required=True);a=p.parse_args();refresh(a.folder,a.csv)
