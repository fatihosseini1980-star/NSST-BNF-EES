"""Monthly point predictions displayed on municipal polygons; no areal averaging.

Uses the saved VI fit, training scalers and original prediction ordering.
Existing station predictions and figures are not overwritten.
"""
from pathlib import Path
import json, calendar
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from nsst_bnf.model import Model, Config
from nsst_bnf.inference import draw_vi

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/tehran_regional_map'; OUT.mkdir(parents=True,exist_ok=True)
geo=json.loads((ROOT/'data/tehran_regions_wgs84.geojson').read_text())
regions=[]
for f in geo['features']:
    polygons=[f['geometry']['coordinates']] if f['geometry']['type']=='Polygon' else f['geometry']['coordinates']
    rings=[np.asarray(p[0]) for p in polygons]
    # Deterministic interior point nearest the polygon's vertex-average center.
    largest=max(range(len(rings)),key=lambda i:len(rings[i]))
    ring=rings[largest]; center=ring.mean(0)
    gx,gy=np.meshgrid(np.linspace(ring[:,0].min(),ring[:,0].max(),101),np.linspace(ring[:,1].min(),ring[:,1].max(),101))
    candidates=np.c_[gx.ravel(),gy.ravel()]
    inside=MPath(ring).contains_points(candidates)
    for hole in polygons[largest][1:]:
        inside &= ~MPath(np.asarray(hole)).contains_points(candidates)
    candidates=candidates[inside]
    point=candidates[np.argmin(((candidates-center)**2).sum(1))]
    regions.append((f['properties']['region'],rings,point))
regions.sort(key=lambda r:int(r[0]))
new=pd.DataFrame([dict(region=r,lon=p[0],lat=p[1],year=2024,month=m) for m in range(1,13) for r,_,p in regions])
new['m_sin']=np.sin(2*np.pi*new.month/12);new['m_cos']=np.cos(2*np.pi*new.month/12)
d=pd.read_csv(ROOT/'data/monthly_qa_screened.csv',encoding='utf-8')
d=d[d.year<=2024].sort_values(['year','month','station']).reset_index(drop=True)
# Reuse the supplied seasonal encoding exactly.
for m in range(1,13):
    for col in ['m_sin','m_cos']:new.loc[new.month==m,col]=d.loc[d.month==m,col].iloc[0]
folder=ROOT/'results/adaptive15'
meta=json.loads((folder/'tehran_run_started.json').read_text());sc=json.loads((folder/'tehran_scalers.json').read_text())
def transform(frame):
    raw=np.c_[frame.lon,frame.lat,12*(frame.year-2014)+frame.month-1]
    u=(raw-np.array(meta['coordinate_origin']))/np.array(meta['coordinate_scale'])
    x=(np.c_[u,frame.m_sin,frame.m_cos]-np.array(sc['feature_mean']))/np.array(sc['feature_sd'])
    return u,x
u,x=transform(d);un,xn=transform(new);tr=np.flatnonzero(d.year<=2022);old=np.flatnonzero(d.year>=2023)
model=Model(u[tr],x[tr],(d.pm25.to_numpy()[tr]-sc['response_mean'])/sc['response_sd'],Config(**meta['configuration']))
z=np.load(folder/'tehran_vi_fit.npz');fit=dict(loc=z['loc'],logsd=z['logsd'],factors=[z[f'factor_{i}'] for i in range(4)])
weights=draw_vi(fit,model.blocks,200,4402)
comps=model.predictive_components(weights,np.vstack([u[old],un]),np.vstack([x[old],xn]),4403)
# Centered TSAVP innovations have zero expectation; average conditional means
# over fitted variational weight draws (Rao-Blackwellized predictive mean).
new['prediction']=comps['calibration_base'][:,len(old):].mean(0)*sc['response_sd']+sc['response_mean']
assert np.isfinite(new.prediction).all() and len(new)==264
new.to_csv(OUT/'regional_point_predictions.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,axs=plt.subplots(3,4,figsize=(13,8.4))
fig.subplots_adjust(left=.055,right=.90,bottom=.075,top=.94,wspace=.10,hspace=.18)
xy=np.concatenate([a for _,rings,_ in regions for a in rings]);norm=Normalize(new.prediction.min(),new.prediction.max());cmap=plt.get_cmap('YlOrRd')
stations=d[d.year==2024].copy()
for month,ax in enumerate(axs.flat,1):
    sub=new[new.month==month].set_index('region')
    for region,rings,_ in regions:
        for ring in rings:ax.add_patch(Polygon(ring,facecolor=cmap(norm(sub.loc[region,'prediction'])),edgecolor='#555555',linewidth=.4))
    current_stations=stations[stations.month==month]
    ax.scatter(current_stations.lon,current_stations.lat,s=7,c='black',linewidths=0,zorder=3)
    ax.set_xlim(xy[:,0].min()-.015,xy[:,0].max()+.015);ax.set_ylim(xy[:,1].min()-.012,xy[:,1].max()+.012)
    ax.set_aspect(1/np.cos(np.deg2rad(35.7)));ax.set_title(f'{calendar.month_abbr[month]} 2024',fontsize=11)
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)
cax=fig.add_axes([.923,.21,.016,.57]);fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,label='Predicted PM2.5 AQI (index points)')
fig.text(.48,.985,'Monthly spatial predictions for Tehran',ha='center',va='top',fontsize=15)
fig.text(.48,.027,'Polygon colors: predictions at one interior point per district. Black dots: monitoring stations.',ha='center',fontsize=9)
for ext in ['png','pdf']:fig.savefig(OUT/f'Tehran_Monthly_Regional_Map.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)
print(new.prediction.describe().to_string())
