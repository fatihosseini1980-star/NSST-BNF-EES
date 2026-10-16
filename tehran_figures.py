"""Reproduce descriptive and fitted Tehran station maps from saved data/checkpoints."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import skew
from nsst_bnf.model import Model,Config
from nsst_bnf.inference import draw_vi
ROOT=Path(__file__).resolve().parent;OUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/tehran_figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
d=pd.read_csv(ROOT/'data/monthly_qa_screened.csv',encoding='utf-8');d=d[d.year<=2024].sort_values(['year','month','station']).reset_index(drop=True)
geo=json.loads((ROOT/'data/tehran_regions_wgs84.geojson').read_text());rings=[]
for f in geo['features']:
 g=f['geometry'];polys=[g['coordinates']] if g['type']=='Polygon' else g['coordinates']
 for poly in polys:rings.append(np.array(poly[0]))
xy=np.concatenate(rings);lim=[xy[:,0].min(),xy[:,0].max(),xy[:,1].min(),xy[:,1].max()]
def base(ax):
 for a in rings:ax.plot(a[:,0],a[:,1],color='#a4adb5',lw=.55,zorder=1)
 ax.set_xlim(lim[:2]);ax.set_ylim(lim[2:]);ax.set_aspect(1/np.cos(np.deg2rad(35.7)));ax.set_xlabel('Longitude');ax.set_ylabel('Latitude');ax.tick_params(labelsize=7)
def save(fig,name):fig.savefig(OUT/(name+'.png'),bbox_inches='tight');plt.close(fig)
def mapvalues(ax,frame,vals,title,vmin=None,vmax=None,cmap='viridis'):
 base(ax);sc=ax.scatter(frame.lon,frame.lat,c=vals,s=28,cmap=cmap,vmin=vmin,vmax=vmax,edgecolor='white',linewidth=.35,zorder=3);ax.set_title(title);return sc
st=d.groupby('station').agg(lon=('lon','first'),lat=('lat','first'),mean=('pm25','mean'),median=('pm25','median'),n=('pm25','size'),p90=('pm25',lambda x:x.quantile(.9)));st['tail_ratio']=st.p90/st['median'];st.to_csv(OUT/'station_descriptives.csv')
summary={'n':len(d),'stations':len(st),'mean':d.pm25.mean(),'median':d.pm25.median(),'IQR':d.pm25.quantile(.75)-d.pm25.quantile(.25),'p90':d.pm25.quantile(.9),'maximum':d.pm25.max(),'skewness':skew(d.pm25,bias=False),'station_mean_min':st['mean'].min(),'station_mean_max':st['mean'].max(),'station_n_min':int(st.n.min()),'station_n_max':int(st.n.max()),'tail_min':st.tail_ratio.min(),'tail_max':st.tail_ratio.max()};(OUT/'descriptives.json').write_text(json.dumps(summary,indent=2))
fig,ax=plt.subplots(figsize=(7,4));base(ax);ax.scatter(st.lon,st.lat,s=30,color='#18558d',edgecolor='white',lw=.4,zorder=3);ax.set_title(f'Tehran monitoring network: {len(st)} stations');save(fig,'tehran_stations')
fig,ax=plt.subplots(figsize=(7,3));ax.hist(d.pm25,bins=26,color='#518aaf',edgecolor='white',density=True);ax.axvline(summary['mean'],color='#b94d38',label=f"Mean {summary['mean']:.2f}");ax.axvline(summary['median'],color='#212c37',ls='--',label=f"Median {summary['median']:.2f}");ax.set_xlabel('Monthly PM2.5 AQI (index points)');ax.set_ylabel('Density');ax.legend();save(fig,'tehran_distribution')
fig,axs=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
for ax,col,title in zip(axs,['mean','tail_ratio'],['Station mean AQI','Station 90th percentile / median']):sc=mapvalues(ax,st,st[col],title);fig.colorbar(sc,ax=ax,shrink=.75)
save(fig,'tehran_observed_spatial')
season=d.groupby('month').pm25.agg(['mean','median','count']);season.to_csv(OUT/'seasonal_descriptives.csv');fig,ax=plt.subplots(figsize=(7,3));ax.plot(season.index,season['mean'],'o-',label='Mean');ax.plot(season.index,season['median'],'s--',label='Median');ax.set_xticks(range(1,13));ax.set_xlabel('Calendar month (2014-2024)');ax.set_ylabel('PM2.5 AQI (index points)');ax.legend();ax.grid(alpha=.2);save(fig,'tehran_seasonal')
folder=ROOT/'results/adaptive15';z=np.load(folder/'tehran_predictive_draws.npz');test=d.iloc[z['test_rows']].copy();samples=z['tsavp'];test['prediction']=samples.mean(0);test['upper95']=np.quantile(samples,.95,axis=0);test['lower05']=np.quantile(samples,.05,axis=0)
fig,axs=plt.subplots(1,3,figsize=(9,2.7),layout='constrained')
for ax,(f,key,label) in zip(axs,[('gaussian15','vi','Gaussian VI'),('stationary15','vi','Stationary skew-t VI'),('adaptive15','tsavp','Adaptive TSAVP')]):
 a=np.load(ROOT/f'results/{f}/tehran_predictive_draws.npz')[key];y=test.pm25.to_numpy();pit=((a<y).sum(0)+.5*(a==y).sum(0))/len(a);ax.hist(pit,bins=np.linspace(0,1,11),density=True,color='#518aaf',edgecolor='white');ax.axhline(1,color='#b94d38',ls='--');ax.set_title(label);ax.set_xlabel('PIT');ax.set_ylabel('Density')
save(fig,'tehran_pit')
dec=test[test.month==12];vmin=min(dec.pm25.min(),dec.prediction.min(),dec.upper95.min());vmax=max(dec.pm25.max(),dec.prediction.max(),dec.upper95.max());fig,axs=plt.subplots(1,3,figsize=(9,3.2),layout='constrained')
for ax,col,title in zip(axs,['pm25','prediction','upper95'],['Observed AQI','Predictive mean','Predictive 95th percentile']):sc=mapvalues(ax,dec,dec[col],title,vmin,vmax)
fig.colorbar(sc,ax=axs,shrink=.7,label='AQI points');save(fig,'tehran_december')
meta=json.loads((folder/'tehran_run_started.json').read_text());scalers=json.loads((folder/'tehran_scalers.json').read_text());tr=np.flatnonzero(d.year<=2022);raw=np.c_[d.lon,d.lat,12*(d.year-2014)+d.month-1];u=(raw-np.array(meta['coordinate_origin']))/np.array(meta['coordinate_scale']);x=np.c_[u,d.m_sin,d.m_cos];x=(x-np.array(scalers['feature_mean']))/np.array(scalers['feature_sd']);yy=(d.pm25.to_numpy()-scalers['response_mean'])/scalers['response_sd'];model=Model(u[tr],x[tr],yy[tr],Config(**meta['configuration']));fitz=np.load(folder/'tehran_vi_fit.npz');fit=dict(loc=fitz['loc'],logsd=fitz['logsd'],factors=[fitz[f'factor_{i}'] for i in range(4)]);weights=draw_vi(fit,model.blocks,200,4402);ids=test.index.to_numpy();fields=np.array([model.fields(w,u[ids],x[ids])[:3] for w in weights]);test['delta_mean']=fields[:,1,:].mean(0);test['phi_mean']=fields[:,2,:].mean(0);dec=test[test.month==12]
fig,axs=plt.subplots(1,2,figsize=(8,3.5),layout='constrained');sc=mapvalues(axs[0],dec,dec.delta_mean,'Mean skewness loading (December)',-1,1,'coolwarm');fig.colorbar(sc,ax=axs[0],shrink=.75,label='Standardized response scale');sc=mapvalues(axs[1],dec,dec.phi_mean,'Mean local range (December)');fig.colorbar(sc,ax=axs[1],shrink=.75,label='Scaled coordinate metric');save(fig,'tehran_fields')
avg=test.groupby('month').agg(observed_mean=('pm25','mean'),observed_median=('pm25','median'),prediction=('prediction','mean'));avg.to_csv(OUT/'test_monthly_descriptives.csv');fig,ax=plt.subplots(figsize=(7,3));
for col,label in [('observed_mean','Observed mean'),('observed_median','Observed median'),('prediction','TSAVP mean')]:ax.plot(avg.index,avg[col],marker='o',label=label)
ax.set_xticks(range(1,13));ax.set_xlabel('Month in 2024');ax.set_ylabel('PM2.5 AQI (index points)');ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.2);save(fig,'tehran_temporal')
fig,axs=plt.subplots(4,3,figsize=(7.5,9),layout='constrained');vmin=test.prediction.min();vmax=test.prediction.max()
for month,ax in enumerate(axs.flat,1):sub=test[test.month==month];sc=mapvalues(ax,sub,sub.prediction,f'2024-{month:02d} (n={len(sub)})',vmin,vmax);ax.set_xlabel('');ax.set_ylabel('')
fig.colorbar(sc,ax=axs,shrink=.55,label='Predictive mean AQI (index points)',location='bottom');save(fig,'tehran_monthly_predictions');test.to_csv(OUT/'mapped_test_predictions.csv',index=False)
print(json.dumps(summary));print(season.to_string())
