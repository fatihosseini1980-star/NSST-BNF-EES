#!/usr/bin/env python3
"""Publication figure: continuous *interpolation* of saved TSAVP point predictions.

No model refitting, no modification of fitted or station predictive values, no use of
2024 observed AQI in the interpolation. This is a deterministic cartographic
secondary visualization of 22 model-derived representative-point predictions
per month. It is NOT direct neural-field prediction on a dense grid and it is
NOT an observed or model-estimated municipal areal mean.

Run from the supplied supplement directory:
  python tehran_interpolated_map.py
Or externally:
  python tehran_interpolated_map.py --supplement-root /path/to/Supplementary_Code_and_Results --output /path/to/output
"""
from pathlib import Path
import argparse
import json
import calendar
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely import affinity
from pyproj import Transformer
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def idw_predict(control, values, query, power=2):
    distances=cdist(query,control)
    zero=distances<1e-10
    wt=1/np.maximum(distances,1e-10)**power
    pred=(wt@values)/wt.sum(axis=1)
    if zero.any():
        i,j=np.nonzero(zero)
        pred[i]=values[j]
    return pred


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supplement-root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--nx',type=int,default=280)
    parser.add_argument('--ny',type=int,default=190)
    args=parser.parse_args()
    root=args.supplement_root
    out=args.output or (root/'results/tehran_interpolation')
    out.mkdir(parents=True,exist_ok=True)
    control=pd.read_csv(root/'results/tehran_regional_map/regional_point_predictions.csv')
    assert len(control)==264 and set(control.region)==set(range(1,23)) and all(control.groupby('month').size()==22)
    assert control[['lon','lat','prediction']].notna().all().all()
    gdf=gpd.read_file(root/'data/tehran_regions_wgs84.geojson').to_crs(32639).sort_values('region')
    assert len(gdf)==22 and set(gdf.region)==set(range(1,23))
    city=gdf.geometry.union_all()
    gdf_km=gdf.copy()
    gdf_km.geometry=gdf.geometry.apply(lambda geom: affinity.scale(geom, xfact=.001, yfact=.001, origin=(0,0)))
    transformer=Transformer.from_crs('EPSG:4326','EPSG:32639',always_xy=True)
    xx,yy=transformer.transform(control.lon.to_numpy(),control.lat.to_numpy())
    control['x_m'],control['y_m']=xx,yy
    assert all(city.covers(shapely.Point(x,y)) for x,y in zip(xx[:22],yy[:22]))
    controls=control[control.month==1].sort_values('region')[['x_m','y_m']].to_numpy()/1000
    assert np.allclose(control[['x_m','y_m']].to_numpy().reshape(12,22,2)/1000, controls)
    # Cartesian metric grid in meters, mask exact city union (all 22 polygons).
    xmin,ymin,xmax,ymax=city.bounds
    pad=.008*(xmax-xmin)
    xaxis=np.linspace(xmin-pad,xmax+pad,args.nx)
    yaxis=np.linspace(ymin-pad,ymax+pad,args.ny)
    mx,my=np.meshgrid(xaxis,yaxis)
    mask=shapely.contains_xy(city,mx.ravel(),my.ravel())
    grid=np.c_[mx.ravel()[mask],my.ravel()[mask]]/1000
    assert mask.any()
    station=pd.read_csv(root/'results/tehran_vecchia15/tehran_predictions.csv')
    mdata=pd.read_csv(root/'data/monthly_supplied.csv',encoding='cp1256')
    locs=mdata.drop_duplicates('station').set_index('station')[['lon','lat']]
    station=station.join(locs,on='station')
    assert not station[['lon','lat']].isna().any().any()
    sx,sy=transformer.transform(station.lon.to_numpy(),station.lat.to_numpy())
    station['x_m'],station['y_m']=sx,sy
    arrays=[];compare=[];knot_max_error=[];grid_values=[]
    for month in range(1,13):
        cc=control.loc[control.month==month].sort_values('region')
        yy=cc.prediction.to_numpy()
        rbf=RBFInterpolator(controls,yy,kernel='thin_plate_spline',smoothing=0)
        # Scipy TPS reproduces saved predictions at its 22 anchor sites exactly.
        anchor_err=np.max(np.abs(rbf(controls)-yy));knot_max_error.append(anchor_err)
        pred=rbf(grid)
        arrays.append(pred)
        block=station[station.month==month].copy()
        query=block[['x_m','y_m']].to_numpy()/1000
        block['RBF_from_22']=rbf(query)
        block['IDW_p2_from_22']=idw_predict(controls,yy,query,power=2)
        block['RBF_minus_station_TSAVP']=block.RBF_from_22-block.TSAVP_mean
        block['IDW_minus_station_TSAVP']=block.IDW_p2_from_22-block.TSAVP_mean
        compare.append(block[['station','year','month','pm25','TSAVP_mean','RBF_from_22','IDW_p2_from_22',
                              'RBF_minus_station_TSAVP','IDW_minus_station_TSAVP']])
        xxg,myg=mx.ravel()[mask],my.ravel()[mask]
        grid_values.append(pd.DataFrame({'year':2024,'month':month,'x_utm_m':xxg,'y_utm_m':myg,'prediction_interpolated':pred}))
    validated=pd.concat(compare,ignore_index=True)
    interp=pd.concat(grid_values,ignore_index=True)
    lo,hi=min(np.min(a) for a in arrays),max(np.max(a) for a in arrays)
    vmin=5*np.floor(lo/5);vmax=5*np.ceil(hi/5)
    norm=Normalize(vmin=vmin,vmax=vmax)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'savefig.facecolor':'white'})
    fig,axs=plt.subplots(4,3,figsize=(10.4,12.8),layout=None)
    fig.subplots_adjust(left=.045,right=.985,bottom=.12,top=.948,wspace=.09,hspace=.095)
    cm=plt.get_cmap('YlOrRd').copy();cm.set_bad((1,1,1,0))
    for month,ax in enumerate(axs.flat,1):
        panel=np.full(mx.size,np.nan,dtype=float);panel[mask]=arrays[month-1]
        field=np.ma.masked_invalid(panel.reshape(mx.shape))
        ax.pcolormesh(mx/1000,my/1000,field,shading='nearest',cmap=cm,norm=norm,rasterized=True,zorder=1)
        gdf_km.boundary.plot(ax=ax, color='#565d64',linewidth=.43,zorder=2)
        block=station[station.month==month]
        ax.scatter(block.x_m/1000,block.y_m/1000,s=10,c='#131313',edgecolor='white',linewidth=.24,zorder=4)
        ax.set_xlim((xmin-pad)/1000,(xmax+pad)/1000)
        ax.set_ylim((ymin-pad)/1000,(ymax+pad)/1000)
        ax.set_aspect('equal','box');ax.axis('off')
        ax.set_title(f'{calendar.month_abbr[month]} 2024   (n={len(block)})',fontsize=10.5,pad=4)
    fig.suptitle('Tehran PM$_{2.5}$ AQI: interpolated TSAVP predictive mean (2024)',fontsize=14.0,y=.989)
    cax=fig.add_axes([.19,.074,.62,.014]); cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm),cax=cax,orientation='horizontal')
    cb.set_label('PM$_{2.5}$ AQI (index points)',fontsize=10);cb.set_ticks(np.arange(vmin,vmax+1,10))
    fig.text(.5,.028,'Thin-plate-spline interpolation of 22 model-predicted interior points per month; not areal averages.',ha='center',fontsize=8.4)
    fig.text(.5,.012,'Black dots: stations with test observations in the corresponding month. Same color scale across panels.',ha='center',fontsize=8.4)
    png=out/'Tehran_2024_interpolated_TSAVP_12_panels.png'
    pdf=out/'Tehran_2024_interpolated_TSAVP_12_panels.pdf'
    fig.savefig(png,dpi=320,bbox_inches='tight')
    fig.savefig(pdf,bbox_inches='tight')
    plt.close(fig)
    validated.to_csv(out/'interpolation_vs_station_predictions.csv',index=False)
    control.to_csv(out/'regional_anchor_predictions_with_utm.csv',index=False)
    interp.to_csv(out/'interpolated_grid_predictions_2024.csv.gz',index=False,compression='gzip')
    errs=validated.RBF_minus_station_TSAVP.to_numpy();err2=validated.IDW_minus_station_TSAVP.to_numpy()
    diagnostics={
      'source':'Saved 22 representative-point posterior predictive means per month from tehran_regional_map.py',
      'interpolation':'Exact thin-plate-spline radial-basis interpolator applied separately for each month; metric UTM EPSG:32639',
      'no_model_refitting':True,'no_observed_2024_targets_used_in_interpolation':True,
      'not':'Not a direct dense-grid neural-field forecast, nor a true polygon/areal average, nor an observed PM2.5 AQI surface.',
      'number_of_anchor_predictions':int(len(control)),'month_station_counts':{int(k):int(v) for k,v in station.groupby('month').size().items()},
      'max_absolute_error_at_22_anchor_points':float(max(knot_max_error)),
      'RBF_reconstruction_of_339_saved_station_predictions_MAE_AQI':float(np.abs(errs).mean()),
      'RBF_reconstruction_of_339_saved_station_predictions_RMSE_AQI':float(np.sqrt(np.mean(errs**2))),
      'RBF_reconstruction_bias_AQI':float(errs.mean()),
      'IDW_p2_reconstruction_of_339_saved_station_predictions_MAE_AQI':float(np.abs(err2).mean()),
      'grid_points_inside_city_per_month':int(mask.sum()),
      'interpolated_global_min_max_AQI':[float(lo),float(hi)],
      'anchor_global_min_max_AQI':[float(control.prediction.min()),float(control.prediction.max())],
      'shared_colorbar_range_AQI':[float(vmin),float(vmax)],
      'warning':'Interpolation can overshoot near city boundaries; it adds no independent predictive information; the station comparison validates approximation to fitted model, not predictive accuracy versus observed outcomes.'
    }
    (out/'interpolation_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    print(json.dumps(diagnostics,indent=2))
    print('outputs:',png,pdf)

if __name__=='__main__': main()
