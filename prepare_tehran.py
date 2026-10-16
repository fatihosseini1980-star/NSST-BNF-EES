"""Build the screened station-month data from the supplied daily AQI source."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def prepare(keep_500=False):
    daily=pd.read_csv(ROOT/'data/source_daily_aqi.csv')
    assert len(daily)==90023
    assert {'Stations','long','lat','PM2.5','year','month'}<=set(daily)
    duplicate=['Sharif','Shahrdari-2']
    daily=daily.loc[~daily.Stations.isin(duplicate)].copy()
    suspect=(daily.Stations.eq('Farmandari-rey') & daily.year.eq(2015) &
             daily.month.eq(9) & daily['PM2.5'].eq(500))
    assert suspect.sum()==1
    if not keep_500: daily=daily.loc[~suspect].copy()
    metadata=pd.read_csv(ROOT/'data/station_metadata.csv')
    avg=daily.groupby(['Stations','year','month'],as_index=False)['PM2.5'].mean()
    avg=avg.rename(columns={'Stations':'station','PM2.5':'pm25'})
    avg=avg.merge(metadata,on='station',how='left',validate='many_to_one')
    assert not avg[['lon','lat','region']].isna().any().any()
    encoding=pd.read_csv(ROOT/'data/month_encoding.csv')
    assert len(encoding)==12 and encoding.month.nunique()==12
    avg=avg.merge(encoding,on='month',how='left',validate='many_to_one')
    avg=avg[['station','lon','lat','year','month','season','m_sin','m_cos','pm25','region']]
    avg=avg.sort_values(['year','month','station']).reset_index(drop=True)
    expected=3348 if not keep_500 else 3348
    assert len(avg)==expected and avg.station.nunique()==36
    dest=ROOT/'data'/('monthly_exclude_two_keep500.csv' if keep_500 else 'monthly_qa_screened.csv')
    avg.to_csv(dest,index=False)
    if not keep_500:
        daily.to_csv(ROOT/'data/daily_quality_screened.csv',index=False)
    print(f'Created {dest.relative_to(ROOT)} with {len(avg)} rows')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--retain-500',action='store_true')
    a=parser.parse_args();prepare(a.retain_500)
