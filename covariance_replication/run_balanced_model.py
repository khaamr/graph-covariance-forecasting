"""Reproduce the selected balanced model without running failed challengers.

Inputs are the pinned realized panel and the five-expert archive produced by
prepare_joint_experts.py. This preserves the established common benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from balanced_covariance import balanced_covariance_forecast
from evaluation import covariance_losses,long_only_weights,point_ratios,portfolio_stats
from literature_models import as_correlation,select_ridge,volatility_features
from models import Forecasts
from run_joint_pool import paired_intervals


MODEL='Stress-reconciled log-ridge covariance'


def run(panel_path,expert_path,output):
    output.mkdir(parents=True,exist_ok=True);panel=np.load(panel_path);experts=np.load(expert_path)
    paths=[panel_path,expert_path]
    (output/'manifest.json').write_text(json.dumps({'model':MODEL,
        'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ['balanced_covariance.py','run_balanced_model.py']},
        'qualification':'Exploratory selected model on the established common benchmark; not fresh confirmation.'},indent=2)+'\n')
    full_dates=pd.DatetimeIndex(experts['dates'])
    selected=np.flatnonzero((np.arange(len(full_dates))>=378)&(full_dates>='2024-01-01'))
    dates=full_dates[selected];positions=experts['positions'][selected]
    np.testing.assert_array_equal(panel['dates'][positions],experts['dates'][selected])
    np.testing.assert_allclose(panel['covariance'][positions],experts['actual'][selected],rtol=1e-12,atol=2e-12)
    clock=pd.DatetimeIndex(panel['reference']).get_indexer(panel['dates'])
    complete=np.zeros(len(clock),dtype=bool);complete[22:]=clock[22:]-clock[:-22]==22
    target=np.log(np.diagonal(panel['covariance'],axis1=-2,axis2=-1))
    features,_=volatility_features(target,tuple(panel['symbols']))
    graph=experts['bank'][selected,2];ridge=np.empty_like(graph);months=dates.to_period('M');folds=[]
    for month in months.unique():
        rows=np.flatnonzero(months==month);origin=positions[rows[0]]
        train=np.arange(max(22,origin-1000),origin);train=train[complete[train]]
        variance,penalty,_=select_ridge(features,target,train,positions[rows],[.0001,.001,.01,.1,1.])
        std=np.sqrt(variance);ridge[rows]=as_correlation(graph[rows])*std[:,:,None]*std[:,None,:]
        folds.append({'month':str(month),'first_forecast':dates[rows[0]],
                      'last_training_label':pd.Timestamp(panel['dates'][train[-1]]),
                      'training_labels':len(train),'ridge_penalty':penalty})
    pd.DataFrame(folds).to_csv(output/'ridge-folds.csv',index=False)
    predicted=[]
    for row,position in enumerate(positions):
        predicted.append(balanced_covariance_forecast(graph[row],ridge[row],panel['covariance'][:position],
                                                      panel['dates'][:position],dates[row]))
        if row%100==0:print('Balanced forecasts:',row+1,'/',len(positions),flush=True)
    predicted=np.asarray(predicted)
    f=Forecasts(dates,positions,experts['actual'][selected],{'HAR-DRD':experts['bank'][selected,0],MODEL:predicted},pd.DataFrame())
    np.savez_compressed(output/'forecasts.npz',dates=dates.to_numpy(),positions=positions,actual=f.actual,
                        log_graph=graph,ridge_graph=ridge,**f.predictions)
    losses=covariance_losses(f);losses.to_csv(output/'daily-losses.csv',index=False)
    returns=panel['returns'][positions];weights={name:long_only_weights(value) for name,value in f.predictions.items()}
    portfolio=pd.DataFrame({name:np.sum(w*returns,axis=1) for name,w in weights.items()},index=dates)
    portfolio.to_csv(output/'portfolio-returns.csv',index_label='date')
    for period,mask in [('development_2024',dates<'2025-01-01'),('reused_2025_2026',dates>='2025-01-01')]:
        folder=output/period;folder.mkdir(exist_ok=True);rows=np.flatnonzero(mask)
        subset=losses[losses.date.isin(dates[rows])];scores=point_ratios(subset)
        risk=pd.DataFrame([{'model':name,'annual_risk_percent':portfolio_stats(w[rows],returns[rows])[0],
                            'turnover':portfolio_stats(w[rows],returns[rows])[1]} for name,w in weights.items()]).set_index('model')
        scores=scores.join(risk);scores.to_csv(folder/'scores.csv')
        intervals=paired_intervals(subset,portfolio.iloc[rows],'HAR-DRD',MODEL,10000,10,20261127)
        intervals.to_csv(folder/'paired-intervals.csv',index=False)
        print(period,'\n'+scores.round(6).to_string(),flush=True)
    print('Finished standalone model reproduction.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['panel','experts','output']:parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    with threadpool_limits(limits=1):run(a.panel,a.experts,a.output)
