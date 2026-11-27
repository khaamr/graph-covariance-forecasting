"""Re-evaluate the selected, recorded balanced candidate through its public API."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from balanced_covariance import balanced_covariance_forecast
from evaluation import covariance_losses,long_only_weights,point_ratios,portfolio_stats
from models import Forecasts
from run_joint_pool import paired_intervals


MODEL='Stress-reconciled log-ridge covariance'


def run(panel_path,controls_path,weight_path,output):
    output.mkdir(parents=True,exist_ok=True)
    panel=np.load(panel_path);controls=np.load(controls_path);old_weights=np.load(weight_path)
    dates=pd.DatetimeIndex(controls['dates']);positions=controls['positions']
    sources={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
             for name in ['balanced_covariance.py','evaluate_balanced_covariance.py']}
    manifest={'model':MODEL,'source_sha256':sources,'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [panel_path,controls_path,weight_path]},
               'qualification':'Previously defined stress-family candidate159, highlighted after retrospective comparison. All live forecasts consume only earlier observations.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    np.testing.assert_array_equal(panel['dates'][positions],controls['dates'])
    predictions=[]
    for row,position in enumerate(positions):
        predictions.append(balanced_covariance_forecast(controls['Log/Fisher GHAR(GL,L)'][row],
                            controls['Ridge log-HAR + graph correlation'][row],panel['covariance'][:position],
                            panel['dates'][:position],dates[row]))
        if row%100==0:print('Public API forecasts checked:',row+1,'/',len(positions),flush=True)
    predictions=np.asarray(predictions);np.linalg.cholesky(predictions)
    names=['HAR-DRD','Uniform covariance pool','Fixed equal log-ridge pool','Fixed stress-gated covariance reconciliation']
    models={name:controls[name] for name in names};models[MODEL]=predictions
    forecast=Forecasts(dates,positions,controls['actual'],models,pd.DataFrame())
    np.savez_compressed(output/'forecasts.npz',dates=dates.to_numpy(),positions=positions,actual=forecast.actual,**models)
    losses=covariance_losses(forecast);losses.to_csv(output/'daily-losses.csv',index=False)
    returns=panel['returns'][positions];weights={name:old_weights[name] for name in names}
    weights[MODEL]=long_only_weights(predictions)
    realized=pd.DataFrame({name:np.sum(w*returns,axis=1) for name,w in weights.items()},index=dates)
    realized.to_csv(output/'portfolio-returns.csv',index_label='date')
    np.savez_compressed(output/'portfolio-weights.npz',**weights)
    for period,mask in [('development_2024',dates<'2025-01-01'),('reused_2025_2026',dates>='2025-01-01')]:
        folder=output/period;folder.mkdir(exist_ok=True);rows=np.flatnonzero(mask)
        subset=losses[losses.date.isin(dates[rows])];scores=point_ratios(subset)
        risk=pd.DataFrame([{'model':name,'annual_risk_percent':portfolio_stats(w[rows],returns[rows])[0],
                            'turnover':portfolio_stats(w[rows],returns[rows])[1]} for name,w in weights.items()]).set_index('model')
        scores=scores.join(risk);scores.to_csv(folder/'scores.csv');frames=[]
        for baseline in ['HAR-DRD','Uniform covariance pool','Fixed equal log-ridge pool']:
            frame=paired_intervals(subset,realized.iloc[rows],baseline,MODEL,10000,10,20261127)
            frame.insert(0,'baseline',baseline);frames.append(frame)
        pd.concat(frames).to_csv(folder/'paired-intervals.csv',index=False)
        print(period,'\n'+scores.round(6).to_string(),flush=True)
        print(frames[0].round(6).to_string(index=False),flush=True)
    print('Finished standalone balanced-model verification.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['panel','controls','control-weights','output']:parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    with threadpool_limits(limits=1):run(a.panel,a.controls,a.control_weights,a.output)
