"""Reproduce diagnostic quantities and exploratory before/after comparisons."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_vhar import causal_clean, factor_decomposition
from market_data import SECTORS
from realized_heavy import as_correlation
from run_joint_pool import paired_intervals


def run(panel_path, repaired_folder, output):
    output.mkdir(parents=True,exist_ok=True)
    panel=np.load(panel_path);forecasts=np.load(repaired_folder/'forecasts.npz')
    mask=forecasts['dates']>=np.datetime64('2025-01-01')
    selected=forecasts['positions'][mask];actual=forecasts['actual'][mask]
    variance=np.diagonal(actual,axis1=-2,axis2=-1);corr=as_correlation(actual)
    records=[]
    for name in forecasts.files:
        if name in ['actual','positions','dates']:continue
        forecast=forecasts[name][mask];v=np.diagonal(forecast,axis1=-2,axis2=-1)
        records.append({'model':name,'mean_trace_ratio':np.trace(forecast,axis1=-2,axis2=-1).mean()/np.trace(actual,axis1=-2,axis2=-1).mean(),
                        'variance_qlike_div':np.mean(np.log(v/variance)+variance/v-1),
                        'correlation_frob':np.linalg.norm(as_correlation(forecast)-corr,axis=(1,2)).mean(),
                        'mean_frob':np.linalg.norm(forecast-actual,axis=(1,2)).mean()})
    pd.DataFrame(records).to_csv(output/'component-diagnostics.csv',index=False)
    covariance=panel['covariance'];factor,beta,residual=factor_decomposition(covariance,np.ones((len(panel['symbols']),1))/len(panel['symbols']))
    blocked=np.zeros_like(residual);sectors=np.array([SECTORS[str(s)] for s in panel['symbols']])
    for sector in dict.fromkeys(sectors):
        indices=np.flatnonzero(sectors==sector)
        blocked[:,indices[:,None],indices]=residual[:,indices[:,None],indices]
    reconstructed=beta.swapaxes(-1,-2)@factor@beta+blocked
    clean,flag=causal_clean(covariance)
    structural={'later_origins':len(selected),'later_flagged':int(flag[selected].sum()),'whole_panel_flagged':int(flag.sum()),
                'same_day_structural_relative_frobenius':float(np.linalg.norm(covariance[selected]-reconstructed[selected],axis=(1,2)).mean()/np.linalg.norm(covariance[selected],axis=(1,2)).mean()),
                'cleaned_to_raw_mean_trace_later':float(np.trace(clean[selected],axis1=-2,axis2=-1).mean()/np.trace(covariance[selected],axis1=-2,axis2=-1).mean()),
                'flagged_dates':[str(d.date()) for d in pd.DatetimeIndex(panel['dates'][selected[flag[selected]]])],
                'qualification':'Same-day realized decompositions diagnose structural approximation only. They are not forecasts or admissible oracle model inputs.'}
    (output/'structural-diagnostics.json').write_text(json.dumps(structural,indent=2)+'\n')
    losses=pd.read_csv(repaired_folder/'daily-losses.csv',parse_dates=['date'])
    returns=pd.read_csv(repaired_folder/'portfolio-returns.csv',index_col='date',parse_dates=True)
    pairs=[('DCC-NL analytical target','Realized DCC-HEAVY M'),
           ('Factor-VHAR market LASSO','Smeared raw factor-VHAR market LASSO')]
    frames=[]
    for period,period_mask in [('development_2024',returns.index<'2025-01-01'),('reused_2025_2026',returns.index>='2025-01-01')]:
        values=returns.loc[period_mask];subset=losses[losses.date.isin(values.index)]
        for baseline,candidate in pairs:
            frame=paired_intervals(subset,values,baseline,candidate,10000,10,20261127)
            frame.insert(0,'candidate',candidate);frame.insert(0,'baseline',baseline);frame.insert(0,'period',period)
            frames.append(frame)
    pd.concat(frames).to_csv(output/'before-after-intervals.csv',index=False)
    (output/'diagnostic-manifest.json').write_text(json.dumps({
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'qualification':'Before/after intervals added as exploratory reporting after candidate evaluation. No model parameters changed.'},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['panel','repaired-folder','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.panel,args.repaired_folder,args.output)
