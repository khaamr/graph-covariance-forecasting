"""Run the fixed joint candidate and retain all endpoint/control results."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from evaluation import METRICS,bootstrap_means,covariance_losses,forecast_table,long_only_weights,portfolio_stats
from joint_pool import MODEL,EXPERT_NAMES,fit_pool
from models import Forecasts


def paired_intervals(losses,portfolio_returns,baseline,candidate,draws,block,seed):
    pieces=[]
    for metric in METRICS:
        daily=losses.pivot(index='date',columns='model',values=metric)
        pieces.extend([daily[baseline].to_numpy(),daily[candidate].to_numpy()])
    a=portfolio_returns[baseline].to_numpy();b=portfolio_returns[candidate].to_numpy()
    pieces.extend([a,b,a*a,b*b]);data=np.column_stack(pieces)
    samples=bootstrap_means(data,draws,block,np.random.default_rng(seed))
    point=data.mean(axis=0)
    rows=[]
    for i,metric in enumerate(METRICS):
        value=100*(point[2*i+1]/point[2*i]-1)
        boot=100*(samples[:,2*i+1]/samples[:,2*i]-1)
        lo,hi=np.quantile(boot,[.025,.975])
        rows.append({'endpoint':metric,'change_percent':value,'lower_95':lo,'upper_95':hi})
    value=100*((point[9]-point[7]**2)/(point[8]-point[6]**2)-1)
    boot=100*((samples[:,9]-samples[:,7]**2)/(samples[:,8]-samples[:,6]**2)-1)
    lo,hi=np.quantile(boot,[.025,.975])
    rows.append({'endpoint':'Long-only portfolio variance','change_percent':value,'lower_95':lo,'upper_95':hi})
    return pd.DataFrame(rows)


def run(expert_path,config_path,output_dir):
    config=json.loads(config_path.read_text());output_dir.mkdir(parents=True,exist_ok=True)
    manifest={'config':config,'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
              'experts_sha256':hashlib.sha256(expert_path.read_bytes()).hexdigest(),
              'source_sha256':{p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest() for p in ['joint_pool.py','run_joint_pool.py','models.py']}}
    (output_dir/'run-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    z=np.load(expert_path);dates=pd.DatetimeIndex(z['dates'])
    pool=fit_pool(dates,z['positions'],z['bank'],z['actual'],z['returns'],config,progress=lambda x:print(x,flush=True))
    f=pool.forecasts
    pool.audit.to_csv(output_dir/'folds.csv',index=False)
    pool.validation.to_csv(output_dir/'validation-candidates.csv',index=False)
    pool.solver_audit.to_csv(output_dir/'selection-solver-audit.csv',index=False)
    np.savez_compressed(output_dir/'forecasts.npz',dates=f.dates.to_numpy(),positions=f.positions,actual=f.actual,**f.predictions)
    aligned=dates.get_indexer(f.dates);returns=z['returns'][aligned]
    assert (aligned>=0).all()
    print('Evaluating exact long-only portfolios',flush=True)
    stock_weights={};portfolio_returns={};risk_records=[]
    for model,matrices in f.predictions.items():
        w=long_only_weights(matrices)
        stock_weights[model]=w
        portfolio_returns[model]=np.sum(w*returns,axis=1)
        risk,turnover=portfolio_stats(w,returns)
        risk_records.append({'model':model,'annual_risk_percent':risk,'turnover':turnover})
    p_returns=pd.DataFrame(portfolio_returns,index=f.dates)
    p_returns.to_csv(output_dir/'portfolio-returns.csv',index_label='date')
    pd.DataFrame(risk_records).to_csv(output_dir/'all-period-portfolio.csv',index=False)
    np.savez_compressed(output_dir/'portfolio-weights.npz',**stock_weights)
    all_losses=covariance_losses(f);all_losses.to_csv(output_dir/'daily-losses.csv',index=False)
    for period,mask in [('development_2024',f.dates<'2025-01-01'),('reused_2025_2026',f.dates>='2025-01-01')]:
        selected=np.flatnonzero(mask)
        if not len(selected):continue
        folder=output_dir/period;folder.mkdir(exist_ok=True)
        losses=all_losses[all_losses.date.isin(f.dates[selected])]
        scores=forecast_table(losses)
        scores.to_csv(folder/'scores.csv')
        risk=[]
        for model,w in stock_weights.items():
            value,turnover=portfolio_stats(w[selected],returns[selected])
            risk.append({'model':model,'origins':len(selected),'annual_risk_percent':value,'turnover':turnover})
        risk=pd.DataFrame(risk).set_index('model');risk.to_csv(folder/'portfolios.csv')
        intervals=paired_intervals(losses,p_returns.iloc[selected],'HAR-DRD',MODEL,config['bootstrap']['draws'],config['bootstrap']['mean_block'],config['bootstrap']['seed'])
        intervals.to_csv(folder/'paired-intervals.csv',index=False)
        # Predefined controls; these comparisons remain exploratory, not primary successes.
        for candidate,label in [('Uniform covariance pool','uniform'),('Log/Fisher GHAR(GL,L)','log-fisher')]:
            control_intervals=paired_intervals(losses,p_returns.iloc[selected],'HAR-DRD',candidate,config['bootstrap']['draws'],config['bootstrap']['mean_block'],config['bootstrap']['seed'])
            control_intervals.to_csv(folder/(label+'-exploratory-intervals.csv'),index=False)
        summary={'origins':len(selected),'first_date':str(f.dates[selected[0]].date()),'last_date':str(f.dates[selected[-1]].date()),
                 'all_four_point_improvements':bool((intervals.change_percent<0).all()),
                 'all_four_upper_intervals_below_zero':bool((intervals.upper_95<0).all()),
                 'qualification':'Retrospective reused data, not fresh confirmation.'}
        (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(period,summary,flush=True)
        print(scores.xs('ratio',axis=1,level=1).join(risk).round(6).to_string(),flush=True)
        print(intervals.round(6).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experts',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.experts,a.config,a.output_dir)
