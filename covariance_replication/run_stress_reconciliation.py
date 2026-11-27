"""Evaluate the fixed causal covariance-reconciliation family and all failures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from covariance_reconciliation import choose_paths
from stress_covariance_reconciliation import make_stress_candidates,ewma_covariance_weights
from evaluation import covariance_losses,long_only_weights,point_ratios,portfolio_stats
from models import Forecasts
from risk_calibration import portfolio_weights
from run_joint_pool import paired_intervals
from run_risk_calibration import prepare_bases


def run(panel_path,expert_path,control_path,weight_path,config_path,output):
    output.mkdir(parents=True,exist_ok=True);config=json.loads(config_path.read_text())
    paths=[panel_path,expert_path,control_path,weight_path,config_path]
    sources={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
             for name in ['covariance_reconciliation.py','stress_covariance_reconciliation.py','risk_calibration.py','run_risk_calibration.py','run_stress_reconciliation.py']}
    (output/'manifest.json').write_text(json.dumps({'config':config,'source_sha256':sources,
               'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2)+'\n')
    panel=np.load(panel_path);experts=np.load(expert_path);controls=np.load(control_path);old_weights=np.load(weight_path)
    dates=pd.DatetimeIndex(controls['dates']);positions=controls['positions'];log=lambda s:print(s,flush=True)
    full=prepare_bases(panel,experts,controls,log);bases=full[:,[1,2,3]]
    drivers=[];driver_audit=[]
    recent,solver=portfolio_weights(panel['covariance'][experts['positions']-1]);drivers.append(recent)
    driver_audit.append({'driver':'yesterday covariance',**solver})
    for half_life in [2,5]:
        weights,solver=ewma_covariance_weights(panel['covariance'],experts['positions'],half_life)
        drivers.append(weights);driver_audit.append({'driver':f'covariance half-life{half_life}',**solver})
    pd.DataFrame(driver_audit).to_csv(output/'driver-solver-audit.csv',index=False)
    covariances,weights,definitions=make_stress_candidates(bases,np.stack(drivers,axis=1),panel['covariance'],experts['positions'],config)
    har,solver=portfolio_weights(full[:,0])
    covariances=np.concatenate([covariances,full[None,:,0]],axis=0)
    weights=np.concatenate([weights,har[None]],axis=0)
    definitions.append({'base':-1,'driver':-1,'gate':'identity','strength':0.,'distance':'HAR-DRD fallback'})
    baseline=len(definitions)-1
    log(f'Constructed {len(definitions)} positive covariance paths with prescribed causal weights.')
    selected,audit,metrics,integrated,portfolio=choose_paths(experts['dates'],covariances,weights,
                                            experts['actual'],experts['returns'],dates,definitions,baseline)
    selected={'Stress '+name:value for name,value in selected.items()}
    audit['model']='Stress '+audit['model']
    audit.to_csv(output/'selection-folds.csv',index=False)
    aligned=pd.DatetimeIndex(experts['dates']).get_indexer(dates)
    primary=next(i for i,d in enumerate(definitions) if d['base']==2 and d['driver']==0
                 and d['gate']=='Volatility ratio1to3' and d['distance']=='Gaussian KL')
    primary_name='Fixed stress-gated covariance reconciliation'
    selected[primary_name]=covariances[primary,aligned]
    pd.DataFrame(definitions).rename_axis('candidate').to_csv(output/'candidate-definitions.csv')
    candidate_records=[]
    n=experts['actual'].shape[-1]
    for period,mask in [('development_2024',dates<'2025-01-01'),('reused_2025_2026',dates>='2025-01-01')]:
        rows=aligned[mask];means=metrics[:,rows].mean(axis=1);base=means[baseline]
        constant=np.linalg.slogdet(experts['actual'][rows])[1].mean()+n
        risks=100*portfolio[:,rows].std(axis=1)*np.sqrt(252)
        iv=integrated[:,rows].mean(axis=1)/integrated[baseline,rows].mean()
        for i,d in enumerate(definitions):
            candidate_records.append({'period':period,'candidate':i,'Euclidean':means[i,0]/base[0],
                'Frobenius':means[i,1]/base[1],'QLIKE':(means[i,2]+constant)/(base[2]+constant),
                'QLIKE_divergence':means[i,2]/base[2],'annual_risk_percent':risks[i],
                'risk_variance_ratio':(risks[i]/risks[baseline])**2,'integrated_risk_ratio':iv[i],**d})
    scores=pd.DataFrame(candidate_records);scores.to_csv(output/'all-candidate-scores.csv',index=False)
    later=scores[scores.period=='reused_2025_2026'].copy();endpoints=['Euclidean','Frobenius','QLIKE_divergence','risk_variance_ratio']
    best=int(later.assign(worst=later[endpoints].max(axis=1)).sort_values(['worst','candidate']).iloc[0].candidate)
    name='Exploratory balanced stress-reconciled candidate';selected[name]=covariances[best,aligned]
    (output/'exploratory-choice.json').write_text(json.dumps({'candidate':best,'definition':definitions[best],
          'qualification':'Chosen after comparing all outcomes; not an independently validated model.'},indent=2)+'\n')
    # Check KKT residuals of prescribed allocations, independent of portfolio returns.
    certificates=[]
    for i,forecast in enumerate(covariances):
        gradient=(forecast@weights[i,...,None])[...,0];value=np.sum(weights[i]*gradient,axis=1)
        gap=2*np.maximum(value-gradient.min(axis=1),0)/value
        if not np.isfinite(gap).all() or gap.max()>1e-6:raise RuntimeError('Reconciled portfolio KKT certificate failed.')
        certificates.append({'candidate':i,'max_relative_duality_gap':float(gap.max())})
    pd.DataFrame(certificates).to_csv(output/'reconciliation-kkt.csv',index=False)
    predictions={key:controls[key] for key in controls.files if key not in ['dates','positions','actual']}
    predictions.update(selected);f=Forecasts(dates,positions,controls['actual'],predictions,pd.DataFrame())
    np.savez_compressed(output/'forecasts.npz',dates=dates.to_numpy(),positions=positions,actual=f.actual,**predictions)
    losses=covariance_losses(f);losses.to_csv(output/'daily-losses.csv',index=False)
    returns=panel['returns'][positions];evaluated_weights={};realized={}
    for name,forecast in predictions.items():
        evaluated_weights[name]=old_weights[name] if name in old_weights.files else long_only_weights(forecast)
        realized[name]=np.sum(evaluated_weights[name]*returns,axis=1)
    realized=pd.DataFrame(realized,index=dates);realized.to_csv(output/'portfolio-returns.csv',index_label='date')
    np.savez_compressed(output/'portfolio-weights.npz',**evaluated_weights)
    for period,mask in [('development_2024',dates<'2025-01-01'),('reused_2025_2026',dates>='2025-01-01')]:
        folder=output/period;folder.mkdir(exist_ok=True);rows=np.flatnonzero(mask)
        subset=losses[losses.date.isin(dates[rows])];scores=point_ratios(subset)
        risk=pd.DataFrame([{'model':name,'annual_risk_percent':portfolio_stats(w[rows],returns[rows])[0],
                    'turnover':portfolio_stats(w[rows],returns[rows])[1]} for name,w in evaluated_weights.items()]).set_index('model')
        scores=scores.join(risk);scores.to_csv(folder/'scores.csv');intervals=[]
        for name in selected:
            frame=paired_intervals(subset,realized.iloc[rows],'HAR-DRD',name,
                    config['bootstrap']['draws'],config['bootstrap']['mean_block'],config['bootstrap']['seed'])
            frame.insert(0,'model',name);intervals.append(frame)
        pd.concat(intervals).to_csv(folder/'paired-intervals.csv',index=False)
        log(period+'\n'+scores.loc[list(selected)].round(6).to_string())
    log('Finished stress covariance reconciliation study.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['panel','experts','controls','control-weights','config','output']:parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    with threadpool_limits(limits=1):run(a.panel,a.experts,a.controls,a.control_weights,a.config,a.output)
