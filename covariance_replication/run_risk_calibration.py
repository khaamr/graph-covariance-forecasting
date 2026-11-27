"""Run the frozen portfolio-risk calibration study and preserve its whole search."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluation import covariance_losses,long_only_weights,point_ratios,portfolio_stats
from literature_models import volatility_features,select_ridge,as_correlation
from models import Forecasts
from risk_calibration import build_candidates,select_paths
from run_joint_pool import paired_intervals


def prepare_bases(panel,experts,controls,progress):
    dates=pd.DatetimeIndex(experts['dates']);positions=experts['positions']
    calendar=pd.DatetimeIndex(panel['reference']).get_indexer(panel['dates'])
    complete=np.zeros(len(calendar),dtype=bool);complete[22:]=calendar[22:]-calendar[:-22]==22
    target=np.log(np.diagonal(panel['covariance'],axis1=-2,axis2=-1))
    features,_=volatility_features(target,tuple(panel['symbols']))
    ridge=np.empty_like(experts['bank'][:,0]);months=dates.to_period('M')
    for month in months.unique():
        rows=np.flatnonzero(months==month);origin=positions[rows[0]]
        train=np.arange(max(22,origin-1000),origin);train=train[complete[train]]
        variance,penalty,_=select_ridge(features,target,train,positions[rows],[.0001,.001,.01,.1,1.])
        std=np.sqrt(variance);ridge[rows]=as_correlation(experts['bank'][rows,2])*std[:,:,None]*std[:,None,:]
    common=dates.get_indexer(controls['dates']);assert (common>=0).all()
    # Preserve earlier evaluated forecasts; the new fitting only supplies the earlier prefix.
    ridge[common]=controls['Ridge log-HAR + graph correlation']
    har,logged,arithmetic=experts['bank'][:,0],experts['bank'][:,2],experts['bank'][:,4]
    np.testing.assert_array_equal(har[common],controls['HAR-DRD'])
    np.testing.assert_array_equal(logged[common],controls['Log/Fisher GHAR(GL,L)'])
    bases=np.stack([har,logged,ridge,(logged+ridge)/2,(logged+ridge+arithmetic)/3,
                    .75*logged+.25*arithmetic],axis=1)
    progress('Prepared six causal base paths including genuine pre-2024 ridge forecasts.')
    return bases


def run(panel_path,expert_path,control_path,weight_path,config_path,output):
    output.mkdir(parents=True,exist_ok=True);config=json.loads(config_path.read_text())
    source={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ['risk_calibration.py','run_risk_calibration.py']}
    manifest={'config':config,'source_sha256':source,'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [panel_path,expert_path,control_path,weight_path,config_path]}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    panel=np.load(panel_path);experts=np.load(expert_path);controls=np.load(control_path);old_weights=np.load(weight_path)
    dates=pd.DatetimeIndex(controls['dates']);positions=controls['positions'];log=lambda s:print(s,flush=True)
    np.testing.assert_array_equal(panel['dates'][positions],controls['dates'])
    np.testing.assert_array_equal(experts['actual'][pd.DatetimeIndex(experts['dates']).get_indexer(dates)],controls['actual'])
    bases=prepare_bases(panel,experts,controls,log)
    bank=build_candidates(experts['dates'],bases,experts['actual'],experts['returns'],config,progress=log)
    bank.folds.to_csv(output/'calibration-folds.csv',index=False);bank.solver_audit.to_csv(output/'solver-audit.csv',index=False)
    selected,selection_audit=select_paths(bank,dates,config);selection_audit.to_csv(output/'selection-folds.csv',index=False)
    predictions={key:controls[key] for key in controls.files if key not in ['dates','positions','actual']}
    predictions.update(selected);new_names=list(selected)
    full_rows=pd.DatetimeIndex(experts['dates']).get_indexer(dates);aligned=bank.dates.get_indexer(dates)
    for j,name in [(3,'Fixed equal log-ridge pool'),(4,'Fixed equal log-ridge-arithmetic pool'),(5,'Fixed 75-log 25-arithmetic pool')]:
        predictions[name]=bases[full_rows,j];new_names.append(name)
    candidate_records=[]
    for period,mask in [('development_2024',dates<'2025-01-01'),('reused_2025_2026',dates>='2025-01-01')]:
        rows=aligned[mask];means=bank.metrics[:,rows].mean(axis=1);baseline=means[0]
        actual=controls['actual'][mask];actual_logdet=np.linalg.slogdet(actual)[1].mean();n=actual.shape[-1]
        ratios=means/baseline
        raw_qlike=(means[:,2]+actual_logdet+n)/(baseline[2]+actual_logdet+n)
        returns=bank.portfolio_returns[:,rows];risks=100*returns.std(axis=1)*np.sqrt(252)
        integrated=bank.integrated_risk[:,rows].mean(axis=1)/bank.integrated_risk[0,rows].mean()
        for j in range(len(means)):
            definition=bank.definitions[j//2]
            candidate_records.append({'period':period,'candidate':j,'case':j//2,'scale':'QLIKE' if j%2 else 'identity',
                                       'Euclidean':ratios[j,0],'Frobenius':ratios[j,1],'QLIKE':raw_qlike[j],
                                       'QLIKE_divergence':ratios[j,2],'annual_risk_percent':risks[j//2],
                                       'risk_variance_ratio':(risks[j//2]/risks[0])**2,
                                       'integrated_risk_ratio':integrated[j//2],**definition})
    candidate_scores=pd.DataFrame(candidate_records);candidate_scores.to_csv(output/'all-candidate-scores.csv',index=False)
    # One post-evaluation descriptive winner; it is explicitly separate from causal selectors.
    later=candidate_scores[candidate_scores.period=='reused_2025_2026'].copy()
    endpoints=['Euclidean','Frobenius','QLIKE_divergence','risk_variance_ratio']
    winner=later.assign(worst=later[endpoints].max(axis=1)).sort_values(['worst','candidate']).iloc[0]
    choice=int(winner.candidate);case=choice//2
    name='Exploratory best balanced calibrated candidate'
    predictions[name]=bank.matrices[case,aligned].copy()
    if choice%2:predictions[name]*=bank.scales[case,aligned,None,None]
    new_names.append(name)
    (output/'exploratory-choice.json').write_text(json.dumps({'candidate':choice,'case':case,'definition':bank.definitions[case],
                 'scale':'QLIKE' if choice%2 else 'identity','qualification':'Chosen after comparing all later outcomes; not a causally selected strategy or independent validation.'},indent=2)+'\n')
    f=Forecasts(dates,positions,controls['actual'],predictions,pd.DataFrame())
    np.savez_compressed(output/'forecasts.npz',dates=dates.to_numpy(),positions=positions,actual=f.actual,**predictions)
    losses=covariance_losses(f);losses.to_csv(output/'daily-losses.csv',index=False)
    returns=panel['returns'][positions];weights={};portfolio={}
    for name,matrices in predictions.items():
        weights[name]=old_weights[name] if name in old_weights.files else long_only_weights(matrices)
        portfolio[name]=np.sum(weights[name]*returns,axis=1)
    portfolio=pd.DataFrame(portfolio,index=dates);portfolio.to_csv(output/'portfolio-returns.csv',index_label='date')
    np.savez_compressed(output/'portfolio-weights.npz',**weights)
    for period,mask in [('development_2024',dates<'2025-01-01'),('reused_2025_2026',dates>='2025-01-01')]:
        folder=output/period;folder.mkdir(exist_ok=True);rows=np.flatnonzero(mask)
        subset=losses[losses.date.isin(dates[rows])];scores=point_ratios(subset)
        risk=pd.DataFrame([{'model':name,'annual_risk_percent':portfolio_stats(w[rows],returns[rows])[0],
                            'turnover':portfolio_stats(w[rows],returns[rows])[1]} for name,w in weights.items()]).set_index('model')
        scores=scores.join(risk);scores.to_csv(folder/'scores.csv');frames=[]
        for name in new_names:
            frame=paired_intervals(subset,portfolio.iloc[rows],'HAR-DRD',name,
                                   config['bootstrap']['draws'],config['bootstrap']['mean_block'],config['bootstrap']['seed'])
            frame.insert(0,'model',name);frames.append(frame)
        pd.concat(frames).to_csv(folder/'paired-intervals.csv',index=False)
        log(period+'\n'+scores.loc[new_names].round(6).to_string())
    log('Finished prequential risk-calibration study.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['panel','experts','controls','control-weights','config','output']:parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    with threadpool_limits(limits=1):run(a.panel,a.experts,a.controls,a.control_weights,a.config,a.output)
