"""Execute fixed diagnostic repairs on the unchanged common-date benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluation import covariance_losses, long_only_weights, point_ratios, portfolio_stats
from factor_vhar import factor_forecasts
from market_data import SECTORS
from models import Forecasts
from realized_heavy import MODEL, as_correlation, heavy_forecasts
from run_joint_pool import paired_intervals


def run(panel_path, controls_path, control_weights_path, config_path, output):
    output.mkdir(parents=True,exist_ok=True)
    config=json.loads(config_path.read_text())
    inputs=[panel_path,controls_path,control_weights_path,config_path]
    source={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ['factor_vhar.py','realized_heavy.py','run_diagnosis_repairs.py']}
    input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (output/'manifest.json').write_text(json.dumps({'config':config,'source_sha256':source,'input_sha256':input_hashes},indent=2)+'\n')
    panel=np.load(panel_path);controls=np.load(controls_path);old_weights=np.load(control_weights_path)
    positions=controls['positions'];dates=pd.DatetimeIndex(controls['dates'])
    np.testing.assert_array_equal(panel['dates'][positions],controls['dates'])
    np.testing.assert_allclose(panel['covariance'][positions],controls['actual'],rtol=1e-12,atol=2e-12)
    predictions={key:controls[key] for key in controls.files if key not in ['actual','dates','positions']}
    log=lambda message:print(message,flush=True)
    heavy,audit,parameters=heavy_forecasts(panel['dates'],panel['covariance'],positions,
                                          panel['reference'],lookback=config['lookback'],progress=log)
    audit.to_csv(output/'heavy-folds.csv',index=False);parameters.to_csv(output/'heavy-parameters.csv',index=False)
    predictions[MODEL]=heavy
    new_names=[MODEL]
    for label,bias in [('Raw factor',False),('Smeared raw factor',True)]:
        factor=factor_forecasts(panel['dates'],panel['covariance'],positions,
                               np.ones((len(panel['symbols']),1))/len(panel['symbols']),
                               [SECTORS[str(s)] for s in panel['symbols']],panel['reference'],
                               lookback=config['lookback'],clean=False,bias_correction=bias,progress=log)
        factor.audit.to_csv(output/(label.lower().replace(' ','-')+'-folds.csv'),index=False)
        for name,values in factor.predictions.items():
            key=name.replace('Factor-VHAR',label+'-VHAR market')
            predictions[key]=values;new_names.append(key)
    graph_correlation=as_correlation(predictions['Log/Fisher GHAR(GL,L)'])
    ridge_variance=np.diagonal(predictions['Ridge log-HAR + graph correlation'],axis1=-2,axis2=-1)
    heavy_variance=np.diagonal(heavy,axis1=-2,axis2=-1)
    for name,variance,correlation in [
        ('HEAVY variance + graph correlation',heavy_variance,graph_correlation),
        ('Ridge variance + HEAVY correlation',ridge_variance,as_correlation(heavy))]:
        std=np.sqrt(variance);predictions[name]=correlation*std[:,:,None]*std[:,None,:];new_names.append(name)
    # Verify the new optional corrections leave the original implementation unchanged.
    selected=np.array([0,len(positions)//2,len(positions)-1])
    default=factor_forecasts(panel['dates'],panel['covariance'],positions[selected],
                             np.ones((len(panel['symbols']),1))/len(panel['symbols']),
                             [SECTORS[str(s)] for s in panel['symbols']],panel['reference'],lookback=1000)
    for name,value in default.predictions.items():
        original=name.replace('Factor-VHAR ','Factor-VHAR market ')
        np.testing.assert_array_equal(value,predictions[original][selected])
    (output/'default-compatibility.json').write_text(json.dumps({'origins_checked':3,'estimators':2,'forecasts_match_exactly':True})+'\n')
    f=Forecasts(dates,positions,controls['actual'],predictions,pd.DataFrame())
    np.savez_compressed(output/'forecasts.npz',dates=dates.to_numpy(),positions=positions,actual=f.actual,**predictions)
    losses=covariance_losses(f);losses.to_csv(output/'daily-losses.csv',index=False)
    returns=panel['returns'][positions];weights={};realized={}
    for name,matrices in predictions.items():
        if name in old_weights.files:
            w=old_weights[name]
            if w.shape!=returns.shape or (w < -1e-9).any() or not np.allclose(w.sum(axis=1),1):
                raise ValueError('Control portfolio cache is invalid.')
        else:
            log('Exact portfolio: '+name);w=long_only_weights(matrices)
        weights[name]=w;realized[name]=np.sum(w*returns,axis=1)
    realized=pd.DataFrame(realized,index=dates);realized.to_csv(output/'portfolio-returns.csv',index_label='date')
    np.savez_compressed(output/'portfolio-weights.npz',**weights)
    for period,mask in [('development_2024',dates<'2025-01-01'),('reused_2025_2026',dates>='2025-01-01')]:
        folder=output/period;folder.mkdir(exist_ok=True);selected=np.flatnonzero(mask)
        subset=losses[losses.date.isin(dates[selected])]
        ratios=point_ratios(subset)
        risk=pd.DataFrame([{'model':name,'annual_risk_percent':portfolio_stats(w[selected],returns[selected])[0],
                            'turnover':portfolio_stats(w[selected],returns[selected])[1]} for name,w in weights.items()]).set_index('model')
        scores=ratios.join(risk);scores.to_csv(folder/'scores.csv')
        frames=[]
        comparisons=[('HAR-DRD',name) for name in new_names]+[(base,MODEL) for base in ['Uniform covariance pool','Log/Fisher GHAR(GL,L)']]
        for baseline,name in comparisons:
            frame=paired_intervals(subset,realized.iloc[selected],baseline,name,
                                   config['bootstrap']['draws'],config['bootstrap']['mean_block'],config['bootstrap']['seed'])
            frame.insert(0,'model',name);frame.insert(0,'baseline',baseline);frames.append(frame)
        pd.concat(frames).to_csv(folder/'paired-intervals.csv',index=False)
        log(f'{period}: {len(selected)} dates\n'+scores.round(6).to_string())
    log('Finished diagnostic repair study.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['panel','controls','control-weights','config','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.panel,args.controls,args.control_weights,args.config,args.output)
