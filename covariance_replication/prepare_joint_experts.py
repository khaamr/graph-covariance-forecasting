"""Build causal experts for the fixed joint-pool experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from market_data import load_market, minute_panel, REVISION
from models import forecast_grid


def run(data_dir,output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    panel=load_market('syncster',data_dir,'2016-01-04','2026-10-23')
    reference=minute_panel('SPY','syncster',data_dir,'2016-01-04','2026-10-23').index
    print('Fitting dense-grid levels experts',flush=True)
    levels=forecast_grid(panel,horizons=(1,),reference_calendar=reference,glasso_alphas=12)[1]
    print('Fitting dense-grid log/Fisher experts',flush=True)
    logged=forecast_grid(panel,horizons=(1,),reference_calendar=reference,glasso_alphas=12,transformation='log_fisher')[1]
    np.testing.assert_array_equal(levels.positions,logged.positions)
    np.testing.assert_allclose(levels.actual,logged.actual,atol=0,rtol=0)
    positions=levels.positions
    prefix=np.concatenate([np.zeros_like(panel.covariance[:1]),np.cumsum(panel.covariance,axis=0)])
    arithmetic=(panel.covariance[positions-1]+(prefix[positions]-prefix[positions-5])/5+(prefix[positions]-prefix[positions-22])/22)/3
    bank=np.stack([levels.predictions['HAR-DRD'],levels.predictions['GHAR(GL,L)'],logged.predictions['GHAR(GL,L)'],levels.predictions['HAR-Cholesky'],arithmetic],axis=1)
    np.linalg.cholesky(bank)
    np.linalg.cholesky(levels.actual)
    np.savez_compressed(output_dir/'experts.npz',bank=bank,actual=levels.actual,returns=panel.portfolio_return[positions],dates=levels.dates.to_numpy(),positions=positions,symbols=panel.symbols)
    levels.audit.to_csv(output_dir/'levels-folds.csv',index=False)
    logged.audit.to_csv(output_dir/'log-fisher-folds.csv',index=False)
    metadata={'dataset_revision':REVISION,'origins':len(positions),'first_forecast':str(levels.dates[0].date()),'last_forecast':str(levels.dates[-1].date()),'graph_alphas':12,'calendar':'complete lag windows relative to screened SPY input','data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data_dir.glob('*.parquet'))},'source_sha256':{p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest() for p in ['models.py','market_data.py','prepare_joint_experts.py']}}
    (output_dir/'expert-manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(metadata['origins'],metadata['first_forecast'],metadata['last_forecast'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.data_dir,a.output_dir)
