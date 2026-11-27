"""Independently score selected GLASSO penalties, outside warm-start CV paths."""
import argparse
import json
from pathlib import Path
from warnings import catch_warnings,simplefilter

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from market_data import load_market


def run(data_dir,audit_dir):
    panel=load_market('syncster',data_dir,'2016-01-04','2024-12-31')
    traces={name:pd.read_csv(audit_dir/name/'lambda-series.csv',parse_dates=['origin']) for name in ['current','denser_cv_grid']}
    rows=[]
    for name,trace in traces.items():
        for row in trace.itertuples():
            origin=panel.dates.get_loc(row.origin)
            x=panel.graph_return[origin-1000:origin]/row.scale
            alpha=row.cv_alpha_return_units/row.scale**2
            scores=[];warnings=0
            for train,test in KFold(n_splits=5,shuffle=False).split(x):
                with catch_warnings(record=True) as issues:
                    simplefilter('always',ConvergenceWarning)
                    fit=GraphicalLasso(alpha=alpha,max_iter=800,tol=1e-6,enet_tol=1e-8).fit(x[train])
                warnings+=sum(issubclass(w.category,ConvergenceWarning) for w in issues)
                scores.append(fit.score(x[test]))
            if not np.isfinite(scores).all():raise ValueError('Independent CV score is not finite.')
            rows.append({'variant':name,'origin':row.origin,'alpha_return_units':row.cv_alpha_return_units,
                         'mean_score_native_units':float(np.mean(scores)),
                         'mean_score_return_units':float(np.mean(scores)-x.shape[1]*np.log(row.scale)),
                         'convergence_warnings':warnings})
    result=pd.DataFrame(rows)
    result.to_csv(audit_dir/'independent-cv-scores.csv',index=False)
    wide=result.pivot(index='origin',columns='variant',values='mean_score_return_units')
    gains=wide.denser_cv_grid-wide.current
    summary={'windows':len(wide),'independent_fits':len(result)*5,
             'dense_better_windows':int(gains.gt(0).sum()),'dense_mean_score_gain':float(gains.mean()),
             'dense_min_score_gain':float(gains.min()),'dense_max_score_gain':float(gains.max()),
             'convergence_warnings':int(result.convergence_warnings.sum()),
             'scoring':'Each penalty fitted independently on identical unshuffled five folds, then scored on held-out returns. Higher log-likelihood is better.'}
    (audit_dir/'independent-cv-verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--audit-dir',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data_dir,args.audit_dir)
