"""Trace GLASSO selection and rolling-boundary sensitivity on a fixed sample.

The authors' variant reproduces their GraphicalLassoCV() call under our pinned
runtime, not their unknown historical environment or unpublished alpha series.
No forecasting defaults are changed; replacements are scoped to each experiment.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from market_data import load_market
from models import forecast_grid, precision_graph
from evaluation import METRICS, covariance_losses, forecast_table


@dataclass(frozen=True)
class Setting:
    scaled: bool = True
    alphas: int = 4
    refinements: int = 4
    iterations: int = 180
    tol: float = 1e-4
    enet_tol: float = 1e-4
    use_cv_precision: bool = False
    refit_every: int | None = None


SETTINGS = {
    'current': Setting(),
    'authors_call_current_runtime': Setting(scaled=False, iterations=100, use_cv_precision=True),
    'raw_scale_only': Setting(scaled=False),
    'denser_cv_grid': Setting(alphas=12),
    'tighter_solver': Setting(iterations=800, tol=1e-6, enet_tol=1e-8),
    'released_22_session_refits': Setting(refit_every=22),
}


def edge_comparison(graph, reference):
    if graph.shape != reference.shape:
        raise ValueError('Graph shapes must match.')
    union = np.count_nonzero(graph | reference)
    return int(np.count_nonzero(graph ^ reference) // 2), (float(np.count_nonzero(graph & reference) / union) if union else 1.)


def alpha_in_return_units(alpha, scale):
    return float(alpha * scale**2)


def fit_trace(returns, setting):
    scale = float(np.sqrt(np.var(returns, axis=0).mean())) if setting.scaled else 1.
    samples = returns / scale
    with catch_warnings(record=True) as cv_warnings:
        simplefilter('always')
        search = GraphicalLassoCV(alphas=setting.alphas, n_refinements=setting.refinements,
                                 cv=5, n_jobs=1, max_iter=setting.iterations,
                                 tol=setting.tol, enet_tol=setting.enet_tol).fit(samples)
    if setting.use_cv_precision:
        fit, alpha, fit_warnings = search, search.alpha_, []
    else:
        sample_cov = np.cov(samples, rowvar=False, ddof=0)
        np.fill_diagonal(sample_cov, 0)
        diagonal_penalty = np.abs(sample_cov).max() * (1 + 1e-6)
        penalties = sorted({min(search.alpha_ * m, diagonal_penalty) for m in (1,1.25,1.5,2,4,8)} | {diagonal_penalty})
        for alpha in penalties:
            with catch_warnings(record=True) as fit_warnings:
                simplefilter('always', ConvergenceWarning)
                fit = GraphicalLasso(alpha=alpha, max_iter=800, enet_tol=1e-8, tol=setting.tol).fit(samples)
            if not any(issubclass(w.category, ConvergenceWarning) for w in fit_warnings):
                break
        else:
            raise RuntimeError('GLASSO did not converge in traced refit.')
    graph = fit.precision_ != 0
    np.fill_diagonal(graph, False)
    curve = pd.DataFrame({
        'alpha_native': search.cv_results_['alphas'],
        'mean_validation_loglikelihood': search.cv_results_['mean_test_score'],
        'std_validation_loglikelihood': search.cv_results_['std_test_score'],
    })
    curve['alpha_return_units'] = curve.alpha_native * scale**2
    positive = curve.alpha_native[curve.alpha_native > 0]
    trace = {'scale': scale, 'cv_alpha_native': float(search.alpha_),
             'cv_alpha_return_units': alpha_in_return_units(search.alpha_, scale),
             'final_alpha_return_units': alpha_in_return_units(alpha, scale),
             'cv_warnings': len(cv_warnings), 'refit_warnings': len(fit_warnings),
             'penalty_increased': bool(alpha > search.alpha_ * (1+1e-4)),
             'edges': int(graph.sum()//2), 'cv_candidates': len(curve),
             'finite_cv_scores': int(np.isfinite(curve.mean_validation_loglikelihood).sum()),
             'selected_at_positive_grid_boundary': bool(len(positive) and (np.isclose(search.alpha_,positive.min(),rtol=1e-8,atol=0) or np.isclose(search.alpha_,positive.max(),rtol=1e-8,atol=0)))}
    return graph, trace, curve


def expected_origins(dates, first_forecast, lookback, interval):
    test = np.arange(lookback + 22, len(dates))
    test = test[dates[test] >= first_forecast]
    months = dates.to_period('M')
    if months[test[0]] == months[test[0]-1]:
        test = test[months[test] != months[test[0]]]
    labels = months[test] if interval is None else np.arange(len(test)) // interval
    return [int(test[np.flatnonzero(labels == label)[0]]) for label in pd.unique(labels)]


def run(data_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {'settings': {k: vars(v) for k,v in SETTINGS.items()},
                'input_period': ['2016-01-04','2024-12-31'], 'first_forecast':'2022-07-01',
                'lookback':1000, 'horizons':[1], 'MCS_draws':10000, 'MCS_block':2,
                'alpha_units':'percentage-log-return squared; scaled penalties multiplied by scale squared',
                'authors_source':'https://github.com/chaozhang-ox/Graph-based-HAR/blob/751f8f0b400a33e2f2fd01501ffdc42a2315f013/GHAR_Var.py',
                'scope':'Existing public sample reused; author alpha series unavailable; no exact historical-environment claim.',
                'source_sha256':{f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in ['models.py','market_data.py','audit_glasso.py']}}
    (output_dir/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    panel = load_market('syncster',data_dir,'2016-01-04','2024-12-31')
    baseline_graphs, baseline_controls, baseline_dates = {}, {}, None
    all_summary = []
    for name, setting in SETTINGS.items():
        print('RUNNING',name,flush=True)
        origins = expected_origins(panel.dates,pd.Timestamp('2022-07-01'),1000,setting.refit_every)
        records, curves = [], []
        def traced_graph(returns):
            origin = origins[len(records)]
            np.testing.assert_array_equal(returns,panel.graph_return[origin-1000:origin])
            graph, trace, curve = fit_trace(returns,setting)
            if origin not in baseline_graphs:
                baseline_graphs[origin] = precision_graph(returns)
            reference, reference_alpha, _, reference_fallback = baseline_graphs[origin]
            if name in ['current','released_22_session_refits']:
                np.testing.assert_array_equal(graph,reference)
                np.testing.assert_allclose(trace['final_alpha_return_units'],reference_alpha,atol=1e-12,rtol=1e-12)
                assert trace['penalty_increased'] == reference_fallback
            changed,jaccard = edge_comparison(graph,reference)
            trace.update({'origin':panel.dates[origin], 'window_first':panel.dates[origin-1000],
                          'window_last':panel.dates[origin-1], 'graph_rows':len(returns),
                          'current_alpha_same_window':float(reference_alpha),
                          'changed_edges_same_window':changed,'jaccard_same_window':jaccard})
            assert trace['window_last'] < trace['origin']
            records.append(trace)
            curves.append(curve.assign(origin=panel.dates[origin]))
            return graph,trace['final_alpha_return_units'],trace['cv_warnings'],trace['penalty_increased']
        with patch('models.precision_graph',new=traced_graph):
            forecast = forecast_grid(panel,horizons=(1,),first_forecast=pd.Timestamp('2022-07-01'),refit_every=setting.refit_every)[1]
        assert len(records)==len(origins)
        if baseline_dates is None:
            baseline_dates=forecast.dates
            baseline_controls={k:forecast.predictions[k] for k in ['HAR-DRD','HAR-Cholesky']}
        assert baseline_dates.equals(forecast.dates)
        if setting.refit_every is None:
            for model,matrices in baseline_controls.items():
                np.testing.assert_allclose(forecast.predictions[model],matrices,atol=1e-12,rtol=1e-12)
        losses=covariance_losses(forecast)
        scores=forecast_table(losses)
        folder=output_dir/name;folder.mkdir(exist_ok=True)
        pd.DataFrame(records).to_csv(folder/'lambda-series.csv',index=False)
        pd.concat(curves,ignore_index=True).to_csv(folder/'cv-curves.csv',index=False)
        forecast.audit.to_csv(folder/'folds.csv',index=False)
        scores.to_csv(folder/'scores.csv')
        losses.to_csv(folder/'daily-losses.csv',index=False)
        selected=scores.loc['GHAR(GL,L)']
        summary={'variant':name,'origins':len(forecast.dates),'refits':len(records),
                 'changed_graph_windows':sum(r['changed_edges_same_window']>0 for r in records),
                 'max_changed_edges':max(r['changed_edges_same_window'] for r in records),
                 'mean_jaccard_same_window':float(np.mean([r['jaccard_same_window'] for r in records])),
                 'fallback_windows':sum(r['penalty_increased'] for r in records)}
        for metric in METRICS:
            summary[metric+'_ratio']=float(selected[metric,'ratio'])
            summary[metric+'_MCS_p']=float(selected[metric,'MCS p'])
        all_summary.append(summary)
        pd.DataFrame(all_summary).to_csv(output_dir/'summary.csv',index=False)
        print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data_dir,args.output_dir)
