"""Past-only covariance pooling selected against four validation objectives."""
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from evaluation import long_only_weights
from models import Forecasts

EXPERT_NAMES=('HAR-DRD','GHAR(GL,L)','Log/Fisher GHAR(GL,L)','HAR-Cholesky','Arithmetic HAR covariance')
MODEL='Joint calibrated covariance pool'
UNIFORM='Uniform covariance pool'
CALIBRATED='Calibrated HAR-DRD'
RECIPES=('identity','Euclidean','Frobenius','QLIKE divergence')


@dataclass(frozen=True)
class PoolResult:
    forecasts: Forecasts
    audit: pd.DataFrame
    validation: pd.DataFrame
    solver_audit: pd.DataFrame


def fixed_covariance_pool(bank):
    """Equal-weight the fixed five causal covariance experts; no outcome fitting."""
    bank=np.asarray(bank)
    if bank.ndim!=4 or bank.shape[1]!=len(EXPERT_NAMES) or bank.shape[-1]!=bank.shape[-2]:
        raise ValueError('Expected dates by five experts by square covariance matrices.')
    if not np.isfinite(bank).all() or not np.allclose(bank,bank.swapaxes(-1,-2),atol=1e-10,rtol=1e-10):
        raise ValueError('Experts must be finite symmetric matrices.')
    np.linalg.cholesky(bank)
    return bank.mean(axis=1)


def simplex_grid(n_experts,denominator=3):
    rows=[np.array(v,dtype=float)/denominator for v in product(range(denominator+1),repeat=n_experts) if sum(v)==denominator]
    uniform=np.full(n_experts,1/n_experts)
    if not any(np.allclose(r,uniform) for r in rows):rows.append(uniform)
    return np.stack(rows)


def project_simplex(values):
    ordered=np.sort(values,axis=-1)[:,::-1]
    cumulative=np.cumsum(ordered,axis=-1)-1
    count=np.arange(1,values.shape[-1]+1)
    active=np.sum(ordered>cumulative/count,axis=-1)-1
    offset=cumulative[np.arange(len(values)),active]/(active+1)
    return np.maximum(values-offset[:,None],0)


def simplex_weights(covariance,tolerance=1e-4,checkpoints=(200,1000,3000)):
    """Accelerated simplex solve with a relative convex-duality-gap certificate."""
    n_dates,n,_=covariance.shape
    largest=np.linalg.eigvalsh(covariance)[:,-1:]
    if (largest<=0).any():raise ValueError('Positive covariance is required.')
    w=np.full((n_dates,n),1/n);y=w.copy();momentum=1.
    for iteration in range(1,max(checkpoints)+1):
        gradient=(covariance@y[...,None])[...,0]
        updated=project_simplex(y-gradient/largest)
        next_momentum=(1+np.sqrt(1+4*momentum**2))/2
        y=updated+(momentum-1)/next_momentum*(updated-w)
        w,momentum=updated,next_momentum
        if iteration in checkpoints:
            gradient=(covariance@w[...,None])[...,0]
            quadratic=np.sum(w*gradient,axis=1)
            relative_gap=2*np.maximum(quadratic-gradient.min(axis=1),0)/quadratic
            if relative_gap.max()<=tolerance:break
    bad=relative_gap>tolerance
    fallbacks=int(bad.sum())
    if fallbacks:
        w[bad]=long_only_weights(covariance[bad])
        gradient=(covariance@w[...,None])[...,0]
        quadratic=np.sum(w*gradient,axis=1)
        relative_gap=2*np.maximum(quadratic-gradient.min(axis=1),0)/quadratic
    if not np.isfinite(w).all() or (w < -1e-10).any() or not np.allclose(w.sum(axis=1),1,atol=1e-9):
        raise ValueError('Portfolio selection solver violated simplex constraints.')
    if relative_gap.max()>max(5*tolerance,1e-5):
        raise ValueError('Portfolio selection solver did not meet its objective-gap tolerance.')
    return w,{'iterations':iteration,'exact_fallbacks':fallbacks,'max_relative_objective_gap_bound':float(relative_gap.max())}


def shrink_diagonal(matrices,intensity):
    if not 0<=intensity<=1:raise ValueError('Shrinkage must lie in [0,1].')
    diagonal=np.diagonal(matrices,axis1=-2,axis2=-1).copy()
    result=(1-intensity)*matrices
    index=np.arange(matrices.shape[-1]);result[:,index,index]+=intensity*diagonal
    return result


def loss_statistics(predicted,actual):
    n=actual.shape[-1];lower=np.tril_indices(n)
    p=predicted[:,lower[0],lower[1]];a=actual[:,lower[0],lower[1]]
    sign,logdet=np.linalg.slogdet(predicted)
    actual_sign,actual_logdet=np.linalg.slogdet(actual)
    if not (np.all(sign>0) and np.all(actual_sign>0)):
        raise ValueError('Positive determinants are required for QLIKE divergence.')
    return {'eu_a':np.sum(p*p,axis=1),'eu_b':np.sum(p*a,axis=1),'eu_c':np.sum(a*a,axis=1),
            'fr_a':np.sum(predicted**2,axis=(1,2)),'fr_b':np.sum(predicted*actual,axis=(1,2)),
            'fr_c':np.sum(actual**2,axis=(1,2)),'logdet':logdet,'actual_logdet':actual_logdet,
            'trace':np.trace(np.linalg.solve(predicted,actual),axis1=-2,axis2=-1),'dimension':n}


def scaled_losses(statistics,scale):
    if scale<=0:raise ValueError('Scale must be positive.')
    s=statistics
    eu=np.sqrt(np.maximum(s['eu_c']-2*scale*s['eu_b']+scale**2*s['eu_a'],0))
    fr=np.sqrt(np.maximum(s['fr_c']-2*scale*s['fr_b']+scale**2*s['fr_a'],0))
    q=s['logdet']+s['dimension']*np.log(scale)-s['actual_logdet']+s['trace']/scale-s['dimension']
    if q.min() < -1e-7:raise ValueError('Negative QLIKE divergence beyond rounding tolerance.')
    return np.column_stack([eu,fr,np.maximum(q,0)])


def scale_from_statistics(statistics,recipe,bounds=(.5,2.)):
    if recipe=='identity':return 1.
    if recipe=='QLIKE divergence':
        return float(np.clip(statistics['trace'].mean()/statistics['dimension'],*bounds))
    metric={'Euclidean':0,'Frobenius':1}.get(recipe)
    if metric is None:raise ValueError('Unknown scale recipe.')
    objective=lambda scale:scaled_losses(statistics,scale)[:,metric].mean()
    result=minimize_scalar(objective,bounds=bounds,method='bounded',options={'xatol':1e-5})
    if not result.success:raise RuntimeError('Scalar calibration failed.')
    return float(min([bounds[0],bounds[1],1.,result.x],key=objective))


def select_candidate(bank,actual,returns,fitting,validation,config):
    baseline=bank[validation,0]
    base_weight,base_solver=simplex_weights(baseline)
    base_return=np.sum(base_weight*returns[validation],axis=1)
    base_metrics=np.r_[scaled_losses(loss_statistics(baseline,actual[validation]),1).mean(axis=0),np.var(base_return)]
    denominator=np.maximum(base_metrics,1e-12)
    candidates=[];solvers=[{'candidate':'baseline',**base_solver}]
    best=None;calibrated=None
    for grid_index,weights in enumerate(simplex_grid(bank.shape[1])):
        mixture=np.einsum('k,tkij->tij',weights,bank)
        for shrinkage in config['diagonal_shrinkage']:
            covariance=shrink_diagonal(mixture,shrinkage)
            fitting_stats=loss_statistics(covariance[fitting],actual[fitting])
            validation_stats=loss_statistics(covariance[validation],actual[validation])
            portfolio,solver=simplex_weights(covariance[validation])
            solvers.append({'candidate':f'{grid_index}:{shrinkage}',**solver})
            risk=float(np.var(np.sum(portfolio*returns[validation],axis=1)))
            for recipe_index,recipe in enumerate(RECIPES):
                scale=scale_from_statistics(fitting_stats,recipe,tuple(config['scale_bounds']))
                metrics=np.r_[scaled_losses(validation_stats,scale).mean(axis=0),risk]
                ratios=metrics/denominator
                complexity=int(np.count_nonzero(weights)-1)+(shrinkage!=0)+(recipe!='identity')
                key=(float(ratios.max()),float(ratios.mean()),complexity,grid_index,recipe_index)
                record={'grid_index':grid_index,'shrinkage':float(shrinkage),'recipe':recipe,'scale':scale,
                        'worst_ratio':key[0],'mean_ratio':key[1],
                        **{f'ratio_{m}':float(v) for m,v in zip(['Euclidean','Frobenius','QLIKE_divergence','portfolio_variance'],ratios)},
                        **{f'weight_{i}':float(v) for i,v in enumerate(weights)}}
                candidates.append(record)
                choice=(key,weights.copy(),float(shrinkage),recipe,record)
                if best is None or key<best[0]:best=choice
                if weights[0]==1 and shrinkage==0 and (calibrated is None or key<calibrated[0]):calibrated=choice
    if best is None or calibrated is None:raise RuntimeError('A valid baseline control was not available.')
    return best,calibrated,candidates,solvers


def fit_pool(dates,positions,bank,actual,returns,config,progress=None):
    dates=pd.DatetimeIndex(dates)
    if not dates.is_unique or not dates.is_monotonic_increasing:raise ValueError('Dates must be ordered and unique.')
    if bank.shape[1]!=len(EXPERT_NAMES):raise ValueError('The fixed expert bank has five members.')
    if not (len(dates)==len(bank)==len(actual)==len(returns)==len(positions)):raise ValueError('Input rows must align.')
    np.linalg.cholesky(bank);np.linalg.cholesky(actual)
    eligible=(np.arange(len(dates))>=config['minimum_history'])&(dates>=config['evaluation_start'])
    periods=dates.to_period('Q')
    outputs={MODEL:[],CALIBRATED:[]};selected=[];audit=[];validation_records=[];solver_records=[]
    for period in periods[eligible].unique():
        test=np.flatnonzero(eligible&(periods==period));origin=test[0]
        train=np.arange(max(0,origin-config['historical_window']),origin)
        validation=train[-config['inner_validation']:];fitting=train[:-config['inner_validation']]
        if len(fitting)<252 or dates[train[-1]]>=dates[origin]:raise ValueError('Training/validation chronology failed.')
        best,calibrated,candidates,solvers=select_candidate(bank,actual,returns,fitting,validation,config)
        for name,choice in [(MODEL,best),(CALIBRATED,calibrated)]:
            _,weights,shrinkage,recipe,record=choice
            mixture=np.einsum('k,tkij->tij',weights,bank)
            covariance=shrink_diagonal(mixture,shrinkage)
            scale=scale_from_statistics(loss_statistics(covariance[train],actual[train]),recipe,tuple(config['scale_bounds']))
            outputs[name].append(scale*covariance[test])
            audit.append({'period':str(period),'model':name,'first_forecast':dates[origin],
                          'last_fit_label':dates[fitting[-1]],'first_validation_label':dates[validation[0]],
                          'last_validation_label':dates[validation[-1]],'last_refit_label':dates[train[-1]],
                          'historical_labels':len(train),'validation_worst_ratio':record['worst_ratio'],
                          'scale_full_history':scale,**{k:v for k,v in record.items() if k not in ['worst_ratio']}})
        validation_records.extend(dict(period=str(period),**r) for r in candidates)
        solver_records.extend(dict(period=str(period),**r) for r in solvers)
        selected.append(test)
        if progress:progress(f'{period}: validation worst ratio {best[4]["worst_ratio"]:.5f}; forecasts {len(test)}')
    if not selected:raise ValueError('No eligible evaluation period.')
    selected=np.concatenate(selected)
    predictions={name:bank[selected,i] for i,name in enumerate(EXPERT_NAMES)}
    predictions[UNIFORM]=fixed_covariance_pool(bank[selected])
    predictions.update({k:np.concatenate(v) for k,v in outputs.items()})
    result=Forecasts(dates[selected],np.asarray(positions)[selected],actual[selected],predictions,pd.DataFrame(audit))
    return PoolResult(result,pd.DataFrame(audit),pd.DataFrame(validation_records),pd.DataFrame(solver_records))
