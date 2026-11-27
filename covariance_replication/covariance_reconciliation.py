"""Positive covariance reconciliation with causal long-only weight forecasts.

This is a project research construction, not the published DCW algorithm.
Let v=Fw, q=w'Fw, and choose h with w'h=1 and h_i>=1 on inactive
coordinates. S=F-vv'/q+c hh' is PD and Sw=ch, so w is its exact simplex
minimum-variance allocation. The free c minimizes a chosen matrix distance.
"""
import numpy as np
import pandas as pd

from risk_calibration import portfolio_weights


def reconcile(covariance,target_weights,distance):
    covariance=np.asarray(covariance,dtype=float);weights=np.asarray(target_weights,dtype=float)
    if covariance.ndim!=3 or weights.shape!=covariance.shape[:2] or not np.isfinite(weights).all():
        raise ValueError('Covariance and target-weight rows must align.')
    if (weights < 0).any() or not np.allclose(weights.sum(axis=1),1,atol=1e-10):
        raise ValueError('Target portfolios must lie on the simplex.')
    np.linalg.cholesky(covariance)
    v=(covariance@weights[...,None])[...,0];q=np.sum(weights*v,axis=1)
    h=np.where(weights>0,1.,np.maximum(1.,v/q[:,None]))
    h/=np.sum(weights*h,axis=1)[:,None]
    if distance=='Gaussian KL':
        solved=np.linalg.solve(covariance,h[...,None])[...,0]
        c=1/np.sum(h*solved,axis=1)
    elif distance=='Reverse Gaussian KL':c=q
    elif distance=='Frobenius':
        c=np.sum(h*v,axis=1)**2/(q*np.sum(h*h,axis=1)**2)
    else:raise ValueError('Unknown reconciliation distance.')
    if not np.isfinite(c).all() or (c<=0).any():raise ValueError('Reconciliation scale must be positive.')
    result=covariance-v[:,:,None]*v[:,None,:]/q[:,None,None]+c[:,None,None]*h[:,:,None]*h[:,None,:]
    result=(result+result.swapaxes(-1,-2))/2
    np.linalg.cholesky(result)
    return result


def smoothed_realized_weights(covariance,half_life):
    if half_life<=0:raise ValueError('A positive half-life is required.')
    realized,audit=portfolio_weights(covariance)
    decay=np.exp2(-1/half_life);output=np.empty_like(realized)
    state=realized[0].copy();output[0]=np.full(realized.shape[1],1/realized.shape[1])
    for t in range(1,len(realized)):
        state=decay*state+(1-decay)*realized[t-1]
        output[t]=state
    return output,audit


def make_candidates(bases,drivers,config):
    candidates=[];weights=[];definitions=[]
    for base in range(bases.shape[1]):
        original,solver=portfolio_weights(bases[:,base])
        candidates.append(bases[:,base]);weights.append(original)
        definitions.append({'base':base,'driver':-1,'strength':0.,'distance':'identity'})
        for driver in range(drivers.shape[1]):
            for strength in config['strengths']:
                target=(1-strength)*original+strength*drivers[:,driver]
                target=np.maximum(target,0);target/=target.sum(axis=1,keepdims=True)
                for distance in config['matrix_distances']:
                    candidates.append(reconcile(bases[:,base],target,distance));weights.append(target)
                    definitions.append({'base':base,'driver':driver,'strength':strength,'distance':distance})
    return np.asarray(candidates),np.asarray(weights),definitions


def choose_paths(dates,covariances,weights,actual,returns,evaluation_dates,definitions,baseline):
    dates=pd.DatetimeIndex(dates);evaluation_dates=pd.DatetimeIndex(evaluation_dates)
    indices=dates.get_indexer(evaluation_dates)
    if (indices<252).any():raise ValueError('Selectors require252 genuine prior outcomes.')
    n=actual.shape[-1];lower=np.tril_indices(n)
    eu=[];fr=[];ql=[]
    for forecast in covariances:
        delta=forecast-actual
        eu.append(np.linalg.norm(delta[:,lower[0],lower[1]],axis=1))
        fr.append(np.linalg.norm(delta,axis=(1,2)))
        ql.append(np.linalg.slogdet(forecast)[1]-np.linalg.slogdet(actual)[1]+np.trace(np.linalg.solve(forecast,actual),axis1=-2,axis2=-1)-n)
    metrics=np.stack([eu,fr,ql],axis=-1)
    integrated=np.einsum('cti,tij,ctj->ct',weights,actual,weights)
    portfolio=np.einsum('cti,ti->ct',weights,returns)
    names=['Integrated-risk reconciled selector','Return-risk reconciled selector','Combined-risk reconciled selector']
    output={name:[] for name in names};audits=[];months=evaluation_dates.to_period('M')
    for month in months.unique():
        test=np.flatnonzero(months==month);origin=indices[test[0]];past=np.arange(origin-252,origin)
        ratios=metrics[:,past].mean(axis=1)/np.maximum(metrics[baseline,past].mean(axis=0),1e-12)
        iv=integrated[:,past].mean(axis=1)/integrated[baseline,past].mean()
        rv=portfolio[:,past].var(axis=1)/portfolio[baseline,past].var()
        for name in names:
            if name==names[0]:score=np.column_stack([ratios,iv])
            elif name==names[1]:score=np.column_stack([ratios,rv])
            else:score=np.column_stack([ratios,iv,1+.5*(rv-1)])
            best=min(range(len(score)),key=lambda i:(float(score[i].max()),float(score[i].mean()),definitions[i]['strength'],i))
            output[name].append(covariances[best,indices[test]])
            audits.append({'model':name,'month':str(month),'first_forecast':evaluation_dates[test[0]],
                           'last_validation_label':dates[past[-1]],'candidate':best,
                           'worst_ratio':float(score[best].max()),**definitions[best]})
    return {name:np.concatenate(parts) for name,parts in output.items()},pd.DataFrame(audits),metrics,integrated,portfolio
