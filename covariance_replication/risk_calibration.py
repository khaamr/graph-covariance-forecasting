"""Past-only return-risk calibration with genuine prequential model selection."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit
from sklearn.covariance import LedoitWolf

from evaluation import long_only_weights


@njit(cache=True)
def active_set_weights(covariance, tolerance=1e-9):
    """Solve strictly convex simplex QPs; return a duality-gap certificate."""
    count,n,_=covariance.shape
    output=np.zeros((count,n));gaps=np.empty(count);iterations=np.empty(count,dtype=np.int64)
    for t in range(count):
        matrix=covariance[t]/np.trace(covariance[t])*n
        index=np.argmin(np.diag(matrix));active=np.empty(n,dtype=np.int64);active[0]=index
        membership=np.zeros(n,dtype=np.bool_);membership[index]=True
        size=1;w=np.zeros(n);w[index]=1.
        for iteration in range(2000):
            block=np.empty((size,size))
            for i in range(size):
                for j in range(size):block[i,j]=matrix[active[i],active[j]]
            raw=np.linalg.solve(block,np.ones(size));solution=raw/raw.sum()
            if solution.min() < -1e-12:
                step=1.
                for i in range(size):
                    if solution[i]<0:
                        step=min(step,w[active[i]]/(w[active[i]]-solution[i]))
                for i in range(size):w[active[i]]+=step*(solution[i]-w[active[i]])
                next_size=0
                for i in range(size):
                    j=active[i]
                    if w[j]<=1e-12:
                        membership[j]=False;w[j]=0.
                    else:
                        active[next_size]=j;next_size+=1
                size=next_size
                continue
            w[:]=0.
            for i in range(size):w[active[i]]=max(solution[i],0.)
            w/=w.sum()
            gradient=matrix@w;value=np.dot(w,gradient)
            candidate=-1;minimum=np.inf
            for j in range(n):
                if not membership[j] and gradient[j]<minimum:
                    candidate=j;minimum=gradient[j]
            if candidate<0 or minimum>=value-tolerance*value:
                break
            active[size]=candidate;size+=1;membership[candidate]=True
        gradient=matrix@w;value=np.dot(w,gradient)
        gaps[t]=2*max(value-gradient.min(),0.)/value
        output[t]=w;iterations[t]=iteration+1
    return output,gaps,iterations


def portfolio_weights(covariance):
    if not np.isfinite(covariance).all():raise ValueError('Finite covariance required.')
    np.linalg.cholesky(covariance)
    weights,gaps,iterations=active_set_weights(np.ascontiguousarray(covariance))
    failed=(gaps>1e-7)|~np.isfinite(gaps)
    if failed.any():
        weights[failed]=long_only_weights(covariance[failed])
        gradient=(covariance[failed]@weights[failed,...,None])[...,0]
        value=np.sum(weights[failed]*gradient,axis=1)
        gaps[failed]=2*np.maximum(value-gradient.min(axis=1),0)/value
    if not np.isfinite(weights).all() or (weights < -1e-10).any() or not np.allclose(weights.sum(axis=1),1):
        raise RuntimeError('Candidate QP violates simplex feasibility.')
    if not np.isfinite(gaps).all() or gaps.max()>1e-6:
        raise RuntimeError('Candidate QP exceeds its relative duality-gap tolerance.')
    return weights,{'max_relative_gap':float(gaps.max()),'fallbacks':int(failed.sum()),
                    'max_iterations':int(iterations.max())}


def return_target(base,returns):
    std=np.sqrt(np.diagonal(base,axis1=-2,axis2=-1))
    residual=100*(returns-returns.mean(axis=0))/std
    estimate=LedoitWolf(assume_centered=True).fit(residual)
    return estimate.covariance_,float(estimate.shrinkage_)


def calibrate(base,target,strength,kind):
    if strength==0:return base.copy()
    if kind=='diagonal':
        scale=np.sqrt(1-strength+strength*np.diag(target))
        return base*scale[None,:,None]*scale[None,None,:]
    if kind!='full':raise ValueError('Unknown return-risk calibration.')
    std=np.sqrt(np.diagonal(base,axis1=-2,axis2=-1))
    return (1-strength)*base+strength*target[None]*std[:,:,None]*std[:,None,:]


def candidate_definitions(n_bases):
    rows=[]
    for base in range(n_bases):
        rows.append({'base':base,'kind':'none','strength':0.,'history':504})
        for kind in ['diagonal','full']:
            for strength in [.1,.25,.5,1.]:
                for history in [252,504]:
                    rows.append({'base':base,'kind':kind,'strength':strength,'history':history})
    return rows


def metric_arrays(covariance,actual):
    n=actual.shape[-1];lower=np.tril_indices(n);delta=covariance-actual
    eu=np.linalg.norm(delta[:,lower[0],lower[1]],axis=1)
    frob=np.linalg.norm(delta,axis=(1,2))
    sign,logdet=np.linalg.slogdet(covariance)
    if not (sign>0).all():raise ValueError('Positive covariance required.')
    trace=np.trace(np.linalg.solve(covariance,actual),axis1=-2,axis2=-1)
    divergence=logdet-np.linalg.slogdet(actual)[1]+trace-n
    if divergence.min() < -1e-7:raise ValueError('Invalid negative QLIKE divergence.')
    return eu,frob,np.maximum(divergence,0),trace,logdet


@dataclass(frozen=True)
class CandidateBank:
    dates: pd.DatetimeIndex
    original_rows: np.ndarray
    matrices: np.ndarray
    scales: np.ndarray
    definitions: list
    metrics: np.ndarray
    portfolio_returns: np.ndarray
    integrated_risk: np.ndarray
    folds: pd.DataFrame
    solver_audit: pd.DataFrame


def build_candidates(dates,bases,actual,returns,config,progress=None):
    dates=pd.DatetimeIndex(dates)
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('Candidate dates must be ordered and unique.')
    if len(bases)!=len(dates) or len(actual)!=len(dates) or len(returns)!=len(dates):
        raise ValueError('Candidate inputs must align.')
    np.linalg.cholesky(bases);np.linalg.cholesky(actual)
    rows=np.arange(config['minimum_calibration_history'],len(dates))
    definitions=candidate_definitions(bases.shape[1]);count=len(definitions)
    matrices=np.empty((count,len(rows),*actual.shape[1:]))
    scales=np.ones((count,len(rows)))
    audit=[];months=dates[rows].to_period('M')
    for month in months.unique():
        output=np.flatnonzero(months==month);test=rows[output];origin=test[0]
        targets={}
        for j,definition in enumerate(definitions):
            base=definition['base'];history=definition['history'];train=np.arange(max(0,origin-history),origin)
            key=(base,history)
            if key not in targets:targets[key]=return_target(bases[train,base],returns[train])
            target,shrinkage=targets[key]
            predicted=calibrate(bases[test,base],target,definition['strength'],definition['kind'])
            historical=calibrate(bases[train,base],target,definition['strength'],definition['kind'])
            scale=float(np.clip(np.trace(np.linalg.solve(historical,actual[train]),axis1=-2,axis2=-1).mean()/actual.shape[-1],.75,1.5))
            matrices[j,output]=predicted;scales[j,output]=scale
            audit.append({'month':str(month),'case':j,'first_forecast':dates[origin],
                          'last_calibration_label':dates[train[-1]],'training_labels':len(train),
                          'ledoit_wolf_shrinkage':shrinkage,'qlike_scale':scale,**definition})
        if progress:progress(f'{month}: {count} covariance calibration paths')
    # Two scalar recipes per structural case share the same portfolio allocation.
    metrics=np.empty((count*2,len(rows),3));portfolio_returns=np.empty((count,len(rows)))
    integrated=np.empty_like(portfolio_returns);solvers=[]
    for j,definition in enumerate(definitions):
        covariance=matrices[j];eu,fr,q,trace,logdet=metric_arrays(covariance,actual[rows])
        metrics[2*j]=np.column_stack([eu,fr,q])
        scale=scales[j];scaled=covariance*scale[:,None,None]
        se,sf,sq,_,_=metric_arrays(scaled,actual[rows])
        metrics[2*j+1]=np.column_stack([se,sf,sq])
        weights,solver=portfolio_weights(covariance)
        portfolio_returns[j]=np.sum(weights*returns[rows],axis=1)
        integrated[j]=np.einsum('ti,tij,tj->t',weights,actual[rows],weights)
        solvers.append({'case':j,**solver})
        if progress and j%10==0:progress(f'Portfolio paths solved: {j+1}/{count}')
    return CandidateBank(dates[rows],rows,matrices,scales,definitions,metrics,portfolio_returns,
                         integrated,pd.DataFrame(audit),pd.DataFrame(solvers))


def select_paths(bank,evaluation_dates,config):
    evaluation_dates=pd.DatetimeIndex(evaluation_dates)
    rows=bank.dates.get_indexer(evaluation_dates)
    if (rows<config['validation']).any():
        raise ValueError('Each selector forecast needs a full genuinely past validation history.')
    months=evaluation_dates.to_period('M');validation_size=config['validation']
    names=['Prequential realized-risk covariance selector','Prequential return-risk covariance selector',
           'Prequential integrated-risk covariance selector']
    outputs={name:[] for name in names};audit=[]
    for month in months.unique():
        test=np.flatnonzero(months==month);origin=rows[test[0]];validation=np.arange(origin-validation_size,origin)
        baseline_metrics=np.maximum(bank.metrics[0,validation].mean(axis=0),1e-12)
        forecast_ratios=bank.metrics[:,validation].mean(axis=1)/baseline_metrics
        integrated=bank.integrated_risk[:,validation].mean(axis=1)/bank.integrated_risk[0,validation].mean()
        return_variance=bank.portfolio_returns[:,validation].var(axis=1)/max(bank.portfolio_returns[0,validation].var(),1e-12)
        integrated=np.repeat(integrated,2);return_variance=np.repeat(return_variance,2)
        for name in names:
            ratios=forecast_ratios.copy()
            if name==names[0]:ratios=np.column_stack([ratios,integrated,1+.5*(return_variance-1)])
            elif name==names[1]:ratios=np.column_stack([ratios,return_variance])
            else:ratios=np.column_stack([ratios,integrated])
            best=min(range(len(ratios)),key=lambda j:(float(ratios[j].max()),float(ratios[j].mean()),
                       bank.definitions[j//2]['strength'],j%2,j))
            structural,scaled=best//2,best%2
            predictions=bank.matrices[structural,rows[test]].copy()
            if scaled:predictions*=bank.scales[structural,rows[test],None,None]
            outputs[name].append(predictions)
            audit.append({'model':name,'month':str(month),'first_forecast':evaluation_dates[test[0]],
                          'last_validation_label':bank.dates[validation[-1]],'case':structural,
                          'scalar_recipe':'QLIKE' if scaled else 'identity',
                          'worst_validation_ratio':float(ratios[best].max()),
                          'mean_validation_ratio':float(ratios[best].mean()),**bank.definitions[structural]})
    return {name:np.concatenate(parts) for name,parts in outputs.items()},pd.DataFrame(audit)
