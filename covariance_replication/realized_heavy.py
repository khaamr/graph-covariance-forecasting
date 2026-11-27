"""Realized-covariance DCC-HEAVY M, Bauwens-Xu equations 16-18 and 43-44.

Fits the realized-covariance process M, not the separate daily-return process H.
Missing reference sessions use the linear conditional-mean multi-step forecast.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize


MODEL = "Realized DCC-HEAVY M"


@njit(cache=True)
def variance_objective(parameters, observations, gaps):
    omega, alpha, beta = parameters
    persistence = alpha + beta
    mean = omega / (1 - persistence)
    mean_derivative = np.array([1 / (1 - persistence), omega / (1 - persistence)**2,
                                omega / (1 - persistence)**2])
    state = mean
    derivative = mean_derivative.copy()
    path = np.empty(len(observations))
    loss = 0.
    gradient = np.zeros(3)
    for t in range(len(observations)):
        if t:
            next_state = omega + alpha * observations[t-1] + beta * state
            next_derivative = beta * derivative
            next_derivative[0] += 1
            next_derivative[1] += observations[t-1]
            next_derivative[2] += state
            power = gaps[t] - 1
            decay = persistence**power
            decay_derivative = power * persistence**(power-1) if power > 0 else 0.
            state = mean + decay * (next_state - mean)
            derivative = mean_derivative + decay * (next_derivative - mean_derivative)
            derivative[1:] += decay_derivative * (next_state - mean)
        if state <= 0 or not np.isfinite(state):
            return np.inf, np.zeros(3), path
        path[t] = state
        loss += np.log(state) + observations[t]/state
        gradient += (1/state-observations[t]/state**2)*derivative
    return loss/len(observations), gradient/len(observations), path


@njit(cache=True)
def correlation_paths(parameters, realized, target, gaps):
    alpha,beta=parameters
    persistence=alpha+beta
    count,n,_=realized.shape
    paths=np.empty_like(realized)
    gradients=np.zeros((2,count,n,n))
    paths[0]=target
    for t in range(1,count):
        power=gaps[t]-1
        decay=persistence**power
        derivative_decay=power*persistence**(power-1) if power>0 else 0.
        for i in range(n):
            for j in range(n):
                next_value=(1-persistence)*target[i,j]+alpha*realized[t-1,i,j]+beta*paths[t-1,i,j]
                paths[t,i,j]=target[i,j]+decay*(next_value-target[i,j])
                next_a=realized[t-1,i,j]-target[i,j]+beta*gradients[0,t-1,i,j]
                next_b=paths[t-1,i,j]-target[i,j]+beta*gradients[1,t-1,i,j]
                gradients[0,t,i,j]=decay*next_a+derivative_decay*(next_value-target[i,j])
                gradients[1,t,i,j]=decay*next_b+derivative_decay*(next_value-target[i,j])
    return paths,gradients


def correlation_objective(parameters,realized,standardized_covariance,target,gaps):
    paths,derivatives=correlation_paths(parameters,realized,target,gaps)
    sign,logdet=np.linalg.slogdet(paths)
    if not np.all(sign>0):
        return np.inf,np.zeros(2)
    precision=np.linalg.inv(paths)
    value=np.mean(logdet+np.einsum('tij,tji->t',precision,standardized_covariance))
    differential=precision-precision@standardized_covariance@precision
    gradient=np.einsum('tij,ptji->p',differential,derivatives)/len(paths)
    return value,gradient


def as_correlation(covariance):
    std=np.sqrt(np.diagonal(covariance,axis1=-2,axis2=-1))
    return covariance/std[..., :,None]/std[...,None,:]


@dataclass(frozen=True)
class HeavyFit:
    variance_parameters: np.ndarray
    last_variance: np.ndarray
    last_realized_variance: np.ndarray
    correlation_parameters: np.ndarray
    target: np.ndarray
    last_correlation: np.ndarray
    last_realized_correlation: np.ndarray
    diagnostics: dict


def fit_heavy(covariance,gaps):
    covariance=np.asarray(covariance,dtype=float)
    gaps=np.asarray(gaps,dtype=np.int64)
    if covariance.ndim!=3 or len(covariance)<50 or len(gaps)!=len(covariance) or (gaps<1).any():
        raise ValueError('HEAVY needs >=50 aligned covariance observations and positive session gaps.')
    np.linalg.cholesky(covariance)
    realized_variance=np.diagonal(covariance,axis1=-2,axis2=-1)
    variance_path=np.empty_like(realized_variance)
    fitted_parameters=[]
    for i in range(covariance.shape[-1]):
        scale=realized_variance[:,i].mean()
        values=np.ascontiguousarray(realized_variance[:,i]/scale)
        objective=lambda p:variance_objective(p,values,gaps)[:2]
        starts=[np.array([1-a-b,a,b]) for a,b in [(.1,.8),(.2,.7),(.05,.9),(.4,.4)]]
        starts.sort(key=lambda p:objective(p)[0])
        fitted=minimize(objective,starts[0],jac=True,method='SLSQP',
                        bounds=[(1e-7,10.),(0,.99998),(0,.99998)],
                        constraints=[{'type':'ineq','fun':lambda p:.99998-p[1:].sum(),
                                      'jac':lambda p:np.array([0.,-1.,-1.])}],
                        options={'maxiter':500,'ftol':1e-10})
        if not fitted.success or not np.isfinite(fitted.fun) or fitted.x[1:].sum()>.99998001:
            raise RuntimeError(f'HEAVY variance optimizer {i}: {fitted.message}')
        variance_path[:,i]=variance_objective(fitted.x,values,gaps)[2]*scale
        params=fitted.x.copy();params[0]*=scale
        fitted_parameters.append(params)
    realized=as_correlation(covariance)
    target=realized.mean(axis=0)
    std=np.sqrt(variance_path)
    normalized=covariance/std[:,:,None]/std[:,None,:]
    objective=lambda p:correlation_objective(p,realized,normalized,target,gaps)
    starts=[np.array([a,p-a]) for a in (.03,.1,.2,.4) for p in (.8,.95,.99) if p>a]
    starts.sort(key=lambda p:objective(p)[0])
    fitted=minimize(objective,starts[0],jac=True,method='SLSQP',bounds=[(0,.99998)]*2,
                    constraints=[{'type':'ineq','fun':lambda p:.99998-p.sum(),
                                  'jac':lambda p:-np.ones(2)}],
                    options={'maxiter':500,'ftol':1e-9})
    if not fitted.success or not np.isfinite(fitted.fun) or fitted.x.sum()>.99998001:
        raise RuntimeError(f'HEAVY correlation optimizer: {fitted.message}')
    path,_=correlation_paths(fitted.x,realized,target,gaps)
    return HeavyFit(np.array(fitted_parameters),variance_path[-1],realized_variance[-1],
                    fitted.x,target,path[-1],realized[-1],
                    {'correlation_alpha':fitted.x[0],'correlation_beta':fitted.x[1],
                     'training_reference_gaps':int(np.count_nonzero(gaps>1)),
                     'correlation_qlike':float(fitted.fun),'optimizer_failures':0})


def advance_heavy(fit,covariance,gaps):
    omega,alpha,beta=fit.variance_parameters.T
    mean=omega/(1-alpha-beta)
    ca,cb=fit.correlation_parameters
    variance,rv=fit.last_variance.copy(),fit.last_realized_variance.copy()
    correlation,rc=fit.last_correlation.copy(),fit.last_realized_correlation.copy()
    output=[]
    for observed,gap in zip(covariance,gaps):
        next_variance=omega+alpha*rv+beta*variance
        variance=mean+(alpha+beta)**(gap-1)*(next_variance-mean)
        next_correlation=(1-ca-cb)*fit.target+ca*rc+cb*correlation
        correlation=fit.target+(ca+cb)**(gap-1)*(next_correlation-fit.target)
        std=np.sqrt(variance)
        output.append(correlation*std[:,None]*std[None,:])
        rv=np.diag(observed).copy()
        rc=as_correlation(observed)
    return np.asarray(output)


def heavy_forecasts(dates,covariance,positions,reference_dates,lookback=1000,progress=None):
    dates,reference=pd.DatetimeIndex(dates),pd.DatetimeIndex(reference_dates)
    positions=np.asarray(positions)
    if not dates.is_unique or not dates.is_monotonic_increasing or not reference.is_unique or not reference.is_monotonic_increasing:
        raise ValueError('Dates and reference must be ordered and unique.')
    clock=reference.get_indexer(dates)
    if (clock<0).any() or len(dates)!=len(covariance) or len(positions)==0 or (np.diff(positions)<=0).any() or positions.min()<lookback or positions.max()>=len(dates):
        raise ValueError('Aligned panel, reference and ordered forecast positions with a full lookback are required.')
    gaps=np.r_[1,np.diff(clock)].astype(np.int64)
    months=dates[positions].to_period('M')
    output=np.empty((len(positions),*covariance.shape[1:]))
    audit=[];parameters=[]
    for month in months.unique():
        rows=np.flatnonzero(months==month);origin,end=positions[rows[0]],positions[rows[-1]]
        fit=fit_heavy(covariance[origin-lookback:origin],gaps[origin-lookback:origin])
        path=advance_heavy(fit,covariance[origin:end+1],gaps[origin:end+1])
        output[rows]=path[positions[rows]-origin]
        audit.append({'month':str(month),'first_forecast':dates[origin],
                      'last_training_label':dates[origin-1],'training_labels':lookback,**fit.diagnostics})
        parameters.extend({'month':str(month),'asset':i,'omega':p[0],'alpha':p[1],'beta':p[2]}
                          for i,p in enumerate(fit.variance_parameters))
        if progress:progress(f'{month}: HEAVY a={fit.correlation_parameters[0]:.5f}, b={fit.correlation_parameters[1]:.5f}')
    np.linalg.cholesky(output)
    return output,pd.DataFrame(audit),pd.DataFrame(parameters)
