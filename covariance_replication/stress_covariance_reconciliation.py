"""Lagged-volatility gates for faster portfolio-consistent covariance forecasts."""
import numpy as np

from covariance_reconciliation import reconcile
from risk_calibration import portfolio_weights


def stress_strength(covariance,positions,rule):
    positions=np.asarray(positions)
    if (positions<22).any() or (positions>=len(covariance)).any():
        raise ValueError('A complete preceding22-observation window is required.')
    if rule.startswith('Fixed'):return np.full(len(positions),float(rule[5:]))
    lower={'Volatility ratio1to3':1.,'Volatility ratio1.5to3':1.5}.get(rule)
    if lower is None:raise ValueError('Unknown stress gate.')
    trace=np.trace(covariance,axis1=-2,axis2=-1)
    prefix=np.r_[0.,np.cumsum(trace)]
    mean=(prefix[positions]-prefix[positions-22])/22
    ratio=trace[positions-1]/mean
    return np.clip(np.log(ratio/lower)/np.log(3/lower),0,1)


def ewma_covariance_weights(covariance,positions,half_life):
    if half_life<=0:raise ValueError('Positive half-life required.')
    if (np.asarray(positions)<1).any():raise ValueError('Previous covariance observations required.')
    decay=np.exp2(-1/half_life);state=covariance[0].copy()
    forecasts=np.empty_like(covariance);forecasts[0]=np.eye(covariance.shape[-1])
    for t in range(1,len(covariance)):
        state=decay*state+(1-decay)*covariance[t-1]
        forecasts[t]=state
    return portfolio_weights(forecasts[positions])


def make_stress_candidates(bases,drivers,panel_covariance,positions,config):
    covariances=[];weights=[];definitions=[]
    gates={rule:stress_strength(panel_covariance,positions,rule) for rule in config['gates']}
    for base in range(bases.shape[1]):
        original,solver=portfolio_weights(bases[:,base])
        covariances.append(bases[:,base]);weights.append(original)
        definitions.append({'base':base,'driver':-1,'gate':'identity','strength':0.,'distance':'identity'})
        for driver in range(drivers.shape[1]):
            for rule,beta in gates.items():
                desired=(1-beta[:,None])*original+beta[:,None]*drivers[:,driver]
                desired=np.maximum(desired,0);desired/=desired.sum(axis=1,keepdims=True)
                for distance in config['matrix_distances']:
                    covariances.append(reconcile(bases[:,base],desired,distance));weights.append(desired)
                    # Tie-break metadata uses a fixed cap, never a future average gate value.
                    cap=float(rule[5:]) if rule.startswith('Fixed') else 1.
                    definitions.append({'base':base,'driver':driver,'gate':rule,'strength':cap,'distance':distance})
    return np.asarray(covariances),np.asarray(weights),definitions
