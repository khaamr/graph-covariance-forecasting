"""Standalone stress-reconciled log/ridge covariance forecast.

Candidate159 from the recorded exploratory stress study. The caller supplies
two causal covariance forecasts and realized observations strictly before the
requested trading date. Selection limitations are documented with the results.
"""
import numpy as np
import pandas as pd

from covariance_reconciliation import reconcile
from risk_calibration import portfolio_weights


def balanced_covariance_forecast(log_graph_forecast,ridge_graph_forecast,
                                 observed_covariance,history_dates,forecast_date):
    log_graph_forecast=np.asarray(log_graph_forecast,dtype=float)
    ridge_graph_forecast=np.asarray(ridge_graph_forecast,dtype=float)
    observed_covariance=np.asarray(observed_covariance,dtype=float)
    dates=pd.DatetimeIndex(history_dates);forecast_date=pd.Timestamp(forecast_date)
    if log_graph_forecast.ndim!=2 or log_graph_forecast.shape[0]!=log_graph_forecast.shape[1] or ridge_graph_forecast.shape!=log_graph_forecast.shape:
        raise ValueError('Two aligned square covariance forecasts are required.')
    if observed_covariance.ndim!=3 or observed_covariance.shape[1:]!=log_graph_forecast.shape or len(observed_covariance)!=len(dates) or len(dates)<22:
        raise ValueError('At least22 aligned realized covariance observations are required.')
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing or not (dates<forecast_date).all():
        raise ValueError('History dates must be ordered, unique and strictly before the forecast.')
    if not all(np.isfinite(value).all() for value in [log_graph_forecast,ridge_graph_forecast,observed_covariance]):
        raise ValueError('Finite covariance matrices are required.')
    np.linalg.cholesky(log_graph_forecast);np.linalg.cholesky(ridge_graph_forecast);np.linalg.cholesky(observed_covariance)
    base=(log_graph_forecast+ridge_graph_forecast)/2
    base_weights,_=portfolio_weights(base[None])
    decay=np.exp2(-1/5);state=observed_covariance[0].copy()
    for observed in observed_covariance[1:]:state=decay*state+(1-decay)*observed
    fast_weights,_=portfolio_weights(state[None])
    trace=np.trace(observed_covariance,axis1=-2,axis2=-1)
    ratio=trace[-1]/trace[-22:].mean();strength=float(np.clip(np.log(ratio)/np.log(3.),0,1))
    desired=(1-strength)*base_weights+strength*fast_weights
    desired=np.maximum(desired,0);desired/=desired.sum(axis=1,keepdims=True)
    return reconcile(base[None],desired,'Gaussian KL')[0]
