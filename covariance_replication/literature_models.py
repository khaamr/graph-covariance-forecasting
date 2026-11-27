"""Causal ridge volatility forecasts and factor-regularized realized correlations.

Literature-inspired challengers, not reproductions of DCC-NL or factor VHAR.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_data import SECTORS
from models import har_lags


PRIMARY = "Factor-regularized ridge log-HAR"
RIDGE_GRAPH = "Ridge log-HAR + graph correlation"
RIDGE_FACTOR = "Ridge log-HAR + factor correlation"
GRAPH_BLEND = "Graph log-variance + factor blend"
GRAPH_FACTOR = "Graph log-variance + factor correlation"
MULTIVARIATE = "Multivariate ridge log-HAR + factor blend"


@dataclass(frozen=True)
class ChallengerResult:
    predictions: dict[str, np.ndarray]
    audit: pd.DataFrame
    validation: pd.DataFrame


def as_correlation(covariance):
    covariance = np.asarray(covariance, dtype=float)
    diagonal = np.diagonal(covariance, axis1=-2, axis2=-1)
    if not np.isfinite(covariance).all() or (diagonal <= 0).any():
        raise ValueError("Finite covariance with positive diagonal is required.")
    std = np.sqrt(diagonal)
    return covariance / (std[..., :, None] * std[..., None, :])


def factor_correlation(fast, vectors, residual_floor=0.05):
    """Project onto a fixed factor basis and add positive idiosyncratic variance."""
    if not 0 < residual_floor < 1:
        raise ValueError("Residual floor must lie strictly between zero and one.")
    common = vectors @ (vectors.T @ fast @ vectors) @ vectors.T
    residual = np.maximum(1 - np.diag(common), residual_floor)
    return as_correlation((common + common.T) / 2 + np.diag(residual))


def ridge_fit(features, target, penalty):
    """Asset-specific standardized ridge; penalty multiplies mean squared loss."""
    if penalty <= 0:
        raise ValueError("Ridge penalty must be positive.")
    x_mean = features.mean(axis=0)
    x_scale = features.std(axis=0)
    x_scale = np.where(x_scale > 1e-12, x_scale, 1)
    y_mean = target.mean(axis=0)
    x = (features - x_mean) / x_scale
    y = target - y_mean
    gram = np.einsum("tnf,tng->nfg", x, x) / len(x)
    rhs = np.einsum("tnf,tn->nf", x, y) / len(x)
    coefficient = np.linalg.solve(gram + penalty * np.eye(x.shape[-1]), rhs)
    residual = y - np.einsum("tnf,nf->tn", x, coefficient)
    # Duan-style training-residual smearing: E(exp(error)), per asset.
    smear = np.log(np.exp(residual).mean(axis=0))
    return x_mean, x_scale, y_mean, coefficient, smear


def ridge_predict(features, fit):
    x_mean, x_scale, y_mean, coefficient, smear = fit
    return np.einsum("tnf,nf->tn", (features - x_mean) / x_scale, coefficient) + y_mean + smear


def select_ridge(features, target, train, test, penalties, validation_size=126):
    """One chronological validation block wholly before the forecast month."""
    if len(train) <= validation_size + 50:
        raise ValueError("Insufficient past observations for ridge validation.")
    fitting, validation = train[:-validation_size], train[-validation_size:]
    records = []
    for penalty in penalties:
        fit = ridge_fit(features[fitting], target[fitting], penalty)
        predicted_log = ridge_predict(features[validation], fit)
        difference = predicted_log - target[validation]
        # Nonnegative univariate Gaussian QLIKE divergence, scale invariant.
        score = np.mean(difference + np.exp(-difference) - 1)
        records.append({"penalty": penalty, "validation_qlike_divergence": float(score)})
    selected = min(records, key=lambda item: (item["validation_qlike_divergence"], -item["penalty"]))
    fit = ridge_fit(features[train], target[train], selected["penalty"])
    return np.exp(ridge_predict(features[test], fit)), selected["penalty"], records


def volatility_features(log_variance, symbols):
    """Own plus leave-one-out market and sector disjoint HAR predictors."""
    n = len(symbols)
    if n < 2 or len(log_variance.shape) != 2 or log_variance.shape[1] != n:
        raise ValueError("Expected multiple assets aligned with the symbol list.")
    market = (log_variance.sum(axis=1, keepdims=True) - log_variance) / (n - 1)
    sector = np.empty_like(log_variance)
    for i, symbol in enumerate(symbols):
        peers = [j for j, other in enumerate(symbols) if j != i and SECTORS[other] == SECTORS[symbol]]
        sector[:, i] = log_variance[:, peers].mean(axis=1) if peers else market[:, i]
    grouped = np.concatenate([har_lags(x) for x in (log_variance, market, sector)], axis=-1)
    all_lags = har_lags(log_variance).reshape(len(log_variance), -1)
    multivariate = np.broadcast_to(all_lags[:, None, :], (len(log_variance), n, all_lags.shape[-1]))
    return grouped, multivariate


def challenger_forecasts(panel_dates, covariance, symbols, reference_dates,
                         forecast_dates, positions, graph_forecasts, config, progress=None):
    panel_dates = pd.DatetimeIndex(panel_dates)
    reference_dates = pd.DatetimeIndex(reference_dates)
    forecast_dates = pd.DatetimeIndex(forecast_dates)
    if any(not x.is_unique or not x.is_monotonic_increasing for x in (panel_dates, reference_dates, forecast_dates)):
        raise ValueError("Dates must be ordered and unique.")
    positions = np.asarray(positions)
    if len(positions) != len(forecast_dates) or not panel_dates[positions].equals(forecast_dates):
        raise ValueError("Forecast positions must align with panel dates.")
    if graph_forecasts.shape != (len(positions), *covariance.shape[1:]):
        raise ValueError("Graph forecasts must align with forecast positions and assets.")
    np.linalg.cholesky(covariance)
    np.linalg.cholesky(graph_forecasts)
    calendar_position = reference_dates.get_indexer(panel_dates)
    if (calendar_position < 0).any():
        raise ValueError("Every panel observation must occur in the reference calendar.")
    complete = np.zeros(len(panel_dates), dtype=bool)
    complete[22:] = calendar_position[22:] - calendar_position[:-22] == 22
    if not complete[positions].all():
        raise ValueError("Every challenger forecast requires a complete preceding HAR window.")
    log_variance = np.log(np.diagonal(covariance, axis1=-2, axis2=-1))
    grouped, multivariate = volatility_features(log_variance, symbols)
    realized_correlation = as_correlation(covariance)
    graph_correlation = as_correlation(graph_forecasts)
    graph_variance = np.diagonal(graph_forecasts, axis1=-2, axis2=-1)
    names = (PRIMARY, RIDGE_GRAPH, RIDGE_FACTOR, GRAPH_BLEND, GRAPH_FACTOR, MULTIVARIATE)
    outputs = {name: np.empty_like(graph_forecasts) for name in names}
    audit, validation_records = [], []
    months = forecast_dates.to_period("M")
    for month in months.unique():
        rows = np.flatnonzero(months == month)
        origin = positions[rows[0]]
        train = np.arange(max(22, origin - config["history"]), origin)
        train = train[complete[train]]
        test = positions[rows]
        variances = {}
        penalties = {}
        for label, features in [("grouped", grouped), ("multivariate", multivariate)]:
            variances[label], penalties[label], records = select_ridge(
                features, log_variance, train, test, config["ridge_grid"])
            validation_records.extend(dict(month=str(month), features=label, **record) for record in records)
        # Loadings are estimated once per month, using only earlier matrices.
        long_run = realized_correlation[max(0, origin - 252):origin].mean(axis=0)
        _, vectors = np.linalg.eigh(long_run)
        vectors = vectors[:, -min(3, len(symbols)):]
        for local, row in enumerate(rows):
            position = positions[row]
            ages = calendar_position[position] - calendar_position[:position]
            recent = np.flatnonzero((ages >= 1) & (ages <= 126))
            weights = np.exp2(-ages[recent] / 20)
            fast = np.einsum("t,tij->ij", weights / weights.sum(), realized_correlation[recent])
            factor = factor_correlation(fast, vectors)
            blend = (factor + graph_correlation[row]) / 2
            for name, variance, correlation in [
                (PRIMARY, variances["grouped"][local], blend),
                (RIDGE_GRAPH, variances["grouped"][local], graph_correlation[row]),
                (RIDGE_FACTOR, variances["grouped"][local], factor),
                (GRAPH_BLEND, graph_variance[row], blend),
                (GRAPH_FACTOR, graph_variance[row], factor),
                (MULTIVARIATE, variances["multivariate"][local], blend),
            ]:
                std = np.sqrt(variance)
                outputs[name][row] = correlation * std[:, None] * std[None, :]
        audit.append({"month": str(month), "first_forecast": forecast_dates[rows[0]],
                      "training_labels": len(train), "last_training_label": panel_dates[train[-1]],
                      "last_fitting_label": panel_dates[train[-127]],
                      "first_validation_label": panel_dates[train[-126]],
                      "last_factor_loading_input": panel_dates[origin - 1],
                      "grouped_penalty": penalties["grouped"],
                      "multivariate_penalty": penalties["multivariate"], "forecasts": len(rows)})
        if progress:
            progress(f"{month}: {len(train)} past labels, penalties {penalties}")
    for matrices in outputs.values():
        if not np.isfinite(matrices).all():
            raise ValueError("Nonfinite challenger prediction.")
        np.linalg.cholesky(matrices)
    return ChallengerResult(outputs, pd.DataFrame(audit), pd.DataFrame(validation_records))
