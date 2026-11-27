"""Factor-VHAR with matrix-log LASSO/adaLASSO, HAR betas and sector blocks.

Implements equations 6-17 and Appendix A of Alves et al. (2023/2024).
Characteristic factors are external, dated portfolio weights, never PCA proxies.
The authorized benchmark uses an explicitly labelled equal-weight market factor.
"""
from dataclasses import dataclass
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from sklearn.linear_model import lars_path


LASSO_MODEL = "Factor-VHAR LASSO"
ADAPTIVE_MODEL = "Factor-VHAR adaptive LASSO"


def symmetric_function(matrices, function):
    values, vectors = np.linalg.eigh(matrices)
    return (vectors * function(values)[..., None, :]) @ vectors.swapaxes(-1, -2)


def matrix_log(matrices):
    """Log positive-definite blocks, repairing only numerical boundary values."""
    values, vectors = np.linalg.eigh(matrices)
    scale = np.maximum(np.abs(np.trace(matrices, axis1=-2, axis2=-1)) / matrices.shape[-1], 1e-15)
    if (values[..., 0] < -1e-10 * scale).any():
        raise ValueError("A factor/residual block has a materially negative eigenvalue.")
    floor = 1e-10 * scale[..., None]
    repairs = int(np.count_nonzero(values < floor))
    values = np.maximum(values, floor)
    return (vectors * np.log(values)[..., None, :]) @ vectors.swapaxes(-1, -2), repairs


def overlap_lags(values):
    """Published overlapping HAR lags: 1, mean(1:5), mean(1:22)."""
    frame = pd.DataFrame(values)
    return np.stack([frame.shift(1).to_numpy(), frame.rolling(5).mean().shift(1).to_numpy(),
                     frame.rolling(22).mean().shift(1).to_numpy()], axis=-1)


def causal_clean(covariance, minimum_history=22):
    """Four-sigma, >25% screen; average preceding ten nonflagged matrices."""
    covariance = np.asarray(covariance, dtype=float)
    n = covariance.shape[-1]
    lower = np.tril_indices(n)
    entries = covariance[:, lower[0], lower[1]]
    mean = np.zeros(entries.shape[1])
    moment = np.zeros_like(mean)
    output = covariance.copy()
    flagged = np.zeros(len(covariance), dtype=bool)
    accepted = []
    for t, entry in enumerate(entries):
        if t >= minimum_history and len(accepted) >= 10:
            deviation = np.sqrt(moment / max(t - 1, 1))
            extreme = np.abs(entry - mean) > 4 * deviation
            if extreme.mean() > .25:
                flagged[t] = True
                output[t] = covariance[accepted[-10:]].mean(axis=0)
        if not flagged[t]:
            accepted.append(t)
        difference = entry - mean
        mean += difference / (t + 1)
        moment += difference * (entry - mean)
    return output, flagged


def factor_decomposition(covariance, weights, dates=None, known_at=None):
    """Sigma_f=W' Sigma W; B=Sigma_f^-1 W' Sigma; residual=Sigma-B'Sigma_f B."""
    covariance = np.asarray(covariance, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if covariance.ndim != 3 or covariance.shape[1] != covariance.shape[2]:
        raise ValueError("Expected dates by square covariance matrices.")
    if weights.ndim == 2:
        weights = np.broadcast_to(weights, (len(covariance), *weights.shape))
    elif weights.ndim == 3:
        if known_at is None or dates is None:
            raise ValueError("Historical factor weights require dates and known_at timestamps.")
        known_at = pd.DatetimeIndex(known_at)
        dates = pd.DatetimeIndex(dates)
        if len(known_at) != len(covariance) or known_at.hasnans or not (known_at < dates).all():
            raise ValueError("Each historical factor weight must be known before its trading date.")
    if weights.ndim != 3 or weights.shape[:2] != covariance.shape[:2] or not np.isfinite(weights).all():
        raise ValueError("Factor portfolio weights must align with dates and assets.")
    if not np.isfinite(covariance).all():
        raise ValueError("Covariance observations must be finite.")
    cross = weights.swapaxes(-1, -2) @ covariance
    factor = cross @ weights
    try:
        np.linalg.cholesky(factor)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Factor portfolios must be linearly independent.") from exc
    beta = np.linalg.solve(factor, cross)
    residual = covariance - beta.swapaxes(-1, -2) @ factor @ beta
    return factor, beta, (residual + residual.swapaxes(-1, -2)) / 2


def _bic_path(x, y):
    """Exact LASSO path knots; intercept unpenalized, columns standardized."""
    if not x.shape[1] or np.max(np.abs(y)) < 1e-14:
        return np.zeros(x.shape[1]), {"alpha": 0., "active": 0, "kkt_error": 0., "path_warnings": 0}
    with catch_warnings(record=True) as warnings:
        simplefilter("always")
        alphas, _, coefficients = lars_path(x, y, method="lasso", max_iter=500,
                                           alpha_min=0, return_path=True)
    residual = y[:, None] - x @ coefficients
    rss = np.maximum(np.sum(residual**2, axis=0), np.finfo(float).tiny)
    active = np.count_nonzero(np.abs(coefficients) > 1e-10, axis=0)
    bic = len(y) * np.log(rss) + active * np.log(len(y))
    selected = int(np.argmin(bic))
    coefficient = coefficients[:, selected]
    gradient = x.T @ (y - x @ coefficient) / len(y)
    nonzero = np.abs(coefficient) > 1e-10
    error = np.where(nonzero, np.abs(gradient - alphas[selected] * np.sign(coefficient)),
                     np.maximum(np.abs(gradient) - alphas[selected], 0)).max(initial=0)
    if not np.isfinite(bic).all() or error > 1e-5 * max(1., np.max(np.abs(x.T @ y / len(y)))):
        raise RuntimeError("LASSO path failed its KKT accuracy check.")
    return coefficient, {"alpha": float(alphas[selected]), "active": int(active[selected]),
                         "kkt_error": float(error), "path_warnings": len(warnings)}


def equation_forecasts(features, target, train, test, *, return_fitted=False):
    """BIC-selected LASSO and two-stage adaptive LASSO for one equation."""
    x_mean = features[train].mean(axis=0)
    scale = features[train].std(axis=0)
    keep = scale > 1e-12
    scale = np.where(keep, scale, 1.)
    x = (features[train][:, keep] - x_mean[keep]) / scale[keep]
    y_mean = target[train].mean()
    y = target[train] - y_mean
    coefficient, first = _bic_path(x, y)
    test_x = (features[test][:, keep] - x_mean[keep]) / scale[keep]
    prediction = test_x @ coefficient + y_mean
    selected = np.abs(coefficient) > 1e-10
    adaptive_x = x[:, selected] * np.abs(coefficient[selected])
    adaptive_coef, second = _bic_path(adaptive_x, y)
    adaptive = test_x[:, selected] @ (np.abs(coefficient[selected]) * adaptive_coef) + y_mean
    if return_fitted:
        fitted = x @ coefficient + y_mean
        fitted_adaptive = x[:, selected] @ (np.abs(coefficient[selected]) * adaptive_coef) + y_mean
        return prediction, adaptive, first, second, fitted, fitted_adaptive
    return prediction, adaptive, first, second


def own_har_forecast(features, target, train, test, *, return_fitted=False):
    predictions = np.empty((len(test), target.shape[1]))
    fitted = np.empty((len(train), target.shape[1])) if return_fitted else None
    for i in range(target.shape[1]):
        design = np.column_stack([np.ones(len(train)), features[train, i]])
        coefficient = lstsq(design, target[train, i], cond=1e-12, lapack_driver="gelsy")[0]
        predictions[:, i] = np.column_stack([np.ones(len(test)), features[test, i]]) @ coefficient
        if return_fitted:
            fitted[:, i] = design @ coefficient
    return (predictions, fitted) if return_fitted else predictions


def matrix_smearing(log_prediction, residual_vectors):
    """Empirical training-error retransformation; preserves positive definiteness.

    Assumes the paired matrix-log residual distribution is transportable to the
    forecast date. This is an experimental multivariate smearing correction.
    """
    n = len(log_prediction)
    lower = np.tril_indices(n)
    residual = np.zeros((len(residual_vectors), n, n))
    centered = residual_vectors - residual_vectors.mean(axis=0)
    residual[:, lower[0], lower[1]] = centered
    residual[:, lower[1], lower[0]] = centered
    return symmetric_function(log_prediction[None] + residual, np.exp).mean(axis=0)


@dataclass(frozen=True)
class FactorResult:
    predictions: dict[str, np.ndarray]
    audit: pd.DataFrame
    equation_audit: pd.DataFrame
    cleaning: pd.DataFrame


def factor_forecasts(dates, covariance, positions, weights, sectors, reference_dates,
                     *, known_at=None, lookback=1000, progress=None, clean=True,
                     bias_correction=False):
    dates, reference = pd.DatetimeIndex(dates), pd.DatetimeIndex(reference_dates)
    positions = np.asarray(positions)
    if not dates.is_unique or not dates.is_monotonic_increasing or not reference.is_unique or not reference.is_monotonic_increasing:
        raise ValueError("Panel and reference dates must be ordered and unique.")
    if len(positions) == 0 or np.any(np.diff(positions) <= 0) or positions.min() < lookback + 22 or positions.max() >= len(dates):
        raise ValueError("Ordered forecast origins require the full lookback and HAR history.")
    covariance = np.asarray(covariance)
    if len(covariance) != len(dates) or len(sectors) != covariance.shape[-1]:
        raise ValueError("Dates, assets and sector labels must align.")
    np.linalg.cholesky(covariance)
    if clean:
        cleaned, flagged = causal_clean(covariance)
    else:
        cleaned, flagged = covariance.copy(), np.zeros(len(covariance), dtype=bool)
    factor, beta, residual = factor_decomposition(cleaned, weights, dates, known_at)
    n, k = covariance.shape[-1], factor.shape[-1]
    calendar = reference.get_indexer(dates)
    if (calendar < 0).any():
        raise ValueError("All observation dates must occur in the reference calendar.")
    complete = np.zeros(len(dates), dtype=bool)
    complete[22:] = calendar[22:] - calendar[:-22] == 22
    if not complete[positions].all():
        raise ValueError("Forecasts require complete preceding 22-session HAR windows.")
    log_factor, repairs = matrix_log(factor)
    lower = np.tril_indices(k)
    y_factor = log_factor[:, lower[0], lower[1]]
    x_factor = overlap_lags(y_factor).reshape(len(dates), -1)
    y_beta = beta.reshape(len(dates), -1)
    x_beta = overlap_lags(y_beta)
    blocks = []
    for sector in dict.fromkeys(sectors):
        indices = np.flatnonzero(np.asarray(sectors) == sector)
        log_block, block_repairs = matrix_log(residual[:, indices[:, None], indices])
        repairs += block_repairs
        triangle = np.tril_indices(len(indices))
        target = log_block[:, triangle[0], triangle[1]]
        x = np.full((len(dates), len(indices)), np.nan)
        x[1:] = np.diagonal(log_block[:-1], axis1=-2, axis2=-1)
        blocks.append((sector, indices, triangle, x, target))
    output = {name: np.empty((len(positions), n, n)) for name in (LASSO_MODEL, ADAPTIVE_MODEL)}
    audit, equation_rows = [], []
    for row, origin in enumerate(positions):
        train = np.arange(origin - lookback, origin)
        train = train[complete[train]]
        if len(train) < 100:
            raise ValueError("Insufficient complete training labels.")
        test = np.array([origin])
        if bias_correction:
            beta_prediction, fitted_beta = own_har_forecast(x_beta, y_beta, train, test, return_fitted=True)
            beta_prediction = beta_prediction[0].reshape(k, n)
            beta_residual = (y_beta[train] - fitted_beta).reshape(len(train), k, n)
        else:
            beta_prediction = own_har_forecast(x_beta, y_beta, train, test)[0].reshape(k, n)
        factor_predictions = [np.empty(len(lower[0])), np.empty(len(lower[0]))]
        factor_residuals = [np.empty((len(train), len(lower[0]))) for _ in range(2)] if bias_correction else None
        records = []
        for equation in range(y_factor.shape[1]):
            equation_result = equation_forecasts(x_factor, y_factor[:, equation], train, test,
                                                return_fitted=bias_correction)
            first, second, a, b = equation_result[:4]
            if bias_correction:
                for estimator in range(2):
                    factor_residuals[estimator][:, equation] = y_factor[train, equation] - equation_result[4+estimator]
            factor_predictions[0][equation], factor_predictions[1][equation] = first[0], second[0]
            records.extend([dict(component="factor", equation=equation, estimator=label, **record)
                            for label, record in [("lasso", a), ("adaptive", b)]])
        forecast_factor = []
        for estimator, values in enumerate(factor_predictions):
            log = np.zeros((k, k))
            log[lower] = values
            log[(lower[1], lower[0])] = values
            forecast_factor.append(matrix_smearing(log, factor_residuals[estimator]) if bias_correction
                                   else symmetric_function(log, np.exp))
        forecast_residual = [np.zeros((n, n)), np.zeros((n, n))]
        for sector, indices, triangle, x, target in blocks:
            block_predictions = [np.empty(target.shape[1]), np.empty(target.shape[1])]
            block_residuals = [np.empty((len(train), target.shape[1])) for _ in range(2)] if bias_correction else None
            for equation in range(target.shape[1]):
                equation_result = equation_forecasts(x, target[:, equation], train, test,
                                                    return_fitted=bias_correction)
                first, second, a, b = equation_result[:4]
                if bias_correction:
                    for estimator in range(2):
                        block_residuals[estimator][:, equation] = target[train, equation] - equation_result[4+estimator]
                block_predictions[0][equation], block_predictions[1][equation] = first[0], second[0]
                records.extend([dict(component=str(sector), equation=equation, estimator=label, **record)
                                for label, record in [("lasso", a), ("adaptive", b)]])
            for estimator, values in enumerate(block_predictions):
                log = np.zeros((len(indices), len(indices)))
                log[triangle] = values
                log[(triangle[1], triangle[0])] = values
                forecast_residual[estimator][np.ix_(indices, indices)] = (
                    matrix_smearing(log, block_residuals[estimator]) if bias_correction
                    else symmetric_function(log, np.exp))
        for estimator, name in enumerate(output):
            output[name][row] = beta_prediction.T @ forecast_factor[estimator] @ beta_prediction + forecast_residual[estimator]
            if bias_correction:
                output[name][row] += np.einsum('tki,kl,tlj->ij', beta_residual,
                                              forecast_factor[estimator], beta_residual) / len(train)
        audit.append({"date": dates[origin], "last_training_label": dates[train[-1]],
                      "training_labels": len(train), "factor_count": k,
                      "max_kkt_error": max(record["kkt_error"] for record in records),
                      "path_warnings": sum(record["path_warnings"] for record in records),
                      "clean_inputs": clean, "bias_correction": bias_correction,
                      "cleaned_training_labels": int(flagged[train].sum())})
        equation_rows.extend(dict(date=dates[origin], **record) for record in records)
        if progress and (row % 25 == 0 or row == len(positions) - 1):
            progress(f"{dates[origin].date()}: {len(train)} factor-VHAR training labels ({row+1}/{len(positions)})")
    for matrices in output.values():
        if not np.isfinite(matrices).all():
            raise RuntimeError("Factor-VHAR produced a nonfinite prediction.")
        np.linalg.cholesky(matrices)
    cleaning = pd.DataFrame({"date": dates, "flagged": flagged})
    cleaning.attrs["log_eigenvalues_repaired"] = repairs
    return FactorResult(output, pd.DataFrame(audit), pd.DataFrame(equation_rows), cleaning)
